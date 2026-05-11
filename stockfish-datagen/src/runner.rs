//! Tier orchestration: spawn N workers, each driving one Stockfish, drain
//! their progress messages, collect results.
//!
//! Partitioning is by **shard id**, not by worker. Each tier has
//! `total_shards = ceil(n_games / shard_size_games)` deterministically-
//! numbered shards; workers pull shard ids from a shared `AtomicU64`
//! counter (`fetch_add(1)`). The per-game seed depends only on the
//! *global* game index within the tier, so `n_workers` is operational
//! only — changing it never changes any game's content, only which
//! worker thread happens to generate it.
//!
//! Multi-pod cooperation: each pod is given a half-open `shard_range`
//! (default `0..total_shards`). The atomic counter starts at the range's
//! `start` and stops at `end`; disjoint pods on different machines
//! produce disjoint shard files and can sync to the same HF dataset
//! folder without name collisions. Per-pod sentinels
//! (`_tier_state-s<A>-s<B>.json`, `_manifest-s<A>-s<B>.json`) keep their
//! resume / completion records separate; a `datagen_reconcile_tier.py`
//! helper merges them into a unified manifest after all pods finish.

use std::collections::BTreeSet;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, anyhow};
use crossbeam_channel::Sender;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::affinity;
use crate::config::{RunConfig, TierConfig};
use crate::game::play_game;
use crate::resume::{
    ShardRange, TierManifest, TierState, detect_resume, enforce_n_games_invariant,
};
use crate::seed;
use crate::shard::{GameRow, ShardWriter};
use crate::stockfish::StockfishProcess;

/// Tier-execution scope: which tiers to run and which shard ids each tier
/// should restrict its work to.
#[derive(Debug, Clone, Default)]
pub struct RunScope {
    /// Subset of tier indices to run; `None` means "all tiers in cfg".
    pub tiers: Option<Vec<usize>>,
    /// Per-tier shard-id range; `None` means "full range `0..total_shards`".
    /// Same range applies to every tier in this run (the multi-pod model
    /// is "this pod owns shard ids `[A, B)` across all of its assigned tiers").
    pub shard_range: Option<ShardRange>,
}

impl RunScope {
    pub fn includes_tier(&self, tier_index: usize) -> bool {
        match &self.tiers {
            Some(t) => t.contains(&tier_index),
            None => true,
        }
    }

    /// Effective shard-id range for `tier` under this scope.
    pub fn effective_shard_range(&self, cfg: &RunConfig, tier: &TierConfig) -> Range<u64> {
        let full = 0..cfg.total_shards(tier);
        match self.shard_range {
            Some(r) => {
                // Clamp to the tier's actual shard count so a multi-pod
                // run that doesn't recompute ranges per-tier stays
                // bounded if one tier has fewer shards than others.
                let start = r.start.min(full.end);
                let end = r.end.min(full.end);
                start..end
            }
            None => full,
        }
    }
}

/// Results from a successful tier run.
#[derive(Debug, Clone)]
pub struct TierResult {
    pub n_games_written: u64,
    pub shards: Vec<PathBuf>,
}

/// Messages workers send to the main thread.
#[derive(Debug)]
pub enum Progress {
    /// Periodic update so the user knows things are alive.
    Heartbeat {
        worker_idx: u32,
        shards_done: u64,
        games_done: u64,
    },
    /// Worker finished cleanly. Final partial shard already closed.
    Done {
        worker_idx: u32,
        shards_done: u64,
        games_written: u64,
        shards: Vec<PathBuf>,
    },
}

/// Run one tier end-to-end under the given `scope`. Spawns `cfg.n_workers`
/// worker threads, each pulling shard ids from a shared atomic counter
/// bounded by `scope.effective_shard_range(cfg, tier)`. Returns summary
/// stats for THIS pod's slice (not the full tier).
///
/// Resume behavior:
/// - If a manifest for this pod's `shard_range` exists with matching
///   `config_fingerprint`, the tier is skipped entirely (returns the
///   recorded stats).
/// - Otherwise existing shard files in `[shard_range.start, shard_range.end)`
///   are scanned and the atomic counter starts at `shard_range.start`;
///   workers skip done shards and regenerate boundary-rewrite shards.
pub fn run_tier(
    cfg: &RunConfig,
    tier_index: usize,
    scope: &RunScope,
) -> anyhow::Result<TierResult> {
    let tier = &cfg.tiers[tier_index];
    let tier_dir = cfg.output_dir.join(&tier.name);
    std::fs::create_dir_all(&tier_dir)
        .with_context(|| format!("creating tier dir {}", tier_dir.display()))?;

    // Cross-pod n_games safety: refuse if any existing `_tier_state*.json`
    // in the dir declares an `n_games` larger than the current config.
    // Runs BEFORE the manifest-skip check so a smaller-n_games pod's own
    // (correct-for-it) manifest can't mask the fact that another pod has
    // committed to producing more games for this tier. See
    // `resume::enforce_n_games_invariant` for the rationale.
    enforce_n_games_invariant(&tier_dir, tier.n_games)?;

    let fingerprint = cfg.tier_fingerprint(tier_index);
    let shard_range = scope.effective_shard_range(cfg, tier);
    // The PRESENCE of `--shard-id-range` (not whether the clamped range
    // happens to equal the full tier) is the explicit signal that this
    // pod is part of a multi-pod cooperation. Always write per-pod
    // sentinels in that case; otherwise a pod that passes an over-wide
    // range (e.g. `0:99999` on a tier with only 1000 shards) would clamp
    // to full → collide on canonical sentinels with a sibling pod that
    // didn't pass the flag. Filename uses the CLAMPED values so reconcile
    // sees the actual work claim (the raw value could lie about coverage).
    let pod_range: Option<ShardRange> = scope.shard_range.as_ref().map(|_| {
        ShardRange { start: shard_range.start, end: shard_range.end }
    });
    // Warn loudly when raw range extends past the tier; without this an
    // operator who overspec'd gets a silent clamp and no signal.
    if let Some(raw) = scope.shard_range.as_ref() {
        if raw.end > cfg.total_shards(tier) {
            eprintln!(
                "[{}] WARNING: --shard-id-range {}..{} extends past tier total of {} shards; clamped to {}..{}",
                tier.name, raw.start, raw.end, cfg.total_shards(tier),
                shard_range.start, shard_range.end,
            );
        }
    }
    let expected_shard_count = shard_range.end.saturating_sub(shard_range.start);

    // Expected-row-count closure: same logic the runner uses when writing
    // a fresh shard, so resume's "done vs boundary-rewrite" classification
    // stays consistent. Computed before the manifest-skip check so the
    // skip can compare game counts (Codex P1, round 2 review).
    let shard_size = cfg.shard_size_games as u64;
    let n_games = tier.n_games;
    let expected_n_rows = |sid: u64| -> u64 {
        let start = sid * shard_size;
        let end = ((sid + 1) * shard_size).min(n_games);
        end.saturating_sub(start)
    };
    let expected_total_games: u64 = (shard_range.start..shard_range.end)
        .map(expected_n_rows)
        .sum();

    // Skip if this pod's slice is already complete. "Complete" requires:
    //   1. fingerprint match (catches config drift),
    //   2. shard count match (catches n_games growth that adds new shards),
    //   3. n_games_written >= expected_total_games (catches n_games growth
    //      that stays WITHIN the last shard — e.g. 900 -> 950 with
    //      shard_size_games=1000 keeps total_shards at 1, so check (2) is
    //      a no-op and we'd silently re-use the old manifest without
    //      regenerating the boundary shard, leaving the dataset short).
    if let Some(manifest) = TierManifest::load(&tier_dir, pod_range.as_ref())? {
        if manifest.config_fingerprint != fingerprint {
            return Err(anyhow!(
                "[{}] manifest exists but tier fingerprint differs (manifest {} vs current {}); \
                 either restore the original config for this tier or delete {}",
                tier.name,
                manifest.config_fingerprint,
                fingerprint,
                TierManifest::path(&tier_dir, pod_range.as_ref()).display(),
            ));
        }
        if manifest.shards.len() as u64 == expected_shard_count
            && manifest.n_games_written >= expected_total_games
        {
            eprintln!(
                "[{}] already complete for range {:?} ({} games, {} shards) — skipping",
                tier.name, shard_range, manifest.n_games_written, manifest.shards.len(),
            );
            return Ok(TierResult {
                n_games_written: manifest.n_games_written,
                shards: manifest
                    .shards
                    .iter()
                    .map(|s| tier_dir.join(s))
                    .collect(),
            });
        }
        // Same fingerprint but shard count or game count don't agree ⇒
        // `tier.n_games` changed since last completion (likely grew). Fall
        // through to the normal run path; it will regenerate the boundary
        // shard and write the new tail. The stale manifest gets
        // overwritten at end of run.
        eprintln!(
            "[{}] manifest covers {} shards / {} games, current config expects {} / {} — regenerating delta",
            tier.name, manifest.shards.len(), manifest.n_games_written,
            expected_shard_count, expected_total_games,
        );
    }

    // Scope detect_resume to this pod's range. Other pods' shards may be
    // visible on disk (multi-pod runs share the directory via the HF
    // primer); scoping inside the function avoids spurious oversize
    // errors on shards we don't own.
    let resume_state = detect_resume(&tier_dir, shard_range.clone(), expected_n_rows)?;
    let done_shards: BTreeSet<u64> = resume_state.done_shards.clone();
    let boundary_rewrites: Vec<(u64, PathBuf)> = resume_state.boundary_rewrites.clone();

    // Tier-state sentinel: validates that any shards on disk in this
    // pod's range were generated under the SAME config we're about to
    // run. Without this, an interrupted run + a config tweak before
    // retry would silently mix old + new bytes into the same tier output.
    let any_existing_in_range = !done_shards.is_empty() || !boundary_rewrites.is_empty();
    match TierState::load(&tier_dir, pod_range.as_ref())? {
        Some(state) if state.config_fingerprint == fingerprint => {
            // Matches — safe to resume.
        }
        Some(state) => {
            return Err(anyhow!(
                "[{}] tier state fingerprint mismatch (started under {}, current {}); \
                 this tier was partly generated under a different config. Either restore \
                 the original config or delete the tier dir ({}) to regenerate from scratch.",
                tier.name,
                state.config_fingerprint,
                fingerprint,
                tier_dir.display(),
            ));
        }
        None if any_existing_in_range => {
            return Err(anyhow!(
                "[{}] shards exist in {} but no tier state file is present; cannot \
                 verify they belong to the current config. Either move them aside or \
                 delete the tier dir to regenerate from scratch.",
                tier.name,
                tier_dir.display(),
            ));
        }
        None => {
            TierState {
                config_fingerprint: fingerprint.clone(),
                started_at: now_iso8601(),
                n_games: tier.n_games,
                shard_range: pod_range,
            }
            .save(&tier_dir)
            .context("writing tier state sentinel")?;
        }
    }

    let tier_seed = seed::tier_seed(cfg.master_seed, &tier.name);

    let total_resume_games: u64 = done_shards.iter().map(|s| expected_n_rows(*s)).sum();
    if total_resume_games > 0 {
        eprintln!(
            "[{}] resuming: {} games already on disk across {} done shard(s){}",
            tier.name,
            total_resume_games,
            done_shards.len(),
            if boundary_rewrites.is_empty() {
                String::new()
            } else {
                format!(" + {} boundary-rewrite shard(s)", boundary_rewrites.len())
            },
        );
    }

    let pod_shards = shard_range.end.saturating_sub(shard_range.start);
    eprintln!(
        "[{}] starting: {} games across {} workers (shard range {:?}, {} shard(s) in this pod)",
        tier.name,
        tier.n_games,
        cfg.n_workers,
        shard_range,
        pod_shards,
    );

    let cfg = Arc::new(cfg.clone());
    let tier = Arc::new(tier.clone());
    let tier_dir = Arc::new(tier_dir);
    let done_shards = Arc::new(done_shards);
    let boundary_rewrites = Arc::new(boundary_rewrites);

    let counter = Arc::new(AtomicU64::new(shard_range.start));
    let counter_end = shard_range.end;

    let (tx, rx) = crossbeam_channel::unbounded::<Progress>();

    let drain_handle = std::thread::Builder::new()
        .name("sfd-progress".into())
        .spawn({
            let tier_name = tier.name.clone();
            move || -> Vec<Progress> {
                let mut completions = Vec::new();
                for msg in rx {
                    match &msg {
                        Progress::Heartbeat { worker_idx, shards_done, games_done } => {
                            eprintln!(
                                "  [{tier_name} w{worker_idx:>2}] {shards_done} shard(s), {games_done} games"
                            );
                        }
                        Progress::Done { worker_idx, shards_done, games_written, .. } => {
                            eprintln!(
                                "  [{tier_name} w{worker_idx:>2}] DONE: {games_written} games across {shards_done} shard(s)"
                            );
                            completions.push(msg);
                            continue;
                        }
                    }
                }
                completions
            }
        })
        .context("spawning progress drain")?;

    let mut handles = Vec::with_capacity(cfg.n_workers as usize);
    let mut spawn_err: Option<anyhow::Error> = None;

    for worker_idx in 0..cfg.n_workers {
        let cfg = Arc::clone(&cfg);
        let tier = Arc::clone(&tier);
        let tier_dir = Arc::clone(&tier_dir);
        let done_shards = Arc::clone(&done_shards);
        let boundary_rewrites = Arc::clone(&boundary_rewrites);
        let counter = Arc::clone(&counter);
        let tx = tx.clone();
        let spawn_result = std::thread::Builder::new()
            .name(format!("sfd-w{worker_idx}"))
            .spawn(move || {
                run_worker(
                    &cfg, &tier, &tier_dir, worker_idx, tier_seed,
                    &counter, counter_end, &done_shards, &boundary_rewrites, tx,
                )
            });
        match spawn_result {
            Ok(h) => handles.push((worker_idx, h)),
            Err(e) => {
                spawn_err = Some(anyhow!(e).context(format!("spawning worker {worker_idx}")));
                break;
            }
        }
    }
    drop(tx);

    let mut first_err: Option<anyhow::Error> = spawn_err;
    for (worker_idx, h) in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let e = e.context(format!("worker {worker_idx}"));
                eprintln!("worker {worker_idx} failed: {e:#}");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
            Err(_) => {
                let e = anyhow!("worker {worker_idx} panicked");
                eprintln!("{e:#}");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }

    let drain_result = drain_handle.join();
    if let Some(e) = first_err {
        return Err(e);
    }
    let _completions = drain_result.map_err(|_| anyhow!("progress drain panicked"))?;

    // Re-scan the dir for this pod's shards (resumed + newly written +
    // boundary rewrites). Use the same expected-rows fn we built earlier
    // so the manifest's shard list reflects only validly-sized files.
    // Scope to this pod's range — `detect_resume` filters out shards
    // owned by other cooperating pods (which may also live in the same
    // shared tier dir).
    let final_state = detect_resume(tier_dir.as_path(), shard_range.clone(), expected_n_rows)?;
    let mut all_shards: Vec<PathBuf> = final_state
        .done_shards
        .iter()
        .copied()
        .map(|sid| crate::shard::shard_final_path(tier_dir.as_path(), sid, expected_n_rows(sid)))
        .collect();
    all_shards.sort();

    // Accounting: every shard in this pod's range should now be `done`.
    // If anything was left in `boundary_rewrites` or beyond, that's a bug.
    let expected_shard_count = pod_shards as usize;
    if all_shards.len() != expected_shard_count {
        return Err(anyhow!(
            "[{}] post-run shard count mismatch: have {} shards on disk in range {:?}, expected {}",
            tier.name, all_shards.len(), shard_range, expected_shard_count,
        ));
    }
    let total_written: u64 = (shard_range.start..shard_range.end).map(expected_n_rows).sum();

    eprintln!(
        "[{}] complete: {} games across {} shards in range {:?}",
        tier.name, total_written, all_shards.len(), shard_range,
    );

    let manifest = TierManifest {
        tier_name: tier.name.clone(),
        config_fingerprint: fingerprint,
        n_games_written: total_written,
        shards: all_shards
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .collect(),
        completed_at: now_iso8601(),
        shard_range: pod_range,
    };
    manifest.save(tier_dir.as_path()).context("writing tier manifest")?;

    Ok(TierResult {
        n_games_written: total_written,
        shards: all_shards,
    })
}

/// Saturating cast from f32 to i16, used to pack `evallegal` /
/// multipv-derived scores into the parquet `LegalMoveEval` struct.
fn f32_to_i16_clamped(x: f32) -> i16 {
    x.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// Pack a worker's per-ply candidate list into the parquet-bound
/// `LegalMoveEval` representation. Lives at module scope (not as a
/// closure inside the per-game hot loop) so the function isn't
/// re-instantiated per game and the reader can find it without
/// hunting through `run_worker`.
fn pack_candidates(
    plies: Vec<Vec<crate::stockfish::Candidate>>,
) -> Vec<Vec<crate::shard::LegalMoveEval>> {
    plies
        .into_iter()
        .map(|cands| {
            cands
                .into_iter()
                .map(|c| crate::shard::LegalMoveEval {
                    move_idx: chess_engine::vocab::uci_to_action(&c.uci)
                        .expect("Stockfish-emitted UCI must be in our action vocab")
                        as i16,
                    score_cp: f32_to_i16_clamped(c.score_cp),
                    score_eval_v: c.score_eval_v.map(f32_to_i16_clamped),
                    score_psqt: c.score_psqt.map(f32_to_i16_clamped),
                    score_positional: c.score_positional.map(f32_to_i16_clamped),
                })
                .collect()
        })
        .collect()
}

fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let epoch_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_epoch_secs(epoch_secs)
}

fn format_epoch_secs(epoch_secs: u64) -> String {
    let days_since_epoch = (epoch_secs / 86_400) as i64;
    let secs_in_day = epoch_secs % 86_400;
    let (h, m, s) = (
        (secs_in_day / 3600) as u32,
        ((secs_in_day / 60) % 60) as u32,
        (secs_in_day % 60) as u32,
    );
    let (y, mo, d) = days_to_ymd(days_since_epoch);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z / 146097 } else { (z - 146096) / 146097 };
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    cfg: &RunConfig,
    tier: &TierConfig,
    tier_dir: &std::path::Path,
    worker_idx: u32,
    tier_seed: u64,
    counter: &AtomicU64,
    counter_end: u64,
    done_shards: &BTreeSet<u64>,
    boundary_rewrites: &[(u64, PathBuf)],
    tx: Sender<Progress>,
) -> anyhow::Result<()> {
    // Pin BEFORE spawning Stockfish so the child inherits affinity via
    // fork() and runs NNUE init / hash-table allocation on the target
    // core's L1/L2. `pick_core` mod-rotates `worker_idx` over the
    // available logical cores, so the worker count being unrelated to
    // the partitioning model has no effect here.
    let core = affinity::pick_core(worker_idx);
    if let Some(c) = core {
        affinity::pin_thread_to(c, worker_idx);
    }

    let budget = if tier.searchless {
        crate::stockfish::GoBudget::EvalLegal
    } else {
        crate::stockfish::GoBudget::Nodes(
            tier.nodes.expect("validated: search-mode tier has nodes"),
        )
    };
    // Per-tier hash_mb override (falls back to top-level default if absent).
    // The effective value is part of the tier fingerprint, so resume only
    // accepts shards produced under the same setting.
    let hash_mb = cfg.effective_hash_mb(tier);
    let mut sf = StockfishProcess::spawn(
        &cfg.stockfish_path,
        &cfg.stockfish_version,
        hash_mb,
        budget,
    )
    .with_context(|| format!("spawning stockfish for worker {worker_idx}"))?;
    if let Some(c) = core {
        affinity::pin_child_to(sf.child_pid(), c, worker_idx);
    }
    if let Some(net) = tier.net_selection {
        sf.set_net_selection(net)
            .with_context(|| format!("worker {worker_idx}: setting NetSelection={:?}", net))?;
    }
    let stockfish_id_name = sf.id_name.clone();
    let stockfish_version: Arc<str> = Arc::from(stockfish_id_name.as_str());

    let shard_size = cfg.shard_size_games as u64;
    let n_games = tier.n_games;
    let mut shards_done = 0u64;
    let mut total_written = 0u64;
    let mut shards = Vec::new();
    // Quick lookup for boundary rewrites: shard_id -> existing-file-to-delete.
    let boundary_paths: std::collections::BTreeMap<u64, PathBuf> =
        boundary_rewrites.iter().cloned().collect();

    loop {
        let shard_id = counter.fetch_add(1, Ordering::Relaxed);
        if shard_id >= counter_end {
            break;
        }
        // Skip shards that are already complete on disk. Boundary
        // rewrites fall through and get regenerated.
        if done_shards.contains(&shard_id) {
            continue;
        }
        let start = shard_id * shard_size;
        let end = ((shard_id + 1) * shard_size).min(n_games);
        if end <= start {
            // Pathological: shard_id beyond n_games. Shouldn't happen
            // given total_shards math but defend in depth.
            continue;
        }

        let mut writer = ShardWriter::create(tier_dir.to_path_buf(), shard_id)?;

        for global_game_index in start..end {
            let game_seed = seed::game_seed(tier_seed, global_game_index);
            let mut rng = ChaCha8Rng::seed_from_u64(game_seed);

            let played = play_game(&mut sf, &mut rng, tier, cfg.max_ply)
                .with_context(|| format!(
                    "playing game {global_game_index} (seed {game_seed})"
                ))?;

            if played.uci_moves.is_empty() {
                return Err(anyhow!(
                    "worker {worker_idx} game {global_game_index} (seed {game_seed}): \
                     play_game returned zero moves — likely terminal-check bug"
                ));
            }

            let refs: Vec<&str> = played.uci_moves.iter().map(|s| s.as_str()).collect();
            let (tokens, san) = chess_engine::uci::uci_to_tokens_and_san(&refs);
            if tokens.len() != played.uci_moves.len() {
                let bad = played.uci_moves.get(tokens.len()).cloned().unwrap_or_default();
                return Err(anyhow!(
                    "worker {worker_idx} game {global_game_index} (seed {game_seed}): \
                     engine rejected move {} of {}: {bad:?}",
                    tokens.len(),
                    played.uci_moves.len(),
                ));
            }
            if let Some(c) = &played.per_ply_candidates {
                anyhow::ensure!(
                    c.len() == played.uci_moves.len(),
                    "worker {worker_idx} game {global_game_index} (seed {game_seed}): \
                     per_ply_candidates len {} != uci_moves len {}",
                    c.len(), played.uci_moves.len(),
                );
            }
            if let Some(c) = &played.per_ply_static_candidates {
                anyhow::ensure!(
                    c.len() == played.uci_moves.len(),
                    "worker {worker_idx} game {global_game_index} (seed {game_seed}): \
                     per_ply_static_candidates len {} != uci_moves len {}",
                    c.len(), played.uci_moves.len(),
                );
            }
            let n = tokens.len();

            let legal_move_evals = played.per_ply_candidates.map(pack_candidates);
            let static_legal_move_evals = played.per_ply_static_candidates.map(pack_candidates);

            let row = GameRow {
                tokens: tokens.into_iter().map(|t| t as i16).collect(),
                san,
                uci: played.uci_moves,
                game_length: n as u16,
                outcome_token: played.outcome.token(),
                result: played.outcome.result_str().into(),
                nodes: tier.nodes.map(|n| n as i32),
                multi_pv: tier.multi_pv.map(|n| n as i32),
                opening_multi_pv: tier.opening_multi_pv.map(|n| n as i32),
                opening_plies: tier.opening_plies.map(|n| n as i32),
                sample_plies: tier.sample_plies.map(|n| n as i32),
                sample_score: tier.sample_score.map(|s| match s {
                    crate::config::SampleScore::Cp => "cp".to_string(),
                    crate::config::SampleScore::V => "v".to_string(),
                }),
                net_selection: tier.net_selection.map(|n| n.as_uci_str().to_string()),
                temperature: tier.temperature,
                global_game_index,
                game_seed,
                // Cheap refcount bump; the underlying string is shared
                // across every game this worker writes.
                stockfish_version: Arc::clone(&stockfish_version),
                legal_move_evals,
                static_legal_move_evals,
            };
            writer.append(&row);
            total_written += 1;

            if total_written % 500 == 0 {
                let _ = tx.send(Progress::Heartbeat {
                    worker_idx,
                    shards_done,
                    games_done: total_written,
                });
            }
        }

        let path = writer.close()?;
        // Boundary-rewrite cleanup: the old (truncated) file had a
        // smaller `r<...>` suffix than the new one, so it's a different
        // path. Delete it now that the new file is durably renamed in.
        // Log failures rather than swallowing — leaving the old file
        // behind would let another pod's `detect_resume` see two files
        // for the same shard_id (and pick the larger one, but the small
        // one then sits as an orphan until manual cleanup).
        if let Some(old) = boundary_paths.get(&shard_id) {
            if old != &path {
                if let Err(e) = std::fs::remove_file(old) {
                    eprintln!(
                        "[w{worker_idx}] boundary-rewrite: failed to delete old shard \
                         {} after writing {}: {e}",
                        old.display(), path.display(),
                    );
                }
            }
        }
        shards.push(path);
        shards_done += 1;
    }

    sf.shutdown();

    let _ = tx.send(Progress::Done {
        worker_idx,
        shards_done,
        games_written: total_written,
        shards,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TierConfig;
    use tempfile::tempdir;

    fn stockfish_path() -> std::path::PathBuf {
        if let Ok(p) = std::env::var("STOCKFISH_PATH") {
            return p.into();
        }
        let default = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("bin/stockfish");
        assert!(
            default.exists(),
            "stockfish binary not found at {} — set STOCKFISH_PATH or install one",
            default.display(),
        );
        default
    }

    fn smoke_config(sf_path: std::path::PathBuf, dir: std::path::PathBuf) -> RunConfig {
        RunConfig {
            stockfish_path: sf_path,
            stockfish_version: "Stockfish".into(),
            output_dir: dir,
            master_seed: 7,
            n_workers: 2,
            max_ply: 512,
            stockfish_hash_mb: 16,
            shard_size_games: 8,
            tiers: vec![TierConfig {
                name: "smoke".into(),
                n_games: 16,
                temperature: 1.0,
                searchless: false,
                store_legal_move_evals: false,
                sample_score: None,
                net_selection: None,
                nodes: Some(1),
                multi_pv: Some(5),
                opening_multi_pv: Some(20),
                opening_plies: Some(1),
                sample_plies: Some(12),
                stockfish_hash_mb: None,
            }],
        }
    }

    #[test]
    fn run_scope_includes_tier_default() {
        let scope = RunScope::default();
        assert!(scope.includes_tier(0));
        assert!(scope.includes_tier(99));
    }

    #[test]
    fn run_scope_filters_tiers() {
        let scope = RunScope { tiers: Some(vec![0, 2]), shard_range: None };
        assert!(scope.includes_tier(0));
        assert!(!scope.includes_tier(1));
        assert!(scope.includes_tier(2));
    }

    #[test]
    fn run_scope_clamps_shard_range_to_tier() {
        let cfg = smoke_config("/tmp/sf".into(), "/tmp/out".into());
        // Tier has 16 games / shard_size 8 = 2 shards.
        let scope = RunScope {
            tiers: None,
            shard_range: Some(ShardRange { start: 1, end: 100 }),
        };
        let r = scope.effective_shard_range(&cfg, &cfg.tiers[0]);
        assert_eq!(r, 1..2);
    }

    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_run_tier_smoke() {
        let sf_path = stockfish_path();
        let dir = tempdir().unwrap();
        let cfg = smoke_config(sf_path, dir.path().to_path_buf());
        let scope = RunScope::default();
        let result = run_tier(&cfg, 0, &scope).unwrap();
        assert_eq!(result.n_games_written, 16);
        // 16 games / shard_size 8 = exactly 2 shards.
        assert_eq!(result.shards.len(), 2);
        for shard in &result.shards {
            assert!(shard.exists(), "shard {} missing", shard.display());
            assert!(shard.extension().unwrap() == "parquet");
        }
    }

    /// Resume regression: delete one shard, re-run, that shard should be
    /// regenerated and the other one untouched.
    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_resume_regenerates_missing_shard() {
        let sf_path = stockfish_path();
        let dir = tempdir().unwrap();
        let cfg = smoke_config(sf_path, dir.path().to_path_buf());
        let scope = RunScope::default();

        let r1 = run_tier(&cfg, 0, &scope).unwrap();
        assert_eq!(r1.n_games_written, 16);

        let tier_dir = cfg.output_dir.join(&cfg.tiers[0].name);
        std::fs::remove_file(tier_dir.join("_manifest.json")).unwrap();
        // Delete shard 1, keep shard 0.
        let s0_bytes_before = std::fs::read(&r1.shards[0]).unwrap();
        std::fs::remove_file(&r1.shards[1]).unwrap();

        let r2 = run_tier(&cfg, 0, &scope).unwrap();
        assert_eq!(r2.n_games_written, 16);
        assert_eq!(r2.shards.len(), 2);
        // Shard 0 must be byte-identical.
        let s0_bytes_after = std::fs::read(&r1.shards[0]).unwrap();
        assert_eq!(s0_bytes_before, s0_bytes_after, "untouched shard should be byte-identical");
    }

    /// Tier fingerprint scoping regression: changing an UNRELATED tier
    /// must NOT invalidate this tier's manifest.
    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_unrelated_tier_change_doesnt_invalidate() {
        let sf_path = stockfish_path();
        let dir = tempdir().unwrap();
        let mut cfg = smoke_config(sf_path, dir.path().to_path_buf());
        cfg.tiers.push(TierConfig {
            name: "second".into(),
            n_games: 8,
            temperature: 1.0,
            searchless: false,
            store_legal_move_evals: false,
            sample_score: None,
            net_selection: None,
            nodes: Some(1),
            multi_pv: Some(5),
            opening_multi_pv: Some(20),
            opening_plies: Some(1),
            sample_plies: Some(12),
            stockfish_hash_mb: None,
        });
        let scope = RunScope::default();

        let r1 = run_tier(&cfg, 0, &scope).unwrap();
        assert_eq!(r1.n_games_written, 16);

        // Modify tier 1's config (dataset-affecting field). Tier 0's
        // manifest must still be honored on re-run.
        cfg.tiers[1].temperature = 0.5;
        let r2 = run_tier(&cfg, 0, &scope).unwrap();
        assert_eq!(r2.n_games_written, 16, "tier 0 should be skipped via manifest");
    }

    /// Codex P1 round-2 regression: growing `tier.n_games` such that
    /// `total_shards` is UNCHANGED (the growth stays within the last
    /// shard) must still regenerate the boundary shard. Without the
    /// `n_games_written` check in the manifest-skip path, the second
    /// run would see "same shard count" and silently re-use the smaller
    /// manifest, leaving the dataset short by the growth delta. We pin
    /// this with starting `n_games=14, shard_size=8` (shards `[0..8)`
    /// full + `[8..14)` partial = 2 shards) and growing to `n_games=16`
    /// (shards `[0..8)` + `[8..16)` full = still 2 shards).
    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_grow_within_last_shard_regenerates_boundary() {
        let sf_path = stockfish_path();
        let dir = tempdir().unwrap();
        let mut cfg = smoke_config(sf_path, dir.path().to_path_buf());
        cfg.tiers[0].n_games = 14;
        let scope = RunScope::default();
        let r1 = run_tier(&cfg, 0, &scope).unwrap();
        assert_eq!(r1.n_games_written, 14);
        assert_eq!(r1.shards.len(), 2, "14 games / shard_size 8 = 2 shards (8 + 6)");

        // Snapshot shard 0 — its rows depend only on tier_seed + global
        // game index, so growth must not change them.
        let s0_bytes = std::fs::read(&r1.shards[0]).unwrap();

        // Grow within the last shard: 14 → 16. Total shards stays 2;
        // shard 1 grows from 6 to 8 rows. The manifest-skip check must
        // detect the n_games mismatch and fall through to regenerate.
        cfg.tiers[0].n_games = 16;
        let r2 = run_tier(&cfg, 0, &scope).unwrap();
        assert_eq!(r2.n_games_written, 16, "must not silently truncate to 14");
        assert_eq!(r2.shards.len(), 2);

        let s0_after = std::fs::read(&r1.shards[0]).unwrap();
        assert_eq!(s0_bytes, s0_after, "shard 0 must be byte-identical (global-index seeding)");
    }

    /// Extension safety: growing n_games adds new shards but doesn't
    /// invalidate existing ones. Boundary shard with a partial row count
    /// gets regenerated to the full row count; new shards beyond the
    /// previous total are produced.
    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_grow_n_games_extends_cleanly() {
        let sf_path = stockfish_path();
        let dir = tempdir().unwrap();
        let mut cfg = smoke_config(sf_path, dir.path().to_path_buf());
        // First run: 16 games / shard_size 8 = 2 shards, no partial last.
        let scope = RunScope::default();
        let r1 = run_tier(&cfg, 0, &scope).unwrap();
        assert_eq!(r1.shards.len(), 2);

        // Snapshot byte contents of existing shards.
        let snapshot: Vec<(PathBuf, Vec<u8>)> = r1.shards.iter()
            .map(|p| (p.clone(), std::fs::read(p).unwrap()))
            .collect();

        // Grow n_games to 30. New total = 4 shards (last has 6 rows).
        // Delete the manifest so resume re-runs.
        let tier_dir = cfg.output_dir.join(&cfg.tiers[0].name);
        std::fs::remove_file(tier_dir.join("_manifest.json")).unwrap();
        cfg.tiers[0].n_games = 30;

        let r2 = run_tier(&cfg, 0, &scope).unwrap();
        assert_eq!(r2.n_games_written, 30);
        assert_eq!(r2.shards.len(), 4);

        // Original shards 0 and 1 (full 8 rows each) must be byte-identical:
        // their content depends only on tier_seed + global_game_index, both
        // unchanged.
        for (orig_path, orig_bytes) in &snapshot {
            let now = std::fs::read(orig_path).unwrap();
            assert_eq!(orig_bytes, &now, "shard {} should be byte-identical after n_games growth", orig_path.display());
        }
    }
}
