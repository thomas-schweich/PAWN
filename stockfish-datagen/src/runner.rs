//! Tier orchestration: spawn N workers, each driving one Stockfish, drain
//! their progress messages, collect results.
//!
//! Workers are independent threads communicating with the main thread via
//! a single mpsc channel. Each worker owns one parquet shard at a time;
//! shards are rotated when they reach `shard_size_games`. On worker
//! completion the final partial shard (if any) is closed and renamed.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use crossbeam_channel::Sender;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::config::{RunConfig, TierConfig};
use crate::game::play_game;
use crate::resume::{TierManifest, TierState, detect_resume};
use crate::seed;
use crate::shard::{GameRow, ShardWriter, shard_path};
use crate::stockfish::StockfishProcess;

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
        worker_id: u32,
        games_done: u64,
    },
    /// Worker finished cleanly. Last shard already closed.
    Done {
        worker_id: u32,
        games_written: u64,
        shards: Vec<PathBuf>,
    },
}

/// Run one tier end-to-end. Spawns `cfg.n_workers` worker threads, each
/// generating its share of `tier.n_games`. Returns summary stats.
///
/// Resume behavior:
/// - If a manifest exists with matching `config_fingerprint`, the tier is
///   skipped entirely (returns the manifest's recorded stats).
/// - Otherwise existing shard files are scanned and each worker resumes
///   from `(games_done, next_chunk_idx)` derived from disk.
pub fn run_tier(
    cfg: &RunConfig,
    tier_index: usize,
) -> anyhow::Result<TierResult> {
    let tier = &cfg.tiers[tier_index];
    let tier_dir = cfg.output_dir.join(&tier.name);
    std::fs::create_dir_all(&tier_dir)
        .with_context(|| format!("creating tier dir {}", tier_dir.display()))?;

    // Per-tier fingerprint: covers the relevant inputs only (this tier's
    // config + tier_index + master_seed + stockfish_version + shard size).
    // Adding or modifying *other* tiers in the run config does NOT
    // invalidate prior tiers' manifests.
    let fingerprint = cfg.tier_fingerprint(tier_index);

    // Skip if already complete.
    if let Some(manifest) = TierManifest::load(&tier_dir)? {
        if manifest.config_fingerprint == fingerprint {
            eprintln!(
                "[{}] already complete ({} games, {} shards) — skipping",
                tier.name, manifest.n_games_written, manifest.shards.len(),
            );
            return Ok(TierResult {
                n_games_written: manifest.n_games_written,
                shards: manifest
                    .shards
                    .iter()
                    .map(|s| tier_dir.join(s))
                    .collect(),
            });
        } else {
            return Err(anyhow!(
                "[{}] manifest exists but tier fingerprint differs (manifest {} vs current {}); \
                 either restore the original config for this tier or delete {}",
                tier.name,
                manifest.config_fingerprint,
                fingerprint,
                TierManifest::path(&tier_dir).display(),
            ));
        }
    }

    let resume_states = detect_resume(&tier_dir, cfg.n_workers)?;

    // Tier-state sentinel: validates that any shards on disk were
    // generated under the SAME config we're about to run. Without this,
    // an interrupted run + a config tweak before retry would silently
    // mix old + new bytes into the same tier output.
    match TierState::load(&tier_dir)? {
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
        None if !resume_states.is_empty() => {
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
            }
            .save(&tier_dir)
            .context("writing tier state sentinel")?;
        }
    }

    let tier_seed = seed::tier_seed(cfg.master_seed, tier_index);
    let split = cfg.games_per_worker(tier);

    // detect_resume already filters stale workers (worker_id >= n_workers),
    // so resume_states represents only the current partition.
    let total_resume_done: u64 = resume_states.values().map(|s| s.games_done).sum();
    if total_resume_done > 0 {
        eprintln!(
            "[{}] resuming: {} games already on disk across {} worker(s)",
            tier.name,
            total_resume_done,
            resume_states.len(),
        );
    }

    eprintln!(
        "[{}] starting: {} games across {} workers ({} max per worker)",
        tier.name,
        tier.n_games,
        cfg.n_workers,
        split.iter().copied().max().unwrap_or(0),
    );

    let cfg = Arc::new(cfg.clone());
    let tier = Arc::new(tier.clone());
    let tier_dir = Arc::new(tier_dir);

    let (tx, rx) = crossbeam_channel::unbounded::<Progress>();

    // Spawn the progress drain BEFORE any workers. If the drain spawn
    // fails, no workers exist yet to leak. (Worker spawn failure after
    // this point is handled by waiting for already-spawned workers below.)
    let drain_handle = std::thread::Builder::new()
        .name("sfd-progress".into())
        .spawn({
            let tier_name = tier.name.clone();
            move || -> Vec<Progress> {
                let mut completions = Vec::new();
                for msg in rx {
                    match &msg {
                        Progress::Heartbeat { worker_id, games_done } => {
                            eprintln!(
                                "  [{tier_name} worker {worker_id:>2}] {games_done:>7} done",
                            );
                        }
                        Progress::Done { worker_id, games_written, shards } => {
                            eprintln!(
                                "  [{tier_name} worker {worker_id:>2}] DONE: {games_written} written, {} shards",
                                shards.len(),
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

    for worker_id in 0..cfg.n_workers {
        let target = split[worker_id as usize];
        let resume = resume_states
            .get(&worker_id)
            .copied()
            .unwrap_or(crate::resume::WorkerResumeState { games_done: 0, next_chunk_idx: 0 });
        if resume.games_done >= target {
            // Worker already finished its share — skip.
            continue;
        }
        let worker_seed = seed::worker_seed(tier_seed, worker_id);
        let cfg = Arc::clone(&cfg);
        let tier = Arc::clone(&tier);
        let tier_dir = Arc::clone(&tier_dir);
        let tx = tx.clone();
        let start_index = resume.games_done;
        let start_chunk = resume.next_chunk_idx;
        let spawn_result = std::thread::Builder::new()
            .name(format!("sfd-w{worker_id}"))
            .spawn(move || {
                run_worker(
                    &cfg, &tier, &tier_dir, worker_id, worker_seed,
                    start_index, start_chunk, target, tx,
                )
            });
        match spawn_result {
            Ok(h) => handles.push((worker_id, h)),
            Err(e) => {
                spawn_err = Some(anyhow!(e).context(format!("spawning worker {worker_id}")));
                break;
            }
        }
    }
    // Drop the original sender so the channel closes once all worker
    // clones are dropped (i.e. when all workers exit). Important to do
    // this even on spawn-error path so the drain thread terminates.
    drop(tx);

    // Wait for all workers, surface the first error if any panicked or
    // returned Err.
    let mut first_err: Option<anyhow::Error> = spawn_err;
    for (worker_id, h) in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let e = e.context(format!("worker {worker_id}"));
                eprintln!("worker {worker_id} failed: {e:#}");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
            Err(_) => {
                let e = anyhow!("worker {worker_id} panicked");
                eprintln!("{e:#}");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }

    // Wait for the drain to finish even on the error path so we don't
    // leak the thread. Suppress drain-panic if we already have a real
    // worker error to surface — the drain panic is almost always a
    // downstream symptom of the worker error.
    let drain_result = drain_handle.join();
    if let Some(e) = first_err {
        return Err(e);
    }
    let completions = drain_result.map_err(|_| anyhow!("progress drain panicked"))?;

    // Accounting invariant: total_resume_done is the count of games
    // already on disk at the start of this run (filtered to in-scope
    // workers). Workers that were already complete are skipped before
    // spawning and never send Progress::Done; only newly-written games
    // increment via Done messages below. So:
    //   total_written = (games already on disk) + (newly written games)
    //                 = total games across the in-scope partition.
    let mut total_written = total_resume_done;
    let mut all_shards: Vec<PathBuf> = Vec::new();
    // Dir-scan picks up both resumed shards and shards just written this
    // run. Filter via the same filename parser used by `detect_resume`
    // so we (a) ignore unrelated `.parquet` files a downstream tool might
    // drop in the dir, and (b) exclude stale shards from worker IDs that
    // are no longer in scope (n_workers reduced between runs). The
    // exclusion keeps `manifest.shards` consistent with `n_games_written`
    // — both reflect only the current partition.
    for entry in std::fs::read_dir(tier_dir.as_path())? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let Some((worker_id, _chunk)) = crate::resume::parse_shard_filename(&name_str)
        else {
            continue;
        };
        if worker_id < cfg.n_workers {
            all_shards.push(entry.path());
        }
    }
    for msg in completions {
        if let Progress::Done { games_written, .. } = msg {
            total_written += games_written;
        }
    }
    all_shards.sort();

    eprintln!(
        "[{}] complete: {} written, {} shards",
        tier.name,
        total_written,
        all_shards.len(),
    );

    // Write the manifest as the very last step — its presence is the
    // signal that the tier is done. The shard list is sorted and stored
    // as filenames relative to the tier dir.
    let manifest = TierManifest {
        tier_name: tier.name.clone(),
        config_fingerprint: fingerprint,
        n_games_written: total_written,
        shards: all_shards
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .collect(),
        completed_at: now_iso8601(),
    };
    manifest.save(tier_dir.as_path()).context("writing tier manifest")?;

    Ok(TierResult {
        n_games_written: total_written,
        shards: all_shards,
    })
}

fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let epoch_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // RFC3339 in UTC, second precision. Manual formatter to avoid pulling
    // in `chrono` for one timestamp.
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

/// Convert days-since-1970 into (year, month, day). Pulled from the
/// classic algorithm; correct for the Gregorian calendar over our range.
fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = if z >= 0 { z / 146097 } else { (z - 146096) / 146097 };
    let doe = (z - era * 146097) as u64; // [0, 146096]
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
    worker_id: u32,
    worker_seed: u64,
    start_index: u64,
    start_chunk: u32,
    target: u64,
    tx: Sender<Progress>,
) -> anyhow::Result<()> {
    let mut sf = StockfishProcess::spawn(
        &expand_tilde(&cfg.stockfish_path),
        &cfg.stockfish_version,
        cfg.stockfish_hash_mb,
        tier.nodes,
    )
    .with_context(|| format!("spawning stockfish for worker {worker_id}"))?;
    let stockfish_id_name = sf.id_name.clone();

    // On resume we always start a *new* shard at start_chunk — the prior
    // run's last shard (if partial) was already closed and counted in
    // start_index. We can't append to a closed parquet, so the next file
    // gets a fresh chunk index.
    // Hoisted out of the per-game loop: same value every row, so cloning
    // the Arc<str> per row is one pointer copy instead of a String alloc.
    let stockfish_version: Arc<str> = Arc::from(stockfish_id_name.as_str());

    let shard_size = cfg.shard_size_games as u64;
    let mut current_chunk = start_chunk;
    let mut games_in_shard = 0u64;
    let mut writer: Option<ShardWriter> = None;
    let mut total_written = 0u64;
    let mut shards = Vec::new();

    for game_index in start_index..target {
        let game_seed = seed::game_seed(worker_seed, game_index);
        let mut rng = ChaCha8Rng::seed_from_u64(game_seed);

        let played = play_game(&mut sf, &mut rng, tier, cfg.max_ply)
            .with_context(|| format!("playing game {game_index} (seed {game_seed})"))?;

        // play_game must produce at least one move — the starting position
        // is never terminal. An empty result indicates a bug in either the
        // pre-move terminal check or play_game itself. Hard-error rather
        // than silently dropping, so dropped games never desync game_index
        // from the row count (which would break resume).
        if played.uci_moves.is_empty() {
            return Err(anyhow!(
                "worker {worker_id} game {game_index} (seed {game_seed}): \
                 play_game returned zero moves — likely terminal-check bug"
            ));
        }

        // Re-tokenize via the canonical encoder. Stockfish should never
        // give us an unparseable move list — if it does, hard-error so the
        // run aborts deterministically and resume picks up from the failed
        // index without skipping it.
        let refs: Vec<&str> = played.uci_moves.iter().map(|s| s.as_str()).collect();
        let (tokens, san) = chess_engine::uci::uci_to_tokens_and_san(&refs);
        if tokens.len() != played.uci_moves.len() {
            let bad = played.uci_moves.get(tokens.len()).cloned().unwrap_or_default();
            return Err(anyhow!(
                "worker {worker_id} game {game_index} (seed {game_seed}): \
                 engine rejected move {} of {}: {bad:?}",
                tokens.len(),
                played.uci_moves.len(),
            ));
        }
        let n = tokens.len();

        let row = GameRow {
            tokens: tokens.into_iter().map(|t| t as i16).collect(),
            san,
            uci: played.uci_moves,
            game_length: n as u16,
            outcome_token: played.outcome.token(),
            result: played.outcome.result_str().into(),
            nodes: tier.nodes as i32,
            multi_pv: tier.multi_pv as i32,
            opening_multi_pv: tier.opening_multi_pv as i32,
            opening_plies: tier.opening_plies as i32,
            sample_plies: tier.sample_plies as i32,
            temperature: tier.temperature,
            worker_id: worker_id as i16,
            game_seed,
            stockfish_version: stockfish_version.to_string(),
        };

        if writer.is_none() {
            writer = Some(ShardWriter::create(shard_path(tier_dir, worker_id, current_chunk))?);
        }
        writer.as_mut().unwrap().append(&row);
        games_in_shard += 1;
        total_written += 1;

        if games_in_shard >= shard_size {
            let path = writer.take().unwrap().close()?;
            shards.push(path);
            current_chunk += 1;
            games_in_shard = 0;
        }

        if (game_index + 1 - start_index) % 500 == 0 {
            let _ = tx.send(Progress::Heartbeat {
                worker_id,
                games_done: game_index + 1 - start_index,
            });
        }
    }

    // Flush any partial trailing shard.
    if let Some(w) = writer.take() {
        if w.n_rows() > 0 {
            let path = w.close()?;
            shards.push(path);
        }
    }

    // shutdown is best-effort; the Drop impl on StockfishProcess is the
    // safety net for early-error returns above.
    sf.shutdown();

    let _ = tx.send(Progress::Done {
        worker_id,
        games_written: total_written,
        shards,
    });
    Ok(())
}

/// Tilde-expand a path. Only handles `~/...` (the common case); doesn't
/// resolve `~user/...` etc. Anything else is returned as-is.
fn expand_tilde(p: &std::path::Path) -> PathBuf {
    if let Ok(s) = p.strip_prefix("~") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(s);
        }
    }
    p.to_path_buf()
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
                nodes: 1,
                n_games: 16,
                multi_pv: 5,
                opening_multi_pv: 20,
                opening_plies: 1,
                sample_plies: 12,
                temperature: 1.0,
            }],
        }
    }

    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_run_tier_smoke() {
        let sf_path = stockfish_path();
        let dir = tempdir().unwrap();
        let cfg = smoke_config(sf_path, dir.path().to_path_buf());
        let result = run_tier(&cfg, 0).unwrap();
        // No drops are tolerated anymore, so written must equal target.
        assert_eq!(result.n_games_written, 16);
        // 2 workers × 8 games each = exactly 1 shard per worker, so 2 total.
        assert_eq!(result.shards.len(), 2);
        for shard in &result.shards {
            assert!(shard.exists(), "shard {} missing", shard.display());
            assert!(shard.extension().unwrap() == "parquet");
        }
    }

    /// Resume regression: pre-populate one worker's shard, then run.
    /// That worker should be skipped entirely; only the other one should
    /// generate, and the final manifest should reflect the combined total.
    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_resume_skips_completed_worker() {
        let sf_path = stockfish_path();
        let dir = tempdir().unwrap();
        let cfg = smoke_config(sf_path, dir.path().to_path_buf());

        // First run — full 16 games across 2 workers.
        let r1 = run_tier(&cfg, 0).unwrap();
        assert_eq!(r1.n_games_written, 16);
        let shard0_paths_before: Vec<_> = r1.shards.clone();

        // Delete the manifest and one worker's shard. Resume should
        // re-run only that worker.
        let tier_dir = cfg.output_dir.join(&cfg.tiers[0].name);
        std::fs::remove_file(tier_dir.join("_manifest.json")).unwrap();
        let w1_shard = tier_dir.join("shard-w001-c0000.parquet");
        let w0_shard = tier_dir.join("shard-w000-c0000.parquet");
        let w0_bytes_before = std::fs::read(&w0_shard).unwrap();
        std::fs::remove_file(&w1_shard).unwrap();

        let r2 = run_tier(&cfg, 0).unwrap();
        assert_eq!(r2.n_games_written, 16, "resumed total must equal target");
        assert!(w1_shard.exists(), "resumed worker should have re-created its shard");
        // Worker 0's shard must be byte-identical: it was skipped, not regenerated.
        let w0_bytes_after = std::fs::read(&w0_shard).unwrap();
        assert_eq!(w0_bytes_before, w0_bytes_after, "completed worker's shard should be untouched");
        // Shard count unchanged — same 2 shards.
        assert_eq!(r2.shards.len(), shard0_paths_before.len());
    }

    /// Full-resume edge case: ALL workers are already complete on disk,
    /// the manifest is missing (e.g. crash after the last shard but
    /// before manifest write), but `_tier_state.json` is present. The
    /// rerun should write a manifest with `n_games_written == target`
    /// without spawning any workers.
    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_resume_all_workers_done_no_manifest() {
        let sf_path = stockfish_path();
        let dir = tempdir().unwrap();
        let cfg = smoke_config(sf_path, dir.path().to_path_buf());

        // First run: full completion.
        let r1 = run_tier(&cfg, 0).unwrap();
        assert_eq!(r1.n_games_written, 16);

        // Delete the manifest, leave shards + tier_state intact.
        let tier_dir = cfg.output_dir.join(&cfg.tiers[0].name);
        std::fs::remove_file(tier_dir.join("_manifest.json")).unwrap();
        assert!(tier_dir.join("_tier_state.json").exists());

        // Snapshot byte-equality of every shard before re-running so we
        // can prove no worker actually ran.
        let snapshot: Vec<(PathBuf, Vec<u8>)> = std::fs::read_dir(&tier_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "parquet"))
            .map(|e| (e.path(), std::fs::read(e.path()).unwrap()))
            .collect();
        assert!(snapshot.len() >= 2);

        let r2 = run_tier(&cfg, 0).unwrap();
        assert_eq!(r2.n_games_written, 16, "all-resumed total must equal target");
        assert_eq!(r2.shards.len(), snapshot.len());

        for (path, before) in &snapshot {
            let after = std::fs::read(path).unwrap();
            assert_eq!(*before, after, "shard {} must be byte-identical", path.display());
        }
        assert!(tier_dir.join("_manifest.json").exists());
    }

    /// Tier-state sentinel: if a user deletes the manifest AND changes
    /// a fingerprint-relevant field (here, master_seed) while shards from
    /// the prior config still exist on disk, the run must abort rather
    /// than silently mixing old + new bytes into the same tier output.
    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_tier_state_catches_config_drift_after_manifest_delete() {
        let sf_path = stockfish_path();
        let dir = tempdir().unwrap();
        let cfg_a = smoke_config(sf_path.clone(), dir.path().to_path_buf());

        // First run: completes cleanly, writes manifest + tier_state.
        run_tier(&cfg_a, 0).unwrap();
        let tier_dir = cfg_a.output_dir.join(&cfg_a.tiers[0].name);
        assert!(tier_dir.join("_manifest.json").exists());
        assert!(tier_dir.join("_tier_state.json").exists());

        // User deletes the manifest, changes master_seed, reruns.
        std::fs::remove_file(tier_dir.join("_manifest.json")).unwrap();
        let mut cfg_b = smoke_config(sf_path, dir.path().to_path_buf());
        cfg_b.master_seed = cfg_a.master_seed + 1;

        let err = run_tier(&cfg_b, 0).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("tier state fingerprint mismatch"),
            "expected tier-state mismatch error, got: {msg}",
        );
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
            nodes: 1,
            n_games: 8,
            multi_pv: 5,
            opening_multi_pv: 20,
            opening_plies: 1,
            sample_plies: 12,
            temperature: 1.0,
        });

        // Run only tier 0.
        let r1 = run_tier(&cfg, 0).unwrap();
        assert_eq!(r1.n_games_written, 16);

        // Modify tier 1's config (n_games up). Tier 0's manifest must
        // still be honored on re-run.
        cfg.tiers[1].n_games = 999;
        let r2 = run_tier(&cfg, 0).unwrap();
        assert_eq!(r2.n_games_written, 16, "tier 0 should be skipped via manifest");
    }
}
