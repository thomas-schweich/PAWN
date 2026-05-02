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
use crate::seed;
use crate::shard::{GameRow, ShardWriter, shard_path};
use crate::stockfish::StockfishProcess;

/// Results from a successful tier run.
#[derive(Debug, Clone)]
pub struct TierResult {
    pub n_games_written: u64,
    pub n_games_dropped: u64,
    pub shards: Vec<PathBuf>,
}

/// Messages workers send to the main thread.
#[derive(Debug)]
pub enum Progress {
    /// Periodic update so the user knows things are alive.
    Heartbeat {
        worker_id: u32,
        games_done: u64,
        games_dropped: u64,
    },
    /// Worker finished cleanly. Last shard already closed.
    Done {
        worker_id: u32,
        games_written: u64,
        games_dropped: u64,
        shards: Vec<PathBuf>,
    },
}

/// Maximum fraction of games a single worker is allowed to drop before we
/// fail the tier. Drops indicate something the engine couldn't tokenize —
/// either a real bug or Stockfish proposing an illegal move. >=0.1% means
/// something is structurally wrong, not just one-off noise.
const MAX_DROP_FRACTION: f64 = 0.001;

/// Run one tier end-to-end. Spawns `cfg.n_workers` worker threads, each
/// generating its share of `tier.n_games`. Returns summary stats.
pub fn run_tier(
    cfg: &RunConfig,
    tier_index: usize,
) -> anyhow::Result<TierResult> {
    let tier = &cfg.tiers[tier_index];
    let tier_dir = cfg.output_dir.join(&tier.name);
    std::fs::create_dir_all(&tier_dir)
        .with_context(|| format!("creating tier dir {}", tier_dir.display()))?;

    let tier_seed = seed::tier_seed(cfg.master_seed, tier_index);
    let split = cfg.games_per_worker(tier);

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
    let mut handles = Vec::with_capacity(cfg.n_workers as usize);

    for worker_id in 0..cfg.n_workers {
        let target = split[worker_id as usize];
        if target == 0 {
            continue; // nothing for this worker — skip spawning entirely
        }
        let worker_seed = seed::worker_seed(tier_seed, worker_id);
        let cfg = Arc::clone(&cfg);
        let tier = Arc::clone(&tier);
        let tier_dir = Arc::clone(&tier_dir);
        let tx = tx.clone();
        let h = std::thread::Builder::new()
            .name(format!("sfd-w{worker_id}"))
            .spawn(move || {
                run_worker(&cfg, &tier, &tier_dir, worker_id, worker_seed, 0, target, tx)
            })
            .with_context(|| format!("spawning worker {worker_id}"))?;
        handles.push((worker_id, h));
    }
    // Drop the original sender so the channel closes once all worker
    // clones are dropped (i.e. when all workers exit).
    drop(tx);

    // Drain progress — runs concurrently with the workers.
    let drain_handle = std::thread::Builder::new()
        .name("sfd-progress".into())
        .spawn(move || -> Vec<Progress> {
            let mut completions = Vec::new();
            for msg in rx {
                match &msg {
                    Progress::Heartbeat { worker_id, games_done, games_dropped } => {
                        eprintln!(
                            "  [worker {worker_id:>2}] {games_done:>7} done, {games_dropped} dropped",
                        );
                    }
                    Progress::Done { worker_id, games_written, games_dropped, shards } => {
                        eprintln!(
                            "  [worker {worker_id:>2}] DONE: {games_written} written, {games_dropped} dropped, {} shards",
                            shards.len(),
                        );
                        completions.push(msg);
                        continue;
                    }
                }
            }
            completions
        })
        .context("spawning progress drain")?;

    // Wait for all workers, surface the first error if any panicked or
    // returned Err.
    let mut first_err: Option<anyhow::Error> = None;
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

    let completions = drain_handle.join().map_err(|_| anyhow!("progress drain panicked"))?;

    if let Some(e) = first_err {
        return Err(e);
    }

    let mut total_written = 0u64;
    let mut total_dropped = 0u64;
    let mut all_shards = Vec::new();
    for msg in completions {
        if let Progress::Done { games_written, games_dropped, shards, .. } = msg {
            total_written += games_written;
            total_dropped += games_dropped;
            all_shards.extend(shards);
        }
    }

    let target_total: u64 = split.iter().sum();
    let drop_frac = if target_total > 0 {
        total_dropped as f64 / target_total as f64
    } else {
        0.0
    };
    if drop_frac > MAX_DROP_FRACTION {
        return Err(anyhow!(
            "[{}] drop rate too high: {} dropped / {} target = {:.3}% > {:.3}%",
            tier.name,
            total_dropped,
            target_total,
            drop_frac * 100.0,
            MAX_DROP_FRACTION * 100.0,
        ));
    }

    eprintln!(
        "[{}] complete: {} written, {} dropped, {} shards",
        tier.name,
        total_written,
        total_dropped,
        all_shards.len(),
    );
    Ok(TierResult {
        n_games_written: total_written,
        n_games_dropped: total_dropped,
        shards: all_shards,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    cfg: &RunConfig,
    tier: &TierConfig,
    tier_dir: &std::path::Path,
    worker_id: u32,
    worker_seed: u64,
    start_index: u64,
    target: u64,
    tx: Sender<Progress>,
) -> anyhow::Result<()> {
    let mut sf = StockfishProcess::spawn(
        &expand_tilde(&cfg.stockfish_path),
        &cfg.stockfish_version,
        cfg.stockfish_hash_mb,
    )
    .with_context(|| format!("spawning stockfish for worker {worker_id}"))?;
    let stockfish_id_name = sf.id_name.clone();

    // Each worker starts on its own chunk based on start_index. With
    // start_index=0 (no resume) every worker begins at chunk 0.
    let shard_size = cfg.shard_size_games as u64;
    let mut current_chunk = (start_index / shard_size) as u32;
    let mut games_in_shard = (start_index % shard_size) as u64;
    let mut writer: Option<ShardWriter> = None;
    let mut total_written = 0u64;
    let mut total_dropped = 0u64;
    let mut shards = Vec::new();

    for game_index in start_index..target {
        let game_seed = seed::game_seed(worker_seed, game_index);
        let mut rng = ChaCha8Rng::seed_from_u64(game_seed);

        let played = play_game(&mut sf, &mut rng, tier, cfg.max_ply)
            .with_context(|| format!("playing game {game_index}"))?;

        if played.uci_moves.is_empty() {
            total_dropped += 1;
            continue;
        }

        // Re-tokenize via the canonical encoder. Stockfish should never
        // give us an unparseable move list, but verify anyway and drop
        // (with counter) if it does.
        let refs: Vec<&str> = played.uci_moves.iter().map(|s| s.as_str()).collect();
        let (tokens, san, n) = chess_engine::uci::uci_to_tokens_and_san(&refs);
        if n != played.uci_moves.len() || n == 0 {
            total_dropped += 1;
            continue;
        }

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
            game_seed: game_seed as i64,
            stockfish_version: stockfish_id_name.clone(),
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

        if (game_index + 1) % 500 == 0 {
            let _ = tx.send(Progress::Heartbeat {
                worker_id,
                games_done: game_index + 1 - start_index,
                games_dropped: total_dropped,
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

    sf.shutdown();

    let _ = tx.send(Progress::Done {
        worker_id,
        games_written: total_written,
        games_dropped: total_dropped,
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

    fn stockfish_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("STOCKFISH_PATH") {
            return Some(p.into());
        }
        let default = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("bin/stockfish");
        if default.exists() { Some(default) } else { None }
    }

    #[test]
    fn live_run_tier_smoke() {
        let Some(sf_path) = stockfish_path() else {
            eprintln!("skipping: no stockfish binary");
            return;
        };
        let dir = tempdir().unwrap();
        let cfg = RunConfig {
            stockfish_path: sf_path,
            stockfish_version: "Stockfish".into(),
            output_dir: dir.path().to_path_buf(),
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
        };
        let result = run_tier(&cfg, 0).unwrap();
        assert_eq!(result.n_games_written + result.n_games_dropped, 16);
        // 2 workers × 8 games each = exactly 1 shard per worker, so 2 total.
        assert_eq!(result.shards.len(), 2);
        for shard in &result.shards {
            assert!(shard.exists(), "shard {} missing", shard.display());
            assert!(shard.extension().unwrap() == "parquet");
        }
    }
}
