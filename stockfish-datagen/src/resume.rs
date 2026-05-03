//! Resume-time scanning + per-tier manifest.
//!
//! On startup, for each tier, we either (a) skip it because a manifest with
//! matching config fingerprint exists, or (b) resume by reading the parquet
//! files on disk to figure out how many games each worker has already
//! produced.
//!
//! Per-worker resume relies on shard files being written in strict
//! chunk-index order, atomically (`.tmp` -> `.parquet` rename), and at
//! exactly `shard_size_games` rows each *except* possibly the highest
//! chunk if the worker had a non-multiple-of-shard-size leftover. We
//! don't need to distinguish "interrupted mid-game" from "completed with
//! a partial last shard" because game seeds are deterministic from
//! `(worker_seed, game_index)` — every shard's contents depend only on
//! its chunk index, so picking up from `last_chunk_idx + 1` is correct
//! either way.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

/// What we know about one worker's prior output in `tier_dir`.
#[derive(Debug, Clone, Copy)]
pub struct WorkerResumeState {
    /// Total rows already on disk for this worker (sum across its shards).
    pub games_done: u64,
    /// Chunk index for the *next* shard this worker should write.
    pub next_chunk_idx: u32,
}

/// Parse `shard-w<NNN>-c<NNNN>.parquet` into (worker_id, chunk_idx).
fn parse_shard_filename(name: &str) -> Option<(u32, u32)> {
    let s = name.strip_prefix("shard-w")?;
    let s = s.strip_suffix(".parquet")?;
    let (w, c) = s.split_once("-c")?;
    Some((w.parse().ok()?, c.parse().ok()?))
}

/// Read just the row count from a parquet file's metadata. Fast — does
/// not decode the data pages.
fn count_parquet_rows(path: &Path) -> anyhow::Result<u64> {
    let file = fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading parquet metadata: {}", path.display()))?;
    Ok(builder.metadata().file_metadata().num_rows() as u64)
}

/// Walk `tier_dir`, identify per-worker resume state. Returns an empty map
/// if the directory doesn't exist or contains no shard files.
pub fn detect_resume(tier_dir: &Path) -> anyhow::Result<HashMap<u32, WorkerResumeState>> {
    if !tier_dir.exists() {
        return Ok(HashMap::new());
    }

    let mut by_worker: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
    for entry in fs::read_dir(tier_dir)
        .with_context(|| format!("listing {}", tier_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some((worker_id, chunk_idx)) = parse_shard_filename(&name) else {
            continue;
        };
        let n_rows = count_parquet_rows(&entry.path())?;
        by_worker.entry(worker_id).or_default().push((chunk_idx, n_rows));
    }

    let mut out = HashMap::new();
    for (worker_id, mut chunks) in by_worker {
        chunks.sort_by_key(|&(c, _)| c);
        // Defensive: chunk indices must be a contiguous prefix starting
        // at 0. Gaps mean someone deleted a middle shard and we can't
        // safely resume — bail with a clear message.
        for (i, &(c, _)) in chunks.iter().enumerate() {
            if c as usize != i {
                return Err(anyhow!(
                    "worker {worker_id} has shard chunk gap: expected chunk {i}, found {c} \
                     (full chunk list: {:?})",
                    chunks.iter().map(|&(c, _)| c).collect::<Vec<_>>(),
                ));
            }
        }
        let games_done: u64 = chunks.iter().map(|&(_, n)| n).sum();
        let next_chunk_idx = chunks.last().map(|&(c, _)| c + 1).unwrap_or(0);
        out.insert(worker_id, WorkerResumeState { games_done, next_chunk_idx });
    }
    Ok(out)
}

/// Per-tier completion record. Presence of this file with a matching
/// `config_fingerprint` tells `run` to skip the tier on restart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierManifest {
    pub tier_name: String,
    /// `RunConfig::tier_fingerprint(tier_index)` — refuse to skip if the
    /// inputs that affect this tier's data have changed.
    pub config_fingerprint: String,
    pub n_games_written: u64,
    /// Shard file names, relative to the tier dir.
    pub shards: Vec<String>,
    /// ISO-8601 UTC. Purely informational.
    pub completed_at: String,
}

const MANIFEST_FILENAME: &str = "_manifest.json";

impl TierManifest {
    pub fn path(tier_dir: &Path) -> PathBuf {
        tier_dir.join(MANIFEST_FILENAME)
    }

    pub fn load(tier_dir: &Path) -> anyhow::Result<Option<Self>> {
        let p = Self::path(tier_dir);
        if !p.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
        let m: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", p.display()))?;
        Ok(Some(m))
    }

    pub fn save(&self, tier_dir: &Path) -> anyhow::Result<()> {
        let p = Self::path(tier_dir);
        let tmp = p.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(self)?;
        // fsync the manifest before rename for the same reason we fsync
        // shards: rename is atomic on the dirent but does not imply data
        // durability. A power failure between rename and journal commit
        // could leave a zero-byte _manifest.json that incorrectly signals
        // "tier complete" on the next run.
        {
            let mut f = fs::File::create(&tmp)
                .with_context(|| format!("creating {}", tmp.display()))?;
            use std::io::Write as _;
            f.write_all(&bytes).with_context(|| format!("writing {}", tmp.display()))?;
            f.sync_all().context("fsyncing manifest")?;
        }
        fs::rename(&tmp, &p).with_context(|| format!("renaming {} -> {}", tmp.display(), p.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::{GameRow, ShardWriter, shard_path};
    use tempfile::tempdir;

    fn fake_row(game_seed: i64) -> GameRow {
        GameRow {
            tokens: vec![1, 2, 3],
            san: vec!["a".into(), "b".into(), "c".into()],
            uci: vec!["e2e4".into(), "e7e5".into(), "g1f3".into()],
            game_length: 3,
            outcome_token: 1969,
            result: "1-0".into(),
            nodes: 1,
            multi_pv: 5,
            opening_multi_pv: 20,
            opening_plies: 1,
            sample_plies: 999,
            temperature: 1.0,
            worker_id: 0,
            game_seed,
            stockfish_version: "Stockfish 18".into(),
        }
    }

    fn write_shard(dir: &Path, worker_id: u32, chunk_idx: u32, n_rows: usize) {
        let p = shard_path(dir, worker_id, chunk_idx);
        let mut w = ShardWriter::create(p).unwrap();
        for i in 0..n_rows {
            w.append(&fake_row(i as i64));
        }
        w.close().unwrap();
    }

    #[test]
    fn parse_shard_filename_basic() {
        assert_eq!(parse_shard_filename("shard-w003-c0017.parquet"), Some((3, 17)));
        assert_eq!(parse_shard_filename("shard-w000-c0000.parquet"), Some((0, 0)));
        assert!(parse_shard_filename("garbage.parquet").is_none());
        assert!(parse_shard_filename("shard-w0-c0.parquet.tmp").is_none());
    }

    #[test]
    fn detect_resume_empty_dir() {
        let dir = tempdir().unwrap();
        let states = detect_resume(dir.path()).unwrap();
        assert!(states.is_empty());
    }

    #[test]
    fn detect_resume_missing_dir() {
        let states = detect_resume(Path::new("/tmp/definitely_not_a_real_dir_xyz_42")).unwrap();
        assert!(states.is_empty());
    }

    #[test]
    fn detect_resume_counts_rows_and_chunks() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 0, 10);
        write_shard(dir.path(), 0, 1, 10);
        write_shard(dir.path(), 0, 2, 4); // partial trailing shard
        write_shard(dir.path(), 1, 0, 10);

        let states = detect_resume(dir.path()).unwrap();
        let w0 = states[&0];
        assert_eq!(w0.games_done, 24);
        assert_eq!(w0.next_chunk_idx, 3);
        let w1 = states[&1];
        assert_eq!(w1.games_done, 10);
        assert_eq!(w1.next_chunk_idx, 1);
    }

    #[test]
    fn detect_resume_rejects_chunk_gaps() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 0, 10);
        // skip chunk 1
        write_shard(dir.path(), 0, 2, 10);
        let err = detect_resume(dir.path()).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("chunk gap"), "expected gap error, got: {msg}");
    }

    #[test]
    fn detect_resume_ignores_non_shard_files() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 0, 10);
        std::fs::write(dir.path().join("_manifest.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("shard-w0-c0.parquet.tmp"), b"orphan").unwrap();
        let states = detect_resume(dir.path()).unwrap();
        assert_eq!(states[&0].games_done, 10);
        assert_eq!(states.len(), 1);
    }

    #[test]
    fn manifest_round_trips() {
        let dir = tempdir().unwrap();
        let m = TierManifest {
            tier_name: "nodes_0001".into(),
            config_fingerprint: "abc123".into(),
            n_games_written: 100,
            shards: vec!["shard-w000-c0000.parquet".into()],
            completed_at: "2026-05-02T00:00:00Z".into(),
        };
        m.save(dir.path()).unwrap();
        let loaded = TierManifest::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.tier_name, m.tier_name);
        assert_eq!(loaded.config_fingerprint, m.config_fingerprint);
        assert_eq!(loaded.n_games_written, m.n_games_written);
        assert_eq!(loaded.shards, m.shards);
    }

    #[test]
    fn manifest_load_missing_returns_none() {
        let dir = tempdir().unwrap();
        let m = TierManifest::load(dir.path()).unwrap();
        assert!(m.is_none());
    }
}
