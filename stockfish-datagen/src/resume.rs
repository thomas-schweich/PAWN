//! Resume-time scanning + per-tier manifest.
//!
//! On startup, for each tier, we either (a) skip it because a manifest with
//! matching config fingerprint exists, or (b) resume by listing the shard
//! files in the tier dir and parsing the row count out of each filename
//! to figure out how many games each worker has already produced.
//!
//! The row count is encoded in the shard filename
//! (`shard-w<NNN>-c<NNNN>-r<NNNNNN>.parquet`), so resume does NOT need to
//! open the parquet files. That matters because it lets a remote-sync tool
//! (e.g. an HF dataset uploader) drop zero-byte placeholder files locally
//! that the resume code reads identically to real shards — enabling
//! "resume on a fresh pod without re-downloading any actual data".
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
use serde::{Deserialize, Serialize};

/// What we know about one worker's prior output in `tier_dir`.
#[derive(Debug, Clone, Copy)]
pub struct WorkerResumeState {
    /// Total rows already on disk for this worker (sum across its shards).
    pub games_done: u64,
    /// Chunk index for the *next* shard this worker should write.
    pub next_chunk_idx: u32,
}

/// Parse `shard-w<NNN>-c<NNNN>-r<NNNNNN>.parquet` into
/// `(worker_id, chunk_idx, n_rows)`. Returns `None` for any other filename
/// (including `.parquet.tmp` orphans, sentinels, and the pre-row-count
/// legacy naming `shard-w<NNN>-c<NNNN>.parquet` — those are unsupported
/// by current resume and will be left untouched by the dir scan).
pub(crate) fn parse_shard_filename(name: &str) -> Option<(u32, u32, u64)> {
    let s = name.strip_prefix("shard-w")?;
    let s = s.strip_suffix(".parquet")?;
    let (w, rest) = s.split_once("-c")?;
    let (c, r) = rest.split_once("-r")?;
    Some((w.parse().ok()?, c.parse().ok()?, r.parse().ok()?))
}

/// Walk `tier_dir`, identify per-worker resume state for workers in
/// `[0, n_workers)`. Returns an empty map if the directory doesn't exist
/// or contains no shard files.
///
/// Stale shards from worker IDs >= `n_workers` (left over after an
/// `n_workers` reduction) are SKIPPED entirely — they're not validated
/// for chunk-gaps and not counted in any worker's resume state. The
/// runner's dir-scan applies the same filter so the manifest stays
/// internally consistent (`shards` and `n_games_written` both reflect
/// only the current partition).
pub fn detect_resume(
    tier_dir: &Path,
    n_workers: u32,
) -> anyhow::Result<HashMap<u32, WorkerResumeState>> {
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
        let Some((worker_id, chunk_idx, n_rows)) = parse_shard_filename(&name) else {
            continue;
        };
        if worker_id >= n_workers {
            // Stale worker from a prior larger n_workers config — ignore.
            continue;
        }
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

/// Per-tier "in-progress" sentinel, written *before* any shards are
/// generated. Carries the tier fingerprint so a resumed run can detect
/// "shards are on disk but they were generated under a different config"
/// — the gap that the manifest alone can't catch (since the manifest is
/// only written on full completion).
///
/// Lifecycle: `run_tier` writes this file at start; the manifest write
/// at end of run leaves it in place (it's harmless once the manifest
/// exists, since the manifest takes precedence). A user who wants to
/// regenerate a tier under a different config must delete BOTH this
/// file and the matching shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierState {
    pub config_fingerprint: String,
    /// ISO-8601 UTC start time. Informational.
    pub started_at: String,
}

const STATE_FILENAME: &str = "_tier_state.json";

impl TierState {
    pub fn path(tier_dir: &Path) -> PathBuf {
        tier_dir.join(STATE_FILENAME)
    }

    pub fn load(tier_dir: &Path) -> anyhow::Result<Option<Self>> {
        let p = Self::path(tier_dir);
        if !p.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
        let s: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", p.display()))?;
        Ok(Some(s))
    }

    pub fn save(&self, tier_dir: &Path) -> anyhow::Result<()> {
        let p = Self::path(tier_dir);
        let tmp = p.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(self)?;
        let mut f = fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        use std::io::Write as _;
        f.write_all(&bytes).with_context(|| format!("writing {}", tmp.display()))?;
        f.sync_all().context("fsyncing tier state")?;
        drop(f);
        fs::rename(&tmp, &p).with_context(|| format!("renaming {} -> {}", tmp.display(), p.display()))?;
        Ok(())
    }
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
    use crate::shard::{GameRow, ShardWriter};
    use tempfile::tempdir;

    fn fake_row(game_seed: u64) -> GameRow {
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
        let mut w = ShardWriter::create(dir.to_path_buf(), worker_id, chunk_idx).unwrap();
        for i in 0..n_rows {
            w.append(&fake_row(i as u64));
        }
        w.close().unwrap();
    }

    #[test]
    fn parse_shard_filename_basic() {
        assert_eq!(
            parse_shard_filename("shard-w003-c0017-r000834.parquet"),
            Some((3, 17, 834)),
        );
        assert_eq!(
            parse_shard_filename("shard-w000-c0000-r000001.parquet"),
            Some((0, 0, 1)),
        );
        // Pre-row-count legacy naming is intentionally rejected.
        assert!(parse_shard_filename("shard-w003-c0017.parquet").is_none());
        assert!(parse_shard_filename("garbage.parquet").is_none());
        assert!(parse_shard_filename("shard-w0-c0.parquet.tmp").is_none());
    }

    #[test]
    fn detect_resume_empty_dir() {
        let dir = tempdir().unwrap();
        let states = detect_resume(dir.path(), 4).unwrap();
        assert!(states.is_empty());
    }

    #[test]
    fn detect_resume_missing_dir() {
        let states = detect_resume(Path::new("/tmp/definitely_not_a_real_dir_xyz_42"), 4).unwrap();
        assert!(states.is_empty());
    }

    #[test]
    fn detect_resume_counts_rows_and_chunks() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 0, 10);
        write_shard(dir.path(), 0, 1, 10);
        write_shard(dir.path(), 0, 2, 4); // partial trailing shard
        write_shard(dir.path(), 1, 0, 10);

        let states = detect_resume(dir.path(), 4).unwrap();
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
        let err = detect_resume(dir.path(), 4).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("chunk gap"), "expected gap error, got: {msg}");
    }

    #[test]
    fn detect_resume_ignores_non_shard_files() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 0, 10);
        std::fs::write(dir.path().join("_manifest.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("shard-w0-c0.parquet.tmp"), b"orphan").unwrap();
        let states = detect_resume(dir.path(), 4).unwrap();
        assert_eq!(states[&0].games_done, 10);
        assert_eq!(states.len(), 1);
    }

    /// The HF-sync placeholder strategy: the primer drops zero-byte files
    /// with the canonical shard naming, and resume must read them as if
    /// they were real shards (since the row count is in the filename).
    #[test]
    fn detect_resume_reads_zero_byte_placeholders() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("shard-w000-c0000-r000010.parquet"), b"").unwrap();
        std::fs::write(dir.path().join("shard-w000-c0001-r000010.parquet"), b"").unwrap();
        std::fs::write(dir.path().join("shard-w001-c0000-r000007.parquet"), b"").unwrap();
        let states = detect_resume(dir.path(), 4).unwrap();
        assert_eq!(states[&0].games_done, 20);
        assert_eq!(states[&0].next_chunk_idx, 2);
        assert_eq!(states[&1].games_done, 7);
        assert_eq!(states[&1].next_chunk_idx, 1);
    }

    #[test]
    fn manifest_round_trips() {
        let dir = tempdir().unwrap();
        let m = TierManifest {
            tier_name: "nodes_0001".into(),
            config_fingerprint: "abc123".into(),
            n_games_written: 100,
            shards: vec!["shard-w000-c0000-r000100.parquet".into()],
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
