//! Resume-time scanning + per-tier sentinels.
//!
//! Under the shard-id partitioning model, a tier has `total_shards`
//! deterministically-numbered shards; shard `s` owns global game indices
//! `[s * shard_size_games, min((s+1) * shard_size_games, n_games))`. Resume
//! walks the tier dir, parses each `shard-s<NNNNNN>-r<NNNNNN>.parquet`
//! filename, and builds two sets:
//!
//! - `done_shards`: shard ids that already exist on disk with their
//!   expected row count (i.e. the row count this run would write for that
//!   shard given the *current* config). Workers skip these.
//! - `boundary_rewrites`: shard ids that exist on disk but with a row count
//!   *less* than what the current config expects. This happens iff `n_games`
//!   grew between runs and the previous run's last shard was truncated.
//!   That one shard is regenerated; its existing-prefix games are
//!   bit-identical because the seed depends only on `tier_seed` +
//!   `global_game_index`, but the file's row count must grow to cover the
//!   new range. Workers pick this shard up via the same atomic counter
//!   and the on-disk truncated file is removed once the new one renames in.
//!
//! Row count is encoded in the shard filename, so resume does NOT need to
//! open the parquet files. That matters because it lets a remote-sync tool
//! (e.g. an HF dataset uploader) drop zero-byte placeholder files locally
//! that the resume code reads identically to real shards — enabling
//! "resume on a fresh pod without re-downloading any actual data".
//!
//! The per-tier `_tier_state.json` and `_manifest.json` sentinels carry
//! the `tier_fingerprint` so a resumed run can detect a config change
//! that would invalidate prior shards. Under multi-pod cooperation
//! (`--shard-id-range A:B`) the sentinels are suffixed with the range
//! so disjoint pods can coexist in the same HF dataset folder.

use std::collections::BTreeSet;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use serde::{Deserialize, Serialize};

/// Parse `shard-s<NNNNNN>-r<NNNNNN>.parquet` into `(shard_id, n_rows)`.
/// Returns `None` for any other filename (including `.parquet.tmp`
/// orphans, sentinels, and the pre-v3 legacy naming
/// `shard-w<NNN>-c<NNNN>-r<NNNNNN>.parquet` — legacy shards are not
/// readable under the new schema and are intentionally ignored).
pub(crate) fn parse_shard_filename(name: &str) -> Option<(u64, u64)> {
    let s = name.strip_prefix("shard-s")?;
    let s = s.strip_suffix(".parquet")?;
    let (sid, r) = s.split_once("-r")?;
    Some((sid.parse().ok()?, r.parse().ok()?))
}

/// What we found on disk in `tier_dir` relative to the current config.
#[derive(Debug, Clone, Default)]
pub struct ShardResumeState {
    /// Shard ids whose existing file row count matches the expected row
    /// count for the *current* `n_games` + `shard_size_games`. Workers
    /// skip these.
    pub done_shards: BTreeSet<u64>,
    /// Shard ids that exist on disk but with a row count smaller than the
    /// current config expects — i.e. previously-truncated last shards
    /// after `n_games` grew. Workers regenerate these and the old short
    /// file is removed at rename time.
    ///
    /// Stored as `(shard_id, old_path)` so we can delete the old file
    /// after the new one is written, without re-scanning the directory.
    pub boundary_rewrites: Vec<(u64, PathBuf)>,
}

/// Walk `tier_dir`, identify per-shard resume state for shard ids in
/// `[0, total_shards)`. Returns an empty state if the directory doesn't
/// exist or contains no shard files.
///
/// `expected_n_rows(s)` is the number of rows shard `s` should contain
/// under the current config — typically `shard_size_games` for all shards
/// except possibly the last (`n_games % shard_size_games` if non-zero).
/// Pass the same closure the runner uses for new shards so the row-count
/// check stays consistent.
///
/// Stale shards with `shard_id >= total_shards` (e.g. left over after a
/// `n_games` shrink) are SKIPPED — not validated, not counted, not
/// deleted. The operator can remove them with a manual prune.
pub fn detect_resume(
    tier_dir: &Path,
    total_shards: u64,
    expected_n_rows: impl Fn(u64) -> u64,
) -> anyhow::Result<ShardResumeState> {
    if !tier_dir.exists() {
        return Ok(ShardResumeState::default());
    }

    let mut state = ShardResumeState::default();
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for entry in fs::read_dir(tier_dir)
        .with_context(|| format!("listing {}", tier_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some((shard_id, n_rows)) = parse_shard_filename(&name) else {
            continue;
        };
        if shard_id >= total_shards {
            // Stale shard from a prior larger `n_games` — leave on disk
            // for the operator to handle.
            continue;
        }
        if !seen.insert(shard_id) {
            // Two files for the same shard id with different row counts
            // (e.g. a truncated old + the new growth-rewrite that hasn't
            // been cleaned up). Refuse to guess which is authoritative.
            return Err(anyhow!(
                "duplicate shard files for shard id {shard_id} in {}; \
                 delete one before resuming",
                tier_dir.display(),
            ));
        }
        let expected = expected_n_rows(shard_id);
        if n_rows == expected {
            state.done_shards.insert(shard_id);
        } else if n_rows < expected {
            state.boundary_rewrites.push((shard_id, entry.path()));
        } else {
            return Err(anyhow!(
                "shard {shard_id} in {} has {n_rows} rows but current config expects {expected}; \
                 shrinking shard contents is not supported (would invalidate already-written data)",
                tier_dir.display(),
            ));
        }
    }
    Ok(state)
}

/// Per-tier "in-progress" sentinel, written *before* any shards are
/// generated. Carries the tier fingerprint so a resumed run can detect
/// "shards are on disk but they were generated under a different config"
/// — the gap that the manifest alone can't catch (since the manifest is
/// only written on full completion).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierState {
    pub config_fingerprint: String,
    /// ISO-8601 UTC start time. Informational.
    pub started_at: String,
    /// Half-open shard-id range owned by the pod that wrote this state
    /// file. `None` ≡ "whole tier" (single-pod, default). When `Some`,
    /// the state file is named `_tier_state-s<A>-s<B>.json` so disjoint
    /// pods can coexist in the same HF dataset folder without colliding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_range: Option<ShardRange>,
}

/// Inclusive-exclusive shard-id range. Serializes as `{"start": A, "end": B}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardRange {
    pub start: u64,
    pub end: u64,
}

impl From<Range<u64>> for ShardRange {
    fn from(r: Range<u64>) -> Self {
        ShardRange { start: r.start, end: r.end }
    }
}

impl From<ShardRange> for Range<u64> {
    fn from(r: ShardRange) -> Self {
        r.start..r.end
    }
}

/// Filename suffix for per-pod sentinel files. `None` ≡ "" (single-pod default).
fn pod_suffix(shard_range: Option<&ShardRange>) -> String {
    match shard_range {
        Some(r) => format!("-s{:06}-s{:06}", r.start, r.end),
        None => String::new(),
    }
}

impl TierState {
    pub fn path(tier_dir: &Path, shard_range: Option<&ShardRange>) -> PathBuf {
        tier_dir.join(format!("_tier_state{}.json", pod_suffix(shard_range)))
    }

    pub fn load(tier_dir: &Path, shard_range: Option<&ShardRange>) -> anyhow::Result<Option<Self>> {
        let p = Self::path(tier_dir, shard_range);
        if !p.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
        let s: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", p.display()))?;
        Ok(Some(s))
    }

    pub fn save(&self, tier_dir: &Path) -> anyhow::Result<()> {
        let p = Self::path(tier_dir, self.shard_range.as_ref());
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
///
/// In multi-pod runs each pod writes its OWN manifest covering its shard
/// range; a separate `scripts/datagen_reconcile_tier.py` helper merges
/// the per-pod manifests into a unified `_manifest.json` covering the
/// full `[0, total_shards)` range.
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
    /// Per-pod shard range, mirrored from the matching `TierState`.
    /// `None` for the canonical single-pod (or reconciled) manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_range: Option<ShardRange>,
}

impl TierManifest {
    pub fn path(tier_dir: &Path, shard_range: Option<&ShardRange>) -> PathBuf {
        tier_dir.join(format!("_manifest{}.json", pod_suffix(shard_range)))
    }

    pub fn load(tier_dir: &Path, shard_range: Option<&ShardRange>) -> anyhow::Result<Option<Self>> {
        let p = Self::path(tier_dir, shard_range);
        if !p.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
        let m: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", p.display()))?;
        Ok(Some(m))
    }

    pub fn save(&self, tier_dir: &Path) -> anyhow::Result<()> {
        let p = Self::path(tier_dir, self.shard_range.as_ref());
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

    fn fake_row(global_game_index: u64) -> GameRow {
        GameRow {
            tokens: vec![1, 2, 3],
            san: vec!["a".into(), "b".into(), "c".into()],
            uci: vec!["e2e4".into(), "e7e5".into(), "g1f3".into()],
            game_length: 3,
            outcome_token: 1969,
            result: "1-0".into(),
            nodes: Some(1),
            multi_pv: Some(5),
            opening_multi_pv: Some(20),
            opening_plies: Some(1),
            sample_plies: Some(999),
            sample_score: None,
            net_selection: None,
            temperature: 1.0,
            global_game_index,
            game_seed: global_game_index, // tests don't care about real seeding
            stockfish_version: "Stockfish 18".into(),
            legal_move_evals: None,
            static_legal_move_evals: None,
        }
    }

    fn write_shard(dir: &Path, shard_id: u64, n_rows: usize) {
        let mut w = ShardWriter::create(dir.to_path_buf(), shard_id).unwrap();
        for i in 0..n_rows {
            w.append(&fake_row(shard_id * 1000 + i as u64));
        }
        w.close().unwrap();
    }

    #[test]
    fn parse_shard_filename_basic() {
        assert_eq!(
            parse_shard_filename("shard-s000017-r000834.parquet"),
            Some((17, 834)),
        );
        assert_eq!(
            parse_shard_filename("shard-s000000-r000001.parquet"),
            Some((0, 1)),
        );
        // Wider field accepted (n_rows > 999_999, shard_id > 999_999).
        assert_eq!(
            parse_shard_filename("shard-s1234567-r1234567.parquet"),
            Some((1_234_567, 1_234_567)),
        );
        // Pre-v3 legacy naming intentionally rejected.
        assert!(parse_shard_filename("shard-w003-c0017-r000834.parquet").is_none());
        assert!(parse_shard_filename("garbage.parquet").is_none());
        assert!(parse_shard_filename("shard-s0.parquet.tmp").is_none());
    }

    #[test]
    fn detect_resume_empty_dir() {
        let dir = tempdir().unwrap();
        let st = detect_resume(dir.path(), 4, |_| 10).unwrap();
        assert!(st.done_shards.is_empty());
        assert!(st.boundary_rewrites.is_empty());
    }

    #[test]
    fn detect_resume_missing_dir() {
        let st = detect_resume(Path::new("/tmp/definitely_not_a_real_dir_xyz_42"), 4, |_| 10).unwrap();
        assert!(st.done_shards.is_empty());
    }

    #[test]
    fn detect_resume_classifies_done_vs_boundary_rewrite() {
        let dir = tempdir().unwrap();
        // 3 full shards, then a truncated last shard (4 rows where 10 expected).
        write_shard(dir.path(), 0, 10);
        write_shard(dir.path(), 1, 10);
        write_shard(dir.path(), 2, 10);
        write_shard(dir.path(), 3, 4);
        // Pretend n_games grew so shard 3 now expects 10 rows.
        let st = detect_resume(dir.path(), 4, |_| 10).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([0, 1, 2]));
        assert_eq!(st.boundary_rewrites.len(), 1);
        assert_eq!(st.boundary_rewrites[0].0, 3);
    }

    #[test]
    fn detect_resume_treats_exact_partial_last_as_done() {
        let dir = tempdir().unwrap();
        // Shard 3 is the partial last shard under the *current* config too
        // (n_games = 34, shard_size = 10 ⇒ last shard has 4 rows). So it's
        // already complete, not a boundary rewrite.
        write_shard(dir.path(), 0, 10);
        write_shard(dir.path(), 1, 10);
        write_shard(dir.path(), 2, 10);
        write_shard(dir.path(), 3, 4);
        let st = detect_resume(dir.path(), 4, |sid| if sid == 3 { 4 } else { 10 }).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([0, 1, 2, 3]));
        assert!(st.boundary_rewrites.is_empty());
    }

    #[test]
    fn detect_resume_rejects_oversized_shard() {
        let dir = tempdir().unwrap();
        // Shard 0 has 15 rows; current config expects 10. Shrinking is
        // unsupported (would invalidate already-written rows).
        write_shard(dir.path(), 0, 15);
        let err = detect_resume(dir.path(), 4, |_| 10).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("shrinking"), "expected shrink error, got: {msg}");
    }

    #[test]
    fn detect_resume_skips_stale_shard_ids() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 10);
        write_shard(dir.path(), 1, 10);
        // Shard 5 is outside [0, total_shards=2) — stale, ignored.
        write_shard(dir.path(), 5, 10);
        let st = detect_resume(dir.path(), 2, |_| 10).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([0, 1]));
        assert!(st.boundary_rewrites.is_empty());
    }

    #[test]
    fn detect_resume_ignores_non_shard_files() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 10);
        std::fs::write(dir.path().join("_manifest.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("shard-s0.parquet.tmp"), b"orphan").unwrap();
        let st = detect_resume(dir.path(), 4, |_| 10).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([0]));
    }

    /// HF-sync placeholder contract: the primer drops zero-byte files with
    /// the canonical shard naming, and resume must read them as if they
    /// were real shards (the row count is in the filename).
    #[test]
    fn detect_resume_reads_zero_byte_placeholders() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("shard-s000000-r000010.parquet"), b"").unwrap();
        std::fs::write(dir.path().join("shard-s000001-r000010.parquet"), b"").unwrap();
        std::fs::write(dir.path().join("shard-s000002-r000007.parquet"), b"").unwrap();
        let st = detect_resume(dir.path(), 4, |sid| if sid == 2 { 7 } else { 10 }).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([0, 1, 2]));
    }

    #[test]
    fn manifest_round_trips_with_shard_range() {
        let dir = tempdir().unwrap();
        let m = TierManifest {
            tier_name: "nodes_0001".into(),
            config_fingerprint: "abc123".into(),
            n_games_written: 100,
            shards: vec!["shard-s000000-r000100.parquet".into()],
            completed_at: "2026-05-02T00:00:00Z".into(),
            shard_range: Some(ShardRange { start: 0, end: 50 }),
        };
        m.save(dir.path()).unwrap();
        // Per-pod manifest lives under the suffixed filename.
        assert_eq!(
            TierManifest::path(dir.path(), m.shard_range.as_ref()).file_name().unwrap(),
            "_manifest-s000000-s000050.json",
        );
        let loaded = TierManifest::load(dir.path(), m.shard_range.as_ref()).unwrap().unwrap();
        assert_eq!(loaded.tier_name, m.tier_name);
        assert_eq!(loaded.shard_range, m.shard_range);
        // The single-pod canonical path is a different file.
        assert!(TierManifest::load(dir.path(), None).unwrap().is_none());
    }

    #[test]
    fn manifest_load_missing_returns_none() {
        let dir = tempdir().unwrap();
        let m = TierManifest::load(dir.path(), None).unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn manifest_default_path_no_suffix() {
        assert_eq!(
            TierManifest::path(Path::new("/x"), None).file_name().unwrap(),
            "_manifest.json",
        );
        assert_eq!(
            TierState::path(Path::new("/x"), None).file_name().unwrap(),
            "_tier_state.json",
        );
    }
}
