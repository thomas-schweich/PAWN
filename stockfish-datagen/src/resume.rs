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

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::Context;
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
/// Shards with `shard_id` outside `validate_range` are SKIPPED — not
/// validated, not counted, not deleted. This covers two cases:
///   * Stale shards from a prior larger `n_games` (`shard_id >= total_shards`):
///     caller passes `0..total_shards` and they fall off the end.
///   * Other pods' shards in a multi-pod run sharing the tier dir
///     (e.g. via the orchestrator primer): caller scopes the range to
///     this pod's `[shard_range.start, shard_range.end)` and another
///     pod's shards are ignored rather than misvalidated against this
///     pod's `expected_n_rows` (which is fine for shards this pod owns
///     but could spuriously flag another pod's last shard as oversized
///     if the two pods happen to disagree on tier-level `n_games` —
///     in normal operation `enforce_n_games_invariant` blocks that, so
///     the scope filter is defense in depth rather than the only line).
pub fn detect_resume(
    tier_dir: &Path,
    validate_range: Range<u64>,
    expected_n_rows: impl Fn(u64) -> u64,
) -> anyhow::Result<ShardResumeState> {
    if !tier_dir.exists() {
        return Ok(ShardResumeState::default());
    }

    // Group by shard_id, keeping the file with the highest `n_rows`. Two
    // files for the same shard id are possible in two benign cases:
    //   * Boundary-rewrite race: a writer in another pod (or this pod's
    //     prior process) has renamed the new larger file into place but
    //     hasn't yet `remove_file`'d the older smaller one. Both live on
    //     disk for milliseconds.
    //   * Crash mid-rewrite: the same window, but never closed by cleanup.
    // Picking the highest-row-count file is always the correct interpretation:
    // the newer file is the larger one (writers grow shards, never shrink),
    // and the smaller file will be deleted shortly (or is an orphan the
    // operator can prune). The lower-row file shows up as a missing
    // boundary-rewrite if it would otherwise have been the canonical entry,
    // so the worker that owns it triggers cleanup on next pass.
    let mut by_shard: BTreeMap<u64, (u64, PathBuf)> = BTreeMap::new();
    for entry in fs::read_dir(tier_dir)
        .with_context(|| format!("listing {}", tier_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some((shard_id, n_rows)) = parse_shard_filename(&name) else {
            continue;
        };
        if !validate_range.contains(&shard_id) {
            // Out of scope for this caller — stale, or owned by another pod.
            continue;
        }
        by_shard
            .entry(shard_id)
            .and_modify(|(cur_rows, cur_path)| {
                if n_rows > *cur_rows {
                    *cur_rows = n_rows;
                    *cur_path = entry.path();
                }
            })
            .or_insert((n_rows, entry.path()));
    }

    let mut state = ShardResumeState::default();
    for (shard_id, (n_rows, path)) in by_shard {
        let expected = expected_n_rows(shard_id);
        if n_rows == expected {
            state.done_shards.insert(shard_id);
        } else if n_rows < expected {
            state.boundary_rewrites.push((shard_id, path));
        } else {
            return Err(anyhow::anyhow!(
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
    /// Tier-level `n_games` at the time this state was written. Excluded
    /// from the per-tier fingerprint (so growth is allowed without
    /// invalidating prior shards), but enforced across pods via
    /// `enforce_n_games_invariant` to prevent two pods cooperating with
    /// different `n_games` from silently producing inconsistent
    /// datasets. See bug-detector finding #2 in the review history.
    #[serde(default)]
    pub n_games: u64,
    /// Half-open shard-id range owned by the pod that wrote this state
    /// file. `None` ≡ "whole tier" (single-pod, default). When `Some`,
    /// the state file is named `_tier_state-s<A>-s<B>.json` so disjoint
    /// pods can coexist in the same HF dataset folder without colliding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_range: Option<ShardRange>,
}

/// Walk `tier_dir` and parse every `_tier_state*.json` (canonical +
/// per-pod variants). Used by `enforce_n_games_invariant` so a new pod
/// can see what every other pod committed to.
pub fn scan_all_tier_states(tier_dir: &Path) -> anyhow::Result<Vec<(PathBuf, TierState)>> {
    if !tier_dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(tier_dir)
        .with_context(|| format!("listing {}", tier_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Match the canonical name and the per-pod variants; reject the
        // `.json.tmp` from atomic-rename in-flight.
        if !(name.starts_with("_tier_state") && name.ends_with(".json")) {
            continue;
        }
        let path = entry.path();
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let state: TierState = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        out.push((path, state));
    }
    Ok(out)
}

/// Refuse to start if any existing `_tier_state*.json` in `tier_dir`
/// declares an `n_games` larger than what this pod's config says. The
/// fingerprint excludes `n_games` (to make tier extension safe), so
/// without this guard a smaller-n_games pod would silently "complete"
/// its slice — its own manifest matches, the larger pod's extra shards
/// fall outside `total_shards` and get skipped as "stale" — while
/// another pod is committed to writing more data. Operator error
/// (two pods configured with different `n_games`), but caught with a
/// clear startup error rather than a silently-truncated dataset.
///
/// Allows the inverse direction: existing state with smaller `n_games`
/// is treated as the prior version of an extended run, which is the
/// supported single-pod extension flow.
pub fn enforce_n_games_invariant(
    tier_dir: &Path,
    current_n_games: u64,
) -> anyhow::Result<()> {
    for (path, state) in scan_all_tier_states(tier_dir)? {
        // Pre-fix state files (written before this field existed)
        // serialize without `n_games`; `#[serde(default)]` parses them
        // as 0. Skip the check in that case — we can't enforce what
        // we don't know, and erroring would block re-runs against
        // partial datasets from this branch's earlier commits.
        if state.n_games == 0 {
            continue;
        }
        if state.n_games > current_n_games {
            anyhow::bail!(
                "{} declares n_games={} but current config has n_games={}; \
                 refusing to start — this run would silently skip {} games \
                 another pod committed to producing. Either raise this \
                 config's n_games to >= {}, or delete the conflicting state \
                 file if you genuinely mean to shrink the dataset.",
                path.display(),
                state.n_games,
                current_n_games,
                state.n_games - current_n_games,
                state.n_games,
            );
        }
    }
    Ok(())
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
        if p.exists() {
            let bytes = fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
            let s: Self = serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", p.display()))?;
            return Ok(Some(s));
        }
        // Fallback: a scoped lookup that found no per-pod state file
        // falls back to the canonical `_tier_state.json` if it exists.
        // This covers the post-reconcile-then-scoped-rerun case: the
        // reconciler deletes all `_tier_state-s<A>-s<B>.json` files
        // after committing the canonical state, so a later operator
        // running with `--shard-id-range A:B` against the reconciled
        // tier would otherwise hit `None` here despite the canonical
        // sentinel existing. The runner's fingerprint check still
        // applies — a non-matching canonical state correctly errors.
        if shard_range.is_some() {
            let canonical = Self::path(tier_dir, None);
            if canonical.exists() {
                let bytes = fs::read(&canonical)
                    .with_context(|| format!("reading {}", canonical.display()))?;
                let s: Self = serde_json::from_slice(&bytes)
                    .with_context(|| format!("parsing {}", canonical.display()))?;
                return Ok(Some(s));
            }
        }
        Ok(None)
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
        let st = detect_resume(dir.path(), 0..4, |_| 10).unwrap();
        assert!(st.done_shards.is_empty());
        assert!(st.boundary_rewrites.is_empty());
    }

    #[test]
    fn detect_resume_missing_dir() {
        let st = detect_resume(Path::new("/tmp/definitely_not_a_real_dir_xyz_42"), 0..4, |_| 10).unwrap();
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
        let st = detect_resume(dir.path(), 0..4, |_| 10).unwrap();
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
        let st = detect_resume(dir.path(), 0..4, |sid| if sid == 3 { 4 } else { 10 }).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([0, 1, 2, 3]));
        assert!(st.boundary_rewrites.is_empty());
    }

    #[test]
    fn detect_resume_rejects_oversized_shard() {
        let dir = tempdir().unwrap();
        // Shard 0 has 15 rows; current config expects 10. Shrinking is
        // unsupported (would invalidate already-written rows).
        write_shard(dir.path(), 0, 15);
        let err = detect_resume(dir.path(), 0..4, |_| 10).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("shrinking"), "expected shrink error, got: {msg}");
    }

    /// Boundary-rewrite race tolerance: a worker that grew a partial
    /// last shard just renamed the larger file into place but hasn't yet
    /// `remove_file`'d the old smaller one. Both files coexist for
    /// milliseconds. Another pod's `detect_resume` (or this same pod's
    /// post-run rescan) MUST pick the canonical entry — the highest
    /// row count — and not error on "duplicate shard files".
    #[test]
    fn detect_resume_tolerates_two_files_for_same_shard() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("shard-s000005-r000005.parquet"), b"").unwrap();
        std::fs::write(dir.path().join("shard-s000005-r000010.parquet"), b"").unwrap();
        // Current config expects 10 rows per shard → the 10-row file is
        // canonical, the 5-row file is the to-be-removed stale rewrite.
        let st = detect_resume(dir.path(), 0..6, |_| 10).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([5]));
        assert!(st.boundary_rewrites.is_empty());
    }

    #[test]
    fn detect_resume_skips_stale_shard_ids() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 10);
        write_shard(dir.path(), 1, 10);
        // Shard 5 is outside [0, total_shards=2) — stale, ignored.
        write_shard(dir.path(), 5, 10);
        let st = detect_resume(dir.path(), 0..2, |_| 10).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([0, 1]));
        assert!(st.boundary_rewrites.is_empty());
    }

    #[test]
    fn detect_resume_ignores_non_shard_files() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 10);
        std::fs::write(dir.path().join("_manifest.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("shard-s0.parquet.tmp"), b"orphan").unwrap();
        let st = detect_resume(dir.path(), 0..4, |_| 10).unwrap();
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
        let st = detect_resume(dir.path(), 0..4, |sid| if sid == 2 { 7 } else { 10 }).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([0, 1, 2]));
    }

    /// Scope filter: shards outside `validate_range` must be skipped
    /// entirely — neither validated nor counted. Covers the multi-pod
    /// case where two pods share a tier dir on a single filesystem
    /// (uncommon, but possible via the HF primer) and one pod's
    /// post-run rescan would otherwise see and error on another pod's
    /// last (oversized-vs-this-pod's-expectation) shard. The same
    /// filter also handles the legacy "stale shard from prior larger
    /// n_games" case: stale shards live outside `0..total_shards`,
    /// caller passes `0..total_shards` as the validate range.
    #[test]
    fn detect_resume_scopes_to_validate_range() {
        let dir = tempdir().unwrap();
        write_shard(dir.path(), 0, 10);
        write_shard(dir.path(), 5, 10);
        write_shard(dir.path(), 7, 15); // oversize under any callers expectations
        // Pod owns only [5, 7) — shard 0 + shard 7 are out of scope.
        let st = detect_resume(dir.path(), 5..7, |_| 10).unwrap();
        assert_eq!(st.done_shards, BTreeSet::from([5]));
        assert!(st.boundary_rewrites.is_empty());
        // Even though shard 7 is technically oversized, we don't error on it
        // because it's outside our scope (another pod owns shard 7).
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

    fn write_tier_state(
        dir: &Path,
        shard_range: Option<ShardRange>,
        n_games: u64,
    ) -> PathBuf {
        let state = TierState {
            config_fingerprint: "fingerprint".into(),
            started_at: "2026-05-10T00:00:00Z".into(),
            n_games,
            shard_range,
        };
        state.save(dir).unwrap();
        TierState::path(dir, shard_range.as_ref())
    }

    /// Post-reconcile scoped-rerun regression: after reconciliation
    /// deletes all `_tier_state-s<A>-s<B>.json` files (leaving only the
    /// canonical `_tier_state.json`), a scoped lookup must fall back to
    /// the canonical state so the operator can re-run / extend the
    /// reconciled tier with `--shard-id-range`. Without the fallback,
    /// `TierState::load(..., Some(range))` returns `None`, and the
    /// runner aborts with "shards exist but no tier state" because the
    /// shard placeholders are still on disk.
    #[test]
    fn tier_state_scoped_load_falls_back_to_canonical() {
        let dir = tempdir().unwrap();
        // Only the canonical state exists (per-pod files deleted by reconcile).
        write_tier_state(dir.path(), None, 1000);

        // A scoped lookup for a range that has no matching per-pod file
        // falls back to canonical instead of returning None.
        let loaded = TierState::load(
            dir.path(),
            Some(&ShardRange { start: 0, end: 50 }),
        )
        .unwrap();
        assert!(loaded.is_some(), "scoped load should fall back to canonical");
        let state = loaded.unwrap();
        assert_eq!(state.n_games, 1000);
        // The canonical state has no shard_range.
        assert!(state.shard_range.is_none());
    }

    /// Negative: when a scoped state file DOES exist, the load returns
    /// that one — never the canonical. Two pods cooperating mid-run
    /// must each see their own scope's state, not each other's nor a
    /// canonical that hasn't been produced yet.
    #[test]
    fn tier_state_scoped_load_prefers_own_per_pod_file() {
        let dir = tempdir().unwrap();
        // Canonical with n_games=1000 — should NOT be returned.
        write_tier_state(dir.path(), None, 1000);
        // Per-pod state for [0, 50) with n_games=500 — SHOULD be returned.
        write_tier_state(dir.path(), Some(ShardRange { start: 0, end: 50 }), 500);

        let loaded = TierState::load(
            dir.path(),
            Some(&ShardRange { start: 0, end: 50 }),
        )
        .unwrap()
        .unwrap();
        assert_eq!(loaded.n_games, 500, "per-pod file must take priority over canonical");
    }

    /// Fallback applies ONLY to scoped lookups. Canonical lookups with
    /// no canonical file present return None as before; they don't
    /// further fall back to any per-pod file.
    #[test]
    fn tier_state_canonical_load_does_not_fall_back_to_per_pod() {
        let dir = tempdir().unwrap();
        write_tier_state(dir.path(), Some(ShardRange { start: 0, end: 50 }), 500);
        // Canonical lookup: no file → None.
        let loaded = TierState::load(dir.path(), None).unwrap();
        assert!(loaded.is_none(), "canonical lookup must not borrow a per-pod state");
    }

    #[test]
    fn scan_all_tier_states_collects_canonical_and_per_pod() {
        let dir = tempdir().unwrap();
        write_tier_state(dir.path(), None, 1000);
        write_tier_state(dir.path(), Some(ShardRange { start: 0, end: 50 }), 1000);
        write_tier_state(dir.path(), Some(ShardRange { start: 50, end: 100 }), 1000);
        // Decoys that must be ignored.
        std::fs::write(dir.path().join("_manifest.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("shard-s000000-r000001.parquet"), b"").unwrap();
        let states = scan_all_tier_states(dir.path()).unwrap();
        assert_eq!(states.len(), 3, "expected 3 tier_state files, got {states:?}");
        for (_, s) in &states {
            assert_eq!(s.n_games, 1000);
        }
    }

    #[test]
    fn enforce_n_games_invariant_passes_on_equal() {
        let dir = tempdir().unwrap();
        write_tier_state(dir.path(), None, 100);
        enforce_n_games_invariant(dir.path(), 100).unwrap();
    }

    #[test]
    fn enforce_n_games_invariant_allows_extension() {
        // The single-pod extension flow: prior run wrote state at 100,
        // current run is growing to 500. Allowed — the larger run will
        // pick up where the prior left off and fill the new tail.
        let dir = tempdir().unwrap();
        write_tier_state(dir.path(), None, 100);
        enforce_n_games_invariant(dir.path(), 500).unwrap();
    }

    #[test]
    fn enforce_n_games_invariant_refuses_when_current_is_smaller() {
        // The bug-detector #2 scenario: a cooperating pod with smaller
        // n_games would silently skip games another pod committed to.
        let dir = tempdir().unwrap();
        write_tier_state(
            dir.path(),
            Some(ShardRange { start: 0, end: 5000 }),
            100_000,
        );
        let err = enforce_n_games_invariant(dir.path(), 50_000).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("refusing to start") && msg.contains("50000 games"),
            "expected refusal-with-delta message, got: {msg}",
        );
    }

    #[test]
    fn enforce_n_games_invariant_refuses_if_any_state_is_larger() {
        // Multiple per-pod state files, ONE of them is larger than the
        // current pod's n_games — must still refuse. The check is
        // "max(existing_n_games)" not "any single one".
        let dir = tempdir().unwrap();
        write_tier_state(
            dir.path(),
            Some(ShardRange { start: 0, end: 5000 }),
            50_000,
        );
        write_tier_state(
            dir.path(),
            Some(ShardRange { start: 5000, end: 10_000 }),
            100_000, // the offending one
        );
        let err = enforce_n_games_invariant(dir.path(), 50_000).unwrap_err();
        assert!(format!("{err:#}").contains("100000"));
    }

    #[test]
    fn enforce_n_games_invariant_skips_zero_n_games_state() {
        // Pre-fix state files (written before n_games existed)
        // serialize without the field; serde's #[default] gives 0.
        // The invariant treats 0 as "unknown, can't enforce" so a
        // partial dataset from this branch's earlier commits doesn't
        // get blocked. Future runs re-write the state with the
        // current n_games and the invariant takes effect from then on.
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("_tier_state.json"),
            br#"{"config_fingerprint": "abc", "started_at": "2026-05-10T00:00:00Z"}"#,
        ).unwrap();
        // Even with a tiny current n_games, the legacy state with
        // n_games=0 doesn't block the run.
        enforce_n_games_invariant(dir.path(), 1).unwrap();
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
