//! Run config — declarative description of a multi-tier generation run.
//!
//! Loaded from a single JSON file via `--config`; everything that isn't a
//! seed is configured here. The config is hashed (sha256) and the digest is
//! stored in each tier's `_manifest.json` so resume can detect a config drift.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Tilde-expand a path. Only handles `~/...` (the common case);
/// `~user/...` is returned as-is.
fn expand_tilde(p: &Path) -> PathBuf {
    if let Ok(s) = p.strip_prefix("~") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(s);
        }
    }
    p.to_path_buf()
}

/// Top-level run config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Path to the Stockfish binary. `~` is expanded.
    pub stockfish_path: PathBuf,

    /// Exact Stockfish version string we will accept (matched against
    /// the `id name <X>` line of the UCI handshake). Pinning this makes
    /// games reproducible from `(game_seed, config)`.
    pub stockfish_version: String,

    /// Output directory. One subdirectory per tier.
    pub output_dir: PathBuf,

    /// Master RNG seed. All per-tier and per-worker / per-game seeds are
    /// derived from this deterministically.
    pub master_seed: u64,

    /// Number of worker threads (each owns one Stockfish subprocess).
    pub n_workers: u32,

    /// Per-game ply cap. Beyond this the game terminates with PLY_LIMIT.
    #[serde(default = "default_max_ply")]
    pub max_ply: u32,

    /// Stockfish `Hash` option (MB).
    #[serde(default = "default_stockfish_hash_mb")]
    pub stockfish_hash_mb: u32,

    /// Games per parquet shard. Smaller = finer resume granularity at the
    /// cost of more shard files.
    #[serde(default = "default_shard_size_games")]
    pub shard_size_games: u32,

    /// Tiers to generate, in declaration order.
    pub tiers: Vec<TierConfig>,
}

/// One tier of self-play games. Fields control sampling / search strength
/// for that tier; mixing tiers in one run yields a heterogeneous corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierConfig {
    /// Human-readable name; used as a subdirectory.
    pub name: String,

    /// Stockfish `go nodes N` budget per move.
    pub nodes: u32,

    /// Total games to generate for this tier (across all workers).
    pub n_games: u64,

    /// MultiPV used for plies in `[opening_plies, sample_plies)`.
    pub multi_pv: u32,

    /// MultiPV used for the very first plies; widens initial branching
    /// without slowing the rest of the game. Set equal to `multi_pv` to
    /// disable the per-ply schedule.
    pub opening_multi_pv: u32,

    /// Number of plies (from move 0) that use `opening_multi_pv`.
    pub opening_plies: u32,

    /// First N plies use MultiPV+softmax; beyond N, take top-1 only.
    /// Set to a very large number (e.g. 999) to keep softmax for the
    /// whole game.
    pub sample_plies: u32,

    /// Softmax temperature over centipawn scores. Scale: 100cp ~ 1 pawn,
    /// so temperature=1.0 means a 1-pawn gap shifts probability by an
    /// e-fold. <=0 falls back to argmax (top-1).
    pub temperature: f32,

    /// When true, drive Stockfish via the `searchless` UCI extension —
    /// `qsearch` is short-circuited so each per-move score is the raw NNUE
    /// static eval after the move. Requires the patched binary built via
    /// `scripts/build_patched_stockfish.sh`. `nodes` is ignored in this
    /// mode (the per-move work is one full NNUE forward, not a node-budget
    /// search). Pair with a wide `multi_pv` (e.g. 256) to score every
    /// legal move per ply.
    #[serde(default)]
    pub searchless: bool,

    /// When true, persist the full Stockfish candidates list for every ply
    /// alongside the played move. The parquet column is
    /// `legal_move_evals: List<List<Struct{move_idx: i16, score_cp: i16}>>`,
    /// indexed per game and per ply. Designed for distillation-style
    /// training (KL loss vs the per-move softmax). Independent of
    /// `searchless` — though the typical use case is `searchless=true`
    /// + `store_legal_move_evals=true` for "tier 0" datasets.
    #[serde(default)]
    pub store_legal_move_evals: bool,
}

fn default_max_ply() -> u32 {
    512
}
fn default_stockfish_hash_mb() -> u32 {
    16
}
fn default_shard_size_games() -> u32 {
    10_000
}

impl RunConfig {
    /// Load + validate a config from a JSON file.
    ///
    /// Path fields (`stockfish_path`, `output_dir`) are tilde-expanded
    /// at load time — `~/sf-data` becomes `$HOME/sf-data` before any
    /// downstream consumer sees the value. Doing this once at the
    /// boundary prevents downstream callers (the CLI's preflight
    /// `create_dir_all`, `print_plan`, the python sync orchestrator)
    /// from each having to remember to expand it themselves.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        let mut cfg: RunConfig = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))?;
        cfg.stockfish_path = expand_tilde(&cfg.stockfish_path);
        cfg.output_dir = expand_tilde(&cfg.output_dir);
        cfg.validate()?;
        Ok(cfg)
    }

    /// Sanity check: tier names unique, counts positive, scheduling consistent.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.n_workers == 0 {
            anyhow::bail!("n_workers must be > 0");
        }
        // worker_id is stored as i16 in the parquet schema.
        if self.n_workers > i16::MAX as u32 {
            anyhow::bail!(
                "n_workers={} exceeds i16::MAX ({}); the parquet `worker_id` column is Int16",
                self.n_workers,
                i16::MAX,
            );
        }
        // game_length is stored as u16. With max_ply=512 (default) we're
        // nowhere near this, but reject pathological configs explicitly.
        if self.max_ply == 0 {
            // play_game's `for ply in 0..max_ply` loop would never run,
            // returning an empty move list that the worker would mis-
            // attribute to a "terminal-check bug" hard error.
            anyhow::bail!("max_ply must be >= 1");
        }
        if self.max_ply > u16::MAX as u32 {
            anyhow::bail!(
                "max_ply={} exceeds u16::MAX ({}); the parquet `game_length` column is UInt16",
                self.max_ply,
                u16::MAX,
            );
        }
        if self.shard_size_games == 0 {
            anyhow::bail!("shard_size_games must be > 0");
        }
        if self.stockfish_hash_mb == 0 {
            // Stockfish silently ignores `setoption Hash 0` and falls
            // back to its built-in default (~16MB), so the configured
            // value would no longer reflect what's actually in use,
            // breaking the fingerprint's "this matches the running run"
            // contract. Reject up-front.
            anyhow::bail!("stockfish_hash_mb must be > 0");
        }
        if self.tiers.is_empty() {
            anyhow::bail!("at least one tier required");
        }
        let mut seen = std::collections::HashSet::new();
        for tier in &self.tiers {
            if !seen.insert(&tier.name) {
                anyhow::bail!("duplicate tier name: {}", tier.name);
            }
            if tier.n_games == 0 {
                anyhow::bail!("tier {}: n_games must be > 0", tier.name);
            }
            if tier.multi_pv == 0 || tier.opening_multi_pv == 0 {
                anyhow::bail!("tier {}: multi_pv values must be >= 1", tier.name);
            }
            if tier.opening_plies > tier.sample_plies {
                anyhow::bail!(
                    "tier {}: opening_plies ({}) must be <= sample_plies ({})",
                    tier.name,
                    tier.opening_plies,
                    tier.sample_plies,
                );
            }
            // `nodes` is only consumed in non-searchless mode (it's the
            // `go nodes K` budget). In searchless mode we send `go depth 1
            // searchless` and nodes is unused — accept any value so users
            // can leave it as a leftover from a non-searchless template.
            if !tier.searchless && tier.nodes == 0 {
                anyhow::bail!("tier {}: nodes must be >= 1 (or set searchless=true)", tier.name);
            }
        }
        Ok(())
    }

    /// Per-tier games-per-worker breakdown. Surplus games go to the
    /// lowest worker indices, matching the Python script's behavior.
    pub fn games_per_worker(&self, tier: &TierConfig) -> Vec<u64> {
        let n = self.n_workers as u64;
        let base = tier.n_games / n;
        let rem = tier.n_games % n;
        (0..n).map(|i| base + if i < rem { 1 } else { 0 }).collect()
    }

    /// Hex-encoded sha256 of the config bytes (after re-serialization, so
    /// formatting differences are normalized away). Stored alongside each
    /// tier's manifest so resume can refuse to continue under a changed
    /// config.
    ///
    /// Prefer [`Self::tier_fingerprint`] for resume-time tier validation —
    /// this whole-config hash is too coarse (adding a new tier to the
    /// config invalidates every existing tier's manifest).
    pub fn fingerprint(&self) -> String {
        let canonical = serde_json::to_vec(self).expect("config is round-trippable");
        let mut h = Sha256::new();
        h.update(&canonical);
        hex(&h.finalize())
    }

    /// Per-tier fingerprint covering only the inputs that would change
    /// the *bytes generated for that tier*. Adding/modifying *other*
    /// tiers leaves this fingerprint untouched, so users can grow their
    /// config over time without invalidating prior runs.
    ///
    /// Why every field matters for resume safety:
    /// - `tier` / `tier_index` / `master_seed`: directly seed every game.
    /// - `stockfish_version`: different NNUE → different game outcomes.
    /// - `stockfish_hash_mb`: Stockfish's transposition-table size can
    ///   affect fixed-node search choices (and therefore moves) at the
    ///   higher tiers (nodes >= 32 or so).
    /// - `shard_size_games`: changes which games end up in which shard
    ///   file, which would silently invalidate the resume row-count math.
    /// - `max_ply`: longer games may now resolve where they previously
    ///   hit `PLY_LIMIT`; outcome tokens differ.
    /// - `n_workers`: changes the (worker_id, game_index) partition AND
    ///   each game's seed (since `game_seed = mix(worker_seed, idx)`),
    ///   so an old shard's games are not a prefix of the new run.
    ///
    /// Implementation note: `serde_json::Value::Object` is `BTreeMap`
    /// (we don't enable the `preserve_order` feature), so the serialized
    /// JSON has keys in alphabetical order regardless of how they're
    /// listed below. This makes the fingerprint stable across builds.
    /// **Do not enable `preserve_order` on `serde_json` without a
    /// fingerprint version bump.** A pinned golden-value test
    /// (`tier_fingerprint_golden`) catches accidental drift.
    pub fn tier_fingerprint(&self, tier_index: usize) -> String {
        let payload = serde_json::json!({
            "tier_index": tier_index,
            "tier": &self.tiers[tier_index],
            "master_seed": self.master_seed,
            "stockfish_version": &self.stockfish_version,
            "stockfish_hash_mb": self.stockfish_hash_mb,
            "shard_size_games": self.shard_size_games,
            "max_ply": self.max_ply,
            "n_workers": self.n_workers,
        });
        let mut h = Sha256::new();
        h.update(payload.to_string().as_bytes());
        hex(&h.finalize())
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> RunConfig {
        RunConfig {
            stockfish_path: "/usr/bin/stockfish".into(),
            stockfish_version: "Stockfish 18".into(),
            output_dir: "out".into(),
            master_seed: 42,
            n_workers: 4,
            max_ply: 512,
            stockfish_hash_mb: 16,
            shard_size_games: 1000,
            tiers: vec![TierConfig {
                name: "nodes_0001".into(),
                nodes: 1,
                n_games: 100,
                multi_pv: 5,
                opening_multi_pv: 20,
                opening_plies: 1,
                sample_plies: 999,
                temperature: 1.0,
                searchless: false,
                store_legal_move_evals: false,
            }],
        }
    }

    #[test]
    fn validate_accepts_minimal_config() {
        minimal_config().validate().unwrap();
    }

    #[test]
    fn load_expands_tilde_in_path_fields() {
        // Pin the contract that `RunConfig::load` is the one place
        // tilde expansion happens — every downstream consumer (CLI
        // preflight, run_tier, print_plan, the Python orchestrator)
        // assumes the path it sees is already absolute.
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("c.json");
        let body = r#"{
            "stockfish_path": "~/bin/stockfish",
            "stockfish_version": "Stockfish",
            "output_dir": "~/sf-data",
            "master_seed": 1,
            "n_workers": 1,
            "max_ply": 16,
            "stockfish_hash_mb": 1,
            "shard_size_games": 4,
            "tiers": [{
                "name": "t",
                "nodes": 1,
                "n_games": 4,
                "multi_pv": 1,
                "opening_multi_pv": 1,
                "opening_plies": 0,
                "sample_plies": 1,
                "temperature": 1.0
            }]
        }"#;
        std::fs::File::create(&cfg_path).unwrap().write_all(body.as_bytes()).unwrap();
        let cfg = RunConfig::load(&cfg_path).unwrap();
        let home = std::env::var("HOME").unwrap();
        assert_eq!(cfg.output_dir, PathBuf::from(&home).join("sf-data"));
        assert_eq!(cfg.stockfish_path, PathBuf::from(&home).join("bin/stockfish"));
    }

    #[test]
    fn validate_rejects_zero_workers() {
        let mut c = minimal_config();
        c.n_workers = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_duplicate_tier_names() {
        let mut c = minimal_config();
        c.tiers.push(c.tiers[0].clone());
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_opening_after_sample() {
        let mut c = minimal_config();
        c.tiers[0].opening_plies = 50;
        c.tiers[0].sample_plies = 10;
        assert!(c.validate().is_err());
    }

    #[test]
    fn games_per_worker_distributes_remainder() {
        let mut c = minimal_config();
        c.n_workers = 3;
        c.tiers[0].n_games = 10;
        let split = c.games_per_worker(&c.tiers[0]);
        assert_eq!(split, vec![4, 3, 3]);
        assert_eq!(split.iter().sum::<u64>(), 10);
    }

    #[test]
    fn fingerprint_is_stable() {
        let c = minimal_config();
        assert_eq!(c.fingerprint(), c.fingerprint());
    }

    #[test]
    fn fingerprint_changes_with_seed() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.master_seed = 1;
        b.master_seed = 2;
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn validate_rejects_zero_shard_size() {
        let mut c = minimal_config();
        c.shard_size_games = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_oversized_n_workers() {
        let mut c = minimal_config();
        c.n_workers = i16::MAX as u32 + 1;
        let err = c.validate().unwrap_err();
        assert!(format!("{err:#}").contains("Int16"), "got {err:#}");
    }

    #[test]
    fn validate_rejects_oversized_max_ply() {
        let mut c = minimal_config();
        c.max_ply = u16::MAX as u32 + 1;
        let err = c.validate().unwrap_err();
        assert!(format!("{err:#}").contains("UInt16"), "got {err:#}");
    }

    #[test]
    fn tier_fingerprint_isolates_tiers() {
        let mut a = minimal_config();
        a.tiers.push(TierConfig {
            name: "second".into(),
            nodes: 32,
            n_games: 50,
            multi_pv: 5,
            opening_multi_pv: 5,
            opening_plies: 0,
            sample_plies: 999,
            temperature: 1.0,
            searchless: false,
            store_legal_move_evals: false,
        });
        let mut b = a.clone();
        // Change only tier 1.
        b.tiers[1].n_games = 75;
        // Tier 0's fingerprint must NOT change just because tier 1 did.
        assert_eq!(a.tier_fingerprint(0), b.tier_fingerprint(0));
        // Tier 1's fingerprint MUST change.
        assert_ne!(a.tier_fingerprint(1), b.tier_fingerprint(1));
    }

    #[test]
    fn tier_fingerprint_changes_with_master_seed() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.master_seed = 1;
        b.master_seed = 2;
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }

    #[test]
    fn tier_fingerprint_changes_with_stockfish_version() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.stockfish_version = "Stockfish 17".into();
        b.stockfish_version = "Stockfish 18".into();
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }

    #[test]
    fn tier_fingerprint_changes_with_max_ply() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.max_ply = 256;
        b.max_ply = 512;
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }

    #[test]
    fn tier_fingerprint_changes_with_n_workers() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.n_workers = 4;
        b.n_workers = 8;
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }

    /// Pin the exact serialized JSON byte-for-byte. If serde or our
    /// payload structure ever changes the emission order or formatting,
    /// this test fires loudly — better than silently invalidating every
    /// existing manifest in production.
    ///
    /// To regenerate after an INTENTIONAL change: run the test, copy
    /// the printed `actual` value below.
    #[test]
    fn tier_fingerprint_golden() {
        let cfg = minimal_config();
        let actual = cfg.tier_fingerprint(0);
        // Updated when `searchless` and `store_legal_move_evals` were added
        // to TierConfig (intentional schema change for distillation tier 0).
        let expected = "62c112888412402b9fe74cbcf18cbedb821e45d316a8e9e0dc448cfa8e1c8693";
        assert_eq!(
            actual, expected,
            "tier_fingerprint changed for minimal_config — verify the change is intentional, then update the expected value"
        );
    }

    #[test]
    fn validate_rejects_zero_max_ply() {
        let mut c = minimal_config();
        c.max_ply = 0;
        let err = c.validate().unwrap_err();
        assert!(format!("{err:#}").contains("max_ply"), "got {err:#}");
    }

    #[test]
    fn validate_rejects_zero_stockfish_hash_mb() {
        let mut c = minimal_config();
        c.stockfish_hash_mb = 0;
        let err = c.validate().unwrap_err();
        assert!(format!("{err:#}").contains("stockfish_hash_mb"), "got {err:#}");
    }

    #[test]
    fn tier_fingerprint_changes_with_stockfish_hash_mb() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.stockfish_hash_mb = 16;
        b.stockfish_hash_mb = 64;
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }
}
