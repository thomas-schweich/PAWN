//! Run config — declarative description of a multi-tier generation run.
//!
//! Loaded from a single JSON file via `--config`; everything that isn't a
//! seed is configured here. The config is hashed (sha256) and the digest is
//! stored in each tier's `_manifest.json` so resume can detect a config drift.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        let cfg: RunConfig = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))?;
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
            if tier.nodes == 0 {
                anyhow::bail!("tier {}: nodes must be >= 1", tier.name);
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
    /// the *bytes generated for that tier*: the tier's own config, its
    /// declaration index, the master seed, the pinned Stockfish version,
    /// and the shard size. Adding/modifying *other* tiers leaves this
    /// fingerprint untouched, so users can grow their config over time
    /// without invalidating prior runs.
    pub fn tier_fingerprint(&self, tier_index: usize) -> String {
        let payload = serde_json::json!({
            "tier_index": tier_index,
            "tier": &self.tiers[tier_index],
            "master_seed": self.master_seed,
            "stockfish_version": &self.stockfish_version,
            "shard_size_games": self.shard_size_games,
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
            }],
        }
    }

    #[test]
    fn validate_accepts_minimal_config() {
        minimal_config().validate().unwrap();
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
}
