//! Deterministic seed derivation. Hierarchy:
//!
//! ```text
//!     master_seed
//!         + tier.name    --> tier_seed
//!         + game_index   --> game_seed
//! ```
//!
//! All steps are pure splitmix64 mixing — fast, deterministic, no PRNG state
//! to thread around. The `tier_seed` is keyed by the **tier name** (hashed
//! to a u64 via the first 8 bytes of its sha256), NOT the tier's index in
//! the config's `tiers` list. This means appending, reordering, or removing
//! other tiers in the config doesn't change a tier's seed-and-thus-bytes;
//! renaming a tier *does* (treat rename as creating a new tier).
//!
//! The `game_seed` depends only on the **global** game index within the
//! tier — independent of `n_workers` or any per-worker partition. That
//! frees `n_workers` to be an operational knob: changing it between runs
//! never changes any game's content, only which worker thread generates
//! it. The shard-id–based partitioning in `runner.rs` relies on this.
//!
//! A `(game_seed, stockfish_version, tier config)` tuple is the full
//! reproduction key for any single game in the dataset.
//!
//! See `RunConfig::tier_fingerprint` for the resume-safety contract that
//! reflects this seeding model.

use sha2::{Digest, Sha256};

/// One step of splitmix64. Standard mixer; good distribution, no biases at
/// our scale (we're using <1B distinct (key, idx) pairs out of 2^64).
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Splitmix64-based 2-input mixer. Public so non-dataset callers (e.g.
/// `tournament.rs`) can derive their own deterministic seed hierarchies
/// without re-implementing the mixer. For dataset seeding use the
/// `tier_seed` / `game_seed` API below.
#[inline]
pub fn mix(parent: u64, key: u64) -> u64 {
    splitmix64(parent ^ splitmix64(key))
}

/// Hash a tier name to a u64 via the first 8 bytes of its sha256. Used as
/// the "key" input to `mix` when deriving `tier_seed`. SHA-256 is overkill
/// for collision avoidance here, but it's already pulled in for the
/// config-fingerprint code path and produces a stable mapping across
/// architectures (unlike `DefaultHasher`, which is intentionally
/// implementation-defined). 8 bytes give us collision odds of ~10^-19 per
/// pair, irrelevant at any realistic tier count.
pub fn tier_name_to_key(tier_name: &str) -> u64 {
    let h = Sha256::digest(tier_name.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&h[..8]);
    u64::from_be_bytes(bytes)
}

/// Derive a tier seed from the master seed and the tier's name. Keying by
/// name (not list index) means reordering, inserting, or removing other
/// tiers in the config doesn't invalidate this tier's data.
#[inline]
pub fn tier_seed(master_seed: u64, tier_name: &str) -> u64 {
    mix(master_seed, tier_name_to_key(tier_name))
}

/// Derive a per-game seed from the tier seed and a **global** game index
/// within the tier (0..n_games). Independent of `n_workers` or any
/// per-worker partition.
#[inline]
pub fn game_seed(tier_seed: u64, global_game_index: u64) -> u64 {
    mix(tier_seed, global_game_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn seeds_are_deterministic() {
        assert_eq!(tier_seed(42, "nodes_0001"), tier_seed(42, "nodes_0001"));
        assert_eq!(game_seed(123, 9_999), game_seed(123, 9_999));
    }

    #[test]
    fn seeds_differ_across_inputs() {
        assert_ne!(tier_seed(42, "nodes_0001"), tier_seed(42, "nodes_0128"));
        assert_ne!(tier_seed(42, "nodes_0001"), tier_seed(43, "nodes_0001"));
        assert_ne!(game_seed(123, 0), game_seed(123, 1));
    }

    #[test]
    fn tier_name_to_key_is_stable() {
        // Pin the exact mapping so a future refactor doesn't silently
        // change the seeds for an existing dataset name. If this fires,
        // verify the mapping change was intentional.
        let k = tier_name_to_key("nodes_0001");
        assert_eq!(k, 13853672269570116244u64);
        // Different name produces a different key (collision-resistant).
        assert_ne!(tier_name_to_key("nodes_0001"), tier_name_to_key("nodes_0002"));
    }

    #[test]
    fn no_collisions_in_realistic_volume() {
        // 1M game seeds from one tier — splitmix is good enough to make
        // collisions astronomically unlikely; if this ever fires, the mixer
        // is broken.
        let ts = tier_seed(0xDEAD_BEEF_CAFE_F00D, "nodes_0001");
        let mut seen = HashSet::with_capacity(1_000_000);
        for i in 0..1_000_000 {
            assert!(seen.insert(game_seed(ts, i)), "collision at game_index {i}");
        }
    }
}
