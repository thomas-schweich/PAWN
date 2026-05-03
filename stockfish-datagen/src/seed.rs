//! Deterministic seed derivation. Hierarchy:
//!
//! ```text
//!     master_seed
//!         + tier_index   --> tier_seed
//!         + worker_id    --> worker_seed
//!         + game_index   --> game_seed
//! ```
//!
//! All steps are pure splitmix64 mixing — fast, deterministic, no PRNG state
//! to thread around. This means resume can recompute a worker's `game_seed`
//! at any `game_index` without replaying the prior games' RNG.
//!
//! A `(game_seed, stockfish_version, tier config)` tuple is the full
//! reproduction key for any single game in the dataset.

/// One step of splitmix64. Standard mixer; good distribution, no biases at
/// our scale (we're using <1B distinct (key, idx) pairs out of 2^64).
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn mix(parent: u64, key: u64) -> u64 {
    splitmix64(parent ^ splitmix64(key))
}

/// Derive a tier seed from the master seed and the tier's declaration index.
/// Using the index (not the name) keeps things robust to renames; renaming
/// `nodes_0128` does not invalidate that tier's existing data.
#[inline]
pub fn tier_seed(master_seed: u64, tier_index: usize) -> u64 {
    mix(master_seed, tier_index as u64)
}

#[inline]
pub fn worker_seed(tier_seed: u64, worker_id: u32) -> u64 {
    mix(tier_seed, worker_id as u64)
}

#[inline]
pub fn game_seed(worker_seed: u64, game_index: u64) -> u64 {
    mix(worker_seed, game_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn seeds_are_deterministic() {
        assert_eq!(tier_seed(42, 0), tier_seed(42, 0));
        assert_eq!(worker_seed(123, 5), worker_seed(123, 5));
        assert_eq!(game_seed(123, 9_999), game_seed(123, 9_999));
    }

    #[test]
    fn seeds_differ_across_inputs() {
        assert_ne!(tier_seed(42, 0), tier_seed(42, 1));
        assert_ne!(tier_seed(42, 0), tier_seed(43, 0));
        assert_ne!(worker_seed(123, 0), worker_seed(123, 1));
        assert_ne!(game_seed(123, 0), game_seed(123, 1));
    }

    #[test]
    fn no_collisions_in_realistic_volume() {
        // 1M game seeds from one worker — splitmix is good enough to make
        // collisions astronomically unlikely; if this ever fires, the mixer
        // is broken.
        let ws = worker_seed(0xDEAD_BEEF_CAFE_F00D, 7);
        let mut seen = HashSet::with_capacity(1_000_000);
        for i in 0..1_000_000 {
            assert!(seen.insert(game_seed(ws, i)), "collision at game_index {i}");
        }
    }
}
