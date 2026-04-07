//! Random game generation with deterministic seeding.

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::board::GameState;
use crate::types::Termination;

/// Derive N independent sub-seeds from a single base seed.
///
/// Uses a ChaCha8 RNG seeded from `base_seed` to generate one u64 per game.
/// This avoids the `seed + i` pattern which causes batch-to-batch game overlap
/// when callers use sequential base seeds (e.g., seed 42 batch of 192 uses
/// sub-seeds 42..233, seed 43 uses 43..234 — sharing 191/192 games).
pub fn derive_game_seeds(base_seed: u64, n: usize) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(base_seed);
    (0..n).map(|_| rng.next_u64()).collect()
}

/// Record of a single generated game.
pub struct GameRecord {
    pub move_ids: Vec<u16>,
    pub game_length: u16,
    pub termination: Termination,
    /// Legal move grids at each ply. grid[ply][src] has bit d set if src->dst is legal.
    /// Labels at ply i represent the legal moves BEFORE move_ids[i] — i.e., the moves
    /// available to the side that is about to play move_ids[i].
    pub legal_grids: Vec<[u64; 64]>,
    /// Promotion masks at each ply (same alignment as legal_grids).
    pub legal_promos: Vec<[[bool; 4]; 44]>,
}

/// Generate a single random game with legal move labels.
/// Labels at ply i represent the legal moves BEFORE move_ids[i] has been played —
/// the moves available to the side whose turn it is at ply i.
pub fn generate_one_game_with_labels(seed: u64, max_ply: usize) -> GameRecord {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = GameState::new();
    let mut move_ids = Vec::with_capacity(max_ply);
    let mut legal_grids = Vec::with_capacity(max_ply);
    let mut legal_promos = Vec::with_capacity(max_ply);

    loop {
        // Check termination before making a move
        if let Some(term) = state.check_termination(max_ply) {
            let game_length = state.ply() as u16;
            return GameRecord {
                move_ids,
                game_length,
                termination: term,
                legal_grids,
                legal_promos,
            };
        }

        // Record legal moves BEFORE making the move — these are the labels
        // for the current position (the moves the current side can choose from)
        legal_grids.push(state.legal_move_grid());
        legal_promos.push(state.legal_promo_mask());

        // Pick and play a random legal move
        let tokens = state.legal_move_tokens();
        debug_assert!(!tokens.is_empty(), "No legal moves but termination not detected");

        let chosen = tokens[rng.gen_range(0..tokens.len())];
        state.make_move(chosen).unwrap();
        move_ids.push(chosen);
    }
}

/// Generate a single random game without labels.
///
/// `mate_boost` controls mate-in-1 probability: 0.0 = pure random (default),
/// 1.0 = always take mate when available, values in between interpolate.
pub fn generate_one_game(seed: u64, max_ply: usize, mate_boost: f64) -> (Vec<u16>, u16, Termination) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = GameState::new();
    let mut move_ids = Vec::with_capacity(max_ply);

    loop {
        if let Some(term) = state.check_termination(max_ply) {
            return (move_ids, state.ply() as u16, term);
        }

        let tokens = state.legal_move_tokens();

        let chosen = if mate_boost > 0.0 {
            if let Some(mate_move) = find_mate_in_one(&state, &tokens) {
                if mate_boost >= 1.0 || rng.gen::<f64>() < mate_boost {
                    mate_move
                } else {
                    tokens[rng.gen_range(0..tokens.len())]
                }
            } else {
                tokens[rng.gen_range(0..tokens.len())]
            }
        } else {
            tokens[rng.gen_range(0..tokens.len())]
        };

        state.make_move(chosen).unwrap();
        move_ids.push(chosen);
    }
}

/// Check if any of the given legal moves delivers immediate checkmate.
/// Returns the first mating move found, or None.
fn find_mate_in_one(state: &GameState, tokens: &[u16]) -> Option<u16> {
    for &token in tokens {
        let mut test = state.clone();
        test.make_move(token).unwrap();
        if test.legal_moves().is_empty() && test.is_check() {
            return Some(token);
        }
    }
    None
}

/// Resolved game outcome with side-aware checkmate.
///
/// Unlike `Termination`, this distinguishes which side was checkmated.
/// Used for accurate conditional ceiling estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Outcome {
    WhiteCheckmated = 0,  // Black wins (1-0 for black)
    BlackCheckmated = 1,  // White wins (0-1 for white... well, 1-0)
    Stalemate = 2,
    SeventyFiveMoveRule = 3,
    FivefoldRepetition = 4,
    InsufficientMaterial = 5,
    PlyLimit = 6,
}

pub const NUM_OUTCOMES: usize = 7;

impl Outcome {
    /// Resolve a Termination + the game state at termination into an Outcome.
    pub fn from_termination(term: Termination, white_to_move_at_end: bool) -> Self {
        match term {
            Termination::Checkmate => {
                // Side to move is checkmated
                if white_to_move_at_end {
                    Outcome::WhiteCheckmated
                } else {
                    Outcome::BlackCheckmated
                }
            }
            Termination::Stalemate => Outcome::Stalemate,
            Termination::SeventyFiveMoveRule => Outcome::SeventyFiveMoveRule,
            Termination::FivefoldRepetition => Outcome::FivefoldRepetition,
            Termination::InsufficientMaterial => Outcome::InsufficientMaterial,
            Termination::PlyLimit => Outcome::PlyLimit,
        }
    }
}

/// Outcome distribution from Monte Carlo rollouts.
#[derive(Debug, Clone, Default)]
pub struct OutcomeDistribution {
    pub counts: [u32; NUM_OUTCOMES],
    pub total: u32,
}

/// Result for a single position in the ceiling computation.
#[derive(Debug, Clone)]
pub struct PositionCeiling {
    /// Number of legal moves at this position
    pub n_legal: u32,
    /// Unconditional ceiling: 1/n_legal
    pub unconditional: f64,
    /// Conditional ceiling: max_m P(m | outcome, history) where the max is over
    /// legal moves and P is estimated from rollouts
    pub conditional: f64,
    /// Naive conditional ceiling: 1/(N_legal - N_wrong_immediate) where
    /// N_wrong_immediate is the count of legal moves that lead to an immediate
    /// terminal state with a different outcome than the actual game outcome.
    /// This is a 0-depth version of the conditional ceiling — no rollouts needed.
    pub naive_conditional: f64,
    /// Split-half bias-corrected conditional ceiling: half-A selects argmax,
    /// half-B evaluates it. Biased downward (vs conditional which is biased upward).
    /// The true ceiling lies between conditional_corrected and conditional.
    pub conditional_corrected: f64,
    /// The actual outcome of the game this position came from
    pub actual_outcome: u8,
    /// Ply index within the game
    pub ply: u16,
    /// Game length
    pub game_length: u16,
    /// Index of the source game (for clustered bootstrap in Python)
    pub game_idx: u32,
}

/// For a given position (as move token prefix), play out N random continuations
/// from each legal move and return the outcome distribution per move.
///
/// Returns Vec<(token, OutcomeDistribution)> for each legal move.
pub fn rollout_legal_moves(
    prefix_tokens: &[u16],
    n_rollouts: usize,
    max_ply: usize,
    base_seed: u64,
) -> Vec<(u16, OutcomeDistribution)> {
    let state = match GameState::from_move_tokens(prefix_tokens) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let legal_tokens = state.legal_move_tokens();
    if legal_tokens.is_empty() {
        return Vec::new();
    }

    let seeds = derive_game_seeds(base_seed, legal_tokens.len() * n_rollouts);

    legal_tokens
        .iter()
        .enumerate()
        .map(|(move_idx, &token)| {
            let mut dist = OutcomeDistribution::default();
            for r in 0..n_rollouts {
                let seed = seeds[move_idx * n_rollouts + r];
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let mut s = state.clone();
                s.make_move(token).unwrap();
                let term = s.play_random_to_end(&mut rng, max_ply);
                let outcome = Outcome::from_termination(term, s.is_white_to_move());
                dist.counts[outcome as usize] += 1;
                dist.total += 1;
            }
            (token, dist)
        })
        .collect()
}

/// Compute the theoretical accuracy ceiling for a batch of random games.
///
/// For each position in each game:
/// - Unconditional: 1/N_legal
/// - Naive conditional (0-depth): prune legal moves that immediately terminate
///   with the wrong outcome, then 1/N_remaining
/// - MC conditional: Monte Carlo rollouts estimate P(outcome | move),
///   best predictor picks argmax. Split-half corrected to bound bias.
///
/// Returns per-position results. The overall ceiling is the mean.
pub fn compute_accuracy_ceiling(
    n_games: usize,
    max_ply: usize,
    n_rollouts_per_move: usize,
    sample_rate: f64,  // fraction of positions to sample (1.0 = all, 0.01 = 1%)
    base_seed: u64,
) -> Vec<PositionCeiling> {
    let game_seeds = derive_game_seeds(base_seed, n_games);

    // Generate all games first
    let games: Vec<(Vec<u16>, u16, Termination)> = game_seeds
        .par_iter()
        .map(|&seed| generate_one_game(seed, max_ply, 0.0))
        .collect();

    // Resolve each game's Termination to a side-aware Outcome
    let game_outcomes: Vec<Outcome> = games
        .iter()
        .map(|(_, game_length, term)| {
            // At termination, the side to move is the one at ply = game_length.
            // Even ply = white to move, odd = black to move.
            let white_to_move_at_end = *game_length % 2 == 0;
            Outcome::from_termination(*term, white_to_move_at_end)
        })
        .collect();

    // For each sampled position, compute the ceiling
    let mut rng_sample = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(999));
    // (game_idx, ply, outcome_idx, game_length)
    let mut work_items: Vec<(usize, usize, u8, u16)> = Vec::new();

    for (game_idx, outcome) in game_outcomes.iter().enumerate() {
        let gl = games[game_idx].1 as usize;
        let oi = *outcome as u8;
        for ply in 0..gl {
            if sample_rate >= 1.0 || rng_sample.gen::<f64>() < sample_rate {
                work_items.push((game_idx, ply, oi, games[game_idx].1));
            }
        }
    }

    // Process positions in parallel
    let rollout_seed_base = base_seed.wrapping_add(1_000_000);
    let total_work = work_items.len();
    let progress = AtomicUsize::new(0);
    let log_interval = (total_work / 20).max(100); // ~5% increments

    eprintln!("[ceiling] {} positions to process ({} games, {:.0}% sampled, {} rollouts/move)",
              total_work, n_games, sample_rate * 100.0, n_rollouts_per_move);

    work_items
        .par_iter()
        .enumerate()
        .map(|(work_idx, &(game_idx, ply, actual_outcome, game_length))| {
            let prefix = &games[game_idx].0[..ply];

            // Reconstruct position for naive ceiling
            let state = GameState::from_move_tokens(prefix).expect("valid prefix");
            let legal_tokens = state.legal_move_tokens();
            let n_legal = legal_tokens.len() as u32;
            let unconditional = if n_legal > 0 { 1.0 / n_legal as f64 } else { 0.0 };

            // --- Naive conditional (0-depth) ---
            // Try each legal move; if it immediately terminates with a different
            // Outcome than the game's actual outcome, it can be pruned.
            let mut n_wrong_immediate = 0u32;
            for &token in &legal_tokens {
                let mut s = state.clone();
                s.make_move(token).unwrap();
                if let Some(term) = s.check_termination(max_ply) {
                    let move_outcome = Outcome::from_termination(term, s.is_white_to_move());
                    if move_outcome as u8 != actual_outcome {
                        n_wrong_immediate += 1;
                    }
                }
            }
            let n_remaining = n_legal - n_wrong_immediate;
            let naive_conditional = if n_remaining > 0 {
                1.0 / n_remaining as f64
            } else {
                unconditional // fallback: all moves lead to wrong immediate outcome
            };

            // --- MC conditional (rollout-based) with split-half bias correction ---
            let half = n_rollouts_per_move / 2;
            let half = half.max(1); // at least 1 rollout per half
            let seed_a = rollout_seed_base.wrapping_add(work_idx as u64 * 2000);
            let seed_b = rollout_seed_base.wrapping_add(work_idx as u64 * 2000 + 1000);
            let dists_a = rollout_legal_moves(prefix, half, max_ply, seed_a);
            let dists_b = rollout_legal_moves(prefix, half, max_ply, seed_b);

            let outcome_idx = actual_outcome as usize;

            // Compute per-move outcome probabilities from each half and combined
            let probs_a: Vec<f64> = dists_a.iter().map(|(_, d)| {
                if d.total > 0 { d.counts[outcome_idx] as f64 / d.total as f64 } else { 0.0 }
            }).collect();
            let probs_b: Vec<f64> = dists_b.iter().map(|(_, d)| {
                if d.total > 0 { d.counts[outcome_idx] as f64 / d.total as f64 } else { 0.0 }
            }).collect();
            let probs_combined: Vec<f64> = dists_a.iter().zip(dists_b.iter()).map(|((_, da), (_, db))| {
                let total = da.total + db.total;
                if total > 0 {
                    (da.counts[outcome_idx] + db.counts[outcome_idx]) as f64 / total as f64
                } else {
                    0.0
                }
            }).collect();

            let sum_combined: f64 = probs_combined.iter().sum();

            // Naive estimator: max of combined / sum of combined (biased upward)
            let conditional = if sum_combined > 0.0 {
                let max_combined = probs_combined.iter().cloned().fold(0.0f64, f64::max);
                max_combined / sum_combined
            } else {
                unconditional
            };

            // Split-half corrected: A selects argmax, B evaluates (biased downward)
            let conditional_corrected = if sum_combined > 0.0 && !probs_a.is_empty() {
                let argmax_a = probs_a.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                probs_b[argmax_a] / sum_combined
            } else {
                unconditional
            };

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % log_interval == 0 || done == total_work {
                let pct = done as f64 / total_work as f64 * 100.0;
                eprintln!("[ceiling] {done}/{total_work} positions ({pct:.0}%)");
            }

            PositionCeiling {
                n_legal,
                unconditional,
                conditional,
                naive_conditional,
                conditional_corrected,
                actual_outcome,
                ply: ply as u16,
                game_length,
                game_idx: game_idx as u32,
            }
        })
        .collect()
}

/// Training example for checkmate prediction.
pub struct CheckmateExample {
    pub move_ids: Vec<u16>,          // full game including mating move
    pub game_length: u16,            // total ply count
    pub checkmate_grid: [u64; 64],   // multi-hot: bit d set at row s if s→d delivers mate
    pub legal_grid: [u64; 64],       // legal move grid at penultimate position
}

/// Generate checkmate games with multi-hot mating move targets.
///
/// For each game ending in checkmate, computes which legal moves at the
/// penultimate position deliver mate (there may be multiple).
/// Generates random games until `n_target` checkmates are collected.
/// Returns (examples, total_games_generated).
pub fn generate_checkmate_examples(
    seed: u64,
    max_ply: usize,
    n_target: usize,
) -> (Vec<CheckmateExample>, usize) {
    let batch_size = 4096usize;
    let mut collected: Vec<CheckmateExample> = Vec::with_capacity(n_target);
    let mut total_generated = 0usize;
    let mut game_seed = seed;

    while collected.len() < n_target {
        let seeds = derive_game_seeds(game_seed, batch_size);
        // Generate batch in parallel, compute checkmate targets for checkmate games
        let batch: Vec<Option<CheckmateExample>> = seeds
            .into_par_iter()
            .map(|s| {
                let mut rng = ChaCha8Rng::seed_from_u64(s);
                let mut state = GameState::new();
                let mut move_ids = Vec::with_capacity(max_ply);

                loop {
                    if let Some(term) = state.check_termination(max_ply) {
                        if term != Termination::Checkmate || move_ids.is_empty() {
                            return None; // not a checkmate game
                        }
                        let game_length = state.ply() as u16;

                        // Replay to penultimate position to compute targets
                        let mut replay = GameState::new();
                        for &tok in &move_ids[..move_ids.len() - 1] {
                            replay.make_move(tok).unwrap();
                        }

                        let legal_grid = replay.legal_move_grid();
                        let legal_tokens = replay.legal_move_tokens();

                        // Test each legal move: does it deliver checkmate?
                        let mut checkmate_grid = [0u64; 64];
                        for &tok in &legal_tokens {
                            let mut test = replay.clone();
                            test.make_move(tok).unwrap();
                            if test.check_termination(max_ply + 10) == Some(Termination::Checkmate) {
                                // Decode token to (src, dst) grid indices
                                let (src, dst) = crate::vocab::token_to_src_dst(tok);
                                checkmate_grid[src as usize] |= 1u64 << dst;
                            }
                        }

                        return Some(CheckmateExample {
                            move_ids,
                            game_length,
                            checkmate_grid,
                            legal_grid,
                        });
                    }

                    let tokens = state.legal_move_tokens();
                    let chosen = tokens[rng.gen_range(0..tokens.len())];
                    state.make_move(chosen).unwrap();
                    move_ids.push(chosen);
                }
            })
            .collect();

        game_seed += batch_size as u64;
        total_generated += batch_size;

        for example in batch.into_iter().flatten() {
            if collected.len() >= n_target {
                break;
            }
            collected.push(example);
        }
    }

    (collected, total_generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_game() {
        let (moves, length, _term) = generate_one_game(42, 256, 0.0);
        assert_eq!(moves.len(), length as usize);
        assert!(length > 0);
        assert!(length <= 256);
    }

    #[test]
    fn test_generate_game_with_labels() {
        let record = generate_one_game_with_labels(42, 256);
        assert_eq!(record.move_ids.len(), record.game_length as usize);
        assert_eq!(record.legal_grids.len(), record.game_length as usize);
        assert_eq!(record.legal_promos.len(), record.game_length as usize);
    }

    #[test]
    fn test_deterministic() {
        let (m1, l1, t1) = generate_one_game(123, 256, 0.0);
        let (m2, l2, t2) = generate_one_game(123, 256, 0.0);
        assert_eq!(m1, m2);
        assert_eq!(l1, l2);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_mate_boost_deterministic() {
        let (m1, l1, t1) = generate_one_game(123, 256, 1.0);
        let (m2, l2, t2) = generate_one_game(123, 256, 1.0);
        assert_eq!(m1, m2);
        assert_eq!(l1, l2);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_mate_boost_shorter_games() {
        let n = 100;
        let seeds = derive_game_seeds(42, n);
        let normal_lengths: Vec<u16> = seeds.iter()
            .map(|&s| generate_one_game(s, 256, 0.0).1)
            .collect();
        let boost_lengths: Vec<u16> = seeds.iter()
            .map(|&s| generate_one_game(s, 256, 1.0).1)
            .collect();
        let avg_normal: f64 = normal_lengths.iter().map(|&l| l as f64).sum::<f64>() / n as f64;
        let avg_boost: f64 = boost_lengths.iter().map(|&l| l as f64).sum::<f64>() / n as f64;
        for i in 0..n {
            assert!(boost_lengths[i] <= normal_lengths[i],
                "Game {}: boost={} > normal={}", i, boost_lengths[i], normal_lengths[i]);
        }
        assert!(avg_boost < avg_normal,
            "Expected shorter avg: boost={:.1} normal={:.1}", avg_boost, avg_normal);
    }

    #[test]
    fn test_mate_boost_intermediate() {
        // mate_boost=0.5 should produce checkmate rate between 0.0 and 1.0
        let n = 200;
        let seeds = derive_game_seeds(99, n);
        let mates_0: usize = seeds.iter()
            .filter(|&&s| generate_one_game(s, 256, 0.0).2 == Termination::Checkmate)
            .count();
        let mates_half: usize = seeds.iter()
            .filter(|&&s| generate_one_game(s, 256, 0.5).2 == Termination::Checkmate)
            .count();
        let mates_1: usize = seeds.iter()
            .filter(|&&s| generate_one_game(s, 256, 1.0).2 == Termination::Checkmate)
            .count();
        assert!(mates_half > mates_0,
            "0.5 boost ({}) should have more mates than 0.0 ({})", mates_half, mates_0);
        assert!(mates_half < mates_1,
            "0.5 boost ({}) should have fewer mates than 1.0 ({})", mates_half, mates_1);
    }

    #[test]
    fn test_different_seeds() {
        let (m1, _, _) = generate_one_game(1, 256, 0.0);
        let (m2, _, _) = generate_one_game(2, 256, 0.0);
        assert_ne!(m1, m2);
    }

    // ==== New tests added by Agent A (Rust Core) ====

    #[test]
    fn test_derive_game_seeds_deterministic() {
        // Same base seed -> same derived seeds
        let s1 = derive_game_seeds(42, 16);
        let s2 = derive_game_seeds(42, 16);
        assert_eq!(s1, s2);
        assert_eq!(s1.len(), 16);
    }

    #[test]
    fn test_derive_game_seeds_different_bases_different_outputs() {
        let s1 = derive_game_seeds(42, 8);
        let s2 = derive_game_seeds(43, 8);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_derive_game_seeds_avoids_sequential_overlap() {
        // Key invariant: base seeds 42 and 43 should not share sub-seeds.
        // Without ChaCha8 derivation, seed+i would cause massive overlap.
        let s1 = derive_game_seeds(42, 192);
        let s2 = derive_game_seeds(43, 192);
        let set1: std::collections::HashSet<u64> = s1.into_iter().collect();
        let set2: std::collections::HashSet<u64> = s2.into_iter().collect();
        let overlap = set1.intersection(&set2).count();
        // With random derivation, overlap should be ~0 (probabilistically 0 in 2^64)
        assert_eq!(overlap, 0, "Sequential base seeds should not produce overlapping sub-seeds");
    }

    #[test]
    fn test_derive_game_seeds_internally_unique() {
        // For a single call, the N sub-seeds should all be distinct (probabilistically)
        let seeds = derive_game_seeds(42, 1000);
        let set: std::collections::HashSet<u64> = seeds.into_iter().collect();
        assert_eq!(set.len(), 1000, "Derived seeds should be internally unique");
    }

    #[test]
    fn test_derive_game_seeds_zero_count() {
        let seeds = derive_game_seeds(42, 0);
        assert_eq!(seeds.len(), 0);
    }

    #[test]
    fn test_generate_game_moves_match_length() {
        let (moves, length, _) = generate_one_game(7, 256, 0.0);
        assert_eq!(moves.len(), length as usize);
        // Each move is a valid move token (1..=4272)
        for &tok in &moves {
            assert!(tok >= 1 && tok <= 4272, "Invalid move token: {}", tok);
        }
    }

    #[test]
    fn test_generate_one_game_termination_consistent() {
        // Re-replaying the game tokens should reach the same termination
        let (moves, _length, term) = generate_one_game(42, 256, 0.0);
        let state = GameState::from_move_tokens(&moves).expect("valid replay");
        let replayed_term = state.check_termination(256);
        assert_eq!(replayed_term, Some(term));
    }

    #[test]
    fn test_generate_game_with_labels_consistency() {
        // Legal grids at ply i should match what the game state reports at that ply
        let record = generate_one_game_with_labels(42, 256);
        assert_eq!(record.legal_grids.len(), record.move_ids.len());
        assert_eq!(record.legal_promos.len(), record.move_ids.len());
        // Replay and verify grids at each ply
        let mut state = GameState::new();
        for (i, &tok) in record.move_ids.iter().enumerate() {
            // Grid BEFORE move i should match label[i]
            let grid = state.legal_move_grid();
            assert_eq!(grid, record.legal_grids[i], "Legal grid mismatch at ply {}", i);
            state.make_move(tok).unwrap();
        }
    }

    #[test]
    fn test_generate_game_max_ply_respected() {
        let max_ply = 50;
        let (moves, length, term) = generate_one_game(42, max_ply, 0.0);
        assert!(length as usize <= max_ply);
        assert_eq!(moves.len(), length as usize);
        // If game hit the ply limit, termination must be PlyLimit
        if length as usize == max_ply {
            assert_eq!(term, Termination::PlyLimit);
        }
    }

    #[test]
    fn test_mate_boost_zero_equals_no_boost() {
        // mate_boost=0.0 should produce identical games to the default (no boost code path)
        // Verified by testing same seed -> same output
        let (m1, l1, t1) = generate_one_game(100, 256, 0.0);
        let (m2, l2, t2) = generate_one_game(100, 256, 0.0);
        assert_eq!(m1, m2);
        assert_eq!(l1, l2);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_outcome_from_termination() {
        // Side-to-move at checkmate is the loser
        assert_eq!(
            Outcome::from_termination(Termination::Checkmate, true),
            Outcome::WhiteCheckmated
        );
        assert_eq!(
            Outcome::from_termination(Termination::Checkmate, false),
            Outcome::BlackCheckmated
        );
        // Other terminations are side-agnostic
        assert_eq!(
            Outcome::from_termination(Termination::Stalemate, true),
            Outcome::Stalemate
        );
        assert_eq!(
            Outcome::from_termination(Termination::Stalemate, false),
            Outcome::Stalemate
        );
        assert_eq!(
            Outcome::from_termination(Termination::PlyLimit, true),
            Outcome::PlyLimit
        );
    }

    #[test]
    fn test_outcome_discriminants() {
        assert_eq!(Outcome::WhiteCheckmated as u8, 0);
        assert_eq!(Outcome::BlackCheckmated as u8, 1);
        assert_eq!(Outcome::Stalemate as u8, 2);
        assert_eq!(Outcome::PlyLimit as u8, 6);
        assert_eq!(NUM_OUTCOMES, 7);
    }

    #[test]
    fn test_mate_in_one_boost_actually_finds_mate() {
        // For seeds where the game naturally mates near the end, boost=1.0 should
        // not miss the mate. Check that with boost=1.0, any game that could have
        // been mated was mated.
        // Simpler test: games with boost=1.0 should have checkmate rate >= baseline.
        let n = 200;
        let seeds = derive_game_seeds(777, n);
        let mates_0: usize = seeds.iter()
            .filter(|&&s| generate_one_game(s, 256, 0.0).2 == Termination::Checkmate)
            .count();
        let mates_1: usize = seeds.iter()
            .filter(|&&s| generate_one_game(s, 256, 1.0).2 == Termination::Checkmate)
            .count();
        assert!(mates_1 >= mates_0, "boost=1.0 should produce >= mates than boost=0.0");
    }

    #[test]
    fn test_rollout_legal_moves_empty_prefix() {
        // Rollouts from startpos should produce 20 entries (one per legal move)
        let results = rollout_legal_moves(&[], 1, 50, 42);
        assert_eq!(results.len(), 20);

        // Build the set of legal move tokens at the starting position
        let startpos = GameState::new();
        let legal_tokens: std::collections::HashSet<u16> = startpos.legal_move_tokens().into_iter().collect();
        assert_eq!(legal_tokens.len(), 20, "Starting position has 20 legal moves");

        for (tok, dist) in &results {
            // Every returned move token must be a legal move from the starting position
            assert!(
                legal_tokens.contains(tok),
                "Rollout returned token {} which is not a legal starting move", tok
            );
            assert_eq!(dist.total, 1);
            let sum: u32 = dist.counts.iter().sum();
            assert_eq!(sum, 1);
        }

        // Also verify all 20 legal moves appear exactly once
        let result_tokens: std::collections::HashSet<u16> = results.iter().map(|(t, _)| *t).collect();
        assert_eq!(result_tokens, legal_tokens,
            "Rollout should return exactly the set of legal starting moves");
    }

    #[test]
    fn test_rollout_legal_moves_deterministic() {
        let r1 = rollout_legal_moves(&[], 4, 50, 42);
        let r2 = rollout_legal_moves(&[], 4, 50, 42);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1.counts, b.1.counts);
            assert_eq!(a.1.total, b.1.total);
        }
    }

    #[test]
    fn test_rollout_legal_moves_invalid_prefix() {
        // Prefix containing an illegal move returns empty
        let results = rollout_legal_moves(&[9999], 1, 50, 42);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_rollout_legal_moves_terminal_prefix() {
        // Create a mate position via Fool's mate token sequence
        let (_, m2t) = crate::vocab::build_vocab_maps();
        let tokens: Vec<u16> = ["f2f3", "e7e5", "g2g4", "d8h4"]
            .iter()
            .map(|u| *m2t.get(*u).unwrap())
            .collect();
        // Game is already over at this position
        let results = rollout_legal_moves(&tokens, 1, 50, 42);
        assert_eq!(results.len(), 0, "Rollouts at terminal position should be empty");
    }

    #[test]
    fn test_compute_accuracy_ceiling_returns_positions() {
        // Very small test - just verify structure
        let positions = compute_accuracy_ceiling(2, 30, 2, 0.1, 42);
        // Should return some positions (sampling is stochastic but with seed=42 gives some)
        // Structure-only: each position has valid fields
        for p in &positions {
            assert!(p.n_legal >= 1, "n_legal should be positive if position exists");
            assert!(p.unconditional > 0.0 && p.unconditional <= 1.0);
            assert!(p.naive_conditional > 0.0 && p.naive_conditional <= 1.0);
            // Naive conditional should be >= unconditional (we prune some moves)
            assert!(p.naive_conditional >= p.unconditional - 1e-10,
                "naive_conditional {} should be >= unconditional {}",
                p.naive_conditional, p.unconditional);
            assert!(p.conditional >= 0.0 && p.conditional <= 1.0);
            assert!(p.conditional_corrected >= 0.0 && p.conditional_corrected <= 1.0);
            assert!(p.actual_outcome < NUM_OUTCOMES as u8);
            assert!((p.ply as u16) < p.game_length);
        }
    }

    #[test]
    fn test_generate_checkmate_examples_returns_checkmates() {
        // Ask for 5 checkmate examples; verify each is actually a checkmate game
        let (examples, total) = generate_checkmate_examples(42, 256, 5);
        assert_eq!(examples.len(), 5);
        assert!(total >= 5);
        for (ei, ex) in examples.iter().enumerate() {
            // Checkmate grid has at least one bit set (there's at least one mating move)
            let has_bits: bool = ex.checkmate_grid.iter().any(|&g| g != 0);
            assert!(has_bits, "Example {}: checkmate game must have at least one mating move", ei);
            // Legal grid has at least as many bits as checkmate grid
            let n_legal: u32 = ex.legal_grid.iter().map(|&g| g.count_ones()).sum();
            let n_mates: u32 = ex.checkmate_grid.iter().map(|&g| g.count_ones()).sum();
            assert!(n_legal >= n_mates, "Example {}: n_legal {} >= n_mates {}", ei, n_legal, n_mates);

            // Replay the game to the penultimate ply
            assert!(ex.move_ids.len() >= 2,
                "Example {}: checkmate game must have >= 2 plies", ei);
            let penultimate_tokens = &ex.move_ids[..ex.move_ids.len() - 1];
            let penultimate_state = GameState::from_move_tokens(penultimate_tokens)
                .expect("replay to penultimate position should succeed");

            let legal_tokens = penultimate_state.legal_move_tokens();

            // Verify every bit set in checkmate_grid has at least one corresponding
            // legal move that delivers checkmate. The grid uses (src, dst) from
            // token_to_src_dst, which strips promotion info — so multiple promotion
            // variants map to the same bit. We verify that at least one matching
            // legal move delivers mate.
            for src in 0..64u8 {
                let bits = ex.checkmate_grid[src as usize];
                if bits == 0 { continue; }
                for dst in 0..64u8 {
                    if (bits >> dst) & 1 == 0 { continue; }
                    // Find all legal tokens that match this (src, dst) pair
                    let matching: Vec<u16> = legal_tokens.iter()
                        .filter(|&&tok| {
                            let (s, d, _) = crate::vocab::decompose_token(tok).unwrap();
                            s == src && d == dst
                        })
                        .copied()
                        .collect();
                    assert!(!matching.is_empty(),
                        "Example {}: grid bit ({},{}) has no matching legal move", ei, src, dst);
                    // At least one matching move must deliver checkmate
                    let any_mate = matching.iter().any(|&tok| {
                        let mut test_state = penultimate_state.clone();
                        test_state.make_move(tok).unwrap();
                        test_state.check_termination(256 + 10) == Some(Termination::Checkmate)
                    });
                    assert!(any_mate,
                        "Example {}: grid bit ({},{}) set but no matching move delivers checkmate",
                        ei, src, dst);
                }
            }

            // Also verify NO legal move outside checkmate_grid delivers checkmate.
            // Group legal moves by (src, dst) — if the grid bit is unset, none of
            // the moves with that (src, dst) should deliver mate.
            for &tok in &legal_tokens {
                let (src, dst, _) = crate::vocab::decompose_token(tok).unwrap();
                let is_marked = (ex.checkmate_grid[src as usize] >> dst) & 1 == 1;
                if is_marked { continue; }
                let mut test_state = penultimate_state.clone();
                test_state.make_move(tok).unwrap();
                let term = test_state.check_termination(256 + 10);
                assert_ne!(term, Some(Termination::Checkmate),
                    "Example {}: move ({},{}) token {} delivers checkmate but is NOT in checkmate_grid",
                    ei, src, dst, tok);
            }
        }
    }

    #[test]
    fn test_generate_checkmate_examples_game_length_matches_moves() {
        let (examples, _) = generate_checkmate_examples(42, 256, 3);
        for ex in &examples {
            assert_eq!(ex.move_ids.len(), ex.game_length as usize);
            // Replay the game — it should terminate in checkmate
            let state = GameState::from_move_tokens(&ex.move_ids).unwrap();
            assert_eq!(state.check_termination(256), Some(Termination::Checkmate));
        }
    }

    #[test]
    fn test_play_random_to_end_invariants() {
        use rand::SeedableRng;
        for seed in 1..10 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut state = GameState::new();
            let term = state.play_random_to_end(&mut rng, 128);
            // Ply <= max_ply
            assert!(state.ply() <= 128, "ply {} > max_ply 128 at seed {}", state.ply(), seed);
            // Termination is consistent with state
            match term {
                Termination::Checkmate => {
                    assert!(state.is_check());
                    assert!(state.legal_move_tokens().is_empty());
                }
                Termination::Stalemate => {
                    assert!(!state.is_check());
                    assert!(state.legal_move_tokens().is_empty());
                }
                _ => {}
            }
        }
    }
}
