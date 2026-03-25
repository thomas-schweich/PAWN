//! Random game generation with deterministic seeding.

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

/// Generate a single random game without labels (utility function).
pub fn generate_one_game(seed: u64, max_ply: usize) -> (Vec<u16>, u16, Termination) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = GameState::new();
    let mut move_ids = Vec::with_capacity(max_ply);

    loop {
        if let Some(term) = state.check_termination(max_ply) {
            return (move_ids, state.ply() as u16, term);
        }

        let tokens = state.legal_move_tokens();
        let chosen = tokens[rng.gen_range(0..tokens.len())];
        state.make_move(chosen).unwrap();
        move_ids.push(chosen);
    }
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
    /// The actual outcome of the game this position came from
    pub actual_outcome: u8,
    /// Ply index within the game
    pub ply: u16,
    /// Game length
    pub game_length: u16,
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
/// - MCTS conditional: Monte Carlo rollouts estimate P(outcome | move),
///   best predictor picks argmax
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
        .map(|&seed| generate_one_game(seed, max_ply))
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

            // --- MCTS conditional (rollout-based) ---
            let rollout_seed = rollout_seed_base.wrapping_add(work_idx as u64 * 1000);
            let move_dists = rollout_legal_moves(prefix, n_rollouts_per_move, max_ply, rollout_seed);

            let outcome_idx = actual_outcome as usize;
            let probs: Vec<f64> = move_dists
                .iter()
                .map(|(_, dist)| {
                    if dist.total > 0 {
                        dist.counts[outcome_idx] as f64 / dist.total as f64
                    } else {
                        0.0
                    }
                })
                .collect();

            let sum_probs: f64 = probs.iter().sum();
            let conditional = if sum_probs > 0.0 {
                let max_prob = probs.iter().cloned().fold(0.0f64, f64::max);
                max_prob / sum_probs
            } else {
                unconditional
            };

            PositionCeiling {
                n_legal,
                unconditional,
                conditional,
                naive_conditional,
                actual_outcome,
                ply: ply as u16,
                game_length,
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
        let (moves, length, term) = generate_one_game(42, 256);
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
        let (m1, l1, t1) = generate_one_game(123, 256);
        let (m2, l2, t2) = generate_one_game(123, 256);
        assert_eq!(m1, m2);
        assert_eq!(l1, l2);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_different_seeds() {
        let (m1, _, _) = generate_one_game(1, 256);
        let (m2, _, _) = generate_one_game(2, 256);
        // Very unlikely to be the same
        assert_ne!(m1, m2);
    }
}
