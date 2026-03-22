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
