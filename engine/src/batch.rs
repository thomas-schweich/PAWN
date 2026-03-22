//! Batch generation with Rayon parallelism.

use rayon::prelude::*;

use crate::random::{derive_game_seeds, generate_one_game, generate_one_game_with_labels, generate_checkmate_examples, GameRecord};
use crate::types::Termination;
use crate::vocab;

/// Output of a training batch generation.
pub struct TrainingBatch {
    pub move_ids: Vec<i16>,        // [batch_size * max_ply], row-major
    pub game_lengths: Vec<i16>,    // [batch_size]
    pub legal_move_grid: Vec<u64>, // [batch_size * max_ply * 64], row-major
    pub legal_promo_mask: Vec<bool>, // [batch_size * max_ply * 44 * 4], row-major
    pub termination_codes: Vec<u8>,  // [batch_size]
    pub batch_size: usize,
    pub max_ply: usize,
}

/// Output of random game generation (no labels).
pub struct GameBatch {
    pub move_ids: Vec<i16>,        // [n_games * max_ply]
    pub game_lengths: Vec<i16>,    // [n_games]
    pub termination_codes: Vec<u8>, // [n_games]
    pub n_games: usize,
    pub max_ply: usize,
}

/// Generate a training batch: random games + legal move labels, fused.
/// Spec §7.2.
pub fn generate_training_batch(batch_size: usize, max_ply: usize, seed: u64) -> TrainingBatch {
    // Derive independent sub-seeds, then generate games in parallel
    let seeds = derive_game_seeds(seed, batch_size);
    let records: Vec<GameRecord> = seeds
        .into_par_iter()
        .map(|s| generate_one_game_with_labels(s, max_ply))
        .collect();

    // Pack into flat arrays
    let total_ply = batch_size * max_ply;
    let mut move_ids = vec![0i16; total_ply];
    let mut game_lengths = Vec::with_capacity(batch_size);
    let mut legal_move_grid = vec![0u64; total_ply * 64];
    let mut legal_promo_mask = vec![false; total_ply * 44 * 4];
    let mut termination_codes = Vec::with_capacity(batch_size);

    for (b, record) in records.iter().enumerate() {
        let length = record.game_length as usize;
        game_lengths.push(record.game_length as i16);
        termination_codes.push(record.termination.as_u8());

        // Copy move_ids with padding
        for t in 0..length {
            move_ids[b * max_ply + t] = record.move_ids[t] as i16;
        }
        // Place EOG token at position game_length
        if length < max_ply {
            move_ids[b * max_ply + length] = vocab::EOG_TOKEN as i16;
        }

        // Copy legal move grids
        for t in 0..length {
            let grid_offset = (b * max_ply + t) * 64;
            debug_assert_eq!(record.legal_grids[t].len(), 64);
            legal_move_grid[grid_offset..grid_offset + 64]
                .copy_from_slice(&record.legal_grids[t]);
        }
        // EOG position: all zeros (already initialized)

        // Copy promotion masks (contiguous layout: [[bool; 4]; 44] = [bool; 176])
        for t in 0..length {
            let promo_offset = (b * max_ply + t) * 44 * 4;
            // Safety: [[bool; 4]; 44] has identical layout to [bool; 176]
            let flat: &[bool; 176] = unsafe {
                &*(&record.legal_promos[t] as *const [[bool; 4]; 44] as *const [bool; 176])
            };
            legal_promo_mask[promo_offset..promo_offset + 176].copy_from_slice(flat);
        }
    }

    TrainingBatch {
        move_ids,
        game_lengths,
        legal_move_grid,
        legal_promo_mask,
        termination_codes,
        batch_size,
        max_ply,
    }
}

/// Generate random games without labels. Spec §7.3.
pub fn generate_random_games(n_games: usize, max_ply: usize, seed: u64) -> GameBatch {
    let seeds = derive_game_seeds(seed, n_games);
    let results: Vec<(Vec<u16>, u16, Termination)> = seeds
        .into_par_iter()
        .map(|s| generate_one_game(s, max_ply))
        .collect();

    let mut move_ids = vec![0i16; n_games * max_ply];
    let mut game_lengths = Vec::with_capacity(n_games);
    let mut termination_codes = Vec::with_capacity(n_games);

    for (b, (moves, length, term)) in results.iter().enumerate() {
        game_lengths.push(*length as i16);
        termination_codes.push(term.as_u8());

        for t in 0..(*length as usize) {
            move_ids[b * max_ply + t] = moves[t] as i16;
        }
        if (*length as usize) < max_ply {
            move_ids[b * max_ply + *length as usize] = vocab::EOG_TOKEN as i16;
        }
    }

    GameBatch {
        move_ids,
        game_lengths,
        termination_codes,
        n_games,
        max_ply,
    }
}

/// Output of checkmate training data generation.
pub struct CheckmateTrainingBatch {
    pub move_ids: Vec<i16>,           // [n_games * max_ply]
    pub game_lengths: Vec<i16>,       // [n_games]
    pub checkmate_targets: Vec<u64>,  // [n_games * 64] — bit-packed multi-hot mating moves
    pub legal_grids: Vec<u64>,        // [n_games * 64] — legal moves at penultimate position
    pub n_games: usize,
    pub max_ply: usize,
    pub total_generated: usize,
}

/// Generate checkmate training examples with multi-hot targets.
pub fn generate_checkmate_training_batch(
    n_games: usize,
    max_ply: usize,
    seed: u64,
) -> CheckmateTrainingBatch {
    let (examples, total_generated) = generate_checkmate_examples(seed, max_ply, n_games);
    let n = examples.len();

    let mut move_ids = vec![0i16; n * max_ply];
    let mut game_lengths = Vec::with_capacity(n);
    let mut checkmate_targets = vec![0u64; n * 64];
    let mut legal_grids = vec![0u64; n * 64];

    for (b, ex) in examples.iter().enumerate() {
        game_lengths.push(ex.game_length as i16);
        for t in 0..(ex.game_length as usize).min(max_ply) {
            move_ids[b * max_ply + t] = ex.move_ids[t] as i16;
        }
        if (ex.game_length as usize) < max_ply {
            move_ids[b * max_ply + ex.game_length as usize] = vocab::EOG_TOKEN as i16;
        }
        checkmate_targets[b * 64..(b + 1) * 64].copy_from_slice(&ex.checkmate_grid);
        legal_grids[b * 64..(b + 1) * 64].copy_from_slice(&ex.legal_grid);
    }

    CheckmateTrainingBatch {
        move_ids,
        game_lengths,
        checkmate_targets,
        legal_grids,
        n_games: n,
        max_ply,
        total_generated,
    }
}

/// Generate checkmate-only games with target counts per winner color.
/// Generates games in parallel batches, discarding non-checkmates in real time.
pub fn generate_checkmate_games(
    n_white_wins: usize,
    n_black_wins: usize,
    max_ply: usize,
    seed: u64,
) -> (GameBatch, usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let batch_size = 4096;
    let target_total = n_white_wins + n_black_wins;

    let mut collected_white: Vec<(Vec<u16>, u16)> = Vec::with_capacity(n_white_wins);
    let mut collected_black: Vec<(Vec<u16>, u16)> = Vec::with_capacity(n_black_wins);
    let mut total_generated: usize = 0;
    let mut game_seed = seed;

    while collected_white.len() < n_white_wins || collected_black.len() < n_black_wins {
        // Generate a batch in parallel, filtering for checkmates
        let batch_seeds = derive_game_seeds(game_seed, batch_size);
        let results: Vec<(Vec<u16>, u16, Termination)> = batch_seeds
            .into_par_iter()
            .map(|s| generate_one_game(s, max_ply))
            .collect();

        game_seed += batch_size as u64;
        total_generated += batch_size;

        for (moves, length, term) in results {
            if term != Termination::Checkmate {
                continue;
            }
            // Odd ply = white made last move = white wins
            if length % 2 == 1 {
                if collected_white.len() < n_white_wins {
                    collected_white.push((moves, length));
                }
            } else {
                if collected_black.len() < n_black_wins {
                    collected_black.push((moves, length));
                }
            }
            if collected_white.len() >= n_white_wins && collected_black.len() >= n_black_wins {
                break;
            }
        }
    }

    // Pack into GameBatch
    let n_games = collected_white.len() + collected_black.len();
    let mut move_ids = vec![0i16; n_games * max_ply];
    let mut game_lengths = Vec::with_capacity(n_games);
    let mut termination_codes = Vec::with_capacity(n_games);

    for (b, (moves, length)) in collected_white.iter().chain(collected_black.iter()).enumerate() {
        game_lengths.push(*length as i16);
        termination_codes.push(Termination::Checkmate.as_u8());
        for t in 0..(*length as usize) {
            move_ids[b * max_ply + t] = moves[t] as i16;
        }
        if (*length as usize) < max_ply {
            move_ids[b * max_ply + *length as usize] = vocab::EOG_TOKEN as i16;
        }
    }

    (GameBatch {
        move_ids,
        game_lengths,
        termination_codes,
        n_games,
        max_ply,
    }, total_generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_batch() {
        let batch = generate_training_batch(4, 256, 42);
        assert_eq!(batch.move_ids.len(), 4 * 256);
        assert_eq!(batch.game_lengths.len(), 4);
        assert_eq!(batch.legal_move_grid.len(), 4 * 256 * 64);
        assert_eq!(batch.legal_promo_mask.len(), 4 * 256 * 44 * 4);
        assert_eq!(batch.termination_codes.len(), 4);

        for &len in &batch.game_lengths {
            assert!(len > 0 && len <= 256);
        }
    }

    #[test]
    fn test_random_games() {
        let batch = generate_random_games(8, 256, 42);
        assert_eq!(batch.move_ids.len(), 8 * 256);
        assert_eq!(batch.game_lengths.len(), 8);
    }

    #[test]
    fn test_eog_token_placement() {
        let batch = generate_training_batch(2, 256, 42);
        for b in 0..2 {
            let len = batch.game_lengths[b] as usize;
            if len < 256 {
                assert_eq!(
                    batch.move_ids[b * 256 + len],
                    vocab::EOG_TOKEN as i16,
                    "EOG token should be at position game_length"
                );
            }
        }
    }

    #[test]
    fn test_batch_deterministic() {
        let b1 = generate_training_batch(4, 256, 99);
        let b2 = generate_training_batch(4, 256, 99);
        assert_eq!(b1.move_ids, b2.move_ids);
        assert_eq!(b1.game_lengths, b2.game_lengths);
        assert_eq!(b1.legal_move_grid, b2.legal_move_grid);
    }
}
