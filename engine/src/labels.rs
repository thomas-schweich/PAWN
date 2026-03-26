//! Legal move mask computation by replaying game sequences. Spec §7.4.

use rayon::prelude::*;

use crate::board::GameState;

/// Replay games and compute legal move masks at each ply.
/// The label at ply t is the legal moves BEFORE move t has been played —
/// i.e., the moves available to the side about to play move_ids[t].
/// Returns (legal_move_grid, legal_promo_mask) as flat arrays.
pub fn compute_legal_move_masks(
    move_ids: &[i16],   // [batch * max_ply]
    game_lengths: &[i16], // [batch]
    max_ply: usize,
) -> (Vec<u64>, Vec<bool>) {
    let batch = game_lengths.len();
    let mut grids = vec![0u64; batch * max_ply * 64];
    let mut promos = vec![false; batch * max_ply * 44 * 4];

    // Process each game in parallel
    let results: Vec<(Vec<[u64; 64]>, Vec<[[bool; 4]; 44]>)> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let length = game_lengths[b] as usize;
            let mut state = GameState::new();
            let mut game_grids = Vec::with_capacity(length);
            let mut game_promos = Vec::with_capacity(length);

            for t in 0..length {
                // Record legal moves BEFORE making the move
                game_grids.push(state.legal_move_grid());
                game_promos.push(state.legal_promo_mask());

                let token = move_ids[b * max_ply + t] as u16;
                state.make_move(token).expect("Move should be legal during replay");
            }

            (game_grids, game_promos)
        })
        .collect();

    // Pack into flat arrays
    for (b, (game_grids, game_promos)) in results.into_iter().enumerate() {
        for (t, grid) in game_grids.iter().enumerate() {
            let offset = (b * max_ply + t) * 64;
            grids[offset..offset + 64].copy_from_slice(grid);
        }
        for (t, promo) in game_promos.iter().enumerate() {
            let offset = (b * max_ply + t) * 44 * 4;
            for pair in 0..44 {
                for pt in 0..4 {
                    promos[offset + pair * 4 + pt] = promo[pair][pt];
                }
            }
        }
    }

    (grids, promos)
}

/// Replay games and produce a dense (batch, max_ply, vocab_size) bool token mask.
///
/// Fuses game replay with token mask construction — no intermediate bitboard grid.
/// Each position's legal moves are converted directly to token IDs and written
/// into the flat output array.  Rayon-parallel over games.
pub fn compute_legal_token_masks(
    move_ids: &[i16],      // [batch * max_ply]
    game_lengths: &[i16],  // [batch]
    max_ply: usize,
    vocab_size: usize,
) -> Vec<bool> {
    let batch = game_lengths.len();
    let stride_game = max_ply * vocab_size;

    // Zero-initialize output (memset — fast)
    let mut masks = vec![false; batch * stride_game];

    // Each game writes to a non-overlapping slice — parallel with no contention.
    masks
        .par_chunks_mut(stride_game)
        .enumerate()
        .for_each(|(b, game_mask)| {
            let length = game_lengths[b] as usize;
            let mut state = GameState::new();

            for t in 0..length {
                let ply_base = t * vocab_size;
                let tokens = state.legal_move_tokens();
                for tok in tokens {
                    let ti = tok as usize;
                    if ti < vocab_size {
                        game_mask[ply_base + ti] = true;
                    }
                }
                let move_tok = move_ids[b * max_ply + t] as u16;
                state.make_move(move_tok).expect("Move should be legal during replay");
            }
        });

    masks
}

/// Sparse variant: return flat i64 indices into a (batch, seq_len, vocab_size) tensor.
///
/// Each index encodes `b * seq_len * vocab_size + t * vocab_size + token_id`,
/// ready for direct GPU scatter via `index_fill_`.  Output is ~2 MB instead of
/// ~70 MB for the dense version (legal moves are <1% of the vocabulary).
pub fn compute_legal_token_masks_sparse(
    move_ids: &[i16],      // [batch * max_ply]
    game_lengths: &[i16],  // [batch]
    max_ply: usize,
    seq_len: usize,        // typically max_ply + 1
    vocab_size: usize,
) -> Vec<i64> {
    let batch = game_lengths.len();

    let per_game: Vec<Vec<i64>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let length = game_lengths[b] as usize;
            let mut state = GameState::new();
            let game_base = (b * seq_len * vocab_size) as i64;
            let mut indices = Vec::with_capacity(length * 32);

            for t in 0..length {
                let ply_base = game_base + (t * vocab_size) as i64;
                for tok in state.legal_move_tokens() {
                    let ti = tok as usize;
                    if ti < vocab_size {
                        indices.push(ply_base + ti as i64);
                    }
                }
                let move_tok = move_ids[b * max_ply + t] as u16;
                state.make_move(move_tok).expect("Move should be legal during replay");
            }

            // At position `length`, the target is PAD (end of game).
            // Include PAD token in the legal mask so loss is finite.
            if length < seq_len {
                let pad_base = game_base + (length * vocab_size) as i64;
                indices.push(pad_base); // PAD_TOKEN = 0
            }

            indices
        })
        .collect();

    // Flatten — total size ~288K for a typical batch
    let total: usize = per_game.iter().map(|v| v.len()).sum();
    let mut flat = Vec::with_capacity(total);
    for v in per_game {
        flat.extend(v);
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::generate_training_batch;
    #[test]
    fn test_labels_match_fused() {
        // Generate a batch with fused labels, then recompute via replay and compare
        let batch = generate_training_batch(4, 256, 42);
        let (grids, promos) = compute_legal_move_masks(
            &batch.move_ids,
            &batch.game_lengths,
            256,
        );
        assert_eq!(grids, batch.legal_move_grid, "Replayed grids must match fused grids");
        assert_eq!(promos, batch.legal_promo_mask, "Replayed promos must match fused promos");
    }

    #[test]
    fn test_token_masks_via_replay() {
        // Verify compute_legal_token_masks matches direct replay with legal_move_tokens()
        let batch_size = 8;
        let max_ply = 256;
        let vocab_size = 4278;
        let batch = generate_training_batch(batch_size, max_ply, 99);

        let token_masks = compute_legal_token_masks(
            &batch.move_ids, &batch.game_lengths, max_ply, vocab_size,
        );

        // Independently replay each game and verify token masks match
        for b in 0..batch_size {
            let gl = batch.game_lengths[b] as usize;
            let mut state = GameState::new();

            for t in 0..gl {
                let legal_tokens = state.legal_move_tokens();
                let mask_off = (b * max_ply + t) * vocab_size;

                // Every legal token should be marked true
                for &tok in &legal_tokens {
                    assert!(
                        token_masks[mask_off + tok as usize],
                        "game {b} ply {t}: legal token {tok} not set in mask"
                    );
                }

                // Count of true entries should match number of legal tokens
                let mask_count: usize = (0..vocab_size)
                    .filter(|&v| token_masks[mask_off + v])
                    .count();
                assert_eq!(
                    mask_count, legal_tokens.len(),
                    "game {b} ply {t}: mask has {mask_count} legal tokens but expected {}",
                    legal_tokens.len()
                );

                let move_tok = batch.move_ids[b * max_ply + t] as u16;
                state.make_move(move_tok).unwrap();
            }

            // Verify positions beyond game_length are all-false
            for t in gl..max_ply {
                let mask_off = (b * max_ply + t) * vocab_size;
                let any_set = (0..vocab_size).any(|v| token_masks[mask_off + v]);
                assert!(!any_set, "game {b} ply {t} (past game end): mask should be all-false");
            }
        }
    }

    #[test]
    fn test_sparse_matches_dense() {
        let batch_size = 8;
        let max_ply = 256;
        let seq_len = max_ply + 1;
        let vocab_size = 4278;
        let batch = generate_training_batch(batch_size, max_ply, 77);

        let dense = compute_legal_token_masks(
            &batch.move_ids, &batch.game_lengths, max_ply, vocab_size,
        );
        let sparse = compute_legal_token_masks_sparse(
            &batch.move_ids, &batch.game_lengths, max_ply, seq_len, vocab_size,
        );

        // Reconstruct dense from sparse and compare
        let mut reconstructed = vec![false; batch_size * seq_len * vocab_size];
        for &idx in &sparse {
            reconstructed[idx as usize] = true;
        }

        // Dense uses (B, max_ply, V), sparse uses (B, seq_len, V) layout.
        // Compare the overlapping region.
        for b in 0..batch_size {
            let gl = batch.game_lengths[b] as usize;
            for t in 0..gl {
                for v in 0..vocab_size {
                    let dense_val = dense[b * max_ply * vocab_size + t * vocab_size + v];
                    let sparse_val = reconstructed[b * seq_len * vocab_size + t * vocab_size + v];
                    assert_eq!(
                        dense_val, sparse_val,
                        "Mismatch at game {b} ply {t} token {v}"
                    );
                }
            }
        }
    }
}
