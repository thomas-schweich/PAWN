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
                // Record legal moves BEFORE making the move — single legal_moves() call
                let (grid, promo) = state.legal_move_grid_and_promo();
                game_grids.push(grid);
                game_promos.push(promo);

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
    fn test_startpos_has_20_legal_moves() {
        // White has exactly 20 legal moves from the starting position.
        // Build a single-move "game" where move_ids[0] = e2e4, game_length = 1.
        // The mask at ply 0 should have 20 true entries.
        let max_ply = 8;
        let vocab_size = crate::vocab::VOCAB_SIZE;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::uci_token("e2e4") as i16; // e2e4
        let game_lengths = vec![1i16];

        let masks = compute_legal_token_masks(&move_ids, &game_lengths, max_ply, vocab_size);
        let count: usize = (0..vocab_size).filter(|&v| masks[v]).count();
        assert_eq!(count, 20, "Starting position should have 20 legal moves");

        // e2e4 should be among them
        let e2e4 = crate::vocab::uci_token("e2e4") as usize;
        assert!(masks[e2e4]);
    }

    #[test]
    fn test_grid_matches_token_mask() {
        // For non-promotion moves, the action maps to a unique (src, dst) pair,
        // and the per-src 64-bit dst grid should match the dense token mask.
        let max_ply = 8;
        let vocab_size = crate::vocab::VOCAB_SIZE;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::uci_token("e2e4") as i16; // e2e4
        let game_lengths = vec![1i16];

        let (grids, _promos) = compute_legal_move_masks(&move_ids, &game_lengths, max_ply);
        let token_masks = compute_legal_token_masks(&move_ids, &game_lengths, max_ply, vocab_size);

        // Count legal pairs from the grid: each bit set in grid[src] = one legal (src,dst).
        let mut grid_count = 0usize;
        for src in 0..64usize {
            grid_count += grids[src].count_ones() as usize;
        }

        // At startpos, no promotions are possible; every (src,dst) pair corresponds
        // to a unique base-grid token. They should be equal.
        let token_count: usize = (0..vocab_size).filter(|&v| token_masks[v]).count();
        assert_eq!(grid_count, token_count, "grid and token counts must match at startpos");
        assert_eq!(grid_count, 20);
    }

    #[test]
    fn test_mask_is_before_move() {
        // Verify alignment: mask[ply=1] should reflect legal moves AFTER move[0].
        // After e2e4, black has 20 legal moves (symmetric).
        let max_ply = 8;
        let vocab_size = crate::vocab::VOCAB_SIZE;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::uci_token("e2e4") as i16; // e2e4
        move_ids[1] = crate::vocab::uci_token("e7e5") as i16; // e7e5
        let game_lengths = vec![2i16];

        let masks = compute_legal_token_masks(&move_ids, &game_lengths, max_ply, vocab_size);

        // mask at ply=0: white's turn, e2e4 is a legal option (20 moves)
        let ply0_count: usize = (0..vocab_size).filter(|&v| masks[v]).count();
        assert_eq!(ply0_count, 20, "ply 0 = startpos, 20 legal moves");
        assert!(masks[crate::vocab::uci_token("e2e4") as usize], "e2e4 legal at ply 0");

        // mask at ply=1: after e2e4, black has 20 moves
        let ply1_off = vocab_size;
        let ply1_count: usize = (0..vocab_size).filter(|&v| masks[ply1_off + v]).count();
        assert_eq!(ply1_count, 20, "ply 1 = after e2e4, black has 20 legal moves");
        assert!(masks[ply1_off + crate::vocab::uci_token("e7e5") as usize], "e7e5 legal at ply 1");
        // e2e4 should NOT be legal at ply 1 (wrong side, pawn moved)
        assert!(!masks[ply1_off + crate::vocab::uci_token("e2e4") as usize]);
    }

    #[test]
    fn test_sparse_padding_pad_token() {
        // Sparse mask must include a PAD token slot at position `length` so loss
        // is finite for that target position. Only PAD should be legal there —
        // no move tokens should appear after game end.
        let max_ply = 8;
        let seq_len = max_ply + 1;
        let vocab_size = crate::vocab::VOCAB_SIZE;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::uci_token("e2e4") as i16;
        let game_lengths = vec![1i16];

        let sparse = compute_legal_token_masks_sparse(
            &move_ids, &game_lengths, max_ply, seq_len, vocab_size,
        );

        // At position length=1, PAD token should be present.
        // PAD_TOKEN = 0, so index = game_base + length * vocab_size + 0 = vocab_size
        let expected_pad_idx = (1 * vocab_size) as i64;
        assert!(sparse.contains(&expected_pad_idx), "sparse must include PAD token at position length");

        // Verify PAD is the ONLY token at position `length` — no move tokens
        // should be legal after game end.
        let pos_start = (1 * vocab_size) as i64;
        let pos_end = (2 * vocab_size) as i64;
        let tokens_at_length: Vec<i64> = sparse.iter()
            .copied()
            .filter(|&idx| idx >= pos_start && idx < pos_end)
            .collect();
        assert_eq!(
            tokens_at_length.len(), 1,
            "exactly one token (PAD) should be legal at position length, found {}",
            tokens_at_length.len()
        );
        assert_eq!(
            tokens_at_length[0], expected_pad_idx,
            "the only token at position length should be PAD (index {}), got {}",
            expected_pad_idx, tokens_at_length[0]
        );
    }

    #[test]
    fn test_sparse_indices_in_range() {
        // All sparse indices must fall within [0, batch * seq_len * vocab_size).
        let batch_size = 4;
        let max_ply = 64;
        let seq_len = max_ply + 1;
        let vocab_size = crate::vocab::VOCAB_SIZE;
        let batch = crate::batch::generate_training_batch(batch_size, max_ply, 123);

        let sparse = compute_legal_token_masks_sparse(
            &batch.move_ids, &batch.game_lengths, max_ply, seq_len, vocab_size,
        );

        let total = (batch_size * seq_len * vocab_size) as i64;
        for &idx in &sparse {
            assert!(idx >= 0 && idx < total, "index {} out of range [0, {})", idx, total);
        }
    }

    #[test]
    fn test_empty_batch() {
        // Zero-size batch should not crash.
        let max_ply = 8;
        let vocab_size = crate::vocab::VOCAB_SIZE;
        let move_ids: Vec<i16> = vec![];
        let game_lengths: Vec<i16> = vec![];

        let dense = compute_legal_token_masks(&move_ids, &game_lengths, max_ply, vocab_size);
        assert_eq!(dense.len(), 0);

        let sparse = compute_legal_token_masks_sparse(
            &move_ids, &game_lengths, max_ply, max_ply + 1, vocab_size,
        );
        assert_eq!(sparse.len(), 0);
    }

    #[test]
    fn test_grids_zero_past_game_length() {
        // Positions beyond game_length should have all-zero grids, but the last
        // valid ply (length-1) should have non-zero legal moves (boundary check).
        let max_ply = 64;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::uci_token("e2e4") as i16; // e2e4
        move_ids[1] = crate::vocab::uci_token("e7e5") as i16; // e7e5
        let game_lengths = vec![2i16];

        let (grids, _) = compute_legal_move_masks(&move_ids, &game_lengths, max_ply);

        // Verify the last valid ply (index length-1 = 1) has a non-zero grid.
        // At ply 1 (after e2e4, black to move), black has 20 legal moves.
        let last_valid_ply = 1;
        let mut any_nonzero = false;
        for src in 0..64 {
            let off = (0 * max_ply + last_valid_ply) * 64 + src;
            if grids[off] != 0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(any_nonzero, "grid at last valid ply (index {}) should NOT be all-zero", last_valid_ply);

        // Beyond ply 2, all grid entries should be 0
        for t in 2..max_ply {
            for src in 0..64 {
                let off = (0 * max_ply + t) * 64 + src;
                assert_eq!(grids[off], 0, "grid at ply {} src {} should be zero", t, src);
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
