//! Board state extraction for probing and diagnostics. Spec §7.5.

use rayon::prelude::*;

use crate::board::GameState;

/// Board state arrays for a batch of games.
pub struct BoardStates {
    pub boards: Vec<i8>,        // [batch * max_ply * 8 * 8]
    pub side_to_move: Vec<bool>,  // [batch * max_ply]
    pub castling_rights: Vec<u8>, // [batch * max_ply]
    pub ep_square: Vec<i8>,       // [batch * max_ply]
    pub is_check: Vec<bool>,      // [batch * max_ply]
    pub halfmove_clock: Vec<u8>,  // [batch * max_ply]
}

/// Extract board states at each ply. The state at ply i is the board BEFORE
/// move_ids[i] is played.
pub fn extract_board_states(
    move_ids: &[i16],     // [batch * max_ply]
    game_lengths: &[i16], // [batch]
    max_ply: usize,
) -> BoardStates {
    let batch = game_lengths.len();

    // Per-game extraction (parallel)
    let results: Vec<_> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let length = game_lengths[b] as usize;
            let mut state = GameState::new();

            let mut boards = vec![0i8; length * 64];
            let mut side_to_move = vec![false; length];
            let mut castling_rights = vec![0u8; length];
            let mut ep_square = vec![-1i8; length];
            let mut is_check = vec![false; length];
            let mut halfmove_clock = vec![0u8; length];

            for t in 0..length {
                // Extract state BEFORE the move
                let board = state.board_array();
                for rank in 0..8 {
                    for file in 0..8 {
                        boards[t * 64 + rank * 8 + file] = board[rank][file];
                    }
                }
                side_to_move[t] = state.is_white_to_move();
                castling_rights[t] = state.castling_rights_bits();
                ep_square[t] = state.ep_square();
                is_check[t] = state.is_check();
                halfmove_clock[t] = std::cmp::min(state.halfmove_clock(), u8::MAX as u32) as u8;

                // Apply the move
                let token = move_ids[b * max_ply + t] as u16;
                state.make_move(token).expect("Move should be legal during replay");
            }

            (boards, side_to_move, castling_rights, ep_square, is_check, halfmove_clock)
        })
        .collect();

    // Pack into flat arrays
    let mut all_boards = vec![0i8; batch * max_ply * 64];
    let mut all_stm = vec![false; batch * max_ply];
    let mut all_cr = vec![0u8; batch * max_ply];
    let mut all_ep = vec![-1i8; batch * max_ply];
    let mut all_check = vec![false; batch * max_ply];
    let mut all_hmc = vec![0u8; batch * max_ply];

    for (b, (boards, stm, cr, ep, check, hmc)) in results.into_iter().enumerate() {
        let length = game_lengths[b] as usize;
        let ply_offset = b * max_ply;

        for t in 0..length {
            let src_offset = t * 64;
            let dst_offset = (ply_offset + t) * 64;
            all_boards[dst_offset..dst_offset + 64]
                .copy_from_slice(&boards[src_offset..src_offset + 64]);
            all_stm[ply_offset + t] = stm[t];
            all_cr[ply_offset + t] = cr[t];
            all_ep[ply_offset + t] = ep[t];
            all_check[ply_offset + t] = check[t];
            all_hmc[ply_offset + t] = hmc[t];
        }
    }

    BoardStates {
        boards: all_boards,
        side_to_move: all_stm,
        castling_rights: all_cr,
        ep_square: all_ep,
        is_check: all_check,
        halfmove_clock: all_hmc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::generate_random_games;

    #[test]
    fn test_extract_initial_position() {
        // Generate a short game and check the initial board state
        let batch = generate_random_games(1, 256, 42);
        let states = extract_board_states(&batch.move_ids, &batch.game_lengths, 256);

        // Ply 0: initial position, white to move
        assert!(states.side_to_move[0]); // White
        assert_eq!(states.castling_rights[0], 0b1111); // All castling rights
        assert_eq!(states.ep_square[0], -1); // No EP
        assert!(!states.is_check[0]); // Not in check

        // Check white pieces on rank 0 (a1..h1)
        // R N B Q K B N R = 4 2 3 5 6 3 2 4
        assert_eq!(states.boards[0], 4); // a1 = white rook
        assert_eq!(states.boards[4], 6); // e1 = white king
    }
}
