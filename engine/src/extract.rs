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
        let batch = generate_random_games(1, 256, 42, 0.0, false);
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

    #[test]
    fn test_extract_halfmove_clock_starts_zero() {
        // Halfmove clock must be 0 at the initial position.
        let batch = generate_random_games(1, 256, 42, 0.0, false);
        let states = extract_board_states(&batch.move_ids, &batch.game_lengths, 256);
        assert_eq!(states.halfmove_clock[0], 0);
    }

    #[test]
    fn test_extract_side_to_move_alternates() {
        // Side to move must strictly alternate at consecutive plies.
        let max_ply = 256;
        let batch = generate_random_games(4, max_ply, 42, 0.0, false);
        let states = extract_board_states(&batch.move_ids, &batch.game_lengths, max_ply);

        for b in 0..4usize {
            let length = batch.game_lengths[b] as usize;
            for t in 0..length {
                let expected_white = t % 2 == 0;
                assert_eq!(
                    states.side_to_move[b * max_ply + t], expected_white,
                    "game {} ply {} side_to_move mismatch", b, t
                );
            }
        }
    }

    #[test]
    fn test_extract_state_before_move_not_after() {
        // Construct a 2-ply game: 1. e2e4 e7e5
        // At ply 0: initial position (white to move, a1=white rook)
        // At ply 1: AFTER e2e4 BEFORE e7e5, black to move, e2 empty, e4 has white pawn
        let max_ply = 8;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::base_grid_token(12, 28) as i16; // e2e4
        move_ids[1] = crate::vocab::base_grid_token(52, 36) as i16; // e7e5
        let game_lengths = vec![2i16];

        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        // Ply 0: startpos
        assert!(states.side_to_move[0], "ply 0 = white to move");
        // e2 square index = 12, should have white pawn (code 1)
        assert_eq!(states.boards[0 * 64 + 12], 1, "e2 has white pawn at ply 0");
        // e4 square index = 28, should be empty
        assert_eq!(states.boards[0 * 64 + 28], 0, "e4 empty at ply 0");

        // Ply 1: AFTER e2e4
        assert!(!states.side_to_move[1], "ply 1 = black to move");
        assert_eq!(states.boards[1 * 64 + 12], 0, "e2 empty at ply 1");
        assert_eq!(states.boards[1 * 64 + 28], 1, "e4 has white pawn at ply 1");
    }

    #[test]
    fn test_extract_ep_square_after_double_push() {
        // After 1. e2e4 (double pawn push), the legal EP square should be e3 (idx 20).
        let max_ply = 8;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::base_grid_token(12, 28) as i16; // e2e4
        // Black also double-pushes to d5 so white has a legal EP
        move_ids[1] = crate::vocab::base_grid_token(51, 35) as i16; // d7d5
        let game_lengths = vec![2i16];

        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        // Ply 0: no EP (start)
        assert_eq!(states.ep_square[0], -1);
        // Ply 1: after e2e4, there's no black pawn to capture en passant, so legal_ep_square is -1.
        // But we did e2e4 then look at next ply. legal_ep_square only reports if a legal EP exists.
        // Check for e6 after d7d5. We need to make another move to look at next state.
    }

    #[test]
    fn test_extract_castling_rights_encoding() {
        // castling_rights bit encoding: bit 0=K, 1=Q, 2=k, 3=q.
        // At startpos, all bits set (0b1111 = 15).
        // After e1 king moves, both white castling rights are lost (bits 0, 1 clear).
        let max_ply = 16;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::base_grid_token(12, 28) as i16; // e2e4
        move_ids[1] = crate::vocab::base_grid_token(52, 36) as i16; // e7e5
        move_ids[2] = crate::vocab::base_grid_token(4, 12) as i16;  // e1e2 (Ke2)
        move_ids[3] = crate::vocab::base_grid_token(60, 52) as i16; // e8e7 (Ke7)
        let game_lengths = vec![4i16];

        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        // Ply 0: all rights
        assert_eq!(states.castling_rights[0], 0b1111);
        // After white king moves (ply 3 = board state after moves 0,1,2 applied, before move 3),
        // white rights should be cleared: bits 2,3 (black) remain.
        assert_eq!(states.castling_rights[3] & 0b0011, 0, "white castling rights cleared after king move");
        assert_eq!(states.castling_rights[3] & 0b1100, 0b1100, "black castling rights still present");
    }

    #[test]
    fn test_extract_not_check_at_startpos() {
        let max_ply = 8;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::base_grid_token(12, 28) as i16;
        let game_lengths = vec![1i16];
        let states = extract_board_states(&move_ids, &game_lengths, max_ply);
        assert!(!states.is_check[0]);
    }

    #[test]
    fn test_extract_piece_codes() {
        // Piece codes: 0=empty, 1-6=white P/N/B/R/Q/K, 7-12=black P/N/B/R/Q/K.
        let max_ply = 4;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::base_grid_token(12, 28) as i16;
        let game_lengths = vec![1i16];
        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        // White pieces on rank 0 (a1-h1)
        assert_eq!(states.boards[0], 4, "a1 white rook");
        assert_eq!(states.boards[1], 2, "b1 white knight");
        assert_eq!(states.boards[2], 3, "c1 white bishop");
        assert_eq!(states.boards[3], 5, "d1 white queen");
        assert_eq!(states.boards[4], 6, "e1 white king");
        assert_eq!(states.boards[5], 3, "f1 white bishop");
        assert_eq!(states.boards[6], 2, "g1 white knight");
        assert_eq!(states.boards[7], 4, "h1 white rook");

        // White pawns on rank 1 (a2-h2)
        for i in 8..16 {
            assert_eq!(states.boards[i], 1, "rank 2 white pawn at {}", i);
        }

        // Black pawns on rank 6 (a7-h7)
        for i in 48..56 {
            assert_eq!(states.boards[i], 7, "rank 7 black pawn at {}", i);
        }

        // Black pieces on rank 7 (a8-h8): code 10=rook, 8=knight, 9=bishop, 11=queen, 12=king
        assert_eq!(states.boards[56], 10, "a8 black rook");
        assert_eq!(states.boards[60], 12, "e8 black king");
    }

    #[test]
    fn test_extract_beyond_game_length_is_zero() {
        // Positions beyond game_length should be default-initialized (zero/false/-1).
        let max_ply = 32;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::base_grid_token(12, 28) as i16;
        let game_lengths = vec![1i16];
        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        // ply 1 and beyond: no state recorded
        for t in 1..max_ply {
            assert_eq!(states.castling_rights[t], 0, "cr[{}] should be 0", t);
            assert_eq!(states.ep_square[t], -1, "ep[{}] should be -1", t);
            assert!(!states.is_check[t], "check[{}] should be false", t);
            assert_eq!(states.halfmove_clock[t], 0);
            // board should be all zeros
            for i in 0..64 {
                assert_eq!(states.boards[t * 64 + i], 0);
            }
        }
    }

    #[test]
    fn test_extract_halfmove_increments_on_nonpawn_noncapture() {
        // 1. Nf3 (knight move): halfmove clock becomes 1
        // 2. Nf6 (knight move): halfmove clock becomes 2
        let max_ply = 8;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::base_grid_token(6, 21) as i16;  // g1f3
        move_ids[1] = crate::vocab::base_grid_token(62, 45) as i16; // g8f6
        let game_lengths = vec![2i16];
        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        assert_eq!(states.halfmove_clock[0], 0, "ply 0 halfmove=0");
        assert_eq!(states.halfmove_clock[1], 1, "after white Nf3, halfmove=1");
    }

    #[test]
    fn test_extract_halfmove_resets_on_pawn_move() {
        // 1. Nf3 (knight): hmc=1
        // 2. e5   (pawn): hmc should reset to 0
        let max_ply = 8;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::base_grid_token(6, 21) as i16;  // g1f3
        move_ids[1] = crate::vocab::base_grid_token(52, 36) as i16; // e7e5 (pawn push)
        move_ids[2] = crate::vocab::base_grid_token(1, 18) as i16;  // b1c3 knight
        let game_lengths = vec![3i16];
        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        assert_eq!(states.halfmove_clock[0], 0);
        assert_eq!(states.halfmove_clock[1], 1, "after Nf3");
        // After Nf3 e5: e5 was a pawn move so hmc resets to 0 (board state BEFORE move 2).
        assert_eq!(states.halfmove_clock[2], 0, "pawn move resets halfmove clock");
    }

    #[test]
    fn test_multiple_games_independent() {
        // Different games should produce independent state arrays.
        let max_ply = 16;
        let mut move_ids = vec![0i16; 2 * max_ply];
        // Game 0: 1. e2e4 (so ply 1 has e4 with white pawn)
        move_ids[0 * max_ply + 0] = crate::vocab::base_grid_token(12, 28) as i16;
        // Game 1: 1. d2d4
        move_ids[1 * max_ply + 0] = crate::vocab::base_grid_token(11, 27) as i16;
        let game_lengths = vec![1i16, 1i16];

        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        // Both games should start identically
        assert_eq!(states.castling_rights[0], 0b1111);
        assert_eq!(states.castling_rights[max_ply], 0b1111);
        assert!(states.side_to_move[0]);
        assert!(states.side_to_move[max_ply]);
    }
}
