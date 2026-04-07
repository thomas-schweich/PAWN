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
        // Set up a position where EP is actually legal:
        // 1. e2e4  a7a6  2. e4e5  d7d5
        // After move 3 (d7d5), white's e5-pawn is adjacent to d5, so EP on d6 is legal.
        // Board state at ply 4 (before a hypothetical 5th move) should have ep_square = d6.
        // d6 in our square mapping: rank=5, file=3 => 5*8+3 = 43.
        //
        // Note: legal_ep_square() only returns Some when an EP capture is actually legal,
        // which requires the capturing pawn to be on the 5th rank (for white) or 4th rank
        // (for black), adjacent to the double-pushed pawn.
        let max_ply = 8;
        let mut move_ids = vec![0i16; max_ply];
        move_ids[0] = crate::vocab::base_grid_token(12, 28) as i16; // e2e4
        move_ids[1] = crate::vocab::base_grid_token(48, 40) as i16; // a7a6 (waiting move)
        move_ids[2] = crate::vocab::base_grid_token(28, 36) as i16; // e4e5
        move_ids[3] = crate::vocab::base_grid_token(51, 35) as i16; // d7d5
        // 5th move: white captures en passant e5d6 (src=36, dst=43)
        move_ids[4] = crate::vocab::base_grid_token(36, 43) as i16; // exd6 (EP capture)
        let game_lengths = vec![5i16];

        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        // Ply 0: no EP at start
        assert_eq!(states.ep_square[0], -1, "no EP at startpos");
        // Ply 1: after e2e4, no adjacent black pawn on 4th rank => legal_ep_square is None
        assert_eq!(states.ep_square[1], -1, "no black pawn adjacent to e4");
        // Ply 2: after a7a6, white's turn. a6 was not a double push => no EP.
        assert_eq!(states.ep_square[2], -1, "a7a6 is single push, no EP");
        // Ply 3: after e4e5, black's turn. e4e5 is a single-square push => no EP.
        assert_eq!(states.ep_square[3], -1, "e4e5 is single push, no EP");
        // Ply 4: after d7d5, white's e5-pawn is adjacent to d5. EP on d6 is legal.
        // d6 = rank 5, file 3 => 5*8+3 = 43.
        assert_eq!(states.ep_square[4], 43, "after d7d5, EP on d6 is legal for white's e5 pawn");
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
        // Different games should produce independent state arrays:
        // same start, but divergent board states after different opening moves.
        let max_ply = 16;
        let mut move_ids = vec![0i16; 2 * max_ply];
        // Game 0: 1. e2e4 e7e5
        move_ids[0 * max_ply + 0] = crate::vocab::base_grid_token(12, 28) as i16; // e2e4
        move_ids[0 * max_ply + 1] = crate::vocab::base_grid_token(52, 36) as i16; // e7e5
        // Game 1: 1. d2d4 d7d5
        move_ids[1 * max_ply + 0] = crate::vocab::base_grid_token(11, 27) as i16; // d2d4
        move_ids[1 * max_ply + 1] = crate::vocab::base_grid_token(51, 35) as i16; // d7d5
        let game_lengths = vec![2i16, 2i16];

        let states = extract_board_states(&move_ids, &game_lengths, max_ply);

        // Ply 0: both games start identically (initial position)
        assert_eq!(states.castling_rights[0], 0b1111);
        assert_eq!(states.castling_rights[max_ply], 0b1111);
        assert!(states.side_to_move[0]);
        assert!(states.side_to_move[max_ply]);
        // Board at ply 0 should be identical between the two games
        for i in 0..64 {
            assert_eq!(
                states.boards[0 * 64 + i],
                states.boards[max_ply * 64 + i],
                "ply 0 boards should be identical at square {}", i
            );
        }

        // Ply 1: after different first moves, boards must diverge.
        // Game 0 ply 1: after e2e4 => e2 empty, e4 has white pawn
        // Game 1 ply 1: after d2d4 => d2 empty, d4 has white pawn
        let g0_ply1_board_base = 1 * 64;           // game 0, ply 1
        let g1_ply1_board_base = (max_ply + 1) * 64; // game 1, ply 1

        // e2=12: game 0 has empty (0), game 1 still has pawn (1)
        assert_eq!(states.boards[g0_ply1_board_base + 12], 0, "game 0: e2 empty after e2e4");
        assert_eq!(states.boards[g1_ply1_board_base + 12], 1, "game 1: e2 still has pawn");
        // d2=11: game 0 still has pawn (1), game 1 has empty (0)
        assert_eq!(states.boards[g0_ply1_board_base + 11], 1, "game 0: d2 still has pawn");
        assert_eq!(states.boards[g1_ply1_board_base + 11], 0, "game 1: d2 empty after d2d4");

        // Confirm the two games' ply-1 boards are actually different
        let mut any_differ = false;
        for i in 0..64 {
            if states.boards[g0_ply1_board_base + i] != states.boards[g1_ply1_board_base + i] {
                any_differ = true;
                break;
            }
        }
        assert!(any_differ, "games with different moves must produce different board states at ply 1");
    }
}
