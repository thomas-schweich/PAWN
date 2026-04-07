//! Game validation: replay and check legality at every ply. Spec §7.6.

use rayon::prelude::*;

use crate::board::GameState;

/// Validate games by replaying and checking legality at every ply.
/// Returns (is_valid, first_illegal) as flat arrays.
pub fn validate_games(
    move_ids: &[i16],     // [batch * max_ply]
    game_lengths: &[i16], // [batch]
    max_ply: usize,
) -> (Vec<bool>, Vec<i16>) {
    let batch = game_lengths.len();

    let results: Vec<(bool, i16)> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let length = game_lengths[b] as usize;
            let mut state = GameState::new();

            for t in 0..length {
                let token = move_ids[b * max_ply + t] as u16;
                if state.make_move(token).is_err() {
                    return (false, t as i16);
                }
            }

            (true, -1i16)
        })
        .collect();

    let (is_valid, first_illegal) = results.into_iter().unzip();
    (is_valid, first_illegal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::generate_random_games;

    #[test]
    fn test_validate_valid_games() {
        let batch = generate_random_games(10, 256, 42, 0.0, false);
        let (valid, first_ill) = validate_games(
            &batch.move_ids,
            &batch.game_lengths,
            256,
        );
        for (i, &v) in valid.iter().enumerate() {
            assert!(v, "Game {} should be valid", i);
            assert_eq!(first_ill[i], -1);
        }
    }

    #[test]
    fn test_validate_invalid_game() {
        // Create a game with an illegal move at ply 1
        let max_ply = 256;
        let mut move_ids = vec![0i16; max_ply];
        // e2e4 is legal
        move_ids[0] = crate::vocab::base_grid_token(12, 28) as i16;
        // e2e4 again is illegal (pawn already moved)
        move_ids[1] = crate::vocab::base_grid_token(12, 28) as i16;
        let game_lengths = vec![2i16];

        let (valid, first_ill) = validate_games(&move_ids, &game_lengths, max_ply);
        assert!(!valid[0]);
        assert_eq!(first_ill[0], 1);
    }

    #[test]
    fn test_validate_illegal_at_ply_0() {
        // The very first move is illegal (e2e5 — pawns can only move one or two squares).
        let max_ply = 16;
        let mut move_ids = vec![0i16; max_ply];
        // e2e5 — pawn can't jump three ranks
        move_ids[0] = crate::vocab::base_grid_token(12, 36) as i16;
        let game_lengths = vec![1i16];

        let (valid, first_ill) = validate_games(&move_ids, &game_lengths, max_ply);
        assert!(!valid[0]);
        assert_eq!(first_ill[0], 0);
    }

    #[test]
    fn test_validate_empty_game() {
        // Zero-length game should be valid (no moves to check).
        let (valid, first_ill) = validate_games(&vec![], &vec![0i16], 8);
        assert!(valid[0]);
        assert_eq!(first_ill[0], -1);
    }

    #[test]
    fn test_validate_mixed_batch() {
        // Mix of valid and invalid games.
        let max_ply = 8;
        let mut move_ids = vec![0i16; 3 * max_ply];
        // Game 0: valid (e2e4, e7e5)
        move_ids[0 * max_ply + 0] = crate::vocab::base_grid_token(12, 28) as i16;
        move_ids[0 * max_ply + 1] = crate::vocab::base_grid_token(52, 36) as i16;
        // Game 1: invalid at ply 0 (e2e5)
        move_ids[1 * max_ply + 0] = crate::vocab::base_grid_token(12, 36) as i16;
        // Game 2: valid single move (d2d4)
        move_ids[2 * max_ply + 0] = crate::vocab::base_grid_token(11, 27) as i16;
        let game_lengths = vec![2i16, 1i16, 1i16];

        let (valid, first_ill) = validate_games(&move_ids, &game_lengths, max_ply);
        assert!(valid[0]);
        assert_eq!(first_ill[0], -1);
        assert!(!valid[1]);
        assert_eq!(first_ill[1], 0);
        assert!(valid[2]);
        assert_eq!(first_ill[2], -1);
    }

    #[test]
    fn test_validate_pad_token_illegal() {
        // PAD_TOKEN (0) is never a legal move; validator should reject immediately.
        let max_ply = 4;
        let move_ids = vec![0i16; max_ply]; // all zeros = PAD
        let game_lengths = vec![1i16];

        let (valid, first_ill) = validate_games(&move_ids, &game_lengths, max_ply);
        assert!(!valid[0]);
        assert_eq!(first_ill[0], 0);
    }

    #[test]
    fn test_validate_generated_random_games_all_valid() {
        // Every game produced by the random generator must validate cleanly.
        let batch = generate_random_games(50, 256, 1337, 0.0, false);
        let (valid, _) = validate_games(&batch.move_ids, &batch.game_lengths, 256);
        assert!(valid.iter().all(|&v| v), "all random-generated games must be valid");
    }
}
