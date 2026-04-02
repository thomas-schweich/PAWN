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
}
