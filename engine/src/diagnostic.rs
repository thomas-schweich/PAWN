//! Diagnostic set generation with quota-controlled sampling. Spec §7.8.

use rayon::prelude::*;

use crate::edgestats;
use crate::random::generate_one_game;
use crate::types::Termination;

/// Output of diagnostic set generation.
pub struct DiagnosticOutput {
    pub move_ids: Vec<i16>,           // [N * max_ply]
    pub game_lengths: Vec<i16>,       // [N]
    pub termination_codes: Vec<u8>,   // [N]
    pub per_ply_stats: Vec<u64>,      // [N * max_ply]
    pub white: Vec<u64>,              // [N]
    pub black: Vec<u64>,              // [N]
    pub quota_assignment_white: Vec<u64>, // [N]
    pub quota_assignment_black: Vec<u64>, // [N]
    pub quotas_filled_white: Vec<i32>,   // [64]
    pub quotas_filled_black: Vec<i32>,   // [64]
    pub n_games: usize,
    pub max_ply: usize,
}

/// Generate a diagnostic game corpus with quota control. Spec §7.8.
pub fn generate_diagnostic_sets(
    quotas_white: &[i32; 64],
    quotas_black: &[i32; 64],
    total_games: usize,
    max_ply: usize,
    seed: u64,
    max_simulated_factor: f64,
) -> DiagnosticOutput {
    let max_simulated = (total_games as f64 * max_simulated_factor) as usize;
    let internal_batch_size = 256; // Generate this many games at a time

    let mut accepted_moves: Vec<Vec<u16>> = Vec::with_capacity(total_games);
    let mut accepted_lengths: Vec<u16> = Vec::with_capacity(total_games);
    let mut accepted_terms: Vec<Termination> = Vec::with_capacity(total_games);
    let mut accepted_white_acc: Vec<u64> = Vec::with_capacity(total_games);
    let mut accepted_black_acc: Vec<u64> = Vec::with_capacity(total_games);
    let mut accepted_ply_bits: Vec<Vec<u64>> = Vec::with_capacity(total_games);
    let mut assignment_white: Vec<u64> = Vec::with_capacity(total_games);
    let mut assignment_black: Vec<u64> = Vec::with_capacity(total_games);

    let mut filled_white = [0i32; 64];
    let mut filled_black = [0i32; 64];

    let mut total_simulated = 0u64;
    let mut game_seed = seed;

    while accepted_moves.len() < total_games && (total_simulated as usize) < max_simulated {
        let batch_count = internal_batch_size.min(max_simulated - total_simulated as usize);

        // Generate games and compute edge stats in parallel
        let games: Vec<(Vec<u16>, u16, Termination, Vec<u64>, u64, u64)> = (0..batch_count)
            .into_par_iter()
            .map(|i| {
                let (moves, length, term) = generate_one_game(game_seed + i as u64, max_ply);
                let (ply_bits, w_acc, b_acc) = compute_game_stats(&moves, length as usize);
                (moves, length, term, ply_bits, w_acc, b_acc)
            })
            .collect();

        game_seed += batch_count as u64;
        total_simulated += batch_count as u64;

        // Decide acceptance (sequential — quota state is mutable)
        for (moves, length, term, ply_bits, w_acc, b_acc) in games {
            if accepted_moves.len() >= total_games {
                break;
            }

            // Check remaining quota need
            let remaining_slots = total_games - accepted_moves.len();
            let remaining_need: usize = (0..64).map(|i| {
                (quotas_white[i] - filled_white[i]).max(0) as usize +
                (quotas_black[i] - filled_black[i]).max(0) as usize
            }).sum();

            let selective = remaining_slots <= remaining_need;

            // Find which quota this game could fill (greedy: greatest remaining need)
            let mut best_bit: Option<usize> = None;
            let mut best_color_is_white = true;
            let mut best_need = 0i32;

            // Deterministic tie-breaking: black is favored on ties (black is checked
            // after white within the same bit, so equal-need ties go to black).
            // This is intentional — it ensures reproducible quota assignment.
            for bit in 0..64usize {
                let mask = 1u64 << bit;
                // White
                if w_acc & mask != 0 {
                    let need = quotas_white[bit] - filled_white[bit];
                    if need > best_need || (need == best_need && need > 0) {
                        best_need = need;
                        best_bit = Some(bit);
                        best_color_is_white = true;
                    }
                }
                // Black
                if b_acc & mask != 0 {
                    let need = quotas_black[bit] - filled_black[bit];
                    if need > best_need || (need == best_need && need > 0) {
                        best_need = need;
                        best_bit = Some(bit);
                        best_color_is_white = false;
                    }
                }
            }

            // Acceptance decision
            let matches_quota = best_need > 0;
            if selective && !matches_quota {
                continue; // Reject: need quota matches only
            }

            // Accept the game
            let mut qa_w: u64 = 0;
            let mut qa_b: u64 = 0;
            if let Some(bit) = best_bit {
                if best_need > 0 {
                    if best_color_is_white {
                        qa_w = 1u64 << bit;
                        filled_white[bit] += 1;
                    } else {
                        qa_b = 1u64 << bit;
                        filled_black[bit] += 1;
                    }
                }
            }

            accepted_moves.push(moves);
            accepted_lengths.push(length);
            accepted_terms.push(term);
            accepted_white_acc.push(w_acc);
            accepted_black_acc.push(b_acc);
            accepted_ply_bits.push(ply_bits);
            assignment_white.push(qa_w);
            assignment_black.push(qa_b);
        }
    }

    let n_games = accepted_moves.len();

    // Pack move_ids and per-ply stats into flat arrays
    let mut move_ids_flat = vec![0i16; n_games * max_ply];
    let mut per_ply_stats = vec![0u64; n_games * max_ply];
    let mut game_lengths_flat = Vec::with_capacity(n_games);

    for (i, moves) in accepted_moves.iter().enumerate() {
        let length = accepted_lengths[i] as usize;
        for t in 0..length {
            move_ids_flat[i * max_ply + t] = moves[t] as i16;
        }
        // Copy pre-computed per-ply bits (length+1 entries, cap at max_ply)
        let ply_bits = &accepted_ply_bits[i];
        let copy_len = ply_bits.len().min(max_ply);
        per_ply_stats[i * max_ply..i * max_ply + copy_len]
            .copy_from_slice(&ply_bits[..copy_len]);
        game_lengths_flat.push(accepted_lengths[i] as i16);
    }

    DiagnosticOutput {
        move_ids: move_ids_flat,
        game_lengths: game_lengths_flat,
        termination_codes: accepted_terms.iter().map(|t| t.as_u8()).collect(),
        per_ply_stats,
        white: accepted_white_acc,
        black: accepted_black_acc,
        quota_assignment_white: assignment_white,
        quota_assignment_black: assignment_black,
        quotas_filled_white: filled_white.to_vec(),
        quotas_filled_black: filled_black.to_vec(),
        n_games,
        max_ply,
    }
}

/// Compute per-ply edge stats + per-game accumulators for a single game.
/// Returns (ply_bits, white_accumulator, black_accumulator).
fn compute_game_stats(moves: &[u16], length: usize) -> (Vec<u64>, u64, u64) {
    let mut move_ids = vec![0i16; length];
    for t in 0..length {
        move_ids[t] = moves[t] as i16;
    }

    let game_lengths = vec![length as i16];
    let max_ply = length.max(1);
    let mut padded = vec![0i16; max_ply];
    padded[..length].copy_from_slice(&move_ids);

    let (per_ply, white, black) = edgestats::compute_edge_stats_per_ply(
        &padded,
        &game_lengths,
        max_ply,
    );

    (per_ply, white[0], black[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_generation_basic() {
        let mut quotas_white = [0i32; 64];
        let mut quotas_black = [0i32; 64];
        // Request 10 games with IN_CHECK for white
        quotas_white[0] = 10;

        let output = generate_diagnostic_sets(
            &quotas_white,
            &quotas_black,
            50, // total games
            256,
            42,
            100.0,
        );

        assert!(output.n_games <= 50);
        assert_eq!(output.move_ids.len(), output.n_games * 256,
            "move_ids length must be n_games * max_ply");
        assert_eq!(output.game_lengths.len(), output.n_games,
            "game_lengths length must be n_games");
        assert_eq!(output.move_ids.len() % output.n_games, 0,
            "move_ids length must be a multiple of n_games");
        assert!(output.quotas_filled_white[0] >= 1,
            "Should find at least some games with white IN_CHECK");
    }
}
