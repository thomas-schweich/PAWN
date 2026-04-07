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
                let (moves, length, term) = generate_one_game(game_seed + i as u64, max_ply, 0.0);
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

    // Recompute accumulators from only the bits that fit in the output stride,
    // so output.white/black always agree with output.per_ply_stats.
    let mut final_white_acc = Vec::with_capacity(n_games);
    let mut final_black_acc = Vec::with_capacity(n_games);

    for (i, moves) in accepted_moves.iter().enumerate() {
        let length = accepted_lengths[i] as usize;
        for t in 0..length {
            move_ids_flat[i * max_ply + t] = moves[t] as i16;
        }
        let ply_bits = &accepted_ply_bits[i];
        let copy_len = ply_bits.len().min(max_ply);
        per_ply_stats[i * max_ply..i * max_ply + copy_len]
            .copy_from_slice(&ply_bits[..copy_len]);
        game_lengths_flat.push(accepted_lengths[i] as i16);

        let mut w = 0u64;
        let mut b = 0u64;
        for t in 0..copy_len {
            if t % 2 == 0 { w |= ply_bits[t]; } else { b |= ply_bits[t]; }
        }
        final_white_acc.push(w);
        final_black_acc.push(b);
    }

    DiagnosticOutput {
        move_ids: move_ids_flat,
        game_lengths: game_lengths_flat,
        termination_codes: accepted_terms.iter().map(|t| t.as_u8()).collect(),
        per_ply_stats,
        white: final_white_acc,
        black: final_black_acc,
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
///
/// The returned ply_bits has `length + 1` entries (including the terminal
/// position). Accumulators are derived from these entries directly so
/// they always agree with the per-ply vector.
fn compute_game_stats(moves: &[u16], length: usize) -> (Vec<u64>, u64, u64) {
    let mut move_ids = vec![0i16; length];
    for t in 0..length {
        move_ids[t] = moves[t] as i16;
    }

    let game_lengths = vec![length as i16];
    let max_ply = (length + 1).max(1);
    let mut padded = vec![0i16; max_ply];
    padded[..length].copy_from_slice(&move_ids);

    let (per_ply, _white, _black) = edgestats::compute_edge_stats_per_ply(
        &padded,
        &game_lengths,
        max_ply,
    );

    // Derive accumulators from the per_ply vector so they are always
    // consistent with what downstream code sees.
    let mut white_acc = 0u64;
    let mut black_acc = 0u64;
    for t in 0..per_ply.len() {
        if t % 2 == 0 {
            white_acc |= per_ply[t];
        } else {
            black_acc |= per_ply[t];
        }
    }

    (per_ply, white_acc, black_acc)
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

    #[test]
    fn test_quota_never_exceeded() {
        // Quota should be an upper bound: filled[i] <= quotas[i] for every bit.
        let mut quotas_white = [0i32; 64];
        let mut quotas_black = [0i32; 64];
        quotas_white[0] = 5;   // IN_CHECK white
        quotas_white[5] = 3;   // PAWN_CAPTURE_AVAILABLE white
        quotas_black[0] = 4;   // IN_CHECK black
        quotas_black[5] = 2;   // PAWN_CAPTURE_AVAILABLE black

        let output = generate_diagnostic_sets(
            &quotas_white, &quotas_black, 30, 256, 42, 50.0,
        );

        for i in 0..64 {
            assert!(output.quotas_filled_white[i] <= quotas_white[i],
                "white quota bit {} exceeded: filled={} > requested={}",
                i, output.quotas_filled_white[i], quotas_white[i]);
            assert!(output.quotas_filled_black[i] <= quotas_black[i],
                "black quota bit {} exceeded: filled={} > requested={}",
                i, output.quotas_filled_black[i], quotas_black[i]);
        }

        // Verify the test is non-vacuous: at least some quotas were actually filled.
        let total_filled: i32 = output.quotas_filled_white.iter().sum::<i32>()
            + output.quotas_filled_black.iter().sum::<i32>();
        assert!(total_filled > 0, "test is vacuous: no quotas were filled at all");
    }

    #[test]
    fn test_total_games_cap_respected() {
        // Number of accepted games must be <= total_games requested.
        let mut quotas_white = [0i32; 64];
        let quotas_black = [0i32; 64];
        quotas_white[0] = 100; // Way more than possible to find

        let output = generate_diagnostic_sets(
            &quotas_white, &quotas_black, 5, 256, 42, 10.0,
        );

        assert!(output.n_games <= 5, "accepted more than total_games");
    }

    #[test]
    fn test_max_simulated_factor_respected() {
        // With a tight max_simulated_factor we may not fill all quotas.
        // But the result should still have valid structure.
        let mut quotas_white = [0i32; 64];
        let quotas_black = [0i32; 64];
        quotas_white[0] = 10;

        let output = generate_diagnostic_sets(
            &quotas_white, &quotas_black, 10, 256, 42, 1.0,
            // max_simulated = 10 * 1.0 = 10 — very tight
        );

        // max_simulated is 10 per total_games, so at most 10 games simulated
        assert!(output.n_games <= 10);
        assert!(output.move_ids.len() == output.n_games * 256);
        assert!(output.per_ply_stats.len() == output.n_games * 256);
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        // Same seed + quotas should produce identical output.
        let mut quotas_white = [0i32; 64];
        let quotas_black = [0i32; 64];
        quotas_white[0] = 5;

        let o1 = generate_diagnostic_sets(&quotas_white, &quotas_black, 20, 256, 1234, 20.0);
        let o2 = generate_diagnostic_sets(&quotas_white, &quotas_black, 20, 256, 1234, 20.0);

        assert_eq!(o1.n_games, o2.n_games);
        assert_eq!(o1.move_ids, o2.move_ids);
        assert_eq!(o1.game_lengths, o2.game_lengths);
        assert_eq!(o1.quota_assignment_white, o2.quota_assignment_white);
        assert_eq!(o1.quota_assignment_black, o2.quota_assignment_black);
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut quotas_white = [0i32; 64];
        let quotas_black = [0i32; 64];
        quotas_white[0] = 5;

        let o1 = generate_diagnostic_sets(&quotas_white, &quotas_black, 20, 256, 1, 20.0);
        let o2 = generate_diagnostic_sets(&quotas_white, &quotas_black, 20, 256, 2, 20.0);

        // At least some moves should differ with different seeds
        assert_ne!(o1.move_ids, o2.move_ids, "different seeds should produce different games");
    }

    #[test]
    fn test_quota_assignment_bit_matches_game_bit() {
        // For an assigned game, the assigned bit should actually be present in
        // the corresponding color's accumulator.
        let mut quotas_white = [0i32; 64];
        let mut quotas_black = [0i32; 64];
        quotas_white[0] = 5;
        quotas_black[0] = 5;

        let output = generate_diagnostic_sets(&quotas_white, &quotas_black, 20, 256, 42, 100.0);

        for i in 0..output.n_games {
            let qa_w = output.quota_assignment_white[i];
            let qa_b = output.quota_assignment_black[i];
            if qa_w != 0 {
                assert_eq!(qa_w.count_ones(), 1, "game {} qa_w should have exactly one bit", i);
                assert_ne!(output.white[i] & qa_w, 0,
                    "game {} assigned white bit {:064b} not in white acc {:064b}", i, qa_w, output.white[i]);
            }
            if qa_b != 0 {
                assert_eq!(qa_b.count_ones(), 1, "game {} qa_b should have exactly one bit", i);
                assert_ne!(output.black[i] & qa_b, 0,
                    "game {} assigned black bit {:064b} not in black acc {:064b}", i, qa_b, output.black[i]);
            }
            // Can't have both colors assigned for same game
            assert!(qa_w == 0 || qa_b == 0, "game {} assigned to both colors", i);
        }
    }

    #[test]
    fn test_terminations_are_valid_codes() {
        let mut quotas_white = [0i32; 64];
        let quotas_black = [0i32; 64];
        quotas_white[0] = 3;
        let output = generate_diagnostic_sets(&quotas_white, &quotas_black, 10, 256, 42, 20.0);

        for &tc in &output.termination_codes {
            // Termination codes should be valid (at most a small range)
            assert!(tc < 10, "invalid termination code {}", tc);
        }
    }

    #[test]
    // Previously BUG-100: fixed by passing max_ply = length + 1 in compute_game_stats.
    fn test_per_ply_stats_match_accumulators() {
        // per_ply_stats should OR by color to match white/black accumulators
        // across accepted games.
        let mut quotas_white = [0i32; 64];
        let quotas_black = [0i32; 64];
        quotas_white[0] = 2;
        let output = generate_diagnostic_sets(&quotas_white, &quotas_black, 5, 256, 42, 20.0);

        for i in 0..output.n_games {
            let length = output.game_lengths[i] as usize;
            let mut w_or = 0u64;
            let mut b_or = 0u64;
            for t in 0..=length {
                if t >= 256 { break; }
                let bits = output.per_ply_stats[i * 256 + t];
                if t % 2 == 0 { w_or |= bits; } else { b_or |= bits; }
            }
            assert_eq!(output.white[i], w_or, "game {} white mismatch", i);
            assert_eq!(output.black[i], b_or, "game {} black mismatch", i);
        }
    }
}
