//! Move vocabulary: the single source of truth for token ↔ UCI string mapping.
//!
//! Token layout (1,980 total):
//!   0..=1967   = searchless_chess actions (1:1 with DeepMind's vocabulary)
//!   1968       = padding
//!   1969..=1979 = outcome tokens (game result)
//!
//! Square indexing: file-major within rank.
//!   a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
//!   file = index % 8, rank = index / 8

use std::collections::HashMap;

use crate::searchless_vocab;
use crate::types::Termination;

pub const NUM_ACTIONS: usize = searchless_vocab::NUM_ACTIONS; // 1968
pub const VOCAB_SIZE: usize = 1980; // 1968 actions + 1 PAD + 11 outcomes
pub const PAD_TOKEN: u16 = 1968;

// Outcome tokens — must match pawn/config.py
pub const OUTCOME_BASE: u16 = 1969;

// Pretraining outcomes (random games — natural terminations)
pub const WHITE_CHECKMATES: u16 = 1969;
pub const BLACK_CHECKMATES: u16 = 1970;
pub const STALEMATE: u16 = 1971;
pub const DRAW_BY_RULE: u16 = 1972;  // 75-move, fivefold rep, insufficient material
pub const PLY_LIMIT: u16 = 1973;     // Hit max plies (also truncated Lichess games)

// Lichess-specific outcomes (finetuning data)
pub const WHITE_RESIGNS: u16 = 1974;
pub const BLACK_RESIGNS: u16 = 1975;
pub const DRAW_BY_AGREEMENT: u16 = 1976;
pub const WHITE_WINS_ON_TIME: u16 = 1977;
pub const BLACK_WINS_ON_TIME: u16 = 1978;
pub const DRAW_BY_TIME: u16 = 1979;

/// Square names in our index order.
pub const SQUARE_NAMES: [&str; 64] = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

pub const PROMO_PIECES: [&str; 4] = ["q", "r", "b", "n"];

// --- Reverse lookup: UCI string → action index ---

static UCI_TO_ACTION: once_cell::sync::Lazy<HashMap<String, u16>> =
    once_cell::sync::Lazy::new(|| {
        let mut map = HashMap::with_capacity(NUM_ACTIONS);
        for (i, &uci) in searchless_vocab::ACTION_TO_UCI.iter().enumerate() {
            map.insert(uci.to_string(), i as u16);
        }
        map
    });

/// Look up the action index for a UCI move string (e.g., "e2e4" → 317).
pub fn uci_to_action(uci: &str) -> Option<u16> {
    UCI_TO_ACTION.get(uci).copied()
}

/// Decompose a token into (src_square, dst_square, promo_type).
/// promo_type: 0=none, 1=q, 2=r, 3=b, 4=n (matches embedding index).
/// Returns None for PAD and outcome tokens.
#[inline]
pub fn decompose_token(token: u16) -> Option<(u8, u8, u8)> {
    if (token as usize) < NUM_ACTIONS {
        Some((
            searchless_vocab::ACTION_TO_SRC[token as usize],
            searchless_vocab::ACTION_TO_DST[token as usize],
            searchless_vocab::ACTION_TO_PROMO[token as usize],
        ))
    } else {
        None // PAD, outcomes, or out-of-range
    }
}

/// Extract (src_square, dst_square) from a move token, ignoring promotion type.
pub fn token_to_src_dst(token: u16) -> (u8, u8) {
    let (src, dst, _) = decompose_token(token).expect("invalid token");
    (src, dst)
}

/// Convert a token to its UCI string representation.
pub fn token_to_uci(token: u16) -> Option<String> {
    if (token as usize) < NUM_ACTIONS {
        Some(searchless_vocab::ACTION_TO_UCI[token as usize].to_string())
    } else {
        None
    }
}

/// Build the full token_to_move and move_to_token maps (actions only, no PAD/outcomes).
pub fn build_vocab_maps() -> (HashMap<u16, String>, HashMap<String, u16>) {
    let mut token_to_move = HashMap::with_capacity(NUM_ACTIONS);
    let mut move_to_token = HashMap::with_capacity(NUM_ACTIONS);

    for (i, &uci) in searchless_vocab::ACTION_TO_UCI.iter().enumerate() {
        let token = i as u16;
        let uci_string = uci.to_string();
        token_to_move.insert(token, uci_string.clone());
        move_to_token.insert(uci_string, token);
    }

    (token_to_move, move_to_token)
}

// --- Outcome token logic ---

/// Map a game termination reason to the corresponding outcome token.
///
/// For checkmate, the winner is determined by game length parity:
/// - Odd game_length (white made the last move) → WHITE_CHECKMATES
/// - Even game_length (black made the last move) → BLACK_CHECKMATES
pub fn termination_to_outcome(term: Termination, game_length: u16) -> u16 {
    match term {
        Termination::Checkmate => {
            if game_length % 2 == 1 { WHITE_CHECKMATES } else { BLACK_CHECKMATES }
        }
        Termination::Stalemate => STALEMATE,
        Termination::SeventyFiveMoveRule
        | Termination::FivefoldRepetition
        | Termination::InsufficientMaterial => DRAW_BY_RULE,
        Termination::PlyLimit => PLY_LIMIT,
    }
}

/// Map a Lichess game to its outcome token.
///
/// Uses the PGN Termination header, Result header, and whether the last
/// move was checkmate/stalemate (determined by replaying the game).
///
/// Returns None for games that should be filtered (Rules infraction,
/// Abandoned, Unterminated).
pub fn lichess_outcome_token(
    termination: &str,
    result: &str,
    is_checkmate: bool,
    is_stalemate: bool,
    truncated: bool,
) -> Option<u16> {
    if truncated {
        return Some(PLY_LIMIT);
    }

    match termination {
        "Normal" => {
            match result {
                "1-0" => Some(if is_checkmate { WHITE_CHECKMATES } else { WHITE_RESIGNS }),
                "0-1" => Some(if is_checkmate { BLACK_CHECKMATES } else { BLACK_RESIGNS }),
                "1/2-1/2" => Some(if is_stalemate { STALEMATE } else { DRAW_BY_AGREEMENT }),
                _ => None,
            }
        }
        "Time forfeit" => {
            match result {
                "1-0" => Some(WHITE_WINS_ON_TIME),
                "0-1" => Some(BLACK_WINS_ON_TIME),
                "1/2-1/2" => Some(DRAW_BY_TIME),
                _ => None,
            }
        }
        "Insufficient material" => Some(DRAW_BY_RULE),
        // Filter out Rules infraction, Abandoned, Unterminated
        _ => None,
    }
}

/// Convenience: look up a UCI string and panic if not found.
/// Useful in tests and static initialization.
pub fn uci_token(uci: &str) -> u16 {
    uci_to_action(uci)
        .unwrap_or_else(|| panic!("UCI move '{}' not found in vocabulary", uci))
}

// --- Legacy helpers kept for grid-based representations (probes, board state) ---

pub const NUM_PROMO_PAIRS: usize = 44;
pub const NUM_PROMO_TYPES: usize = 4;

/// The 44 promotion-eligible (src_square, dst_square) pairs.
/// Kept for grid-based legal move representations used by probes.
static PROMO_PAIRS_ARRAY: once_cell::sync::Lazy<[(u8, u8); NUM_PROMO_PAIRS]> =
    once_cell::sync::Lazy::new(|| {
        let mut pairs = [(0u8, 0u8); NUM_PROMO_PAIRS];
        let mut idx = 0;

        // White straight pushes
        for file in 0u8..8 { pairs[idx] = (6 * 8 + file, 7 * 8 + file); idx += 1; }
        // White left captures
        for file in 1u8..8 { pairs[idx] = (6 * 8 + file, 7 * 8 + (file - 1)); idx += 1; }
        // White right captures
        for file in 0u8..7 { pairs[idx] = (6 * 8 + file, 7 * 8 + (file + 1)); idx += 1; }
        // Black straight pushes
        for file in 0u8..8 { pairs[idx] = (1 * 8 + file, 0 * 8 + file); idx += 1; }
        // Black left captures
        for file in 1u8..8 { pairs[idx] = (1 * 8 + file, 0 * 8 + (file - 1)); idx += 1; }
        // Black right captures
        for file in 0u8..7 { pairs[idx] = (1 * 8 + file, 0 * 8 + (file + 1)); idx += 1; }
        assert_eq!(idx, NUM_PROMO_PAIRS);
        pairs
    });

pub fn promo_pairs() -> &'static [(u8, u8); NUM_PROMO_PAIRS] {
    &PROMO_PAIRS_ARRAY
}

static PROMO_PAIR_INDEX: once_cell::sync::Lazy<HashMap<(u8, u8), usize>> =
    once_cell::sync::Lazy::new(|| {
        let pairs = promo_pairs();
        let mut map = HashMap::with_capacity(NUM_PROMO_PAIRS);
        for (i, &(s, d)) in pairs.iter().enumerate() {
            map.insert((s, d), i);
        }
        map
    });

pub fn promo_pair_index(src: u8, dst: u8) -> Option<usize> {
    PROMO_PAIR_INDEX.get(&(src, dst)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_constants() {
        assert_eq!(NUM_ACTIONS, 1968);
        assert_eq!(PAD_TOKEN, 1968);
        assert_eq!(OUTCOME_BASE, 1969);
        assert_eq!(VOCAB_SIZE, 1980);
        assert_eq!(WHITE_CHECKMATES, 1969);
        assert_eq!(DRAW_BY_TIME, 1979);
        // 11 outcome tokens: 1969..=1979
        assert_eq!(DRAW_BY_TIME as usize - OUTCOME_BASE as usize + 1, 11);
        // Layout: actions [0, 1967], PAD [1968], outcomes [1969, 1979]
        assert_eq!(NUM_ACTIONS as u16, PAD_TOKEN);
        assert_eq!(PAD_TOKEN + 1, OUTCOME_BASE);
    }

    #[test]
    fn test_all_outcome_tokens_distinct_and_in_range() {
        let outcomes = [
            WHITE_CHECKMATES, BLACK_CHECKMATES, STALEMATE, DRAW_BY_RULE, PLY_LIMIT,
            WHITE_RESIGNS, BLACK_RESIGNS, DRAW_BY_AGREEMENT,
            WHITE_WINS_ON_TIME, BLACK_WINS_ON_TIME, DRAW_BY_TIME,
        ];
        for &t in &outcomes {
            assert!(t >= OUTCOME_BASE, "outcome {} < OUTCOME_BASE", t);
            assert!((t as usize) < VOCAB_SIZE, "outcome {} >= VOCAB_SIZE", t);
        }
        let mut sorted = outcomes.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), outcomes.len(), "outcome tokens must be distinct");
    }

    #[test]
    fn test_square_indexing() {
        assert_eq!(SQUARE_NAMES[0], "a1");
        assert_eq!(SQUARE_NAMES[7], "h1");
        assert_eq!(SQUARE_NAMES[8], "a2");
        assert_eq!(SQUARE_NAMES[63], "h8");
    }

    #[test]
    fn test_square_names_indexing_exhaustive() {
        for i in 0..64 {
            let name = SQUARE_NAMES[i];
            let chars: Vec<char> = name.chars().collect();
            assert_eq!(chars.len(), 2);
            let file = chars[0] as u8 - b'a';
            let rank = chars[1] as u8 - b'1';
            assert_eq!(file as usize, i % 8, "file mismatch for idx {}: {}", i, name);
            assert_eq!(rank as usize, i / 8, "rank mismatch for idx {}: {}", i, name);
        }
    }

    #[test]
    fn test_decompose_roundtrip() {
        for action in 0..NUM_ACTIONS as u16 {
            let (src, dst, promo) = decompose_token(action).unwrap();
            assert!(src < 64, "invalid src {} for action {}", src, action);
            assert!(dst < 64, "invalid dst {} for action {}", dst, action);
            assert!(promo <= 4, "invalid promo {} for action {}", promo, action);
            // Reconstruct UCI and look up action
            let uci = token_to_uci(action).unwrap();
            assert_eq!(uci_to_action(&uci), Some(action),
                "roundtrip failed for action {}: uci={}", action, uci);
        }
    }

    #[test]
    fn test_decompose_pad_and_outcomes() {
        assert!(decompose_token(PAD_TOKEN).is_none());
        for token in OUTCOME_BASE..=DRAW_BY_TIME {
            assert!(decompose_token(token).is_none(),
                "outcome token {} should not decompose", token);
        }
        // Out of range
        assert!(decompose_token(VOCAB_SIZE as u16).is_none());
        assert!(decompose_token(5000).is_none());
        assert!(decompose_token(u16::MAX).is_none());
    }

    #[test]
    fn test_specific_uci() {
        // e2e4: src=e2=12, dst=e4=28
        let e2e4 = uci_to_action("e2e4").unwrap();
        assert_eq!(token_to_uci(e2e4).unwrap(), "e2e4");
        let (src, dst, promo) = decompose_token(e2e4).unwrap();
        assert_eq!(src, 12);
        assert_eq!(dst, 28);
        assert_eq!(promo, 0);

        // e1g1 (kingside castling)
        let e1g1 = uci_to_action("e1g1").unwrap();
        assert_eq!(token_to_uci(e1g1).unwrap(), "e1g1");

        // a7a8q (promotion)
        let a7a8q = uci_to_action("a7a8q").unwrap();
        assert_eq!(token_to_uci(a7a8q).unwrap(), "a7a8q");
        let (src, dst, promo) = decompose_token(a7a8q).unwrap();
        assert_eq!(src, 48); // a7
        assert_eq!(dst, 56); // a8
        assert_eq!(promo, 1); // queen
    }

    #[test]
    fn test_uci_to_action_invalid() {
        assert!(uci_to_action("a1a1").is_none()); // impossible move
        assert!(uci_to_action("a1b4").is_none()); // impossible move
        assert!(uci_to_action("xxxx").is_none());
        assert!(uci_to_action("").is_none());
    }

    #[test]
    fn test_token_to_uci_none_for_special() {
        assert!(token_to_uci(PAD_TOKEN).is_none());
        assert!(token_to_uci(OUTCOME_BASE).is_none());
        assert!(token_to_uci(DRAW_BY_TIME).is_none());
    }

    #[test]
    fn test_vocab_maps_roundtrip() {
        let (t2m, m2t) = build_vocab_maps();
        assert_eq!(t2m.len(), NUM_ACTIONS);
        assert_eq!(m2t.len(), NUM_ACTIONS);
        for (&token, uci) in &t2m {
            assert_eq!(m2t.get(uci), Some(&token));
        }
    }

    #[test]
    fn test_vocab_maps_uci_format() {
        let (_, m2t) = build_vocab_maps();
        for (uci, _) in &m2t {
            assert!(uci.len() == 4 || uci.len() == 5,
                "UCI should be 4-5 chars: {}", uci);
        }
    }

    #[test]
    fn test_termination_to_outcome() {
        assert_eq!(termination_to_outcome(Termination::Checkmate, 11), WHITE_CHECKMATES);
        assert_eq!(termination_to_outcome(Termination::Checkmate, 1), WHITE_CHECKMATES);
        assert_eq!(termination_to_outcome(Termination::Checkmate, 12), BLACK_CHECKMATES);
        assert_eq!(termination_to_outcome(Termination::Checkmate, 2), BLACK_CHECKMATES);
        assert_eq!(termination_to_outcome(Termination::Stalemate, 50), STALEMATE);
        assert_eq!(termination_to_outcome(Termination::SeventyFiveMoveRule, 100), DRAW_BY_RULE);
        assert_eq!(termination_to_outcome(Termination::FivefoldRepetition, 80), DRAW_BY_RULE);
        assert_eq!(termination_to_outcome(Termination::InsufficientMaterial, 60), DRAW_BY_RULE);
        assert_eq!(termination_to_outcome(Termination::PlyLimit, 255), PLY_LIMIT);
    }

    #[test]
    fn test_lichess_outcome_token_normal_results() {
        assert_eq!(lichess_outcome_token("Normal", "1-0", true, false, false), Some(WHITE_CHECKMATES));
        assert_eq!(lichess_outcome_token("Normal", "0-1", true, false, false), Some(BLACK_CHECKMATES));
        assert_eq!(lichess_outcome_token("Normal", "1-0", false, false, false), Some(WHITE_RESIGNS));
        assert_eq!(lichess_outcome_token("Normal", "0-1", false, false, false), Some(BLACK_RESIGNS));
        assert_eq!(lichess_outcome_token("Normal", "1/2-1/2", false, true, false), Some(STALEMATE));
        assert_eq!(lichess_outcome_token("Normal", "1/2-1/2", false, false, false), Some(DRAW_BY_AGREEMENT));
    }

    #[test]
    fn test_lichess_outcome_token_time_forfeit() {
        assert_eq!(lichess_outcome_token("Time forfeit", "1-0", false, false, false), Some(WHITE_WINS_ON_TIME));
        assert_eq!(lichess_outcome_token("Time forfeit", "0-1", false, false, false), Some(BLACK_WINS_ON_TIME));
        assert_eq!(lichess_outcome_token("Time forfeit", "1/2-1/2", false, false, false), Some(DRAW_BY_TIME));
    }

    #[test]
    fn test_lichess_outcome_token_truncated() {
        assert_eq!(lichess_outcome_token("Normal", "1-0", true, false, true), Some(PLY_LIMIT));
        assert_eq!(lichess_outcome_token("Time forfeit", "0-1", false, false, true), Some(PLY_LIMIT));
        assert_eq!(lichess_outcome_token("Abandoned", "1-0", false, false, true), Some(PLY_LIMIT));
    }

    #[test]
    fn test_lichess_outcome_token_filtered() {
        assert!(lichess_outcome_token("Abandoned", "1-0", false, false, false).is_none());
        assert!(lichess_outcome_token("Rules infraction", "0-1", false, false, false).is_none());
        assert!(lichess_outcome_token("Unterminated", "*", false, false, false).is_none());
        assert!(lichess_outcome_token("Normal", "*", false, false, false).is_none());
    }

    // --- Promo pairs (kept for grid-based representations) ---

    #[test]
    fn test_promo_pairs_count() {
        let pairs = promo_pairs();
        assert_eq!(pairs.len(), 44);
        for i in 0..22 {
            let (src, _) = pairs[i];
            assert_eq!(src / 8, 6, "White promo src must be rank 6");
        }
        for i in 22..44 {
            let (src, _) = pairs[i];
            assert_eq!(src / 8, 1, "Black promo src must be rank 1");
        }
    }

    #[test]
    fn test_promo_pair_index_lookup() {
        for (i, &(s, d)) in promo_pairs().iter().enumerate() {
            assert_eq!(promo_pair_index(s, d), Some(i));
        }
        assert!(promo_pair_index(12, 28).is_none()); // e2e4 is not a promo
    }

    #[test]
    fn test_token_to_src_dst() {
        let e2e4 = uci_to_action("e2e4").unwrap();
        assert_eq!(token_to_src_dst(e2e4), (12, 28));

        let a7a8q = uci_to_action("a7a8q").unwrap();
        assert_eq!(token_to_src_dst(a7a8q), (48, 56));
    }
}
