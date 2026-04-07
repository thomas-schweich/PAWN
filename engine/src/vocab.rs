//! Move vocabulary: the single source of truth for token ↔ UCI string mapping.
//!
//! Token layout (4,284 total):
//!   0        = padding
//!   1..=4096 = base grid (64×64 src×dst pairs)
//!   4097..=4272 = promotions (44 eligible pairs × 4 piece types)
//!   4273..=4283 = outcome tokens (game result)
//!
//! Square indexing: file-major within rank.
//!   a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
//!   file = index % 8, rank = index / 8

use std::collections::HashMap;

use crate::types::Termination;

pub const VOCAB_SIZE: usize = 4284;
pub const PAD_TOKEN: u16 = 0;
pub const BASE_GRID_START: u16 = 1;
pub const BASE_GRID_END: u16 = 4096; // inclusive
pub const PROMO_START: u16 = 4097;
pub const PROMO_END: u16 = 4272; // inclusive
pub const NUM_PROMO_PAIRS: usize = 44;
pub const NUM_PROMO_TYPES: usize = 4;

// Outcome tokens — must match pawn/config.py
pub const OUTCOME_BASE: u16 = 4273;

// Pretraining outcomes (random games — natural terminations)
pub const WHITE_CHECKMATES: u16 = 4273;
pub const BLACK_CHECKMATES: u16 = 4274;
pub const STALEMATE: u16 = 4275;
pub const DRAW_BY_RULE: u16 = 4276;  // 75-move, fivefold rep, insufficient material
pub const PLY_LIMIT: u16 = 4277;     // Hit max plies (also truncated Lichess games)

// Lichess-specific outcomes (finetuning data)
pub const WHITE_RESIGNS: u16 = 4278;
pub const BLACK_RESIGNS: u16 = 4279;
pub const DRAW_BY_AGREEMENT: u16 = 4280;
pub const WHITE_WINS_ON_TIME: u16 = 4281;
pub const BLACK_WINS_ON_TIME: u16 = 4282;
pub const DRAW_BY_TIME: u16 = 4283;

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

/// The 44 promotion-eligible (src_square, dst_square) pairs.
/// 22 per color: 8 straight pushes + 7 left captures + 7 right captures.
/// White promotes: src on rank 6 (indices 48..56), dst on rank 7 (indices 56..64).
/// Black promotes: src on rank 1 (indices 8..16), dst on rank 0 (indices 0..8).
///
/// Order: white straight, white left-capture, white right-capture,
///        black straight, black left-capture, black right-capture.
static PROMO_PAIRS_ARRAY: once_cell::sync::Lazy<[(u8, u8); NUM_PROMO_PAIRS]> =
    once_cell::sync::Lazy::new(|| {
        let mut pairs = [(0u8, 0u8); NUM_PROMO_PAIRS];
        let mut idx = 0;

        // White straight pushes: src file f, rank 6 -> dst same file, rank 7
        for file in 0u8..8 {
            let src = 6 * 8 + file; // rank 6
            let dst = 7 * 8 + file; // rank 7
            pairs[idx] = (src, dst);
            idx += 1;
        }
        // White left captures: file-1 (dst file < src file)
        for file in 1u8..8 {
            let src = 6 * 8 + file;
            let dst = 7 * 8 + (file - 1);
            pairs[idx] = (src, dst);
            idx += 1;
        }
        // White right captures: file+1 (dst file > src file)
        for file in 0u8..7 {
            let src = 6 * 8 + file;
            let dst = 7 * 8 + (file + 1);
            pairs[idx] = (src, dst);
            idx += 1;
        }
        // Black straight pushes: src rank 1 -> dst rank 0
        for file in 0u8..8 {
            let src = 1 * 8 + file; // rank 1
            let dst = 0 * 8 + file; // rank 0
            pairs[idx] = (src, dst);
            idx += 1;
        }
        // Black left captures: file-1
        for file in 1u8..8 {
            let src = 1 * 8 + file;
            let dst = 0 * 8 + (file - 1);
            pairs[idx] = (src, dst);
            idx += 1;
        }
        // Black right captures: file+1
        for file in 0u8..7 {
            let src = 1 * 8 + file;
            let dst = 0 * 8 + (file + 1);
            pairs[idx] = (src, dst);
            idx += 1;
        }
        assert_eq!(idx, NUM_PROMO_PAIRS);
        pairs
    });

pub fn promo_pairs() -> &'static [(u8, u8); NUM_PROMO_PAIRS] {
    &PROMO_PAIRS_ARRAY
}

/// Lookup table: for a given (src, dst) pair, what is the index into promo_pairs?
/// Returns None if this pair is not promotion-eligible.
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

/// Convert a base-grid (src, dst) pair to its token index (1..=4096).
#[inline]
pub fn base_grid_token(src: u8, dst: u8) -> u16 {
    debug_assert!(src < 64 && dst < 64);
    (src as u16) * 64 + (dst as u16) + 1
}

/// Convert a promotion move to its token index (4097..=4272).
/// promo_type: 0=q, 1=r, 2=b, 3=n
pub fn promo_token(src: u8, dst: u8, promo_type: u8) -> Option<u16> {
    if promo_type >= NUM_PROMO_TYPES as u8 {
        return None;
    }
    let pair_idx = promo_pair_index(src, dst)?;
    Some(PROMO_START + (pair_idx as u16) * 4 + (promo_type as u16))
}

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

/// Decompose a token into (src_square, dst_square, promo_type).
/// promo_type: 0=none, 1=q, 2=r, 3=b, 4=n (matches embedding index).
/// Returns None for PAD and outcome tokens.
pub fn decompose_token(token: u16) -> Option<(u8, u8, u8)> {
    if token == PAD_TOKEN || token >= OUTCOME_BASE {
        return None;
    }
    if token >= BASE_GRID_START && token <= BASE_GRID_END {
        let t = token - 1;
        let src = (t / 64) as u8;
        let dst = (t % 64) as u8;
        return Some((src, dst, 0)); // promo=0 means none
    }
    if token >= PROMO_START && token <= PROMO_END {
        let t = token - PROMO_START;
        let pair_idx = (t / 4) as usize;
        let promo_type = (t % 4) as u8 + 1; // 1=q, 2=r, 3=b, 4=n
        let (src, dst) = promo_pairs()[pair_idx];
        return Some((src, dst, promo_type));
    }
    None
}

/// Extract (src_square, dst_square) from a move token, ignoring promotion type.
pub fn token_to_src_dst(token: u16) -> (u8, u8) {
    let (src, dst, _) = decompose_token(token).expect("invalid token");
    (src, dst)
}

/// Convert a token to its UCI string representation.
pub fn token_to_uci(token: u16) -> Option<String> {
    let (src, dst, promo) = decompose_token(token)?;
    let mut s = String::with_capacity(5);
    s.push_str(SQUARE_NAMES[src as usize]);
    s.push_str(SQUARE_NAMES[dst as usize]);
    if promo > 0 {
        s.push_str(PROMO_PIECES[(promo - 1) as usize]);
    }
    Some(s)
}

/// Build the full token_to_move and move_to_token maps.
pub fn build_vocab_maps() -> (HashMap<u16, String>, HashMap<String, u16>) {
    let mut token_to_move = HashMap::with_capacity(VOCAB_SIZE);
    let mut move_to_token = HashMap::with_capacity(VOCAB_SIZE);

    // Base grid: 1..=4096
    for src in 0u8..64 {
        for dst in 0u8..64 {
            let token = base_grid_token(src, dst);
            let uci = format!("{}{}", SQUARE_NAMES[src as usize], SQUARE_NAMES[dst as usize]);
            token_to_move.insert(token, uci.clone());
            move_to_token.insert(uci, token);
        }
    }

    // Promotions: 4097..=4272
    let pairs = promo_pairs();
    for (pair_idx, &(src, dst)) in pairs.iter().enumerate() {
        for promo_idx in 0u8..4 {
            let token = PROMO_START + (pair_idx as u16) * 4 + (promo_idx as u16);
            let uci = format!(
                "{}{}{}",
                SQUARE_NAMES[src as usize],
                SQUARE_NAMES[dst as usize],
                PROMO_PIECES[promo_idx as usize]
            );
            token_to_move.insert(token, uci.clone());
            move_to_token.insert(uci, token);
        }
    }

    (token_to_move, move_to_token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_indexing() {
        assert_eq!(SQUARE_NAMES[0], "a1");
        assert_eq!(SQUARE_NAMES[7], "h1");
        assert_eq!(SQUARE_NAMES[8], "a2");
        assert_eq!(SQUARE_NAMES[63], "h8");
    }

    #[test]
    fn test_base_grid_tokens() {
        // a1a1 = src=0, dst=0 -> token 1
        assert_eq!(base_grid_token(0, 0), 1);
        // a1h8 = src=0, dst=63 -> token 64
        assert_eq!(base_grid_token(0, 63), 64);
        // h8h8 = src=63, dst=63 -> token 4096
        assert_eq!(base_grid_token(63, 63), 4096);
    }

    #[test]
    fn test_decompose_roundtrip() {
        for token in 1..=4096u16 {
            let (src, dst, promo) = decompose_token(token).unwrap();
            assert_eq!(promo, 0);
            assert_eq!(base_grid_token(src, dst), token);
        }
        for token in 4097..=4272u16 {
            let (src, dst, promo) = decompose_token(token).unwrap();
            assert!(promo >= 1 && promo <= 4);
            assert_eq!(promo_token(src, dst, promo - 1), Some(token));
        }
    }

    #[test]
    fn test_vocab_maps_roundtrip() {
        let (t2m, m2t) = build_vocab_maps();
        assert_eq!(t2m.len(), 4096 + 176); // base grid + promotions
        for (token, uci) in &t2m {
            assert_eq!(m2t.get(uci), Some(token));
        }
    }

    #[test]
    fn test_promo_pairs_count() {
        let pairs = promo_pairs();
        assert_eq!(pairs.len(), 44);
        // Verify white pairs: first 22
        for i in 0..22 {
            let (src, dst) = pairs[i];
            let src_rank = src / 8;
            let dst_rank = dst / 8;
            assert_eq!(src_rank, 6, "White promo src must be rank 6");
            assert_eq!(dst_rank, 7, "White promo dst must be rank 7");
        }
        // Verify black pairs: last 22
        for i in 22..44 {
            let (src, dst) = pairs[i];
            let src_rank = src / 8;
            let dst_rank = dst / 8;
            assert_eq!(src_rank, 1, "Black promo src must be rank 1");
            assert_eq!(dst_rank, 0, "Black promo dst must be rank 0");
        }
    }

    #[test]
    fn test_specific_uci() {
        // e2e4 = src=e2=12, dst=e4=28
        assert_eq!(token_to_uci(base_grid_token(12, 28)).unwrap(), "e2e4");
        // e1g1 (castling) = src=e1=4, dst=g1=6
        assert_eq!(token_to_uci(base_grid_token(4, 6)).unwrap(), "e1g1");
        // a7a8q (promotion)
        let token = promo_token(48, 56, 0).unwrap(); // src=a7=48, dst=a8=56, q=0
        assert_eq!(token_to_uci(token).unwrap(), "a7a8q");
    }

    #[test]
    fn test_pad_outcome_decompose() {
        assert!(decompose_token(PAD_TOKEN).is_none());
        // All 5 outcome tokens should return None
        for token in OUTCOME_BASE..=PLY_LIMIT {
            assert!(decompose_token(token).is_none(),
                "outcome token {} should not decompose", token);
        }
    }

    #[test]
    fn test_termination_to_outcome() {
        use crate::types::Termination;

        // Checkmate with odd game_length = white wins
        assert_eq!(termination_to_outcome(Termination::Checkmate, 11), WHITE_CHECKMATES);
        assert_eq!(termination_to_outcome(Termination::Checkmate, 1), WHITE_CHECKMATES);

        // Checkmate with even game_length = black wins
        assert_eq!(termination_to_outcome(Termination::Checkmate, 12), BLACK_CHECKMATES);
        assert_eq!(termination_to_outcome(Termination::Checkmate, 2), BLACK_CHECKMATES);

        // Other terminations
        assert_eq!(termination_to_outcome(Termination::Stalemate, 50), STALEMATE);
        assert_eq!(termination_to_outcome(Termination::SeventyFiveMoveRule, 100), DRAW_BY_RULE);
        assert_eq!(termination_to_outcome(Termination::FivefoldRepetition, 80), DRAW_BY_RULE);
        assert_eq!(termination_to_outcome(Termination::InsufficientMaterial, 60), DRAW_BY_RULE);
        assert_eq!(termination_to_outcome(Termination::PlyLimit, 255), PLY_LIMIT);
    }

    // ==== New tests added by Agent A (Rust Core) ====

    #[test]
    fn test_vocab_constants() {
        // VOCAB_SIZE must match docs: 1 PAD + 4096 grid + 176 promo + 11 outcome = 4284
        assert_eq!(VOCAB_SIZE, 4284);
        assert_eq!(PAD_TOKEN, 0);
        assert_eq!(BASE_GRID_START, 1);
        assert_eq!(BASE_GRID_END, 4096);
        assert_eq!(PROMO_START, 4097);
        assert_eq!(PROMO_END, 4272);
        // 4272 - 4097 + 1 = 176 promotion tokens
        assert_eq!(PROMO_END - PROMO_START + 1, 176);
        assert_eq!(NUM_PROMO_PAIRS, 44);
        assert_eq!(NUM_PROMO_TYPES, 4);
        assert_eq!(NUM_PROMO_PAIRS * NUM_PROMO_TYPES, 176);
        // Outcome base starts right after promo end
        assert_eq!(OUTCOME_BASE, PROMO_END + 1);
        assert_eq!(OUTCOME_BASE, 4273);
        // 11 outcome tokens: 4273..=4283
        assert_eq!(DRAW_BY_TIME, 4283);
        assert_eq!(DRAW_BY_TIME as usize - OUTCOME_BASE as usize + 1, 11);
    }

    #[test]
    fn test_all_outcome_tokens_distinct_and_in_range() {
        // All 11 outcome tokens distinct and within [OUTCOME_BASE, VOCAB_SIZE)
        let outcomes = [
            WHITE_CHECKMATES, BLACK_CHECKMATES, STALEMATE, DRAW_BY_RULE, PLY_LIMIT,
            WHITE_RESIGNS, BLACK_RESIGNS, DRAW_BY_AGREEMENT,
            WHITE_WINS_ON_TIME, BLACK_WINS_ON_TIME, DRAW_BY_TIME,
        ];
        for &t in &outcomes {
            assert!(t >= OUTCOME_BASE, "outcome {} < OUTCOME_BASE", t);
            assert!((t as usize) < VOCAB_SIZE, "outcome {} >= VOCAB_SIZE", t);
        }
        // Check distinctness
        let mut sorted = outcomes.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), outcomes.len(), "outcome tokens must be distinct");
    }

    #[test]
    fn test_square_names_indexing_exhaustive() {
        // Verify ALL 64 square names match the file-major formula: file=i%8, rank=i/8
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
    fn test_base_grid_token_exhaustive_roundtrip() {
        // Every (src, dst) pair in [0..64, 0..64] => token => roundtrip back
        for src in 0u8..64 {
            for dst in 0u8..64 {
                let token = base_grid_token(src, dst);
                assert!(token >= BASE_GRID_START && token <= BASE_GRID_END);
                let (s, d, p) = decompose_token(token).unwrap();
                assert_eq!(s, src);
                assert_eq!(d, dst);
                assert_eq!(p, 0);
            }
        }
    }

    #[test]
    fn test_promo_pair_index_lookup() {
        // Every pair in the array should be findable via the lookup table
        for (i, &(s, d)) in promo_pairs().iter().enumerate() {
            assert_eq!(promo_pair_index(s, d), Some(i));
        }
        // A non-promotion pair (e.g. e2e4: src=12, dst=28) returns None
        assert!(promo_pair_index(12, 28).is_none());
        // Mid-board src should not be a promotion
        assert!(promo_pair_index(32, 40).is_none());
    }

    #[test]
    fn test_promo_pairs_all_unique() {
        let pairs = promo_pairs();
        let mut seen = std::collections::HashSet::new();
        for &(s, d) in pairs.iter() {
            assert!(seen.insert((s, d)), "Duplicate promo pair ({}, {})", s, d);
        }
        assert_eq!(seen.len(), NUM_PROMO_PAIRS);
    }

    #[test]
    fn test_promo_pair_ordering() {
        // Pair indices 0..8: white straight (file f -> file f)
        let pairs = promo_pairs();
        for f in 0..8u8 {
            assert_eq!(pairs[f as usize], (48 + f, 56 + f));
        }
        // Pairs 8..15: white left-capture (file f -> file f-1, f=1..=7)
        for (i, f) in (1..8u8).enumerate() {
            assert_eq!(pairs[8 + i], (48 + f, 56 + f - 1));
        }
        // Pairs 15..22: white right-capture (file f -> file f+1, f=0..=6)
        for (i, f) in (0..7u8).enumerate() {
            assert_eq!(pairs[15 + i], (48 + f, 56 + f + 1));
        }
        // Pairs 22..30: black straight (file f -> file f), src rank 1 -> dst rank 0
        for f in 0..8u8 {
            assert_eq!(pairs[22 + f as usize], (8 + f, f));
        }
    }

    #[test]
    fn test_promo_token_exhaustive_roundtrip() {
        // For every promo pair and every promo type, token -> decompose -> reconstruct
        for (pair_idx, &(src, dst)) in promo_pairs().iter().enumerate() {
            for promo_type in 0u8..4 {
                let token = promo_token(src, dst, promo_type).unwrap();
                let expected = PROMO_START + (pair_idx as u16) * 4 + (promo_type as u16);
                assert_eq!(token, expected);
                assert!(token >= PROMO_START && token <= PROMO_END);
                // Decompose returns promo_type+1 (1..=4) since 0 means "no promo"
                let (s, d, p) = decompose_token(token).unwrap();
                assert_eq!(s, src);
                assert_eq!(d, dst);
                assert_eq!(p, promo_type + 1);
            }
        }
    }

    #[test]
    fn test_promo_token_invalid_pair() {
        // Non-promotion pair returns None
        assert!(promo_token(0, 0, 0).is_none()); // a1a1 is not a promo
        assert!(promo_token(12, 28, 0).is_none()); // e2e4 is not a promo
    }

    #[test]
    fn test_promo_token_invalid_type() {
        // White straight a7a8 (src=48, dst=56) is valid
        assert!(promo_token(48, 56, 0).is_some());
        // promo_type >= 4 is invalid
        assert!(promo_token(48, 56, 4).is_none());
        assert!(promo_token(48, 56, 99).is_none());
    }

    #[test]
    fn test_token_to_uci_specific_cases() {
        // Base grid
        assert_eq!(token_to_uci(1).unwrap(), "a1a1"); // src=0, dst=0
        assert_eq!(token_to_uci(64).unwrap(), "a1h8"); // src=0, dst=63
        assert_eq!(token_to_uci(65).unwrap(), "b1a1"); // src=1, dst=0
        assert_eq!(token_to_uci(4096).unwrap(), "h8h8"); // src=63, dst=63

        // a7a8q - first white promo
        let t_a7a8q = promo_token(48, 56, 0).unwrap();
        assert_eq!(t_a7a8q, PROMO_START);
        assert_eq!(token_to_uci(t_a7a8q).unwrap(), "a7a8q");

        // a7a8n - first white promo, knight
        let t_a7a8n = promo_token(48, 56, 3).unwrap();
        assert_eq!(token_to_uci(t_a7a8n).unwrap(), "a7a8n");

        // a2a1q - first black promo
        let t_a2a1q = promo_token(8, 0, 0).unwrap();
        assert_eq!(token_to_uci(t_a2a1q).unwrap(), "a2a1q");
    }

    #[test]
    fn test_token_to_uci_all_promo_types() {
        // Verify order q,r,b,n for promo types 0..4
        let expected = ["q", "r", "b", "n"];
        for promo_type in 0u8..4 {
            let token = promo_token(48, 56, promo_type).unwrap();
            let uci = token_to_uci(token).unwrap();
            assert!(uci.ends_with(expected[promo_type as usize]),
                "promo_type {} -> {} should end with {}", promo_type, uci, expected[promo_type as usize]);
        }
    }

    #[test]
    fn test_token_to_uci_none_for_invalid() {
        assert!(token_to_uci(PAD_TOKEN).is_none());
        assert!(token_to_uci(OUTCOME_BASE).is_none());
        assert!(token_to_uci(DRAW_BY_TIME).is_none());
        assert!(token_to_uci(PLY_LIMIT).is_none());
    }

    #[test]
    fn test_decompose_token_all_outcomes_none() {
        // All 11 outcome tokens (4273..=4283) should return None
        for token in OUTCOME_BASE..=DRAW_BY_TIME {
            assert!(decompose_token(token).is_none(),
                "outcome token {} should return None", token);
        }
    }

    #[test]
    fn test_decompose_token_out_of_range() {
        // Tokens beyond vocab size should return None
        assert!(decompose_token(VOCAB_SIZE as u16).is_none());
        assert!(decompose_token(5000).is_none());
        assert!(decompose_token(u16::MAX).is_none());
    }

    #[test]
    fn test_build_vocab_maps_bijective() {
        let (t2m, m2t) = build_vocab_maps();
        // 4096 grid + 176 promo = 4272 entries
        assert_eq!(t2m.len(), 4272);
        assert_eq!(m2t.len(), 4272);
        // Verify all mappings are consistent
        for (&token, uci) in &t2m {
            assert_eq!(m2t.get(uci), Some(&token),
                "Inconsistent mapping: token {} -> {}", token, uci);
        }
        for (uci, &token) in &m2t {
            assert_eq!(t2m.get(&token), Some(uci),
                "Inconsistent mapping: {} -> token {}", uci, token);
        }
    }

    #[test]
    fn test_build_vocab_maps_uci_format() {
        let (_, m2t) = build_vocab_maps();
        // Every UCI in the map is either 4 chars (base) or 5 chars (promo)
        for (uci, &token) in &m2t {
            if token >= PROMO_START {
                assert_eq!(uci.len(), 5, "promo UCI should be 5 chars: {}", uci);
            } else {
                assert_eq!(uci.len(), 4, "base UCI should be 4 chars: {}", uci);
            }
        }
    }

    #[test]
    fn test_token_to_src_dst_ignores_promo() {
        // src=48 (a7), dst=56 (a8), promo=queen
        let token = promo_token(48, 56, 0).unwrap();
        assert_eq!(token_to_src_dst(token), (48, 56));
        // Same for non-promo
        let token = base_grid_token(12, 28);
        assert_eq!(token_to_src_dst(token), (12, 28));
    }

    #[test]
    fn test_promo_pairs_cover_all_rank_7_destinations() {
        // White promotes: every file has a straight push and capture combinations
        // All 22 white pairs must have dst on rank 7 (indices 56..64)
        // And each rank-7 dst must be reached from at least one src
        let pairs = promo_pairs();
        let mut white_dsts = std::collections::HashSet::new();
        for i in 0..22 {
            let (_, dst) = pairs[i];
            assert!(dst >= 56 && dst < 64);
            white_dsts.insert(dst);
        }
        // All 8 files on rank 7 are reachable
        assert_eq!(white_dsts.len(), 8);
        // Same for black: all dsts on rank 0
        let mut black_dsts = std::collections::HashSet::new();
        for i in 22..44 {
            let (_, dst) = pairs[i];
            assert!(dst < 8);
            black_dsts.insert(dst);
        }
        assert_eq!(black_dsts.len(), 8);
    }

    #[test]
    fn test_lichess_outcome_token_normal_results() {
        // Normal termination - checkmate wins
        assert_eq!(lichess_outcome_token("Normal", "1-0", true, false, false), Some(WHITE_CHECKMATES));
        assert_eq!(lichess_outcome_token("Normal", "0-1", true, false, false), Some(BLACK_CHECKMATES));
        // Normal - resignation (not checkmate)
        assert_eq!(lichess_outcome_token("Normal", "1-0", false, false, false), Some(WHITE_RESIGNS));
        assert_eq!(lichess_outcome_token("Normal", "0-1", false, false, false), Some(BLACK_RESIGNS));
        // Normal - stalemate vs draw agreement
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
    fn test_lichess_outcome_token_insufficient_material() {
        assert_eq!(lichess_outcome_token("Insufficient material", "1/2-1/2", false, false, false), Some(DRAW_BY_RULE));
    }

    #[test]
    fn test_lichess_outcome_token_truncated() {
        // Truncated games (max_ply exceeded) always become PLY_LIMIT regardless
        assert_eq!(lichess_outcome_token("Normal", "1-0", true, false, true), Some(PLY_LIMIT));
        assert_eq!(lichess_outcome_token("Time forfeit", "0-1", false, false, true), Some(PLY_LIMIT));
        assert_eq!(lichess_outcome_token("Abandoned", "1-0", false, false, true), Some(PLY_LIMIT));
    }

    #[test]
    fn test_lichess_outcome_token_filtered() {
        // Rules infraction, Abandoned, Unterminated are filtered out
        assert!(lichess_outcome_token("Abandoned", "1-0", false, false, false).is_none());
        assert!(lichess_outcome_token("Rules infraction", "0-1", false, false, false).is_none());
        assert!(lichess_outcome_token("Unterminated", "*", false, false, false).is_none());
        // Unknown results with Normal termination are None
        assert!(lichess_outcome_token("Normal", "*", false, false, false).is_none());
    }

    #[test]
    fn test_token_ranges_dont_overlap() {
        // PAD=0 is alone, base grid 1..=4096, promo 4097..=4272, outcome 4273..=4283
        assert_eq!(PAD_TOKEN + 1, BASE_GRID_START);
        assert_eq!(BASE_GRID_END + 1, PROMO_START);
        assert_eq!(PROMO_END + 1, OUTCOME_BASE);
        // Total vocab = 4284 accounts for all: 0 + [1,4283] = 4284 tokens
    }
}
