//! Move vocabulary: the single source of truth for token ↔ UCI string mapping.
//!
//! Token layout (4,278 total):
//!   0        = padding
//!   1..=4096 = base grid (64×64 src×dst pairs)
//!   4097..=4272 = promotions (44 eligible pairs × 4 piece types)
//!   4273..=4277 = outcome tokens (game result)
//!
//! Square indexing: file-major within rank.
//!   a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
//!   file = index % 8, rank = index / 8

use std::collections::HashMap;

use crate::types::Termination;

pub const VOCAB_SIZE: usize = 4278;
pub const PAD_TOKEN: u16 = 0;
pub const BASE_GRID_START: u16 = 1;
pub const BASE_GRID_END: u16 = 4096; // inclusive
pub const PROMO_START: u16 = 4097;
pub const PROMO_END: u16 = 4272; // inclusive
pub const NUM_PROMO_PAIRS: usize = 44;
pub const NUM_PROMO_TYPES: usize = 4;

// Outcome tokens — must match pawn/config.py
pub const OUTCOME_BASE: u16 = 4273;
pub const WHITE_CHECKMATES: u16 = 4273;
pub const BLACK_CHECKMATES: u16 = 4274;
pub const STALEMATE: u16 = 4275;
pub const DRAW_BY_RULE: u16 = 4276;
pub const PLY_LIMIT: u16 = 4277;

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
}
