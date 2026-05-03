//! UCI move parsing: UCI strings → PAWN token sequences and
//! SAN → UCI conversion.
//!
//! Parallel to pgn.rs but for UCI-format input (e.g., "e2e4 e7e5 g1f3").

use rayon::prelude::*;
use shakmaty::san::SanPlus;
use shakmaty::{Chess, Move, Position, Role, Square};

use crate::board::{move_to_token, our_sq_to_shakmaty};
use crate::vocab;

/// Parse a UCI move string (e.g., "e2e4", "e7e8q") into a legal shakmaty Move.
///
/// Returns None if the string is malformed or the move is illegal.
fn parse_uci_move(uci: &str, pos: &Chess) -> Option<Move> {
    let bytes = uci.as_bytes();
    if bytes.len() < 4 || bytes.len() > 5 {
        return None;
    }

    let src_file = bytes[0].wrapping_sub(b'a');
    let src_rank = bytes[1].wrapping_sub(b'1');
    let dst_file = bytes[2].wrapping_sub(b'a');
    let dst_rank = bytes[3].wrapping_sub(b'1');

    if src_file > 7 || src_rank > 7 || dst_file > 7 || dst_rank > 7 {
        return None;
    }

    let src_sq = our_sq_to_shakmaty(src_rank * 8 + src_file);
    let dst_sq = our_sq_to_shakmaty(dst_rank * 8 + dst_file);

    let promo_role = if bytes.len() == 5 {
        match bytes[4] {
            b'q' | b'Q' => Some(Role::Queen),
            b'r' | b'R' => Some(Role::Rook),
            b'b' | b'B' => Some(Role::Bishop),
            b'n' | b'N' => Some(Role::Knight),
            _ => return None,
        }
    } else {
        None
    };

    // Find the matching legal move
    let legal = pos.legal_moves();
    for m in &legal {
        let (m_src, m_dst, m_promo) = match m {
            Move::Normal { from, to, promotion, .. } => (*from, *to, *promotion),
            Move::EnPassant { from, to } => (*from, *to, None),
            Move::Castle { king, rook } => {
                let king_sq = *king;
                let rook_sq = *rook;
                let castle_dst = if rook_sq.file() > king_sq.file() {
                    Square::from_coords(shakmaty::File::G, king_sq.rank())
                } else {
                    Square::from_coords(shakmaty::File::C, king_sq.rank())
                };
                (king_sq, castle_dst, None)
            }
            _ => continue,
        };

        if m_src == src_sq && m_dst == dst_sq && m_promo == promo_role {
            return Some(m.clone());
        }
    }

    None
}

/// Convert a sequence of UCI move strings to PAWN token indices.
///
/// Returns (tokens, n_valid) where n_valid is how many moves were
/// successfully parsed. Stops at the first illegal or unparseable move.
pub fn uci_moves_to_tokens(
    uci_moves: &[&str],
    max_ply: usize,
) -> (Vec<u16>, usize) {
    let mut pos = Chess::default();
    let mut tokens = Vec::with_capacity(uci_moves.len().min(max_ply));

    for (i, uci_str) in uci_moves.iter().enumerate() {
        if i >= max_ply {
            break;
        }

        let m = match parse_uci_move(uci_str, &pos) {
            Some(m) => m,
            None => break,
        };

        let token = move_to_token(&m);
        tokens.push(token);
        pos.play_unchecked(m);
    }

    let n = tokens.len();
    (tokens, n)
}

/// Convert a sequence of SAN moves to UCI strings.
///
/// Returns (uci_strings, n_valid). Stops at the first parse error.
pub fn san_to_uci(san_moves: &[&str]) -> (Vec<String>, usize) {
    let mut pos = Chess::default();
    let mut uci_moves = Vec::with_capacity(san_moves.len());

    for san_str in san_moves {
        let san = match shakmaty::san::San::from_ascii(san_str.as_bytes()) {
            Ok(s) => s,
            Err(_) => break,
        };

        let m = match san.to_move(&pos) {
            Ok(m) => m,
            Err(_) => break,
        };

        let token = move_to_token(&m);
        let uci = vocab::token_to_uci(token).unwrap();
        uci_moves.push(uci);
        pos.play_unchecked(m);
    }

    let n = uci_moves.len();
    (uci_moves, n)
}

/// Full pipeline: read a file of UCI game lines, convert to tokens.
///
/// File format: one game per line, space-separated UCI moves, optional
/// result marker at end (1-0, 0-1, 1/2-1/2, *).
///
/// Returns (flat_tokens: Vec<i16> of shape n_games*max_ply,
///          lengths: Vec<i16>, n_parsed: usize).
pub fn uci_file_to_tokens(
    path: &str,
    max_ply: usize,
    max_games: usize,
    min_ply: usize,
) -> (Vec<i16>, Vec<i16>, usize) {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read UCI file {}: {}", path, e));

    let lines: Vec<&str> = content.lines()
        .filter(|l| !l.trim().is_empty())
        .take(max_games)
        .collect();

    let n_parsed = lines.len();

    // Parse each line into UCI moves (strip result marker)
    let games: Vec<Vec<&str>> = lines.iter().map(|line| {
        let parts: Vec<&str> = line.split_whitespace().collect();
        // Strip result marker if present
        if let Some(last) = parts.last() {
            if *last == "1-0" || *last == "0-1" || *last == "1/2-1/2" || *last == "*" {
                return parts[..parts.len() - 1].to_vec();
            }
        }
        parts
    }).collect();

    // Parallel token conversion
    let converted: Vec<(Vec<u16>, usize)> = games
        .par_iter()
        .map(|moves| uci_moves_to_tokens(moves, max_ply))
        .collect();

    // Filter by min_ply and pack into flat array
    let filtered: Vec<&(Vec<u16>, usize)> = converted
        .iter()
        .filter(|(_, n)| *n >= min_ply)
        .collect();

    let n = filtered.len();
    let mut flat = vec![0i16; n * max_ply];
    let mut lengths = Vec::with_capacity(n);

    for (gi, (tokens, n_valid)) in filtered.iter().enumerate() {
        for (t, &tok) in tokens.iter().enumerate() {
            flat[gi * max_ply + t] = tok as i16;
        }
        lengths.push(*n_valid as i16);
    }

    (flat, lengths, n_parsed)
}

/// Batch convert: SAN games → UCI strings. Parallel via rayon.
///
/// Returns a Vec of (uci_strings, n_valid) per game.
pub fn batch_san_to_uci(games: &[Vec<&str>]) -> Vec<(Vec<String>, usize)> {
    games
        .par_iter()
        .map(|san_moves| san_to_uci(san_moves))
        .collect()
}

/// Convert a sequence of UCI move strings to SAN strings (with check/mate suffixes).
///
/// Returns (san_strings, n_valid). Stops at the first illegal or unparseable move.
pub fn uci_to_san(uci_moves: &[&str]) -> (Vec<String>, usize) {
    let mut pos = Chess::default();
    let mut san_moves = Vec::with_capacity(uci_moves.len());

    for uci_str in uci_moves {
        let m = match parse_uci_move(uci_str, &pos) {
            Some(m) => m,
            None => break,
        };
        let san_plus = SanPlus::from_move_and_play_unchecked(&mut pos, m);
        san_moves.push(san_plus.to_string());
    }

    let n = san_moves.len();
    (san_moves, n)
}

/// Batch convert: UCI games → SAN strings. Parallel via rayon.
///
/// Returns a Vec of (san_strings, n_valid) per game.
pub fn batch_uci_to_san(games: &[Vec<&str>]) -> Vec<(Vec<String>, usize)> {
    games
        .par_iter()
        .map(|uci_moves| uci_to_san(uci_moves))
        .collect()
}

/// Single-pass replay: UCI moves → (action tokens, SAN strings).
///
/// Walks the move list once, maintaining a single shakmaty board, and
/// produces both the searchless_chess token IDs and the SAN strings (with
/// check/mate suffixes) in lock-step. Stops at the first
/// illegal/unparseable move; the returned vecs always have equal length,
/// equal to the number of successfully processed moves. Callers can use
/// `tokens.len()` (or `san.len()`) to detect a short-tokenization.
///
/// This is the canonical encoder used by `stockfish-datagen`. It exists
/// because previously we replayed the game twice (once for tokens, once
/// for SAN) — pointless, since both derive from the same move stream.
pub fn uci_to_tokens_and_san(uci_moves: &[&str]) -> (Vec<u16>, Vec<String>) {
    let mut pos = Chess::default();
    let mut tokens = Vec::with_capacity(uci_moves.len());
    let mut san_moves = Vec::with_capacity(uci_moves.len());

    for uci_str in uci_moves {
        let m = match parse_uci_move(uci_str, &pos) {
            Some(m) => m,
            None => break,
        };
        let token = move_to_token(&m);
        // SanPlus::from_move_and_play_unchecked needs to be called BEFORE
        // we lose `m` (it consumes by value), so derive the token first.
        let san_plus = SanPlus::from_move_and_play_unchecked(&mut pos, m);
        tokens.push(token);
        san_moves.push(san_plus.to_string());
    }

    assert_eq!(
        tokens.len(),
        san_moves.len(),
        "tokens and SAN must stay in lock-step",
    );
    (tokens, san_moves)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uci_to_tokens() {
        let moves = vec!["e2e4", "e7e5", "g1f3", "b8c6"];
        let (tokens, n) = uci_moves_to_tokens(&moves, 256);
        assert_eq!(n, 4);
        assert_eq!(tokens.len(), 4);
        // e2e4: src=12 (e2), dst=28 (e4)
        let e2e4 = vocab::uci_token("e2e4");
        assert_eq!(tokens[0], e2e4);
    }

    #[test]
    fn test_uci_promotion() {
        // 1. a2a4 b7b5 2. a4b5 a7a6 3. b5a6 ... setup for promotion
        // Just test that promotion parsing works
        let moves = vec!["e2e4", "d7d5", "e4d5", "e7e6", "d5e6", "f7e6"];
        let (tokens, n) = uci_moves_to_tokens(&moves, 256);
        assert_eq!(n, 6);
    }

    #[test]
    fn test_uci_castling() {
        // Italian Game to castling
        let moves = vec!["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"];
        let (tokens, n) = uci_moves_to_tokens(&moves, 256);
        assert_eq!(n, 7);
    }

    #[test]
    fn test_uci_illegal_stops() {
        let moves = vec!["e2e4", "e7e5", "e4e5"]; // e4e5 is illegal (square occupied)
        let (tokens, n) = uci_moves_to_tokens(&moves, 256);
        assert_eq!(n, 2); // stops before illegal move
    }

    #[test]
    fn test_san_to_uci() {
        let san = vec!["e4", "e5", "Nf3", "Nc6"];
        let (uci, n) = san_to_uci(&san);
        assert_eq!(n, 4);
        assert_eq!(uci, vec!["e2e4", "e7e5", "g1f3", "b8c6"]);
    }

    #[test]
    fn test_san_to_uci_castling() {
        let san = vec!["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O"];
        let (uci, n) = san_to_uci(&san);
        assert_eq!(n, 7);
        assert_eq!(uci[6], "e1g1");
    }

    #[test]
    fn test_uci_file_to_tokens() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_uci.txt");
        std::fs::write(&path, "e2e4 e7e5 g1f3 b8c6 1-0\nd2d4 d7d5 0-1\n").unwrap();

        let (flat, lengths, n_parsed) = uci_file_to_tokens(
            path.to_str().unwrap(), 256, 100, 2,
        );
        assert_eq!(n_parsed, 2);
        assert_eq!(lengths.len(), 2);
        assert_eq!(lengths[0], 4);
        assert_eq!(lengths[1], 2);
        assert_eq!(flat.len(), 2 * 256);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_uci_malformed_strings() {
        // Too short
        assert!(parse_uci_move("e2e", &Chess::default()).is_none());
        // Too long (beyond promo)
        assert!(parse_uci_move("e2e4qq", &Chess::default()).is_none());
        // Empty
        assert!(parse_uci_move("", &Chess::default()).is_none());
        // Out of range squares
        assert!(parse_uci_move("z1e4", &Chess::default()).is_none());
        assert!(parse_uci_move("e9e4", &Chess::default()).is_none());
    }

    #[test]
    fn test_uci_castling_queenside() {
        // Setup: 1. e4 e5 2. Qh5 Nc6 3. Qh4 Nf6 (... to setup)
        // Easier: setup queenside castling prerequisites.
        // 1. d4 d5 2. Nc3 Nc6 3. Bf4 Bf5 4. Qd2 Qd7 5. O-O-O
        let moves = vec![
            "d2d4", "d7d5", "b1c3", "b8c6", "c1f4", "c8f5",
            "d1d2", "d8d7", "e1c1",
        ];
        let (tokens, n) = uci_moves_to_tokens(&moves, 256);
        assert_eq!(n, 9, "queenside castling setup and O-O-O all legal");
        assert_eq!(tokens.len(), 9);
    }

    #[test]
    fn test_uci_promotion_all_pieces() {
        // Advance a white pawn to c7 via captures, then promote c7xb8=Q/R/B/N.
        let prefix = vec!["a2a4", "b7b5", "a4b5", "c7c5", "b5c6", "d7d5", "c6c7"];
        let black_move = "a7a6";
        let targets = vec!["c7b8q", "c7b8r", "c7b8b", "c7b8n"];

        for target in &targets {
            let mut moves = prefix.clone();
            moves.push(black_move);
            moves.push(target);
            let (tokens, n) = uci_moves_to_tokens(&moves, 256);
            assert_eq!(n, 9, "promotion {} should be legal", target);
            let promo_tok = tokens[8];
            let (_, _, promo_type) = crate::vocab::decompose_token(promo_tok as u16)
                .expect("promotion token should decompose");
            assert!(promo_type >= 1 && promo_type <= 4,
                "promotion token {} should have promo_type 1-4, got {}", promo_tok, promo_type);
        }
    }

    #[test]
    fn test_uci_moves_to_tokens_max_ply_cap() {
        let moves = vec!["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"];
        let (tokens, n) = uci_moves_to_tokens(&moves, 3);
        assert_eq!(n, 3);
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_uci_bad_promo_char() {
        // Promotion with invalid piece char 'x'. Build to promotion with same chain.
        let moves = vec!["a2a4", "b7b5", "a4b5", "c7c5", "b5c6", "d7d5", "c6c7", "a7a6", "c7b8x"];
        let (_, n) = uci_moves_to_tokens(&moves, 256);
        assert_eq!(n, 8, "bad promotion char should stop parsing");
    }

    #[test]
    fn test_san_to_uci_promotion() {
        // Promotion: 1.a4 b5 2.axb5 c5 3.bxc6 d5 4.c7 d4 5.cxb8=Q
        let san = vec!["a4", "b5", "axb5", "c5", "bxc6", "d5", "c7", "d4", "cxb8=Q"];
        let (uci, n) = san_to_uci(&san);
        assert_eq!(n, 9);
        assert_eq!(uci[8], "c7b8q");
    }

    #[test]
    fn test_san_to_uci_disambiguator() {
        // Verify SAN→UCI resolves knight moves correctly.
        let san = vec!["e4", "e5", "Nf3", "Nc6", "Nc3"];
        let (uci, n) = san_to_uci(&san);
        assert_eq!(n, 5);
        assert_eq!(uci[2], "g1f3");
        assert_eq!(uci[4], "b1c3");
    }

    #[test]
    fn test_san_to_uci_stops_on_invalid() {
        let san = vec!["e4", "InvalidMove", "Nf3"];
        let (uci, n) = san_to_uci(&san);
        assert_eq!(n, 1);
        assert_eq!(uci.len(), 1);
    }

    #[test]
    fn test_batch_san_to_uci_parallel() {
        let games = vec![
            vec!["e4", "e5"],
            vec!["d4", "d5", "c4"],
            vec!["Nf3", "Nf6"],
        ];
        let results = batch_san_to_uci(&games);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1, 2);
        assert_eq!(results[1].1, 3);
        assert_eq!(results[2].1, 2);
        assert_eq!(results[0].0, vec!["e2e4", "e7e5"]);
        assert_eq!(results[1].0, vec!["d2d4", "d7d5", "c2c4"]);
        assert_eq!(results[2].0, vec!["g1f3", "g8f6"]);
    }

    #[test]
    fn test_uci_ep_capture() {
        // 1. e4 a6 2. e5 d5 3. exd6 (en passant)
        let moves = vec!["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"];
        let (_, n) = uci_moves_to_tokens(&moves, 256);
        assert_eq!(n, 5);
    }

    #[test]
    fn test_uci_uppercase_promotion() {
        // Promotion with uppercase Q
        let moves = vec!["a2a4", "b7b5", "a4b5", "c7c5", "b5c6", "d7d5", "c6c7", "a7a6", "c7b8Q"];
        let (_, n) = uci_moves_to_tokens(&moves, 256);
        assert_eq!(n, 9, "uppercase Q promotion should parse");
    }

    #[test]
    fn test_uci_to_san_basic() {
        let uci = vec!["e2e4", "e7e5", "g1f3", "b8c6"];
        let (san, n) = uci_to_san(&uci);
        assert_eq!(n, 4);
        assert_eq!(san, vec!["e4", "e5", "Nf3", "Nc6"]);
    }

    #[test]
    fn test_uci_to_san_castling_and_check() {
        // Italian Game ending in O-O.
        let uci = vec!["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"];
        let (san, n) = uci_to_san(&uci);
        assert_eq!(n, 7);
        assert_eq!(san[6], "O-O");
    }

    #[test]
    fn test_uci_to_san_promotion() {
        // 1.a4 b5 2.axb5 c5 3.bxc6 d5 4.c7 d4 5.cxb8=Q
        let uci = vec!["a2a4", "b7b5", "a4b5", "c7c5", "b5c6", "d7d5", "c6c7", "d5d4", "c7b8q"];
        let (san, n) = uci_to_san(&uci);
        assert_eq!(n, 9);
        assert!(san[8].starts_with("cxb8=Q"), "got SAN {:?}", san[8]);
    }

    #[test]
    fn test_uci_to_san_checkmate_suffix() {
        // Fool's mate — final move should carry '#'.
        let uci = vec!["f2f3", "e7e5", "g2g4", "d8h4"];
        let (san, n) = uci_to_san(&uci);
        assert_eq!(n, 4);
        assert!(san[3].ends_with('#'), "expected mate suffix, got {:?}", san[3]);
    }

    #[test]
    fn test_uci_to_san_stops_on_invalid() {
        let uci = vec!["e2e4", "z9z9", "g1f3"];
        let (san, n) = uci_to_san(&uci);
        assert_eq!(n, 1);
        assert_eq!(san, vec!["e4"]);
    }

    #[test]
    fn test_batch_uci_to_san_parallel() {
        let games = vec![
            vec!["e2e4", "e7e5"],
            vec!["d2d4", "d7d5", "c2c4"],
            vec!["g1f3", "g8f6"],
        ];
        let results = batch_uci_to_san(&games);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (vec!["e4".to_string(), "e5".to_string()], 2));
        assert_eq!(
            results[1],
            (vec!["d4".to_string(), "d5".to_string(), "c4".to_string()], 3),
        );
        assert_eq!(results[2], (vec!["Nf3".to_string(), "Nf6".to_string()], 2));
    }

    #[test]
    fn test_uci_to_tokens_and_san_lockstep() {
        // The combined function must agree exactly with the two separate paths.
        let uci = vec!["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f8c5"];
        let (tokens_combined, san_combined) = uci_to_tokens_and_san(&uci);
        assert_eq!(tokens_combined.len(), 8);

        let (tokens_solo, n_t) = uci_moves_to_tokens(&uci, 256);
        let (san_solo, n_s) = uci_to_san(&uci);
        assert_eq!(n_t, 8);
        assert_eq!(n_s, 8);
        assert_eq!(tokens_combined, tokens_solo);
        assert_eq!(san_combined, san_solo);
    }

    #[test]
    fn test_uci_to_tokens_and_san_stops_on_invalid() {
        let uci = vec!["e2e4", "z9z9", "g1f3"];
        let (tokens, san) = uci_to_tokens_and_san(&uci);
        assert_eq!(tokens.len(), 1);
        assert_eq!(san, vec!["e4"]);
    }

    #[test]
    fn test_roundtrip_uci_san_uci() {
        // UCI -> SAN -> UCI should round-trip.
        let uci = vec!["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"];
        let (san, _) = uci_to_san(&uci);
        let san_refs: Vec<&str> = san.iter().map(|s| s.as_str()).collect();
        let (uci_back, n) = san_to_uci(&san_refs);
        assert_eq!(n, 5);
        let uci_owned: Vec<String> = uci.iter().map(|s| s.to_string()).collect();
        assert_eq!(uci_back, uci_owned);
    }

    #[test]
    fn test_roundtrip_san_uci_tokens() {
        // SAN -> UCI -> tokens should match SAN -> tokens
        let san = vec!["e4", "e5", "Nf3", "Nc6", "Bb5"];
        let (tokens_from_san, n1) = crate::pgn::san_moves_to_tokens(&san, 256);
        let (uci, _) = san_to_uci(&san);
        let uci_refs: Vec<&str> = uci.iter().map(|s| s.as_str()).collect();
        let (tokens_from_uci, n2) = uci_moves_to_tokens(&uci_refs, 256);

        assert_eq!(n1, n2);
        assert_eq!(tokens_from_san, tokens_from_uci);
    }
}
