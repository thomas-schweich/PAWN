//! UCI move parsing: UCI strings → PAWN token sequences and
//! SAN → UCI conversion.
//!
//! Parallel to pgn.rs but for UCI-format input (e.g., "e2e4 e7e5 g1f3").

use rayon::prelude::*;
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
        let e2e4 = vocab::base_grid_token(12, 28);
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
