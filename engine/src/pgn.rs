//! PGN game parsing: file → SAN moves → PAWN token sequences.
//!
//! Full pipeline in Rust: reads PGN files, extracts SAN move strings,
//! converts to PAWN tokens via shakmaty. Uses rayon for parallel
//! token conversion.

use std::fs;
use rayon::prelude::*;
use shakmaty::{Chess, Position};
use shakmaty::san::San;

use crate::board::move_to_token;

/// Convert a sequence of SAN move strings to PAWN token indices.
///
/// Returns (tokens, n_valid) where tokens has length up to max_ply,
/// and n_valid is how many moves were successfully parsed.
/// Stops at the first parse error or illegal move.
pub fn san_moves_to_tokens(
    san_moves: &[&str],
    max_ply: usize,
) -> (Vec<u16>, usize) {
    let mut pos = Chess::default();
    let mut tokens = Vec::with_capacity(san_moves.len().min(max_ply));

    for (i, san_str) in san_moves.iter().enumerate() {
        if i >= max_ply {
            break;
        }

        let san = match San::from_ascii(san_str.as_bytes()) {
            Ok(s) => s,
            Err(_) => break,
        };

        let m = match san.to_move(&pos) {
            Ok(m) => m,
            Err(_) => break,
        };

        let token = move_to_token(&m);
        tokens.push(token);
        pos.play_unchecked(m);
    }

    let n = tokens.len();
    (tokens, n)
}

/// Batch convert: multiple games, each as a list of SAN moves.
/// Returns a flat (n_games * max_ply) i16 array (0-padded) + lengths.
pub fn batch_san_to_tokens(
    games: &[Vec<&str>],
    max_ply: usize,
) -> (Vec<i16>, Vec<i16>) {
    let n = games.len();
    let mut flat = vec![0i16; n * max_ply];
    let mut lengths = Vec::with_capacity(n);

    for (gi, san_moves) in games.iter().enumerate() {
        let (tokens, n_valid) = san_moves_to_tokens(san_moves, max_ply);
        for (t, &tok) in tokens.iter().enumerate() {
            flat[gi * max_ply + t] = tok as i16;
        }
        lengths.push(n_valid as i16);
    }

    (flat, lengths)
}

/// Parse a PGN file and extract SAN move lists for each game.
///
/// Handles standard PGN: skips headers ([...]), strips move numbers,
/// comments ({...}), NAGs ($N), and result markers.
fn parse_pgn_to_san(content: &str, max_games: usize) -> Vec<Vec<String>> {
    let mut games = Vec::new();
    let mut movetext_lines: Vec<&str> = Vec::new();
    let mut in_movetext = false;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            if in_movetext && !movetext_lines.is_empty() {
                let text: String = movetext_lines.join(" ");
                if let Some(moves) = extract_san_moves(&text) {
                    if !moves.is_empty() {
                        games.push(moves);
                        if games.len() >= max_games {
                            break;
                        }
                    }
                }
                movetext_lines.clear();
                in_movetext = false;
            }
            continue;
        }

        if line.starts_with('[') {
            in_movetext = false;
            continue;
        }

        in_movetext = true;
        movetext_lines.push(line);
    }

    // Handle last game
    if !movetext_lines.is_empty() && games.len() < max_games {
        let text: String = movetext_lines.join(" ");
        if let Some(moves) = extract_san_moves(&text) {
            if !moves.is_empty() {
                games.push(moves);
            }
        }
    }

    games
}

/// Extract SAN moves from a PGN movetext string.
fn extract_san_moves(text: &str) -> Option<Vec<String>> {
    let mut moves = Vec::new();

    // First strip comments { ... } (can span multiple words)
    let mut cleaned = String::with_capacity(text.len());
    let mut in_comment = false;
    for ch in text.chars() {
        if ch == '{' { in_comment = true; continue; }
        if ch == '}' { in_comment = false; continue; }
        if !in_comment { cleaned.push(ch); }
    }

    for token in cleaned.split_whitespace() {
        // Skip NAGs: $1, $2, etc.
        if token.starts_with('$') {
            continue;
        }

        // Result markers — stop parsing
        if token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*" {
            break;
        }

        // Skip move numbers: "1.", "1...", "23."
        let stripped = token.trim_end_matches('.');
        if !stripped.is_empty() && stripped.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }

        moves.push(token.to_string());
    }

    Some(moves)
}

/// Full pipeline: read PGN file → parse → convert to tokens (parallel).
///
/// Returns (flat_tokens: Vec<i16> of shape n_games*max_ply, lengths: Vec<i16>).
pub fn pgn_file_to_tokens(
    path: &str,
    max_ply: usize,
    max_games: usize,
    min_ply: usize,
) -> (Vec<i16>, Vec<i16>, usize) {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read PGN file {}: {}", path, e));

    let san_games = parse_pgn_to_san(&content, max_games);
    let n_parsed = san_games.len();

    // Parallel token conversion with rayon
    let converted: Vec<(Vec<u16>, usize)> = san_games
        .par_iter()
        .map(|moves| {
            let refs: Vec<&str> = moves.iter().map(|s| s.as_str()).collect();
            san_moves_to_tokens(&refs, max_ply)
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_san_to_tokens() {
        let moves = vec!["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"];
        let (tokens, n) = san_moves_to_tokens(&moves, 256);
        assert_eq!(n, 7);
        assert_eq!(tokens.len(), 7);
        let e2e4 = crate::vocab::base_grid_token(12, 28);
        assert_eq!(tokens[0], e2e4);
    }

    #[test]
    fn test_san_to_tokens_max_ply() {
        let moves = vec!["e4", "e5", "Nf3", "Nc6"];
        let (tokens, n) = san_moves_to_tokens(&moves, 2);
        assert_eq!(n, 2);
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_extract_san_moves() {
        let text = "1. e4 e5 2. Nf3 Nc6 3. Bb5 {Spanish} a6 1-0";
        let moves = extract_san_moves(text).unwrap();
        assert_eq!(moves, vec!["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]);
    }

    #[test]
    fn test_extract_san_with_nags() {
        let text = "1. e4 $1 e5 2. Nf3 $2 Nc6 0-1";
        let moves = extract_san_moves(text).unwrap();
        assert_eq!(moves, vec!["e4", "e5", "Nf3", "Nc6"]);
    }

    #[test]
    fn test_parse_pgn_to_san() {
        let pgn = r#"[Event "Test"]
[White "Alice"]
[Black "Bob"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Test2"]

1. d4 d5 0-1
"#;
        let games = parse_pgn_to_san(pgn, 100);
        assert_eq!(games.len(), 2);
        assert_eq!(games[0], vec!["e4", "e5", "Nf3", "Nc6"]);
        assert_eq!(games[1], vec!["d4", "d5"]);
    }

    #[test]
    fn test_pgn_file_to_tokens_inline() {
        // Test the full pipeline with a temp file
        let dir = std::env::temp_dir();
        let path = dir.join("test_pgn.pgn");
        fs::write(&path, r#"[Event "Test"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Test2"]

1. d4 d5 0-1
"#).unwrap();

        let (flat, lengths, n_parsed) = pgn_file_to_tokens(
            path.to_str().unwrap(), 256, 100, 2
        );
        assert_eq!(n_parsed, 2);
        assert_eq!(lengths.len(), 2);
        assert_eq!(lengths[0], 4);
        assert_eq!(lengths[1], 2);
        assert_eq!(flat.len(), 2 * 256);

        fs::remove_file(path).ok();
    }
}
