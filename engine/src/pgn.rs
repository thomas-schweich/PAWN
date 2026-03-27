//! PGN game parsing: file → SAN moves → PAWN token sequences.
//!
//! Full pipeline in Rust: reads PGN files, extracts SAN move strings,
//! converts to PAWN tokens via shakmaty. Uses rayon for parallel
//! token conversion.
//!
//! Also provides enriched parsing that extracts clock annotations,
//! eval annotations, and PGN headers for dataset construction.

use std::collections::{HashMap, HashSet};
use std::fs;
use rayon::prelude::*;
use shakmaty::{Chess, Position};
use shakmaty::san::San;

use crate::board::move_to_token;

// ---------------------------------------------------------------------------
// Enriched PGN parsing — extracts moves, clocks, evals, and headers
// ---------------------------------------------------------------------------

/// A fully parsed game with move tokens, annotations, and metadata.
pub struct EnrichedGame {
    /// PAWN token indices for each ply (not padded).
    pub tokens: Vec<u16>,
    /// Seconds remaining on clock after each ply (0 = no annotation).
    pub clocks: Vec<u16>,
    /// Centipawns from white's perspective after each ply.
    /// Mate scores: ±(32767-N). No annotation: 0x8000 (-32768 as i16).
    pub evals: Vec<i16>,
    /// Number of valid plies.
    pub game_length: usize,
    /// PGN header fields (e.g., "White" -> "alice", "WhiteElo" -> "1873").
    pub headers: HashMap<String, String>,
}

/// Parse a PGN string into enriched games.
///
/// Extracts SAN moves (tokenized), `[%clk h:mm:ss]` annotations,
/// `[%eval ±N.NN]` / `[%eval #±N]` annotations, and all PGN headers.
/// Tokenization uses shakmaty and is parallelized with rayon.
pub fn parse_pgn_enriched(
    content: &str,
    max_ply: usize,
    max_games: usize,
    min_ply: usize,
) -> Vec<EnrichedGame> {
    let raw_games = parse_raw_games(content, max_games, None, None);

    // Phase 2: parallel tokenization + annotation extraction
    raw_games
        .into_par_iter()
        .filter_map(|raw| {
            let (san_moves, clocks_raw, evals_raw) = extract_moves_and_annotations(&raw.movetext);
            if san_moves.len() < min_ply {
                return None;
            }

            // Tokenize SAN moves via shakmaty
            let refs: Vec<&str> = san_moves.iter().map(|s| s.as_str()).collect();
            let (tokens, n_valid) = san_moves_to_tokens(&refs, max_ply);
            if n_valid < min_ply {
                return None;
            }

            // Trim annotations to match token count (moves may have failed to parse).
            let clocks = clocks_raw.into_iter().take(n_valid).collect();
            let evals = evals_raw.into_iter().take(n_valid).collect();

            Some(EnrichedGame {
                tokens,
                clocks,
                evals,
                game_length: n_valid,
                headers: raw.headers,
            })
        })
        .collect()
}

/// Count games in a PGN string whose UTCDate falls within [start, end].
///
/// Header-only scan — no movetext parsing, no tokenization.
/// Returns (count_in_range, offset) where offset is the running game index
/// that should be passed to the next chunk for correct global indexing.
pub fn count_games_in_date_range(
    content: &str,
    date_start: &str,
    date_end: &str,
) -> usize {
    let mut count = 0;
    let mut current_date: Option<String> = None;
    let mut in_movetext = false;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            if in_movetext {
                // End of game — check if the date was in range
                if let Some(ref d) = current_date {
                    if d.as_str() >= date_start && d.as_str() <= date_end {
                        count += 1;
                    }
                }
                current_date = None;
                in_movetext = false;
            }
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') {
            if let Some((key, value)) = parse_header_line(line) {
                if key == "UTCDate" {
                    current_date = Some(value);
                }
            }
            in_movetext = false;
        } else {
            in_movetext = true;
        }
    }

    // Handle last game
    if in_movetext {
        if let Some(ref d) = current_date {
            if d.as_str() >= date_start && d.as_str() <= date_end {
                count += 1;
            }
        }
    }

    count
}

/// Parse a PGN string, but only tokenize games at specific indices within a
/// date range. Used for uniform random sampling: Python counts games in the
/// date range (via `count_games_in_date_range`), generates a random index
/// set, then calls this to parse only those games.
///
/// `indices` are 0-based within the date-range-matching games of this chunk.
/// `game_offset` is the number of date-matching games seen in previous chunks,
/// so global index = game_offset + local_index.
pub fn parse_pgn_enriched_sampled(
    content: &str,
    max_ply: usize,
    min_ply: usize,
    date_start: &str,
    date_end: &str,
    indices: &HashSet<usize>,
    game_offset: usize,
) -> Vec<EnrichedGame> {
    let raw_games = parse_raw_games(content, usize::MAX, Some((date_start, date_end)), Some((indices, game_offset)));

    raw_games
        .into_par_iter()
        .filter_map(|raw| {
            let (san_moves, clocks_raw, evals_raw) = extract_moves_and_annotations(&raw.movetext);
            if san_moves.len() < min_ply {
                return None;
            }

            let refs: Vec<&str> = san_moves.iter().map(|s| s.as_str()).collect();
            let (tokens, n_valid) = san_moves_to_tokens(&refs, max_ply);
            if n_valid < min_ply {
                return None;
            }

            let clocks = clocks_raw.into_iter().take(n_valid).collect();
            let evals = evals_raw.into_iter().take(n_valid).collect();

            Some(EnrichedGame {
                tokens,
                clocks,
                evals,
                game_length: n_valid,
                headers: raw.headers,
            })
        })
        .collect()
}

/// Raw game data before tokenization.
struct RawGame {
    headers: HashMap<String, String>,
    movetext: String,
}

/// Single-threaded PGN line scanner. Extracts headers and raw movetext.
///
/// If `date_range` is Some((start, end)), only games whose UTCDate falls
/// within [start, end] are included. If `sample` is Some((indices, offset)),
/// only games whose (offset + local_index) is in the index set are kept.
fn parse_raw_games(
    content: &str,
    max_games: usize,
    date_range: Option<(&str, &str)>,
    sample: Option<(&HashSet<usize>, usize)>,
) -> Vec<RawGame> {
    let mut games = Vec::new();
    let mut headers: HashMap<String, String> = HashMap::new();
    let mut movetext_lines: Vec<&str> = Vec::new();
    let mut in_movetext = false;
    let mut date_excluded = false;   // UTCDate outside date_range
    let mut has_utc_date = false;    // saw a UTCDate header for this game
    let mut date_matched_idx = 0usize; // count of date-matching games seen

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            if in_movetext {
                // End of movetext — game boundary.
                // Exclude if date is out of range OR if date_range is active
                // but no UTCDate header was found (consistent with count_games_in_date_range).
                let excluded = date_excluded || (date_range.is_some() && !has_utc_date);
                if !excluded && !movetext_lines.is_empty() {
                    // Game passed date filter. Check sample if present.
                    let keep = match sample {
                        Some((indices, offset)) => indices.contains(&(offset + date_matched_idx)),
                        None => true,
                    };
                    date_matched_idx += 1;

                    if keep {
                        games.push(RawGame {
                            headers: std::mem::take(&mut headers),
                            movetext: movetext_lines.join(" "),
                        });
                        if games.len() >= max_games {
                            break;
                        }
                    }
                }
                movetext_lines.clear();
                headers.clear();
                in_movetext = false;
                date_excluded = false;
                has_utc_date = false;
            }
            // Blank line between headers and movetext: don't reset state
            continue;
        }

        // Header line: [Key "Value"]
        if line.starts_with('[') && line.ends_with(']') {
            if let Some((key, value)) = parse_header_line(line) {
                if key == "UTCDate" {
                    has_utc_date = true;
                    if let Some((start, end)) = date_range {
                        if value.as_str() < start || value.as_str() > end {
                            date_excluded = true;
                        }
                    }
                }
                if !date_excluded {
                    headers.insert(key, value);
                }
            }
            in_movetext = false;
            continue;
        }

        if !date_excluded {
            in_movetext = true;
            movetext_lines.push(line);
        }
    }

    // Handle last game
    let last_excluded = date_excluded || (date_range.is_some() && !has_utc_date);
    if in_movetext && !last_excluded && !movetext_lines.is_empty() && games.len() < max_games {
        let keep = match sample {
            Some((indices, offset)) => indices.contains(&(offset + date_matched_idx)),
            None => true,
        };
        if keep {
            games.push(RawGame {
                headers: std::mem::take(&mut headers),
                movetext: movetext_lines.join(" "),
            });
        }
    }

    games
}

/// Parse a PGN header line like `[White "alice"]` into ("White", "alice").
fn parse_header_line(line: &str) -> Option<(String, String)> {
    // Strip surrounding brackets
    let inner = line.strip_prefix('[')?.strip_suffix(']')?.trim();
    let space = inner.find(' ')?;
    let key = inner[..space].to_string();
    let value_part = inner[space..].trim();
    // Strip surrounding quotes
    let value = value_part
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(value_part)
        .to_string();
    Some((key, value))
}

/// Sentinel for "no clock annotation" (0x8000 as u16 = 32768).
const CLOCK_NONE: u16 = 0x8000;
/// Sentinel for "no eval annotation" (0x8000 as i16 = -32768).
const EVAL_NONE: i16 = -0x8000; // i16::MIN

/// Extract SAN moves, clock annotations, and eval annotations from movetext.
///
/// Returns (san_moves, clocks, evals) where clocks[i] is the clock after
/// move i (CLOCK_NONE if no annotation) and evals[i] is centipawns after
/// move i (EVAL_NONE if no annotation).
fn extract_moves_and_annotations(text: &str) -> (Vec<String>, Vec<u16>, Vec<i16>) {
    let mut moves = Vec::new();
    let mut clocks = Vec::new();
    let mut evals = Vec::new();

    let bytes = text.as_bytes();
    let len = bytes.len();

    // Lichess format: move { comment } move { comment } ...
    // The comment annotates the move immediately before it.
    let mut i = 0;
    while i < len {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Comment: { ... } — applies to the last pushed move
        if bytes[i] == b'{' {
            i += 1;
            let start = i;
            while i < len && bytes[i] != b'}' {
                i += 1;
            }
            let comment = &text[start..i];
            if i < len { i += 1; }

            // Apply to last move
            if let Some(last_clk) = clocks.last_mut() {
                let mut clk = CLOCK_NONE;
                let mut ev = EVAL_NONE;
                parse_comment(comment, &mut clk, &mut ev);
                if clk != CLOCK_NONE { *last_clk = clk; }
                if ev != EVAL_NONE {
                    if let Some(last_ev) = evals.last_mut() {
                        *last_ev = ev;
                    }
                }
            }
            continue;
        }

        let start = i;
        while i < len && !bytes[i].is_ascii_whitespace() && bytes[i] != b'{' {
            i += 1;
        }
        let token = &text[start..i];

        if token.starts_with('$') { continue; }
        if token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*" { break; }

        let stripped = token.trim_end_matches('.');
        if !stripped.is_empty() && stripped.bytes().all(|b| b.is_ascii_digit()) { continue; }

        moves.push(token.to_string());
        clocks.push(CLOCK_NONE);
        evals.push(EVAL_NONE);
    }

    (moves, clocks, evals)
}

/// Parse a PGN comment body for clock and eval annotations.
///
/// Lichess format: `[%clk 0:03:00]` and `[%eval 1.23]` or `[%eval #-3]`.
fn parse_comment(comment: &str, clock: &mut u16, eval: &mut i16) {
    // Clock: [%clk H:MM:SS]
    if let Some(pos) = comment.find("[%clk ") {
        let rest = &comment[pos + 6..];
        if let Some(end) = rest.find(']') {
            let clk_str = rest[..end].trim();
            if let Some(secs) = parse_clock(clk_str) {
                *clock = secs;
            }
        }
    }

    // Eval: [%eval 1.23] or [%eval #-3]
    if let Some(pos) = comment.find("[%eval ") {
        let rest = &comment[pos + 7..];
        if let Some(end) = rest.find(']') {
            let eval_str = rest[..end].trim();
            if let Some(cp) = parse_eval(eval_str) {
                *eval = cp;
            }
        }
    }
}

/// Parse "H:MM:SS" into total seconds as u16.
fn parse_clock(s: &str) -> Option<u16> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 { return None; }
    let h: u32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    let s: u32 = parts[2].parse().ok()?;
    let total = h * 3600 + m * 60 + s;
    // Cap at 0x7FFF (32767) to avoid collision with CLOCK_NONE (0x8000)
    Some(total.min(0x7FFF) as u16)
}

/// Parse eval string into centipawns (i16).
/// "1.23" → 123, "-0.50" → -50.
/// Mate scores: "#N" → 32767-N, "#-N" → -(32767-N).
/// Bit 14 is always set for mates, making them detectable via bitmask.
/// Centipawn values are clamped to ±16383 to avoid overlap with the mate range.
fn parse_eval(s: &str) -> Option<i16> {
    if s.starts_with('#') {
        let rest = &s[1..];
        let n: i32 = rest.parse().ok()?;
        let abs_n = n.unsigned_abs().max(1) as i16;
        let mate_val = 32767 - abs_n;
        Some(if n > 0 { mate_val } else { -mate_val })
    } else {
        let f: f64 = s.parse().ok()?;
        let cp = (f * 100.0).round() as i32;
        Some(cp.clamp(-16383, 16383) as i16)
    }
}

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

    // --- Enriched parsing tests ---

    #[test]
    fn test_parse_clock() {
        assert_eq!(parse_clock("0:10:00"), Some(600));
        assert_eq!(parse_clock("1:30:00"), Some(5400));
        assert_eq!(parse_clock("0:00:05"), Some(5));
        assert_eq!(parse_clock("0:03:00"), Some(180));
        assert_eq!(parse_clock("bad"), None);
    }

    #[test]
    fn test_parse_eval() {
        assert_eq!(parse_eval("0.23"), Some(23));
        assert_eq!(parse_eval("-1.50"), Some(-150));
        assert_eq!(parse_eval("0.00"), Some(0));
        // Mate scores: 32767 - N
        assert_eq!(parse_eval("#1"), Some(32766));
        assert_eq!(parse_eval("#-1"), Some(-32766));
        assert_eq!(parse_eval("#3"), Some(32764));
        assert_eq!(parse_eval("#-3"), Some(-32764));
        assert_eq!(parse_eval("#10"), Some(32757));
        // Bit 14 (0x4000 = 16384) is set for all mate values
        assert!(parse_eval("#1").unwrap() & 0x4000 != 0);
        assert!(parse_eval("#100").unwrap() & 0x4000 != 0);
        // Centipawns clamped to ±16383 to avoid mate range
        assert_eq!(parse_eval("200.00"), Some(16383));
        assert_eq!(parse_eval("-200.00"), Some(-16383));
    }

    #[test]
    fn test_parse_header_line() {
        assert_eq!(
            parse_header_line(r#"[White "alice"]"#),
            Some(("White".to_string(), "alice".to_string()))
        );
        assert_eq!(
            parse_header_line(r#"[WhiteElo "1873"]"#),
            Some(("WhiteElo".to_string(), "1873".to_string()))
        );
        assert_eq!(
            parse_header_line(r#"[Opening "Bird Opening: Dutch Variation"]"#),
            Some(("Opening".to_string(), "Bird Opening: Dutch Variation".to_string()))
        );
    }

    #[test]
    fn test_extract_moves_and_annotations() {
        let text = r#"1. e4 { [%clk 0:10:00] [%eval 0.23] } 1... e5 { [%clk 0:09:58] [%eval 0.31] } 2. Nf3 { [%clk 0:09:55] } 1-0"#;
        let (moves, clocks, evals) = extract_moves_and_annotations(text);
        assert_eq!(moves, vec!["e4", "e5", "Nf3"]);
        assert_eq!(clocks, vec![600, 598, 595]);
        assert_eq!(evals, vec![23, 31, EVAL_NONE]);
    }

    #[test]
    fn test_extract_moves_no_annotations() {
        let text = "1. e4 e5 2. Nf3 Nc6 1-0";
        let (moves, clocks, evals) = extract_moves_and_annotations(text);
        assert_eq!(moves, vec!["e4", "e5", "Nf3", "Nc6"]);
        assert_eq!(clocks, vec![CLOCK_NONE, CLOCK_NONE, CLOCK_NONE, CLOCK_NONE]);
        assert_eq!(evals, vec![EVAL_NONE; 4]);
    }

    #[test]
    fn test_extract_moves_mate_eval() {
        let text = r#"1. e4 { [%eval 0.23] } 1... e5 { [%eval #-3] } 1-0"#;
        let (moves, _clocks, evals) = extract_moves_and_annotations(text);
        assert_eq!(moves, vec!["e4", "e5"]);
        assert_eq!(evals, vec![23, -32764]);
    }

    #[test]
    fn test_enriched_full_game() {
        let pgn = r#"[Event "Rated Rapid game"]
[Site "https://lichess.org/abc123"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1873"]
[BlackElo "1844"]
[WhiteRatingDiff "+6"]
[BlackRatingDiff "-26"]
[ECO "C20"]
[Opening "King's Pawn Game"]
[TimeControl "600+0"]
[Termination "Normal"]
[UTCDate "2025.01.15"]
[UTCTime "12:30:00"]

1. e4 { [%clk 0:10:00] [%eval 0.23] } 1... e5 { [%clk 0:09:58] [%eval 0.31] } 2. Nf3 { [%clk 0:09:50] [%eval 0.25] } 2... Nc6 { [%clk 0:09:45] [%eval 0.30] } 1-0
"#;
        let games = parse_pgn_enriched(pgn, 256, 100, 2);
        assert_eq!(games.len(), 1);
        let g = &games[0];
        assert_eq!(g.game_length, 4);
        assert_eq!(g.clocks, vec![600, 598, 590, 585]);
        assert_eq!(g.evals, vec![23, 31, 25, 30]);
        assert_eq!(g.headers.get("White").unwrap(), "alice");
        assert_eq!(g.headers.get("WhiteElo").unwrap(), "1873");
        assert_eq!(g.headers.get("Site").unwrap(), "https://lichess.org/abc123");
        assert_eq!(g.headers.get("ECO").unwrap(), "C20");
        assert_eq!(g.headers.get("TimeControl").unwrap(), "600+0");
    }

    #[test]
    fn test_enriched_tokens_match_legacy() {
        // Enriched parsing should produce the same tokens as the legacy pipeline
        let pgn = r#"[Event "Test"]

1. e4 { [%clk 0:10:00] } 1... e5 { [%clk 0:09:58] } 2. Nf3 { [%clk 0:09:50] } 2... Nc6 { [%clk 0:09:45] } 1-0
"#;
        let enriched = parse_pgn_enriched(pgn, 256, 100, 2);
        let legacy = parse_pgn_to_san(pgn, 100);

        assert_eq!(enriched.len(), 1);
        assert_eq!(legacy.len(), 1);

        // Convert legacy SAN to tokens for comparison
        let refs: Vec<&str> = legacy[0].iter().map(|s| s.as_str()).collect();
        let (legacy_tokens, legacy_n) = san_moves_to_tokens(&refs, 256);

        assert_eq!(enriched[0].tokens, legacy_tokens);
        assert_eq!(enriched[0].game_length, legacy_n);
    }

    #[test]
    fn test_count_games_in_date_range() {
        let pgn = r#"[Event "Game 1"]
[UTCDate "2023.12.05"]

1. e4 e5 1-0

[Event "Game 2"]
[UTCDate "2023.12.20"]

1. d4 d5 0-1

[Event "Game 3"]
[UTCDate "2025.01.15"]

1. e4 c5 1-0
"#;
        assert_eq!(count_games_in_date_range(pgn, "2023.12.01", "2023.12.31"), 2);
        assert_eq!(count_games_in_date_range(pgn, "2023.12.01", "2023.12.14"), 1);
        assert_eq!(count_games_in_date_range(pgn, "2023.12.15", "2023.12.31"), 1);
        assert_eq!(count_games_in_date_range(pgn, "2025.01.01", "2025.01.31"), 1);
        assert_eq!(count_games_in_date_range(pgn, "2024.01.01", "2024.12.31"), 0);
    }

    #[test]
    fn test_sampled_parsing() {
        let pgn = r#"[Event "Game 1"]
[UTCDate "2023.12.05"]

1. e4 e5 1-0

[Event "Game 2"]
[UTCDate "2023.12.10"]

1. d4 d5 0-1

[Event "Game 3"]
[UTCDate "2023.12.20"]

1. e4 c5 1-0

[Event "Game 4"]
[UTCDate "2025.01.15"]

1. Nf3 d5 1-0
"#;
        // 3 games match Dec 2023 (indices 0, 1, 2 within the date range)
        assert_eq!(count_games_in_date_range(pgn, "2023.12.01", "2023.12.31"), 3);

        // Sample only index 1 (Game 2)
        let indices: HashSet<usize> = HashSet::from([1]);
        let sampled = parse_pgn_enriched_sampled(
            pgn, 256, 2, "2023.12.01", "2023.12.31", &indices, 0,
        );
        assert_eq!(sampled.len(), 1);
        assert_eq!(sampled[0].headers.get("UTCDate").unwrap(), "2023.12.10");

        // Sample indices 0 and 2 (Game 1 and Game 3)
        let indices: HashSet<usize> = HashSet::from([0, 2]);
        let sampled = parse_pgn_enriched_sampled(
            pgn, 256, 2, "2023.12.01", "2023.12.31", &indices, 0,
        );
        assert_eq!(sampled.len(), 2);
        assert_eq!(sampled[0].headers.get("UTCDate").unwrap(), "2023.12.05");
        assert_eq!(sampled[1].headers.get("UTCDate").unwrap(), "2023.12.20");

        // Sample with offset: simulating a second chunk where previous chunk had 1 match.
        // Global index 2 = offset 1 + local index 1 => selects Game 2 (local idx 1).
        let indices: HashSet<usize> = HashSet::from([2]);
        let sampled = parse_pgn_enriched_sampled(
            pgn, 256, 2, "2023.12.01", "2023.12.31", &indices, 1,
        );
        assert_eq!(sampled.len(), 1);
        assert_eq!(sampled[0].headers.get("UTCDate").unwrap(), "2023.12.10");

        // Offset that skips all local games: offset=3 means local indices are 3,4,5
        // but we only ask for global index 0, which isn't in this chunk.
        let indices: HashSet<usize> = HashSet::from([0]);
        let sampled = parse_pgn_enriched_sampled(
            pgn, 256, 2, "2023.12.01", "2023.12.31", &indices, 3,
        );
        assert_eq!(sampled.len(), 0);
    }
}
