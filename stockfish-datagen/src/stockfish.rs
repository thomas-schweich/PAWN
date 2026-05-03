//! Thin synchronous wrapper around a Stockfish UCI subprocess.
//!
//! One `StockfishProcess` per worker thread. The wrapper owns the child
//! process + buffered stdin/stdout pipes; it knows how to parse the
//! handful of UCI responses we actually care about (`uciok`, `readyok`,
//! `info ... multipv ...`, `bestmove`).
//!
//! Versioning: the constructor extracts the `id name <X>` line during the
//! UCI handshake and refuses to run if `<X>` doesn't match the expected
//! version. This is part of the per-game reproducibility guarantee — a
//! different Stockfish ships a different NNUE and would silently produce
//! different games from the same `(game_seed, config)` pair.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum StockfishError {
    #[error("spawning stockfish at {path}: {source}")]
    Spawn { path: String, source: std::io::Error },
    #[error("stockfish I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("stockfish died unexpectedly (no more output)")]
    UnexpectedEof,
    #[error(
        "stockfish version mismatch: expected {expected:?}, got {actual:?}. \
         Pin a different version in the config or install the matching binary."
    )]
    VersionMismatch { expected: String, actual: String },
    #[error("stockfish returned bestmove (none) but the position has legal moves; \
             this likely indicates a bug in our pre-move terminal check.")]
    UnexpectedNoneBestmove,
    #[error("stockfish protocol violation: {0}")]
    Protocol(String),
}

/// One candidate move surfaced by `go nodes N` with `MultiPV >= 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub uci: String,
    /// Centipawns from side-to-move's perspective. Mate scores are folded
    /// to ±30000 so downstream softmax never sees +inf.
    pub score_cp: f32,
}

/// Result of a `go nodes N` call.
///
/// `terminal` is set when Stockfish reports `bestmove (none)` — by then the
/// position has no legal moves (mate or stalemate). With our in-process
/// pre-move terminal check we shouldn't normally see this, but we surface
/// it explicitly so callers can decide whether to treat it as a bug or a
/// fallback.
#[derive(Debug, Clone)]
pub struct CandidatesResult {
    pub candidates: Vec<Candidate>,
    pub terminal: Option<TerminalKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    Checkmate,
    Stalemate,
}

pub struct StockfishProcess {
    child: Child,
    /// Buffered to coalesce the multiple small writes per `send` into one
    /// kernel write per command. `ChildStdin` is unbuffered by default.
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    line_buf: String,
    /// Reused per-call scratch for `position startpos moves ...`. Maintained
    /// incrementally — appended to per move, reset on `new_game()` — so we
    /// don't allocate or recopy O(ply²) text per game.
    position_cmd: String,
    /// Pre-rendered `go nodes N` (constant for the lifetime of a process).
    go_cmd: String,
    /// Banner line value (e.g. `Stockfish 18 by the Stockfish developers`)
    /// captured from the `id name <X>` line. Stored for diagnostics; the
    /// version check itself happens in the constructor.
    pub id_name: String,
}

impl Drop for StockfishProcess {
    /// Safety net: ensure the Stockfish subprocess is reaped on any error
    /// path. `Child` does NOT kill on drop in std (Rust stdlib guarantee),
    /// so without this, every `?`-returned error in a worker would leak a
    /// Stockfish process.
    fn drop(&mut self) {
        // Best effort: try a graceful quit, then kill if still alive.
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

impl StockfishProcess {
    /// Spawn Stockfish, complete the UCI handshake, set the standard options,
    /// and verify the version against `expected_version`.
    ///
    /// The expected string is matched at a word boundary against the
    /// child's `id name` line: equal, or `"<expected> "` is a prefix of
    /// `id_name`. This ensures `"Stockfish 1"` does NOT match `"Stockfish 18"`,
    /// which a naive `starts_with` would.
    ///
    /// `nodes` here is the per-process budget — every `candidates()` call
    /// uses it as the `go nodes N` value, so it's pre-rendered into
    /// `self.go_cmd` to avoid per-ply formatting.
    pub fn spawn(
        path: &Path,
        expected_version: &str,
        hash_mb: u32,
        nodes: u32,
    ) -> Result<Self, StockfishError> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| StockfishError::Spawn {
                path: path.display().to_string(),
                source: e,
            })?;

        let stdin = BufWriter::new(child.stdin.take().expect("piped"));
        let stdout = BufReader::new(child.stdout.take().expect("piped"));

        let mut sf = Self {
            child,
            stdin,
            stdout,
            line_buf: String::with_capacity(512),
            position_cmd: String::with_capacity(64 + 5 * 256),
            go_cmd: format!("go nodes {nodes}"),
            id_name: String::new(),
        };

        sf.send("uci")?;
        sf.complete_uci_handshake(expected_version)?;
        sf.send(&format!("setoption name Hash value {hash_mb}"))?;
        sf.send("setoption name Threads value 1")?;
        sf.send("isready")?;
        sf.wait_for_token("readyok")?;

        Ok(sf)
    }

    /// Set MultiPV. No-op if Stockfish is already at this value (caller's
    /// responsibility to track — sending duplicate `setoption` to Stockfish
    /// is allowed but slow because we must wait for `readyok`).
    pub fn set_multi_pv(&mut self, n: u32) -> Result<(), StockfishError> {
        self.send(&format!("setoption name MultiPV value {n}"))?;
        self.send("isready")?;
        self.wait_for_token("readyok")
    }

    /// Tell Stockfish a new game is starting; resets internal heuristics
    /// and our incremental `position_cmd` buffer.
    pub fn new_game(&mut self) -> Result<(), StockfishError> {
        self.position_cmd.clear();
        self.position_cmd.push_str("position startpos");
        self.send("ucinewgame")?;
        self.send("isready")?;
        self.wait_for_token("readyok")
    }

    /// Run `go nodes N` from the position reached by playing `moves` from
    /// the start position. Returns up to MultiPV candidates with their
    /// centipawn scores.
    ///
    /// `moves` must be the FULL move list from the start; this method
    /// rebuilds its incremental buffer to match. If you'd prefer the
    /// fast incremental path (one move at a time), call `play_move`
    /// after each ply and use [`Self::candidates_after_play_moves`].
    pub fn candidates(
        &mut self,
        moves: &[String],
    ) -> Result<CandidatesResult, StockfishError> {
        // Rebuild the position cmd from scratch. With incremental usage
        // (callers using play_move), this branch is rarely the hot path.
        self.position_cmd.clear();
        self.position_cmd.push_str("position startpos");
        if !moves.is_empty() {
            self.position_cmd.push_str(" moves");
            for m in moves {
                self.position_cmd.push(' ');
                self.position_cmd.push_str(m);
            }
        }
        self.candidates_after_play_moves()
    }

    /// Append a single UCI move to the cached position so the next
    /// [`Self::candidates_after_play_moves`] call doesn't have to rebuild
    /// the move list from scratch.
    pub fn play_move(&mut self, uci: &str) {
        // First move after `new_game()` needs the " moves" preamble.
        if !self.position_cmd.contains(" moves") {
            self.position_cmd.push_str(" moves");
        }
        self.position_cmd.push(' ');
        self.position_cmd.push_str(uci);
    }

    /// Issue `go nodes N` against the cached position (built incrementally
    /// via [`Self::play_move`] or rebuilt by [`Self::candidates`]).
    pub fn candidates_after_play_moves(
        &mut self,
    ) -> Result<CandidatesResult, StockfishError> {
        // Write position + go through the BufWriter, then a single flush.
        // Splitting the writes lets the BufWriter coalesce the two short
        // commands into one syscall pair instead of three.
        self.stdin.write_all(self.position_cmd.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.write_all(self.go_cmd.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        // Map keyed by multipv index. Later (deeper) info lines for the
        // same index overwrite earlier ones.
        let mut by_pv: std::collections::BTreeMap<u32, Candidate> = std::collections::BTreeMap::new();
        let mut last_score_was_mate0 = false;

        loop {
            self.line_buf.clear();
            let n = self.stdout.read_line(&mut self.line_buf)?;
            if n == 0 {
                return Err(StockfishError::UnexpectedEof);
            }
            let line = self.line_buf.trim_end();

            if line.starts_with("bestmove") {
                let mut parts = line.split_whitespace();
                let _ = parts.next(); // "bestmove"
                let mv = parts.next().unwrap_or("");
                if mv == "(none)" {
                    // No legal moves. The earlier expression had a dead
                    // disjunct (`A || (B && A)` collapses to `A`). Real
                    // intent: any positive mate-score signal in the info
                    // lines (or no info lines at all, which is what
                    // Stockfish emits when there are no moves to score)
                    // means checkmate; otherwise stalemate.
                    let terminal = if last_score_was_mate0 || by_pv.is_empty() {
                        Some(TerminalKind::Checkmate)
                    } else {
                        Some(TerminalKind::Stalemate)
                    };
                    return Ok(CandidatesResult {
                        candidates: by_pv.into_values().collect(),
                        terminal,
                    });
                }
                if by_pv.is_empty() {
                    // Fallback: no multipv info lines but we have a bestmove.
                    // Score 0 since we don't actually know it.
                    return Ok(CandidatesResult {
                        candidates: vec![Candidate {
                            uci: mv.to_string(),
                            score_cp: 0.0,
                        }],
                        terminal: None,
                    });
                }
                return Ok(CandidatesResult {
                    candidates: by_pv.into_values().collect(),
                    terminal: None,
                });
            }

            if let Some(parsed) = parse_info_multipv(line) {
                if parsed.score_was_mate_zero {
                    last_score_was_mate0 = true;
                }
                by_pv.insert(
                    parsed.pv_idx,
                    Candidate {
                        uci: parsed.first_move,
                        score_cp: parsed.score_cp,
                    },
                );
            }
        }
    }

    /// Send `quit` and wait briefly. Best-effort; killed forcibly on timeout.
    pub fn shutdown(mut self) {
        let _ = self.send("quit");
        match self.child.try_wait() {
            Ok(Some(_)) => {}
            _ => {
                std::thread::sleep(std::time::Duration::from_millis(200));
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    fn send(&mut self, cmd: &str) -> Result<(), StockfishError> {
        self.stdin.write_all(cmd.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn wait_for_token(&mut self, tok: &str) -> Result<(), StockfishError> {
        loop {
            self.line_buf.clear();
            let n = self.stdout.read_line(&mut self.line_buf)?;
            if n == 0 {
                return Err(StockfishError::UnexpectedEof);
            }
            if self.line_buf.trim_end().starts_with(tok) {
                return Ok(());
            }
        }
    }

    fn complete_uci_handshake(&mut self, expected: &str) -> Result<(), StockfishError> {
        loop {
            self.line_buf.clear();
            let n = self.stdout.read_line(&mut self.line_buf)?;
            if n == 0 {
                return Err(StockfishError::UnexpectedEof);
            }
            let line = self.line_buf.trim_end();
            if let Some(rest) = line.strip_prefix("id name ") {
                self.id_name = rest.to_string();
            }
            if line.starts_with("uciok") {
                break;
            }
        }
        if self.id_name.is_empty() {
            return Err(StockfishError::Protocol(
                "no `id name` line in UCI handshake".into(),
            ));
        }
        // Word-boundary match. Naive `starts_with(expected)` would
        // accept e.g. `"Stockfish 1"` as a prefix of `"Stockfish 18"`.
        // Require equality OR `<expected> ` (so the next char must be a
        // space, signalling the version token has ended).
        let matches = self.id_name == expected
            || self.id_name.starts_with(&format!("{expected} "));
        if !matches {
            return Err(StockfishError::VersionMismatch {
                expected: expected.to_string(),
                actual: self.id_name.clone(),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ParsedInfo {
    pv_idx: u32,
    first_move: String,
    score_cp: f32,
    score_was_mate_zero: bool,
}

/// Parse one `info ... multipv ...` line, returning the multipv index, the
/// first move of the principal variation, and the score in centipawns
/// (mate scores folded to ±30000). Returns `None` if the line doesn't carry
/// the three fields we need.
///
/// Streams over the line — no `Vec<&str>` allocation. With Stockfish at
/// higher node counts emitting hundreds of `info` lines per `go`, this
/// matters cumulatively.
fn parse_info_multipv(line: &str) -> Option<ParsedInfo> {
    if !line.starts_with("info ") || !line.contains(" multipv ") {
        return None;
    }
    let mut it = line.split_whitespace();

    let mut pv_idx: Option<u32> = None;
    let mut score: Option<f32> = None;
    let mut score_was_mate_zero = false;
    let mut first_move: Option<String> = None;

    while let Some(tok) = it.next() {
        match tok {
            "multipv" => {
                pv_idx = it.next().and_then(|s| s.parse().ok());
            }
            "score" => {
                let kind = it.next();
                let val = it.next();
                match (kind, val) {
                    (Some("cp"), Some(v)) => {
                        score = v.parse::<f32>().ok();
                    }
                    (Some("mate"), Some(v)) => {
                        if let Ok(m) = v.parse::<i32>() {
                            score = Some(if m > 0 { 30_000.0 } else { -30_000.0 });
                            if m == 0 {
                                score_was_mate_zero = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
            "pv" => {
                // First move of the PV — and we're done; later tokens
                // are the rest of the line we don't care about.
                first_move = it.next().map(|s| s.to_string());
                break;
            }
            _ => {}
        }
    }

    Some(ParsedInfo {
        pv_idx: pv_idx?,
        first_move: first_move?,
        score_cp: score?,
        score_was_mate_zero,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_info_extracts_cp_score() {
        let line = "info depth 1 seldepth 1 multipv 1 score cp 12 nodes 1 nps 1000 \
                    hashfull 0 tbhits 0 time 0 pv e2e4 e7e5";
        let p = parse_info_multipv(line).unwrap();
        assert_eq!(p.pv_idx, 1);
        assert_eq!(p.first_move, "e2e4");
        assert!((p.score_cp - 12.0).abs() < 1e-6);
        assert!(!p.score_was_mate_zero);
    }

    #[test]
    fn parse_info_extracts_mate_score() {
        let line = "info depth 1 multipv 2 score mate 1 nodes 5 pv h2h3";
        let p = parse_info_multipv(line).unwrap();
        assert_eq!(p.pv_idx, 2);
        assert_eq!(p.first_move, "h2h3");
        assert!((p.score_cp - 30_000.0).abs() < 1e-6);
    }

    #[test]
    fn parse_info_returns_none_without_multipv() {
        let line = "info depth 1 score cp 0 nodes 1 pv e2e4";
        assert!(parse_info_multipv(line).is_none());
    }

    #[test]
    fn parse_info_handles_trailing_pv_moves() {
        // Should pick the FIRST move after `pv`.
        let line = "info depth 5 multipv 1 score cp 25 pv e2e4 e7e5 g1f3 b8c6";
        let p = parse_info_multipv(line).unwrap();
        assert_eq!(p.first_move, "e2e4");
    }

    /// Live test against a real Stockfish binary. Skipped automatically if
    /// the binary isn't where we expect (so CI without Stockfish still
    /// passes); set `STOCKFISH_PATH=...` to point at an alternate binary.
    fn stockfish_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("STOCKFISH_PATH") {
            return Some(p.into());
        }
        let default = std::path::PathBuf::from(
            std::env::var("HOME").unwrap_or_default(),
        )
        .join("bin/stockfish");
        if default.exists() {
            Some(default)
        } else {
            None
        }
    }

    #[test]
    fn live_handshake_and_starting_candidates() {
        let Some(path) = stockfish_path() else {
            eprintln!("skipping: no stockfish binary at $HOME/bin/stockfish");
            return;
        };
        let mut sf = StockfishProcess::spawn(&path, "Stockfish", 16, 1).unwrap();
        assert!(sf.id_name.starts_with("Stockfish"), "got id_name={:?}", sf.id_name);
        sf.set_multi_pv(5).unwrap();
        sf.new_game().unwrap();
        let res = sf.candidates(&[]).unwrap();
        assert!(res.terminal.is_none());
        assert!(!res.candidates.is_empty(), "expected at least one candidate from start position");
        // First-move candidates from start position should look like UCI
        // squares (4 chars, files a-h, ranks 1-8).
        for c in &res.candidates {
            assert_eq!(c.uci.len(), 4, "candidate {:?} not 4 chars", c.uci);
        }
        sf.shutdown();
    }

    #[test]
    fn live_version_mismatch_rejected() {
        let Some(path) = stockfish_path() else {
            eprintln!("skipping: no stockfish binary at $HOME/bin/stockfish");
            return;
        };
        match StockfishProcess::spawn(&path, "Stockfish 9999", 16, 1) {
            Err(StockfishError::VersionMismatch { .. }) => {}
            Err(e) => panic!("expected VersionMismatch, got {e:?}"),
            Ok(_) => panic!("expected VersionMismatch, got Ok"),
        }
    }
}
