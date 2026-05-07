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

/// One candidate move surfaced by either `go nodes N` (multipv parsing) or
/// `evallegal` (per-legal-move NNUE).
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub uci: String,
    /// Normalized centipawns, mover-POV (`UCIEngine::to_cp(v, pos)` —
    /// 100 cp ≈ "1 pawn equivalent"). Mate scores are folded to ±30000 so
    /// downstream softmax never sees +inf. Available from both protocols.
    pub score_cp: f32,
    /// Raw internal NNUE Value, mover-POV. Only the `evallegal` protocol
    /// surfaces this directly; `go nodes N` (multipv) reports cp only, so
    /// this is `None` for that path. Distillation targets should use this
    /// when present (it's the network's actual output, before win-rate
    /// normalization shrinks magnitudes by `a / 100` ≈ 2–3.5×).
    pub score_v: Option<f32>,
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

/// What command to issue per ply. Pre-rendered into `StockfishProcess::play_cmd`
/// at spawn time — picking between these is a tier-level choice, not per-ply.
///
/// `EvalLegal` requires a Stockfish binary built from
/// `scripts/build_patched_stockfish.sh` (the patched binary adds the
/// `evallegal` UCI command). On a vanilla Stockfish the command yields
/// `Unknown command: 'evallegal'` and the worker fails fast — the spawn-time
/// probe in [`StockfishProcess::spawn`] catches the mismatch before it could
/// silently corrupt a tier.
#[derive(Debug, Clone, Copy)]
pub enum GoBudget {
    /// `go nodes N`. Standard search-tree budget; per-move scores are
    /// qsearch-resolved. Output: zero or more `info ... multipv ...` lines
    /// followed by `bestmove`.
    Nodes(u32),
    /// `evallegal`. Pure NNUE static eval per legal move, no search loop,
    /// every legal move scored. Output: a single
    /// `info string evallegal <status> [<uci> <cp>]...` line.
    EvalLegal,
}

impl GoBudget {
    fn render(self) -> String {
        match self {
            GoBudget::Nodes(n) => format!("go nodes {n}"),
            GoBudget::EvalLegal => "evallegal".to_string(),
        }
    }
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
    /// Pre-rendered per-ply command (`go nodes N` or `evallegal`).
    play_cmd: String,
    /// Which budget produced `play_cmd`. Drives output-parser dispatch in
    /// [`Self::candidates_after_play_moves`]: `Nodes` reads multipv info
    /// lines until `bestmove`, `EvalLegal` reads exactly one
    /// `info string evallegal ...` line.
    budget: GoBudget,
    /// Banner line value (e.g. `Stockfish 18 by the Stockfish developers`)
    /// captured from the `id name <X>` line. Stored for diagnostics; the
    /// version check itself happens in the constructor.
    pub id_name: String,
    /// True iff the binary recognizes the `evallegal` UCI command — the marker
    /// for our patched binary built via `scripts/build_patched_stockfish.sh`.
    /// Set by the post-handshake probe in [`Self::spawn`]. Used by the runner
    /// to fail loudly when a tier requests `EvalLegal` against a vanilla
    /// Stockfish (which would emit `Unknown command: 'evallegal'` and have no
    /// way to score the position).
    pub is_patched: bool,
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
    /// `budget` is the per-process go-command shape — every `candidates()`
    /// call uses it as-is, so it's pre-rendered into `self.go_cmd` to avoid
    /// per-ply formatting.
    pub fn spawn(
        path: &Path,
        expected_version: &str,
        hash_mb: u32,
        budget: GoBudget,
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

        // evallegal output for a 218-legal-move position is ~3 KB — bump the
        // line buffer above the parquet 512-default to avoid the first ply
        // forcing a regrowth.
        let mut sf = Self {
            child,
            stdin,
            stdout,
            line_buf: String::with_capacity(4096),
            position_cmd: String::with_capacity(64 + 5 * 256),
            play_cmd: budget.render(),
            budget,
            id_name: String::new(),
            is_patched: false,
        };

        sf.send("uci")?;
        sf.complete_uci_handshake(expected_version)?;
        sf.send(&format!("setoption name Hash value {hash_mb}"))?;
        sf.send("setoption name Threads value 1")?;
        sf.send("isready")?;
        sf.wait_for_token("readyok")?;

        // Probe for the patched binary by sending `evallegal` against the
        // start position and tagging `is_patched` based on whether the
        // response shape matches our protocol. `isready`/`readyok` is the
        // synchronization barrier — vanilla SF emits `Unknown command:
        // 'evallegal'` (one stdout line, then nothing more until readyok),
        // patched SF emits `info string evallegal <status> ...`. Either way,
        // we read until readyok so the channel is drained.
        sf.send("position startpos")?;
        sf.send("evallegal")?;
        sf.send("isready")?;
        loop {
            sf.line_buf.clear();
            let n = sf.stdout.read_line(&mut sf.line_buf)?;
            if n == 0 {
                return Err(StockfishError::UnexpectedEof);
            }
            let line = sf.line_buf.trim_end();
            if line.starts_with("info string evallegal ") {
                sf.is_patched = true;
            }
            if line.starts_with("readyok") {
                break;
            }
        }

        Ok(sf)
    }

    /// PID of the spawned Stockfish process. Used by the runner to pin
    /// the child to the same core as its driving worker thread.
    pub fn child_pid(&self) -> u32 {
        self.child.id()
    }

    /// Set MultiPV. No-op if Stockfish is already at this value (caller's
    /// responsibility to track — sending duplicate `setoption` to Stockfish
    /// is allowed but slow because we must wait for `readyok`).
    pub fn set_multi_pv(&mut self, n: u32) -> Result<(), StockfishError> {
        self.send(&format!("setoption name MultiPV value {n}"))?;
        self.send("isready")?;
        self.wait_for_token("readyok")
    }

    /// Set the patched binary's `NetSelection` UCI option. Vanilla SF18
    /// silently ignores unknown setoption names, so calling this on an
    /// unpatched binary is a no-op rather than an error — the caller is
    /// expected to have run the preflight check (`is_patched`) before
    /// calling this on a tier that meaningfully needs the override.
    pub fn set_net_selection(
        &mut self,
        choice: crate::config::NetSelection,
    ) -> Result<(), StockfishError> {
        self.send(&format!("setoption name NetSelection value {}", choice.as_uci_str()))?;
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
    ///
    /// **Contract:** the caller must have established a valid base
    /// position first, either via [`Self::new_game`] (which seeds the
    /// cache with `position startpos`) or via [`Self::candidates`]
    /// (which rebuilds the cache from a full move list). Calling
    /// `play_move` against a freshly-spawned process with no prior
    /// `new_game` call would produce a malformed UCI command.
    pub fn play_move(&mut self, uci: &str) {
        debug_assert!(
            !self.position_cmd.is_empty(),
            "play_move called before new_game / candidates — no base position",
        );
        // First move after `new_game()` needs the " moves" preamble.
        if !self.position_cmd.contains(" moves") {
            self.position_cmd.push_str(" moves");
        }
        self.position_cmd.push(' ');
        self.position_cmd.push_str(uci);
    }

    /// Issue the per-ply command (`go nodes N` or `evallegal`, depending on
    /// `budget`) against the cached position and parse the response.
    pub fn candidates_after_play_moves(
        &mut self,
    ) -> Result<CandidatesResult, StockfishError> {
        // Write position + per-ply command through the BufWriter, then a
        // single flush. Splitting the writes lets the BufWriter coalesce the
        // two short commands into one syscall pair instead of three.
        self.stdin.write_all(self.position_cmd.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.write_all(self.play_cmd.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        match self.budget {
            GoBudget::Nodes(_) => self.read_go_response(),
            GoBudget::EvalLegal => self.read_evallegal_response(),
        }
    }

    /// Parse the `info ... multipv ...` + `bestmove` stream produced by
    /// `go ...`. Used for every search-tree-budgeted tier.
    fn read_go_response(&mut self) -> Result<CandidatesResult, StockfishError> {
        // Map keyed by multipv index. Later (deeper) info lines for the
        // same index overwrite earlier ones. BTreeMap is fine at our
        // typical node budgets (1k–10k → at most a few entries kept).
        // If we ever drive Stockfish at much higher nodes where it emits
        // hundreds of `info` lines per `go`, a `Vec<(u32, Candidate)>`
        // sorted at the end would avoid the per-insert tree-node alloc;
        // not worth doing speculatively.
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
                    // Stockfish reports no legal moves. The
                    // discriminator is whether ANY info line carried
                    // `score mate 0` — true checkmate emits this even
                    // without a multipv field; stalemate emits either
                    // `score cp 0` or no info at all. Don't use
                    // `by_pv.is_empty()` as a heuristic here: stalemate
                    // also produces an empty `by_pv`, so that would
                    // misclassify it as checkmate.
                    let terminal = if last_score_was_mate0 {
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
                            score_v: None,
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
                        score_v: None,
                    },
                );
            } else if line.starts_with("info ")
                && (line.contains(" score mate 0 ")
                    || line.ends_with(" score mate 0"))
            {
                // Non-multipv `info ... score mate 0 ...` lines accompany
                // a `bestmove (none)` for true checkmate. Stalemate emits
                // either no info line or `score cp 0` with no mate marker;
                // without parsing this case we'd misclassify checkmate
                // as stalemate (or vice-versa) when MultiPV emits no
                // candidates because there are no legal moves.
                //
                // Use a word-boundary check (space-or-EOL after the `0`)
                // rather than a bare `contains(" score mate 0")` so
                // `" score mate 0X"` (a hypothetical leading-zero
                // emission) wouldn't falsely match. Stockfish 18 doesn't
                // emit such lines, but custom builds might.
                last_score_was_mate0 = true;
            }
        }
    }

    /// Parse the single `info string evallegal <status> [<uci> <cp>]...` line
    /// produced by the patched binary's `evallegal` command. Other `info`
    /// lines that may have leaked through (e.g. nnue init banners on first
    /// call) are skipped.
    fn read_evallegal_response(&mut self) -> Result<CandidatesResult, StockfishError> {
        loop {
            self.line_buf.clear();
            let n = self.stdout.read_line(&mut self.line_buf)?;
            if n == 0 {
                return Err(StockfishError::UnexpectedEof);
            }
            let line = self.line_buf.trim_end();
            let Some(rest) = line.strip_prefix("info string evallegal ") else {
                // Should not happen with a patched binary on a well-formed
                // position. If a vanilla SF slipped through preflight,
                // the response would be `Unknown command: 'evallegal'` —
                // treat as protocol violation.
                if line.starts_with("Unknown command") {
                    return Err(StockfishError::Protocol(format!(
                        "evallegal not supported by this binary: {line}"
                    )));
                }
                continue;
            };
            return Ok(parse_evallegal_payload(rest));
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

/// Parse the payload portion of an `info string evallegal <payload>` line —
/// `<payload>` is everything after the `evallegal ` prefix:
///
/// - `mate` / `stalemate` → empty candidates + matching `TerminalKind`
/// - `none <uci> <cp> <v> <uci> <cp> <v> ...` → candidates list, no terminal
/// - `check <uci> <cp> <v> ...` → candidates list, no terminal (in-check
///   positions are NNUE-OOD; the caller is expected to flag/discard if
///   they care, but we don't drop them on the engine's behalf)
///
/// Malformed payloads (unknown status, missing trailing tokens, unparseable
/// integers) degrade gracefully — we keep whatever well-formed triplets we
/// got and stop. Protocol-level violations should be impossible from our
/// patched binary, and a stricter error here would risk killing a worker
/// on the rare malformed line.
fn parse_evallegal_payload(payload: &str) -> CandidatesResult {
    let mut it = payload.split_whitespace();
    let status = match it.next() {
        Some(s) => s,
        None => return CandidatesResult { candidates: Vec::new(), terminal: None },
    };
    match status {
        "mate" => {
            return CandidatesResult {
                candidates: Vec::new(),
                terminal: Some(TerminalKind::Checkmate),
            };
        }
        "stalemate" => {
            return CandidatesResult {
                candidates: Vec::new(),
                terminal: Some(TerminalKind::Stalemate),
            };
        }
        "none" | "check" => {} // fall through
        _ => return CandidatesResult { candidates: Vec::new(), terminal: None },
    }

    let mut candidates = Vec::with_capacity(32);
    while let Some(uci) = it.next() {
        let Some(cp_tok) = it.next() else { break };
        let Some(v_tok) = it.next() else { break };
        let Ok(cp) = cp_tok.parse::<i32>() else { break };
        let Ok(v) = v_tok.parse::<i32>() else { break };
        candidates.push(Candidate {
            uci: uci.to_string(),
            score_cp: cp as f32,
            score_v: Some(v as f32),
        });
    }
    CandidatesResult { candidates, terminal: None }
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

    #[test]
    fn parse_evallegal_none_triplets() {
        let r = parse_evallegal_payload("none e2e4 27 75 d2d4 24 67 g1f3 -3 -8");
        assert!(r.terminal.is_none());
        assert_eq!(r.candidates.len(), 3);
        assert_eq!(r.candidates[0].uci, "e2e4");
        assert!((r.candidates[0].score_cp - 27.0).abs() < 1e-6);
        assert!((r.candidates[0].score_v.unwrap() - 75.0).abs() < 1e-6);
        assert_eq!(r.candidates[2].uci, "g1f3");
        assert!((r.candidates[2].score_cp - -3.0).abs() < 1e-6);
        assert!((r.candidates[2].score_v.unwrap() - -8.0).abs() < 1e-6);
    }

    #[test]
    fn parse_evallegal_check_triplets() {
        let r = parse_evallegal_payload("check a8a7 -3 -10");
        assert!(r.terminal.is_none()); // in-check is NOT terminal
        assert_eq!(r.candidates.len(), 1);
        assert!((r.candidates[0].score_v.unwrap() - -10.0).abs() < 1e-6);
    }

    #[test]
    fn parse_evallegal_mate_and_stalemate() {
        assert_eq!(
            parse_evallegal_payload("mate").terminal,
            Some(TerminalKind::Checkmate),
        );
        assert!(parse_evallegal_payload("mate").candidates.is_empty());
        assert_eq!(
            parse_evallegal_payload("stalemate").terminal,
            Some(TerminalKind::Stalemate),
        );
    }

    #[test]
    fn parse_evallegal_truncated_triplet_drops_partial() {
        // Two well-formed triplets; the third is missing its v token. We keep
        // the first two and stop — better than panicking on bad engine output.
        let r = parse_evallegal_payload("none e2e4 10 35 d2d4 8 28 g1f3 -3");
        assert_eq!(r.candidates.len(), 2);
        assert_eq!(r.candidates[1].uci, "d2d4");
    }

    #[test]
    fn parse_evallegal_unparseable_v_drops_remainder() {
        let r = parse_evallegal_payload("none e2e4 10 35 d2d4 8 deadbeef");
        assert_eq!(r.candidates.len(), 1);
        assert_eq!(r.candidates[0].uci, "e2e4");
    }

    /// Resolve the Stockfish binary for a live test. Tests that use this
    /// are marked `#[ignore]` so they only run when the user explicitly
    /// opts in (`cargo test -- --include-ignored` or
    /// `cargo test -- --ignored`); inside the test body we expect the
    /// binary to be present and panic with a clear message otherwise.
    fn stockfish_path() -> std::path::PathBuf {
        if let Ok(p) = std::env::var("STOCKFISH_PATH") {
            return p.into();
        }
        let default = std::path::PathBuf::from(
            std::env::var("HOME").unwrap_or_default(),
        )
        .join("bin/stockfish");
        assert!(
            default.exists(),
            "stockfish binary not found at {} — set STOCKFISH_PATH or install one",
            default.display(),
        );
        default
    }

    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_handshake_and_starting_candidates() {
        let path = stockfish_path();
        let mut sf = StockfishProcess::spawn(&path, "Stockfish", 16, crate::stockfish::GoBudget::Nodes(1)).unwrap();
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
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_version_mismatch_rejected() {
        let path = stockfish_path();
        match StockfishProcess::spawn(&path, "Stockfish 9999", 16, crate::stockfish::GoBudget::Nodes(1)) {
            Err(StockfishError::VersionMismatch { .. }) => {}
            Err(e) => panic!("expected VersionMismatch, got {e:?}"),
            Ok(_) => panic!("expected VersionMismatch, got Ok"),
        }
    }
}
