//! UCI engine self-play: spawn external engines, play games, return results.
//!
//! Each rayon worker gets its own engine subprocess. Games are played via
//! the UCI protocol (stdin/stdout pipes). Supports MultiPV + softmax
//! temperature sampling for move diversity.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// A single UCI engine process.
struct UciEngine {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
}

impl UciEngine {
    fn new(path: &str, hash_mb: u32, multi_pv: u32) -> Self {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("Failed to start engine at {}: {}", path, e));

        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);

        let mut eng = UciEngine { child, reader };

        // Check if engine died immediately
        if let Some(status) = eng.child.try_wait().ok().flatten() {
            // Read stderr for error message
            let mut stderr = String::new();
            if let Some(ref mut err) = eng.child.stderr {
                use std::io::Read;
                let _ = err.read_to_string(&mut stderr);
            }
            panic!("Engine exited immediately with {}: {}", status, stderr.trim());
        }

        eng.send("uci");
        eng.wait_for("uciok");
        eng.send(&format!("setoption name Hash value {}", hash_mb));
        eng.send("setoption name Threads value 1");
        if multi_pv > 1 {
            eng.send(&format!("setoption name MultiPV value {}", multi_pv));
        }
        eng.send("isready");
        eng.wait_for("readyok");
        eng
    }

    fn send(&mut self, cmd: &str) {
        let stdin = self.child.stdin.as_mut().unwrap();
        writeln!(stdin, "{}", cmd).unwrap();
        stdin.flush().unwrap();
    }

    fn wait_for(&mut self, token: &str) -> Vec<String> {
        use std::time::{Duration, Instant};
        let timeout = Duration::from_secs(60);
        let start = Instant::now();
        let mut lines = Vec::new();
        let mut buf = String::new();
        loop {
            buf.clear();
            match self.reader.read_line(&mut buf) {
                Ok(0) => {
                    // EOF — engine closed stdout
                    let mut stderr = String::new();
                    if let Some(ref mut err) = self.child.stderr {
                        use std::io::Read;
                        let _ = err.read_to_string(&mut stderr);
                    }
                    panic!(
                        "Engine closed stdout while waiting for '{}'. Lines so far: {:?}. Stderr: {}",
                        token, lines, stderr.trim()
                    );
                }
                Ok(_) => {
                    let line = buf.trim_end().to_string();
                    let done = line.starts_with(token);
                    lines.push(line);
                    if done {
                        break;
                    }
                }
                Err(e) => panic!("Error reading from engine: {}", e),
            }
            if start.elapsed() > timeout {
                panic!(
                    "Timeout waiting for '{}' after {:?}. Lines so far: {:?}",
                    token, timeout, lines
                );
            }
        }
        lines
    }

    fn set_multi_pv(&mut self, n: u32) {
        self.send(&format!("setoption name MultiPV value {}", n));
        self.send("isready");
        self.wait_for("readyok");
    }

    fn new_game(&mut self) {
        self.send("ucinewgame");
        self.send("isready");
        self.wait_for("readyok");
    }

    /// Run a search and return (candidates, is_terminal, terminal_type).
    /// Candidates: Vec<(uci_move, score_centipawns)>.
    fn candidates(&mut self, moves: &[String], nodes: u32) -> (Vec<(String, f64)>, Option<&'static str>) {
        let pos = if moves.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves.join(" "))
        };
        self.send(&pos);
        self.send(&format!("go nodes {}", nodes));
        let lines = self.wait_for("bestmove");

        // Parse MultiPV info lines — keep last (deepest) per PV index
        let mut best_by_pv: std::collections::BTreeMap<u32, (String, f64)> =
            std::collections::BTreeMap::new();

        for line in &lines {
            if !line.starts_with("info") || !line.contains(" multipv ") {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            let pv_idx = Self::find_val(&parts, "multipv")
                .and_then(|s| s.parse::<u32>().ok());
            let score = Self::parse_score(&parts);
            let pv_move = Self::find_val(&parts, "pv");

            if let (Some(idx), Some(sc), Some(mv)) = (pv_idx, score, pv_move) {
                best_by_pv.insert(idx, (mv.to_string(), sc));
            }
        }

        if !best_by_pv.is_empty() {
            let cands: Vec<(String, f64)> = best_by_pv.into_values().collect();
            return (cands, None);
        }

        // Fallback: parse bestmove directly
        for line in &lines {
            if line.starts_with("bestmove") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(mv) = parts.get(1) {
                    // Detect terminal: "(none)" (Stockfish) or self-move like "a1a1" (Lc0)
                    let is_terminal = *mv == "(none)" || Self::is_self_move(mv);
                    if !is_terminal {
                        return (vec![(mv.to_string(), 0.0)], None);
                    }
                    // Terminal — check if checkmate or stalemate from info lines
                    let mut terminal = "stalemate";
                    for info_line in &lines {
                        if info_line.starts_with("info") && info_line.contains("score") {
                            let info_parts: Vec<&str> = info_line.split_whitespace().collect();
                            if Self::find_val(&info_parts, "score") == Some("mate") {
                                terminal = "checkmate";
                            }
                        }
                    }
                    return (vec![], Some(terminal));
                }
            }
        }

        (vec![], Some("stalemate"))
    }

    fn find_val<'a>(parts: &'a [&'a str], key: &str) -> Option<&'a str> {
        parts.iter()
            .position(|&p| p == key)
            .and_then(|i| parts.get(i + 1))
            .copied()
    }

    /// Check if a UCI move is a self-move (src == dst), which Lc0 uses
    /// to signal no legal moves (instead of Stockfish's "(none)").
    fn is_self_move(uci: &str) -> bool {
        let b = uci.as_bytes();
        b.len() >= 4 && b[0] == b[2] && b[1] == b[3]
    }

    fn parse_score(parts: &[&str]) -> Option<f64> {
        let si = parts.iter().position(|&p| p == "score")?;
        match parts.get(si + 1)? {
            &"cp" => parts.get(si + 2)?.parse::<f64>().ok(),
            &"mate" => {
                let mate_in = parts.get(si + 2)?.parse::<i32>().ok()?;
                Some(if mate_in > 0 { 30_000.0 } else { -30_000.0 })
            }
            _ => None,
        }
    }

    fn close(mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

/// Softmax sample from candidates using centipawn scores.
fn softmax_sample(candidates: &[(String, f64)], temperature: f64, rng: &mut StdRng) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 || temperature <= 0.0 {
        return Some(candidates.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?.0.clone());
    }

    let max_s = candidates.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = candidates
        .iter()
        .map(|c| ((c.1 - max_s) / (100.0 * temperature)).exp())
        .collect();
    let total: f64 = exps.iter().sum();

    let r: f64 = rng.gen::<f64>() * total;
    let mut cumulative = 0.0;
    for (i, e) in exps.iter().enumerate() {
        cumulative += e;
        if r <= cumulative {
            return Some(candidates[i].0.clone());
        }
    }
    Some(candidates.last()?.0.clone())
}

/// Play one self-play game. Returns (uci_moves, result_string).
fn play_game(
    engine: &mut UciEngine,
    nodes: u32,
    rng: &mut StdRng,
    temperature: f64,
    multi_pv: u32,
    sample_plies: u32,
    max_ply: u32,
) -> (Vec<String>, String) {
    engine.new_game();
    if multi_pv > 1 {
        engine.set_multi_pv(multi_pv);
    }

    let mut moves: Vec<String> = Vec::new();
    let mut switched = false;

    for ply in 0..max_ply {
        if !switched && ply >= sample_plies {
            engine.set_multi_pv(1);
            switched = true;
        }

        let (cands, terminal) = engine.candidates(&moves, nodes);

        let chosen = if switched {
            cands.first().map(|c| c.0.clone())
        } else {
            softmax_sample(&cands, temperature, rng)
        };

        match chosen {
            Some(mv) => moves.push(mv),
            None => {
                // Terminal position
                let n = moves.len();
                let result = match terminal {
                    Some("checkmate") => {
                        if n % 2 == 0 { "0-1" } else { "1-0" }
                    }
                    _ => "1/2-1/2",
                };
                return (moves, result.to_string());
            }
        }
    }

    // Hit max ply
    (moves, "1/2-1/2".to_string())
}

/// A single game result returned to Python.
pub struct GameResult {
    pub uci: String,       // space-joined UCI moves
    pub result: String,    // "1-0", "0-1", "1/2-1/2"
    pub n_ply: u16,
}

/// Worker function: play num_games with a dedicated engine, return results.
fn worker_play(
    engine_path: &str,
    nodes: u32,
    num_games: u32,
    hash_mb: u32,
    seed: u64,
    temperature: f64,
    multi_pv: u32,
    sample_plies: u32,
    max_ply: u32,
    worker_id: u32,
) -> Vec<GameResult> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut engine = UciEngine::new(engine_path, hash_mb, multi_pv);
    let mut results = Vec::with_capacity(num_games as usize);

    for i in 0..num_games {
        let (moves, result) = play_game(
            &mut engine, nodes, &mut rng, temperature,
            multi_pv, sample_plies, max_ply,
        );
        let n_ply = moves.len() as u16;
        results.push(GameResult {
            uci: moves.join(" "),
            result,
            n_ply,
        });

        if (i + 1) % 500 == 0 {
            eprintln!("  [worker {:>2}] {:>6}/{:>6}", worker_id, i + 1, num_games);
        }
    }

    engine.close();
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_self_move_detects_a1a1() {
        assert!(UciEngine::is_self_move("a1a1"));
        assert!(UciEngine::is_self_move("e4e4"));
        assert!(UciEngine::is_self_move("h8h8"));
    }

    #[test]
    fn test_is_self_move_rejects_actual_moves() {
        assert!(!UciEngine::is_self_move("e2e4"));
        assert!(!UciEngine::is_self_move("g1f3"));
        assert!(!UciEngine::is_self_move("a1h8"));
    }

    #[test]
    fn test_is_self_move_short_string() {
        // Needs at least 4 chars
        assert!(!UciEngine::is_self_move(""));
        assert!(!UciEngine::is_self_move("a1"));
        assert!(!UciEngine::is_self_move("abc"));
    }

    #[test]
    fn test_is_self_move_with_promotion() {
        // Self-move with promotion suffix (technically not a real move but we accept the 4 char check)
        assert!(UciEngine::is_self_move("a1a1q"));
    }

    #[test]
    fn test_find_val_basic() {
        let parts = vec!["info", "depth", "10", "score", "cp", "42", "pv", "e2e4"];
        assert_eq!(UciEngine::find_val(&parts, "depth"), Some("10"));
        assert_eq!(UciEngine::find_val(&parts, "score"), Some("cp"));
        assert_eq!(UciEngine::find_val(&parts, "pv"), Some("e2e4"));
        assert_eq!(UciEngine::find_val(&parts, "nope"), None);
    }

    #[test]
    fn test_find_val_trailing_key() {
        // Key is last in list (no val)
        let parts = vec!["info", "depth"];
        assert_eq!(UciEngine::find_val(&parts, "depth"), None);
    }

    #[test]
    fn test_parse_score_cp() {
        let parts = vec!["info", "score", "cp", "42", "pv", "e2e4"];
        assert_eq!(UciEngine::parse_score(&parts), Some(42.0));
    }

    #[test]
    fn test_parse_score_cp_negative() {
        let parts = vec!["info", "score", "cp", "-150", "pv", "e2e4"];
        assert_eq!(UciEngine::parse_score(&parts), Some(-150.0));
    }

    #[test]
    fn test_parse_score_mate_winning() {
        let parts = vec!["info", "score", "mate", "3", "pv", "e2e4"];
        assert_eq!(UciEngine::parse_score(&parts), Some(30_000.0));
    }

    #[test]
    fn test_parse_score_mate_losing() {
        let parts = vec!["info", "score", "mate", "-2", "pv", "e2e4"];
        assert_eq!(UciEngine::parse_score(&parts), Some(-30_000.0));
    }

    #[test]
    fn test_parse_score_missing() {
        let parts = vec!["info", "depth", "10"];
        assert_eq!(UciEngine::parse_score(&parts), None);
    }

    #[test]
    fn test_parse_score_unknown_type() {
        // Unknown score type (e.g., "xx")
        let parts = vec!["info", "score", "xx", "42"];
        assert_eq!(UciEngine::parse_score(&parts), None);
    }

    #[test]
    fn test_softmax_sample_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        assert!(softmax_sample(&[], 1.0, &mut rng).is_none());
    }

    #[test]
    fn test_softmax_sample_single() {
        let mut rng = StdRng::seed_from_u64(42);
        let cands = vec![("e2e4".to_string(), 0.5)];
        assert_eq!(softmax_sample(&cands, 1.0, &mut rng), Some("e2e4".to_string()));
    }

    #[test]
    fn test_softmax_sample_zero_temp_picks_best() {
        let mut rng = StdRng::seed_from_u64(42);
        let cands = vec![
            ("e2e4".to_string(), 10.0),
            ("d2d4".to_string(), 50.0),
            ("c2c4".to_string(), 25.0),
        ];
        // T=0 → argmax
        assert_eq!(softmax_sample(&cands, 0.0, &mut rng), Some("d2d4".to_string()));
    }

    #[test]
    fn test_softmax_sample_negative_temp_picks_best() {
        let mut rng = StdRng::seed_from_u64(42);
        let cands = vec![
            ("e2e4".to_string(), 10.0),
            ("d2d4".to_string(), 50.0),
        ];
        // Negative T: treated like T<=0 → argmax
        assert_eq!(softmax_sample(&cands, -1.0, &mut rng), Some("d2d4".to_string()));
    }

    #[test]
    fn test_softmax_sample_distribution() {
        // At high temperature, sampling should sometimes hit lower-scored candidates.
        let cands = vec![
            ("m1".to_string(), 0.0),
            ("m2".to_string(), 0.0),
            ("m3".to_string(), 0.0),
        ];
        // Equal scores: uniform distribution, all 3 should be reachable.
        let mut seen = std::collections::HashSet::new();
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let r = softmax_sample(&cands, 1.0, &mut rng);
            seen.insert(r.unwrap());
        }
        assert_eq!(seen.len(), 3, "uniform sampling should cover all candidates");
    }

    #[test]
    fn test_softmax_deterministic_with_seed() {
        let cands = vec![
            ("m1".to_string(), 0.0),
            ("m2".to_string(), 10.0),
            ("m3".to_string(), 20.0),
        ];
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            let r1 = softmax_sample(&cands, 0.5, &mut rng1);
            let r2 = softmax_sample(&cands, 0.5, &mut rng2);
            assert_eq!(r1, r2, "same seed → same sample");
        }
    }

    #[test]
    #[ignore = "requires stockfish binary"]
    fn test_generate_engine_games_integration() {
        // Would call generate_engine_games with a real Stockfish binary.
        // Skipped in unit tests to avoid requiring an external engine.
    }
}

/// Generate self-play games using an external UCI engine (Stockfish, Lc0, etc).
///
/// Spawns `n_workers` engine processes, each playing its share of games.
/// Uses rayon for orchestration but each worker is I/O-bound (engine subprocess),
/// so we use a dedicated thread pool sized to n_workers.
///
/// Returns Vec of (uci_string, result, n_ply, worker_id, seed).
pub fn generate_engine_games(
    engine_path: &str,
    nodes: u32,
    total_games: u32,
    n_workers: u32,
    base_seed: u64,
    temperature: f64,
    multi_pv: u32,
    sample_plies: u32,
    hash_mb: u32,
    max_ply: u32,
) -> Vec<GameResult> {
    let base = total_games / n_workers;
    let remainder = total_games % n_workers;

    eprintln!("Generating {} games with {} workers (engine: {})",
              total_games, n_workers, engine_path);
    eprintln!("  nodes={}, temperature={}, multi_pv={}, sample_plies={}",
              nodes, temperature, multi_pv, sample_plies);

    // Build a custom rayon pool sized to n_workers (these are I/O-bound threads)
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers as usize)
        .build()
        .unwrap();

    let path = engine_path.to_string();

    let all_results: Vec<Vec<GameResult>> = pool.install(|| {
        (0..n_workers)
            .into_par_iter()
            .map(|i| {
                let games = base + if i < remainder { 1 } else { 0 };
                let seed = base_seed + i as u64;
                worker_play(&path, nodes, games, hash_mb, seed,
                            temperature, multi_pv, sample_plies, max_ply, i)
            })
            .collect()
    });

    let mut flat: Vec<GameResult> = Vec::with_capacity(total_games as usize);
    for worker_results in all_results {
        flat.extend(worker_results);
    }

    eprintln!("Generated {} games total", flat.len());
    flat
}
