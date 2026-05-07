//! cp-vs-v (or any two SampleScore × temperature combinations) match runner.
//!
//! Drives N parallel workers, each spawning one patched-Stockfish via the
//! evallegal protocol. Openings are generated deterministically from
//! `master_seed` so that the same opening is played twice (color-swapped),
//! cancelling first-move advantage. Outputs aggregate W/D/L plus an Elo
//! difference with a Wilson 95% CI.
//!
//! Config schema: see `TournamentConfig`. CLI: `stockfish-datagen tournament
//! --config tournament.json`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use shakmaty::Color;

use crate::affinity;
use crate::config::SampleScore;
use crate::match_game::{MatchOutcome, SideConfig as MatchSide, generate_opening, play_match_game};
use crate::seed;
use crate::stockfish::{GoBudget, StockfishProcess};

/// Whole-tournament config, JSON-loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentConfig {
    pub stockfish_path: PathBuf,
    pub stockfish_version: String,
    #[serde(default = "default_hash_mb")]
    pub stockfish_hash_mb: u32,
    pub master_seed: u64,
    pub n_workers: u32,
    /// Number of distinct opening positions to play. Total games = 2 ×
    /// `n_pairs` (one per color assignment).
    pub n_pairs: u32,
    #[serde(default = "default_max_ply")]
    pub max_ply: u32,
    /// How many random plies of opening to play before handing the game
    /// off to the two sides. 4 plies (= 2 each side) gives plenty of
    /// position diversity without straying into pathological positions.
    #[serde(default = "default_opening_plies")]
    pub opening_plies: u32,
    pub side_a: TournamentSide,
    pub side_b: TournamentSide,
    /// Optional path to write per-game results JSON (one row per game).
    /// Useful for downstream Elo-curve analysis or sanity-checking
    /// individual games. Skipped when None.
    #[serde(default)]
    pub output_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentSide {
    pub name: String,
    pub sample_score: SampleScore,
    pub temperature: f32,
}

fn default_hash_mb() -> u32 { 16 }
fn default_max_ply() -> u32 { 512 }
fn default_opening_plies() -> u32 { 4 }

impl TournamentConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: TournamentConfig = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.n_workers == 0 { anyhow::bail!("n_workers must be > 0"); }
        if self.n_pairs == 0 { anyhow::bail!("n_pairs must be > 0"); }
        if self.max_ply == 0 { anyhow::bail!("max_ply must be > 0"); }
        // `max_ply` is the absolute cap on total plies (matches play_game).
        // `opening_plies` consumes plies from that budget, so configs with
        // `opening_plies >= max_ply` would skip the play loop entirely and
        // every game would be labeled as `PlyLimit` with the actual game
        // being just the opening — silently producing meaningless ~50% Elo.
        if self.opening_plies >= self.max_ply {
            anyhow::bail!(
                "opening_plies ({}) must be < max_ply ({}); the opening prefix \
                 consumes from the play budget and an >=-cap value would skip \
                 the play loop entirely",
                self.opening_plies, self.max_ply,
            );
        }
        // sample_score=V requires the patched binary's evallegal protocol —
        // both sides drive that protocol regardless of mode (we always
        // spawn with GoBudget::EvalLegal in tournament mode), so this is
        // satisfied as long as the binary is patched. The preflight in the
        // CLI handler catches an unpatched binary; nothing to validate here.
        if self.side_a.name == self.side_b.name {
            anyhow::bail!("side_a and side_b need distinct names for output disambiguation");
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> String {
        // Whole-config sha256, same approach as RunConfig::fingerprint.
        use sha2::{Digest, Sha256};
        let canonical = serde_json::to_vec(self).expect("config is round-trippable");
        let mut h = Sha256::new();
        h.update(&canonical);
        format!("{:x}", h.finalize())
    }
}

/// One row in the per-game output. Persistent format — additions only;
/// don't reorder or drop fields once we've shipped any results.
#[derive(Debug, Clone, Serialize)]
pub struct GameRecord {
    pub pair_idx: u32,
    pub a_color: &'static str, // "w" or "b"
    pub winner: Option<&'static str>, // "w", "b", or None
    pub reason: &'static str,
    pub n_plies: usize,
    pub opening: Vec<String>,
    pub moves: Vec<String>,
}

/// Aggregate result of a tournament run.
#[derive(Debug, Clone)]
pub struct TournamentResult {
    pub a_wins: u32,
    pub b_wins: u32,
    pub draws: u32,
    pub total: u32,
    pub records: Vec<GameRecord>,
}

impl TournamentResult {
    /// Side A's score in chess terms: 1 per win, 0.5 per draw, 0 per loss.
    pub fn a_score(&self) -> f64 {
        self.a_wins as f64 + 0.5 * self.draws as f64
    }

    /// Side A's win rate (0..1) used for Elo conversion.
    pub fn a_win_rate(&self) -> f64 {
        if self.total == 0 { return 0.5; }
        self.a_score() / self.total as f64
    }

    /// Wilson 95% interval on `a_win_rate()`. Returned as `(low, high)`.
    pub fn a_win_rate_ci95(&self) -> (f64, f64) {
        wilson_interval(self.a_win_rate(), self.total as f64, 1.96)
    }

    /// Elo difference (A − B). Positive ⇒ A stronger.
    pub fn a_elo(&self) -> f64 {
        elo_from_win_rate(self.a_win_rate())
    }

    /// 95% CI on the Elo diff via the Wilson interval on the win rate.
    pub fn a_elo_ci95(&self) -> (f64, f64) {
        let (lo, hi) = self.a_win_rate_ci95();
        (elo_from_win_rate(lo), elo_from_win_rate(hi))
    }
}

/// `Elo = -400 · log10(1/w − 1)`. Saturates at ±∞ for win rates of 0/1
/// (unbounded; caller should display "n/a" or similar).
fn elo_from_win_rate(w: f64) -> f64 {
    if w >= 1.0 { return f64::INFINITY; }
    if w <= 0.0 { return f64::NEG_INFINITY; }
    -400.0 * (1.0 / w - 1.0).log10()
}

/// Wilson score interval — the 95% (z=1.96) CI for a binomial proportion.
/// Better-behaved than the normal approximation for small n or extreme p.
fn wilson_interval(p: f64, n: f64, z: f64) -> (f64, f64) {
    if n <= 0.0 { return (0.0, 1.0); }
    let denom = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / denom;
    let margin = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denom;
    (
        (center - margin).clamp(0.0, 1.0),
        (center + margin).clamp(0.0, 1.0),
    )
}

/// Run the configured tournament. Spawns workers, generates openings,
/// drains outcomes, returns aggregate stats + per-game records.
pub fn run_tournament(cfg: &TournamentConfig) -> anyhow::Result<TournamentResult> {
    // Generate openings up front in the main thread — deterministic,
    // cheap, and lets workers consume them by index without coordinating.
    eprintln!("generating {} openings ({} plies each)...", cfg.n_pairs, cfg.opening_plies);
    let openings: Vec<Vec<String>> = (0..cfg.n_pairs)
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed::tier_seed(cfg.master_seed, i as usize));
            generate_opening(&mut rng, cfg.opening_plies)
        })
        .collect();

    let cfg = Arc::new(cfg.clone());
    let openings = Arc::new(openings);
    let (tx, rx) = crossbeam_channel::unbounded::<(u32, MatchOutcome, Vec<String>)>();

    eprintln!(
        "starting tournament: {} pairs × 2 = {} games, {} workers",
        cfg.n_pairs, 2 * cfg.n_pairs, cfg.n_workers,
    );

    let mut handles = Vec::with_capacity(cfg.n_workers as usize);
    for worker_id in 0..cfg.n_workers {
        let cfg = Arc::clone(&cfg);
        let openings = Arc::clone(&openings);
        let tx = tx.clone();
        let h = std::thread::Builder::new()
            .name(format!("tournament-w{worker_id}"))
            .spawn(move || run_worker(worker_id, cfg, openings, tx))
            .with_context(|| format!("spawning tournament worker {worker_id}"))?;
        handles.push((worker_id, h));
    }
    drop(tx); // close the sending side so the recv loop exits when workers finish

    let mut a_wins = 0u32;
    let mut b_wins = 0u32;
    let mut draws = 0u32;
    let mut records: Vec<GameRecord> = Vec::with_capacity((2 * cfg.n_pairs) as usize);
    for (pair_idx, outcome, opening) in rx {
        let scored_for = match outcome.winner {
            Some(c) if c == outcome.a_color => Some("a"),
            Some(_) => Some("b"),
            None => None,
        };
        match scored_for {
            Some("a") => a_wins += 1,
            Some("b") => b_wins += 1,
            _ => draws += 1,
        }
        records.push(GameRecord {
            pair_idx,
            a_color: color_str(outcome.a_color),
            winner: outcome.winner.map(color_str),
            reason: reason_str(outcome.reason),
            n_plies: outcome.n_plies,
            opening,
            moves: outcome.moves,
        });
    }

    let mut had_err: Option<anyhow::Error> = None;
    for (worker_id, h) in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                had_err.get_or_insert(e.context(format!("worker {worker_id}")));
            }
            Err(_) => {
                had_err.get_or_insert(anyhow!("worker {worker_id} panicked"));
            }
        }
    }
    if let Some(e) = had_err { return Err(e); }

    // Sort records by pair_idx so the output file is deterministic across runs.
    records.sort_by_key(|r| (r.pair_idx, r.a_color));

    let total = a_wins + b_wins + draws;
    let result = TournamentResult { a_wins, b_wins, draws, total, records };

    if let Some(out) = &cfg.output_path {
        let payload = serde_json::json!({
            "fingerprint": cfg.fingerprint(),
            "side_a": &cfg.side_a,
            "side_b": &cfg.side_b,
            "n_pairs": cfg.n_pairs,
            "opening_plies": cfg.opening_plies,
            "max_ply": cfg.max_ply,
            "master_seed": cfg.master_seed,
            "summary": {
                "a_wins": result.a_wins,
                "b_wins": result.b_wins,
                "draws": result.draws,
                "total": result.total,
                "a_win_rate": result.a_win_rate(),
                "a_win_rate_ci95": result.a_win_rate_ci95(),
                "a_elo": result.a_elo(),
                "a_elo_ci95": result.a_elo_ci95(),
            },
            "games": &result.records,
        });
        std::fs::write(out, serde_json::to_vec_pretty(&payload)?)
            .with_context(|| format!("writing tournament output to {}", out.display()))?;
    }

    Ok(result)
}

fn run_worker(
    worker_id: u32,
    cfg: Arc<TournamentConfig>,
    openings: Arc<Vec<Vec<String>>>,
    tx: crossbeam_channel::Sender<(u32, MatchOutcome, Vec<String>)>,
) -> anyhow::Result<()> {
    let mut sf = StockfishProcess::spawn(
        &cfg.stockfish_path,
        &cfg.stockfish_version,
        cfg.stockfish_hash_mb,
        GoBudget::EvalLegal,
    )
    .with_context(|| format!("spawning stockfish for tournament worker {worker_id}"))?;
    affinity::pin_pair(sf.child_pid(), worker_id);

    if !sf.is_patched {
        anyhow::bail!(
            "tournament requires the patched binary (evallegal command) — \
             {} is vanilla SF. Build via scripts/build_patched_stockfish.sh.",
            cfg.stockfish_path.display(),
        );
    }

    let side_a = MatchSide {
        name: cfg.side_a.name.clone(),
        sample_score: cfg.side_a.sample_score,
        temperature: cfg.side_a.temperature,
    };
    let side_b = MatchSide {
        name: cfg.side_b.name.clone(),
        sample_score: cfg.side_b.sample_score,
        temperature: cfg.side_b.temperature,
    };

    // Each worker takes a strided slice of pairs: pair_idx % n_workers == worker_id.
    // No coordination needed — the modular partition is disjoint and exhaustive.
    for pair_idx in (worker_id..cfg.n_pairs).step_by(cfg.n_workers as usize) {
        let opening = &openings[pair_idx as usize];

        // Game 0 of the pair: side_a as White, side_b as Black.
        let game0_seed = seed::worker_seed(
            seed::tier_seed(cfg.master_seed.wrapping_add(0xA17EE7), pair_idx as usize),
            0,
        );
        let mut rng0 = ChaCha8Rng::seed_from_u64(game0_seed);
        let outcome0 = play_match_game(
            &mut sf, &mut rng0, opening, Color::White, &side_a, &side_b, cfg.max_ply,
        )
        .with_context(|| format!("worker {worker_id} pair {pair_idx} game 0"))?;
        tx.send((pair_idx, outcome0, opening.clone()))
            .map_err(|_| anyhow!("tournament drain dropped before worker {worker_id} finished pair {pair_idx} game 0"))?;

        // Game 1: side_a as Black, side_b as White. Different seed so the
        // RNG draws don't accidentally line up.
        let game1_seed = seed::worker_seed(
            seed::tier_seed(cfg.master_seed.wrapping_add(0xB17EE7), pair_idx as usize),
            1,
        );
        let mut rng1 = ChaCha8Rng::seed_from_u64(game1_seed);
        let outcome1 = play_match_game(
            &mut sf, &mut rng1, opening, Color::Black, &side_a, &side_b, cfg.max_ply,
        )
        .with_context(|| format!("worker {worker_id} pair {pair_idx} game 1"))?;
        tx.send((pair_idx, outcome1, opening.clone()))
            .map_err(|_| anyhow!("tournament drain dropped before worker {worker_id} finished pair {pair_idx} game 1"))?;
    }

    sf.shutdown();
    Ok(())
}

fn color_str(c: Color) -> &'static str {
    match c { Color::White => "w", Color::Black => "b" }
}

fn reason_str(r: crate::outcome::OutcomeReason) -> &'static str {
    use crate::outcome::OutcomeReason::*;
    match r {
        WhiteCheckmate => "white_checkmate",
        BlackCheckmate => "black_checkmate",
        Stalemate => "stalemate",
        InsufficientMaterial => "insufficient_material",
        ThreefoldRepetition => "threefold_repetition",
        FiftyMoveRule => "fifty_move_rule",
        PlyLimit => "ply_limit",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elo_zero_at_50pct() {
        assert!((elo_from_win_rate(0.5)).abs() < 1e-9);
    }

    #[test]
    fn elo_positive_when_winning_more() {
        assert!(elo_from_win_rate(0.6) > 0.0);
        assert!(elo_from_win_rate(0.55) > 0.0);
    }

    #[test]
    fn elo_negative_when_losing_more() {
        assert!(elo_from_win_rate(0.4) < 0.0);
    }

    #[test]
    fn elo_70pct_is_about_147() {
        // Standard reference from chess Elo math.
        let e = elo_from_win_rate(0.70);
        assert!((e - 147.19).abs() < 0.5, "got {e}");
    }

    #[test]
    fn wilson_collapses_to_unit_interval_when_n_zero() {
        let (lo, hi) = wilson_interval(0.5, 0.0, 1.96);
        assert_eq!((lo, hi), (0.0, 1.0));
    }

    #[test]
    fn wilson_brackets_observed_proportion() {
        // For 60/100 the standard Wilson 95% CI is roughly (0.502, 0.690).
        let (lo, hi) = wilson_interval(0.6, 100.0, 1.96);
        assert!(lo > 0.49 && lo < 0.51, "lo={lo}");
        assert!(hi > 0.68 && hi < 0.70, "hi={hi}");
    }

    #[test]
    fn tournament_result_arithmetic() {
        let r = TournamentResult {
            a_wins: 60,
            b_wins: 30,
            draws: 10,
            total: 100,
            records: Vec::new(),
        };
        assert!((r.a_score() - 65.0).abs() < 1e-9);
        assert!((r.a_win_rate() - 0.65).abs() < 1e-9);
        assert!(r.a_elo() > 0.0);
    }

    fn minimal_tournament_config() -> TournamentConfig {
        TournamentConfig {
            stockfish_path: "/usr/bin/stockfish".into(),
            stockfish_version: "Stockfish 18".into(),
            stockfish_hash_mb: 16,
            master_seed: 1,
            n_workers: 1,
            n_pairs: 1,
            max_ply: 64,
            opening_plies: 4,
            side_a: TournamentSide {
                name: "a".into(),
                sample_score: SampleScore::Cp,
                temperature: 0.0,
            },
            side_b: TournamentSide {
                name: "b".into(),
                sample_score: SampleScore::V,
                temperature: 0.0,
            },
            output_path: None,
        }
    }

    #[test]
    fn validate_rejects_opening_plies_at_or_above_max_ply() {
        let mut cfg = minimal_tournament_config();
        cfg.max_ply = 10;
        cfg.opening_plies = 10; // equal to max_ply
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("opening_plies"), "got: {err}");

        cfg.opening_plies = 20; // greater than max_ply
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("opening_plies"), "got: {err}");
    }

    #[test]
    fn validate_accepts_opening_plies_below_max_ply() {
        let cfg = minimal_tournament_config();
        // 4 < 64, the default minimal-config combo.
        cfg.validate().unwrap();
    }
}
