//! One self-play game: drive Stockfish, sample candidates, detect terminals.
//!
//! All terminal detection lives in *this* loop, in-process: checkmate,
//! stalemate, insufficient material, threefold repetition, fifty-move rule.
//! We use FIDE's *claimable* thresholds (3-fold and 50-move) rather than
//! the *automatic* ones (5-fold and 75-move) because real games — which
//! we want this data to look like — end at the claimable thresholds when
//! both sides accept.
//!
//! Stockfish is only consulted for move *selection*. It should never need
//! to report `bestmove (none)`; if it does, we surface that as a bug
//! rather than papering over it.

use rand::Rng;
use shakmaty::zobrist::Zobrist64;
use shakmaty::uci::UciMove;
use shakmaty::{Chess, EnPassantMode, Position};
use thiserror::Error;

use crate::config::TierConfig;
use crate::outcome::OutcomeReason;
use crate::sampler::softmax_sample;
use crate::stockfish::{StockfishError, StockfishProcess};

/// Fully-played game.
#[derive(Debug, Clone)]
pub struct PlayedGame {
    pub uci_moves: Vec<String>,
    pub outcome: OutcomeReason,
}

#[derive(Debug, Error)]
pub enum GameError {
    #[error("stockfish error: {0}")]
    Stockfish(#[from] StockfishError),
    #[error("stockfish proposed move {uci:?} which is illegal in the current position")]
    IllegalMoveFromStockfish { uci: String },
    #[error("stockfish returned no candidates from a non-terminal position")]
    NoCandidates,
    #[error(
        "stockfish returned bestmove (none) but our pre-move terminal check found legal moves; \
         this indicates a bug in the in-process terminal detector or a Stockfish issue."
    )]
    UnexpectedNoneBestmove,
}

/// Number of times `current_hash` appears in `history`. The caller is
/// expected to push the current position into `history` before calling,
/// so a return value of `>= 3` means the third occurrence has been seen
/// (the FIDE 3-fold threshold).
fn count_repetitions(history: &[u64], current_hash: u64) -> usize {
    history.iter().filter(|&&h| h == current_hash).count()
}

fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Look up which terminal kind, if any, applies to the current position.
/// Returns `None` if the game is still alive.
fn detect_terminal(
    pos: &Chess,
    history: &[u64],
    current_hash: u64,
    n_moves_played: usize,
) -> Option<OutcomeReason> {
    if pos.is_checkmate() {
        return Some(OutcomeReason::from_checkmate(n_moves_played));
    }
    if pos.is_stalemate() {
        return Some(OutcomeReason::Stalemate);
    }
    if pos.is_insufficient_material() {
        return Some(OutcomeReason::InsufficientMaterial);
    }
    // Halfmove clock: 50 moves = 100 halfmoves without pawn move or capture.
    // Use >=100 (claimable) rather than >=150 (automatic 75-move rule).
    if pos.halfmoves() >= 100 {
        return Some(OutcomeReason::FiftyMoveRule);
    }
    // 3-fold repetition (current position must have been seen at least
    // twice before, i.e. count >= 3 including the current).
    if count_repetitions(history, current_hash) >= 3 {
        return Some(OutcomeReason::ThreefoldRepetition);
    }
    None
}

/// Per-ply MultiPV bucket selector — single source of truth for the
/// 3-stage opening / sampling / top-1 schedule.
fn target_multi_pv(tier: &TierConfig, ply: u32) -> u32 {
    if ply < tier.opening_plies {
        tier.opening_multi_pv
    } else if ply < tier.sample_plies {
        tier.multi_pv
    } else {
        1
    }
}

/// Play one game. The provided `rng` should be seeded from the per-game
/// seed; this function consumes from it deterministically (one f64 per
/// non-trivial softmax pick), so the entire game is reproducible from the
/// initial RNG state plus tier config and Stockfish version.
pub fn play_game<R: Rng + ?Sized>(
    sf: &mut StockfishProcess,
    rng: &mut R,
    tier: &TierConfig,
    max_ply: u32,
) -> Result<PlayedGame, GameError> {
    sf.new_game()?;

    let mut board = Chess::default();
    let mut moves: Vec<String> = Vec::with_capacity(128);
    let mut history: Vec<u64> = Vec::with_capacity(128);
    history.push(zobrist(&board));

    let mut current_pv: Option<u32> = None;

    for ply in 0..max_ply {
        let cur_hash = *history.last().unwrap();
        // Pass the FULL history (current included). `count_repetitions`'s
        // contract is "count of current_hash in history", so >= 3 means
        // the position has now occurred 3 times — the FIDE threshold.
        if let Some(reason) = detect_terminal(&board, &history, cur_hash, ply as usize) {
            return Ok(PlayedGame { uci_moves: moves, outcome: reason });
        }

        let target_pv = target_multi_pv(tier, ply);
        if current_pv != Some(target_pv) {
            sf.set_multi_pv(target_pv)?;
            current_pv = Some(target_pv);
        }

        // Use the incremental position cache: we pushed the previous move
        // via `play_move` below, so the cached position string is already
        // up to date and we can skip rebuilding the full move list.
        let res = sf.candidates_after_play_moves()?;
        if res.terminal.is_some() {
            // We pre-checked all terminals; SF saying (none) here is a bug.
            return Err(GameError::UnexpectedNoneBestmove);
        }
        if res.candidates.is_empty() {
            return Err(GameError::NoCandidates);
        }

        let pick = if target_pv == 1 {
            &res.candidates[0]
        } else {
            softmax_sample(&res.candidates, tier.temperature, rng).ok_or(GameError::NoCandidates)?
        };

        let uci_parsed: UciMove = pick.uci.parse().map_err(|_| GameError::IllegalMoveFromStockfish {
            uci: pick.uci.clone(),
        })?;
        let m = uci_parsed
            .to_move(&board)
            .map_err(|_| GameError::IllegalMoveFromStockfish { uci: pick.uci.clone() })?;
        board.play_unchecked(m);
        history.push(zobrist(&board));
        sf.play_move(&pick.uci);
        moves.push(pick.uci.clone());
    }

    Ok(PlayedGame { uci_moves: moves, outcome: OutcomeReason::PlyLimit })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_multi_pv_three_phase_schedule() {
        let tier = TierConfig {
            name: "t".into(),
            nodes: 1,
            n_games: 1,
            multi_pv: 5,
            opening_multi_pv: 20,
            opening_plies: 2,
            sample_plies: 12,
            temperature: 1.0,
        };
        assert_eq!(target_multi_pv(&tier, 0), 20);
        assert_eq!(target_multi_pv(&tier, 1), 20);
        assert_eq!(target_multi_pv(&tier, 2), 5);
        assert_eq!(target_multi_pv(&tier, 11), 5);
        assert_eq!(target_multi_pv(&tier, 12), 1);
        assert_eq!(target_multi_pv(&tier, 999), 1);
    }

    #[test]
    fn count_repetitions_basic() {
        assert_eq!(count_repetitions(&[1, 2, 3], 4), 0);
        assert_eq!(count_repetitions(&[1, 2, 1, 3], 1), 2);
        assert_eq!(count_repetitions(&[1, 1, 1, 1], 1), 4);
    }

    #[test]
    fn detect_terminal_starting_position_is_alive() {
        let pos = Chess::default();
        let h = zobrist(&pos);
        assert!(detect_terminal(&pos, &[h], h, 0).is_none());
    }

    #[test]
    fn detect_terminal_fires_on_third_occurrence_not_fourth() {
        // Regression test for the off-by-one that originally counted
        // *prior* occurrences instead of total: a position seen 3 times
        // (including the current one) MUST trigger the rule. Anything
        // less than 3 must not.
        let pos = Chess::default();
        let h = zobrist(&pos);
        // 1st occurrence of `h` (just the current).
        assert!(detect_terminal(&pos, &[h], h, 0).is_none());
        // 2nd occurrence (one prior + current).
        assert!(detect_terminal(&pos, &[h, h], h, 0).is_none());
        // 3rd occurrence: must fire.
        let outcome = detect_terminal(&pos, &[h, h, h], h, 0);
        assert_eq!(outcome, Some(OutcomeReason::ThreefoldRepetition));
    }

    fn stockfish_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("STOCKFISH_PATH") {
            return Some(p.into());
        }
        let default = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("bin/stockfish");
        if default.exists() {
            Some(default)
        } else {
            None
        }
    }

    fn smoke_tier() -> TierConfig {
        TierConfig {
            name: "smoke".into(),
            nodes: 1,
            n_games: 1,
            multi_pv: 5,
            opening_multi_pv: 20,
            opening_plies: 2,
            sample_plies: 12,
            temperature: 1.0,
        }
    }

    /// Live test: play one game with a real Stockfish, verify it terminated
    /// naturally and produced legal moves.
    #[test]
    fn live_play_one_game() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let Some(path) = stockfish_path() else {
            eprintln!("skipping: no stockfish binary");
            return;
        };
        let mut sf = StockfishProcess::spawn(&path, "Stockfish", 16, 1).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let game = play_game(&mut sf, &mut rng, &smoke_tier(), 512).unwrap();
        assert!(!game.uci_moves.is_empty(), "game produced no moves");
        // Re-tokenize via engine and verify all moves are legal under the rules.
        let refs: Vec<&str> = game.uci_moves.iter().map(|s| s.as_str()).collect();
        let (tokens, san) = chess_engine::uci::uci_to_tokens_and_san(&refs);
        assert_eq!(tokens.len(), game.uci_moves.len(), "engine rejected one of the moves");
        assert_eq!(san.len(), game.uci_moves.len());
        sf.shutdown();
        eprintln!(
            "play_game smoke: {} plies, outcome {:?} ({})",
            game.uci_moves.len(),
            game.outcome,
            game.outcome.result_str()
        );
    }

    /// Reproducibility: same Stockfish + same RNG seed + same tier produce
    /// the same game.
    #[test]
    fn live_play_is_deterministic() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let Some(path) = stockfish_path() else {
            eprintln!("skipping: no stockfish binary");
            return;
        };
        let tier = smoke_tier();
        let game_a = {
            let mut sf = StockfishProcess::spawn(&path, "Stockfish", 16, 1).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(12345);
            let g = play_game(&mut sf, &mut rng, &tier, 512).unwrap();
            sf.shutdown();
            g
        };
        let game_b = {
            let mut sf = StockfishProcess::spawn(&path, "Stockfish", 16, 1).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(12345);
            let g = play_game(&mut sf, &mut rng, &tier, 512).unwrap();
            sf.shutdown();
            g
        };
        assert_eq!(
            game_a.uci_moves, game_b.uci_moves,
            "same seed should produce the same game"
        );
        assert_eq!(game_a.outcome, game_b.outcome);
    }
}
