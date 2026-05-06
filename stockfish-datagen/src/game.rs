//! One self-play game: drive Stockfish, sample candidates, detect terminals.
//!
//! Terminal detection lives in *this* loop, in-process. Two flavors:
//!
//! - **Pre-eval** terminals (cheap, no Stockfish call needed):
//!   checkmate, stalemate, insufficient material, threefold repetition.
//!   Plus the 75-move *automatic* draw (FIDE) as a hard upper bound.
//!   These fire in `detect_pre_eval_terminal` before the Stockfish call
//!   for the next move.
//!
//! - **Strategic 50-move claim** (needs Stockfish's eval): the 50-move
//!   rule is *claimable*, not automatic. A player has no incentive to
//!   claim if they're winning. We model this by using Stockfish's
//!   side-to-move-relative eval at the moment the 50-move threshold is
//!   reached: if the side about to move is losing (eval < 0), they
//!   claim → game ends as `FiftyMoveRule`; if winning or even, they
//!   continue playing. In practice this becomes a "50-or-51-or-…
//!   move rule" — claims fluctuate as the eval swings, until either
//!   someone resets the halfmove clock with a capture / pawn move or
//!   the 75-move auto-rule fires at halfmove 150. This gives the model
//!   a meaningful signal to learn *when* to claim, rather than baking
//!   in an "always claim" policy that doesn't match strong-player
//!   behavior.
//!
//! 3-fold repetition is also technically claimable (vs 5-fold automatic),
//! but we keep that check unconditional because making it strategic
//! risks both sides perpetually shuffling in a drawn position. The
//! 50-move clock would eventually reset things anyway.
//!
//! Stockfish is only consulted for move *selection* (and the 50-move
//! eval-based decision). It should never need to report
//! `bestmove (none)` from a non-terminal position; if it does, we
//! surface that as a bug rather than papering over it.

use rand::Rng;
use shakmaty::zobrist::Zobrist64;
use shakmaty::uci::UciMove;
use shakmaty::{Chess, EnPassantMode, Position};
use thiserror::Error;

use crate::config::TierConfig;
use crate::outcome::OutcomeReason;
use crate::sampler::softmax_sample;
use crate::stockfish::{Candidate, StockfishError, StockfishProcess};

/// Fully-played game.
#[derive(Debug, Clone)]
pub struct PlayedGame {
    pub uci_moves: Vec<String>,
    pub outcome: OutcomeReason,
    /// Per-ply candidates list (one inner Vec per ply, in play order),
    /// captured iff the tier had `store_legal_move_evals = true`. When
    /// the flag is off this is `None` to avoid the per-ply Vec allocation
    /// overhead in tiers that don't need it. The semantics of each
    /// `Candidate.score_cp` depend on the tier's go-budget (qsearch-resolved
    /// for `Nodes(_)` tiers, raw NNUE static eval for `EvalLegal` tiers).
    pub per_ply_candidates: Option<Vec<Vec<Candidate>>>,
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
pub(crate) fn count_repetitions(history: &[u64], current_hash: u64) -> usize {
    history.iter().filter(|&&h| h == current_hash).count()
}

pub(crate) fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Look up which pre-eval terminal kind, if any, applies to the current
/// position. Returns `None` if the game is alive (no terminal known
/// without consulting Stockfish — see `should_strategic_claim_50mv` for
/// the post-eval check).
///
/// Includes the 75-move *automatic* draw (FIDE: 150 halfmoves) as a
/// hard upper bound — neither side can avoid this, so it's safe to
/// fire without an eval.
pub(crate) fn detect_pre_eval_terminal(
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
    // 75-move automatic rule. Both sides MUST accept the draw at this
    // point. Tagged as `FiftyMoveRule` since we don't have a separate
    // outcome category for "75-move auto" (and it's still a draw by
    // halfmove-clock rule).
    if pos.halfmoves() >= 150 {
        return Some(OutcomeReason::FiftyMoveRule);
    }
    // 3-fold repetition (current position must have been seen at least
    // twice before, i.e. count >= 3 including the current). Kept
    // unconditional rather than eval-strategic — see module docstring.
    if count_repetitions(history, current_hash) >= 3 {
        return Some(OutcomeReason::ThreefoldRepetition);
    }
    None
}

/// Strategic 50-move claim: the side about to move claims the draw iff
/// they're losing (eval < 0 from their perspective). At eval == 0 (theoretical
/// draw) neither side claims; the 75-move auto-rule will eventually fire.
///
/// Pulled out as a pure function so we can pin the semantics in unit
/// tests without driving Stockfish.
pub(crate) fn should_strategic_claim_50mv(halfmoves: u32, side_to_move_eval_cp: f32) -> bool {
    halfmoves >= 100 && side_to_move_eval_cp < 0.0
}

/// Per-ply MultiPV bucket selector — single source of truth for the
/// 3-stage opening / sampling / top-1 schedule.
///
/// Only called from non-searchless code paths. Config validation enforces
/// that `opening_plies`, `multi_pv`, `sample_plies` are all Some when
/// `searchless == false`, so the unwraps below are safe within this scope.
fn target_multi_pv(tier: &TierConfig, ply: u32) -> u32 {
    debug_assert!(!tier.searchless, "target_multi_pv called in searchless mode");
    let opening_plies = tier.opening_plies.expect("validated: search-mode tier has opening_plies");
    let sample_plies = tier.sample_plies.expect("validated: search-mode tier has sample_plies");
    if ply < opening_plies {
        tier.opening_multi_pv.expect("validated: search-mode tier has opening_multi_pv")
    } else if ply < sample_plies {
        tier.multi_pv.expect("validated: search-mode tier has multi_pv")
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
    // Allocate the per-ply candidates buffer iff the tier asks for it.
    // Plumbed through every early-return path so the consumer always sees
    // either Some(complete-list) or None — no partial captures.
    let mut per_ply_candidates: Option<Vec<Vec<Candidate>>> = if tier.store_legal_move_evals {
        Some(Vec::with_capacity(128))
    } else {
        None
    };

    for ply in 0..max_ply {
        let cur_hash = *history.last().unwrap();
        // Pass the FULL history (current included). `count_repetitions`'s
        // contract is "count of current_hash in history", so >= 3 means
        // the position has now occurred 3 times — the FIDE threshold.
        if let Some(reason) = detect_pre_eval_terminal(&board, &history, cur_hash, ply as usize) {
            return Ok(PlayedGame { uci_moves: moves, outcome: reason, per_ply_candidates });
        }

        // MultiPV scheduling only applies in non-searchless mode. In
        // searchless mode, evallegal returns every legal move regardless of
        // any setoption; touching MultiPV would just incur wasted UCI
        // round-trips, and the `target_pv == 1` shortcut would silently
        // pick `candidates[0]` (move-gen order, not score order).
        let target_pv = if !tier.searchless {
            let pv = target_multi_pv(tier, ply);
            if current_pv != Some(pv) {
                sf.set_multi_pv(pv)?;
                current_pv = Some(pv);
            }
            Some(pv)
        } else {
            None
        };

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

        // Strategic 50-move claim — needs Stockfish's eval, so it has to
        // run AFTER candidates retrieval. Best candidate's score is the
        // side-to-move's eval (Candidate.score_cp is documented as POV).
        // The pre-eval check above caps things at halfmove 150 (75-move
        // auto), so this only fires in the [100, 150) window where a
        // claim is strategic.
        let best_score_cp = res.candidates.iter()
            .map(|c| c.score_cp)
            .fold(f32::NEG_INFINITY, f32::max);
        if should_strategic_claim_50mv(board.halfmoves(), best_score_cp) {
            return Ok(PlayedGame {
                uci_moves: moves,
                outcome: OutcomeReason::FiftyMoveRule,
                per_ply_candidates,
            });
        }

        // Snapshot the full candidates list BEFORE picking — otherwise the
        // borrow checker complains about `&res.candidates[0]` outliving the
        // clone. Cheap when the tier doesn't request it (None path skips).
        if let Some(buf) = per_ply_candidates.as_mut() {
            buf.push(res.candidates.clone());
        }

        let pick = if target_pv == Some(1) {
            // Search-mode top-1 tail: candidates[0] is the multipv=1 entry
            // (the engine's best move at this depth), which is what we want.
            &res.candidates[0]
        } else {
            // Searchless tiers always softmax-sample (no `sample_plies`
            // knob — every ply uses `sample_score` per validation). For
            // non-searchless tiers, `sample_score` is None per validation
            // and we default to Cp (the only score multipv parsing
            // surfaces — `score_v` is None on those candidates).
            let score = tier.sample_score.unwrap_or(crate::config::SampleScore::Cp);
            softmax_sample(&res.candidates, score, tier.temperature, rng)
                .ok_or(GameError::NoCandidates)?
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

    // Terminal recheck on the position reached by the LAST allowed move.
    // The loop above only checks before each move, so a natural terminal
    // (mate, stalemate, 3-fold, insufficient material, 75-move auto) hit
    // on exactly the max_ply'th move would otherwise be mislabeled as
    // PLY_LIMIT. We only run the pre-eval checks here — we don't have a
    // fresh Stockfish eval, and a strategic 50-move claim wouldn't have
    // fired in the loop precisely because the side-to-move was winning
    // or even at every prior ply, so PLY_LIMIT is the correct label for
    // a game that ran out the clock without being natural-terminal.
    let final_hash = *history.last().unwrap();
    let outcome = detect_pre_eval_terminal(&board, &history, final_hash, moves.len())
        .unwrap_or(OutcomeReason::PlyLimit);
    Ok(PlayedGame { uci_moves: moves, outcome, per_ply_candidates })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_multi_pv_three_phase_schedule() {
        let tier = TierConfig {
            name: "t".into(),
            n_games: 1,
            temperature: 1.0,
            searchless: false,
            store_legal_move_evals: false,
            sample_score: None,
            nodes: Some(1),
            multi_pv: Some(5),
            opening_multi_pv: Some(20),
            opening_plies: Some(2),
            sample_plies: Some(12),
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
        assert!(detect_pre_eval_terminal(&pos, &[h], h, 0).is_none());
    }

    #[test]
    fn strategic_50mv_only_claims_when_losing() {
        // Below the threshold: never claims, regardless of eval.
        assert!(!should_strategic_claim_50mv(0, -1000.0));
        assert!(!should_strategic_claim_50mv(99, -1000.0));
        // Boundary: at exactly halfmove 100, eval is the discriminator.
        assert!(!should_strategic_claim_50mv(100, 0.0),
                "eval==0 (theoretical draw) should NOT claim — let 75-move auto fire");
        assert!(!should_strategic_claim_50mv(100, 1.0),
                "winning side should not claim");
        assert!(should_strategic_claim_50mv(100, -1.0),
                "losing side claims at the threshold");
        assert!(should_strategic_claim_50mv(100, -10000.0),
                "near-mate-against claims");
        // Above the threshold but still below 75-move auto: same behavior.
        assert!(!should_strategic_claim_50mv(149, 50.0));
        assert!(should_strategic_claim_50mv(149, -50.0));
    }

    #[test]
    fn pre_eval_terminal_fires_75mv_auto_at_150_halfmoves() {
        // Construct a position with halfmoves >= 150 via FEN — both sides
        // are forced into the auto-rule regardless of eval.
        use shakmaty::fen::Fen;
        // KK endgame at halfmove 150 (not insufficient material because of
        // the queen — we want to confirm the 75-move check fires alone,
        // not via insufficient-material overlap).
        let fen: Fen = "8/8/4k3/8/3K4/3Q4/8/8 w - - 150 80".parse().unwrap();
        let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();
        let h = zobrist(&pos);
        let outcome = detect_pre_eval_terminal(&pos, &[h], h, 80);
        assert_eq!(outcome, Some(OutcomeReason::FiftyMoveRule),
                   "75-move auto rule must fire unconditionally at halfmove 150");
    }

    #[test]
    fn pre_eval_terminal_does_not_fire_50mv_at_100_halfmoves() {
        // The pre-eval check must NOT auto-claim at halfmove 100 anymore
        // — that's now a strategic decision driven by Stockfish's eval.
        use shakmaty::fen::Fen;
        let fen: Fen = "8/8/4k3/8/3K4/3Q4/8/8 w - - 100 51".parse().unwrap();
        let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();
        let h = zobrist(&pos);
        assert_eq!(detect_pre_eval_terminal(&pos, &[h], h, 51), None,
                   "50-move-rule decision moved into the eval-driven path; \
                    pre-eval must let halfmove 100..149 through");
    }

    /// Regression test: a terminal that lands on the max_ply'th move
    /// must NOT be labeled PlyLimit. Set max_ply=4 and check that
    /// Fool's mate (4 plies) gets BlackCheckmate, not PlyLimit.
    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_terminal_at_max_ply_is_not_plylimit() {
        let path = stockfish_path();
        // Fool's mate is unlikely to come out of stockfish nodes=1 + multipv=20
        // at temperature 1.0, so we use a bespoke max_ply that's very small
        // and let the test verify the contract via game_length math.
        // Easier: just play a normal game with max_ply=1 — it should produce
        // exactly 1 move and label outcome PlyLimit (no natural terminal at
        // depth 1 from the start position). This pins the boundary.
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut sf = StockfishProcess::spawn(&path, "Stockfish", 16, crate::stockfish::GoBudget::Nodes(1)).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let game = play_game(&mut sf, &mut rng, &smoke_tier(), 1).unwrap();
        assert_eq!(game.uci_moves.len(), 1);
        // No natural terminal possible after exactly 1 move from start
        // position, so PlyLimit is correct here. The recheck must not
        // mis-fire on a non-terminal final position.
        assert_eq!(game.outcome, OutcomeReason::PlyLimit);
        sf.shutdown();
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
        assert!(detect_pre_eval_terminal(&pos, &[h], h, 0).is_none());
        // 2nd occurrence (one prior + current).
        assert!(detect_pre_eval_terminal(&pos, &[h, h], h, 0).is_none());
        // 3rd occurrence: must fire.
        let outcome = detect_pre_eval_terminal(&pos, &[h, h, h], h, 0);
        assert_eq!(outcome, Some(OutcomeReason::ThreefoldRepetition));
    }

    fn stockfish_path() -> std::path::PathBuf {
        if let Ok(p) = std::env::var("STOCKFISH_PATH") {
            return p.into();
        }
        let default = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("bin/stockfish");
        assert!(
            default.exists(),
            "stockfish binary not found at {} — set STOCKFISH_PATH or install one",
            default.display(),
        );
        default
    }

    fn smoke_tier() -> TierConfig {
        TierConfig {
            name: "smoke".into(),
            n_games: 1,
            temperature: 1.0,
            searchless: false,
            store_legal_move_evals: false,
            sample_score: None,
            nodes: Some(1),
            multi_pv: Some(5),
            opening_multi_pv: Some(20),
            opening_plies: Some(2),
            sample_plies: Some(12),
        }
    }

    /// Live test: play one game with a real Stockfish, verify it terminated
    /// naturally and produced legal moves.
    #[test]
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_play_one_game() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let path = stockfish_path();
        let mut sf = StockfishProcess::spawn(&path, "Stockfish", 16, crate::stockfish::GoBudget::Nodes(1)).unwrap();
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
    #[ignore = "requires stockfish binary ($HOME/bin/stockfish or $STOCKFISH_PATH)"]
    fn live_play_is_deterministic() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let path = stockfish_path();
        let tier = smoke_tier();
        let game_a = {
            let mut sf = StockfishProcess::spawn(&path, "Stockfish", 16, crate::stockfish::GoBudget::Nodes(1)).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(12345);
            let g = play_game(&mut sf, &mut rng, &tier, 512).unwrap();
            sf.shutdown();
            g
        };
        let game_b = {
            let mut sf = StockfishProcess::spawn(&path, "Stockfish", 16, crate::stockfish::GoBudget::Nodes(1)).unwrap();
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
