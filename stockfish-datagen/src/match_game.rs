//! One match game between two sampling configs.
//!
//! Mirrors `play_game` but parameterized by two `SideConfig`s instead of a
//! single tier. Each ply consults the side-to-move's sampler config (score
//! field + temperature) on the same engine output. Optional opening prefix
//! is replayed onto both shakmaty's board and the engine's position cache
//! before the play loop starts, so both color-swap games of a pair start
//! from byte-identical state.
//!
//! Drives the engine via the `evallegal` protocol (the patched binary's
//! per-legal-move NNUE eval). Multipv setting is irrelevant — evallegal
//! returns ALL legal moves regardless. The caller is responsible for
//! spawning the StockfishProcess with `GoBudget::EvalLegal`.

use rand::Rng;
use shakmaty::uci::UciMove;
use shakmaty::{Chess, Color, Position};

use crate::config::SampleScore;
use crate::game::{
    GameError, detect_pre_eval_terminal, should_strategic_claim_50mv, zobrist,
};
use crate::outcome::OutcomeReason;
use crate::sampler::softmax_sample;
use crate::stockfish::StockfishProcess;

/// Sampling identity for one side of a match.
#[derive(Debug, Clone)]
pub struct SideConfig {
    pub name: String,
    pub sample_score: SampleScore,
    pub temperature: f32,
}

/// Result of one match game.
#[derive(Debug, Clone)]
pub struct MatchOutcome {
    pub reason: OutcomeReason,
    /// `Some(color)` if that color won, `None` for any draw.
    pub winner: Option<Color>,
    /// Total plies actually played, including the opening prefix.
    pub n_plies: usize,
    /// Which color side `a` was assigned in this game. The pair-runner uses
    /// this to attribute wins to the right side regardless of which color
    /// the side happened to be in the current half of the pair.
    pub a_color: Color,
    /// Full move list from startpos (UCI), including the opening prefix.
    /// Useful for spot-checking + replay; cheap (one Vec per game).
    pub moves: Vec<String>,
}

/// Play one game between `side_a` and `side_b`, with `a_color` deciding
/// which color side_a takes. The opening prefix is a UCI move list applied
/// from the start position; pass `&[]` to play from scratch.
///
/// `rng` should be seeded from the per-game seed so the entire game is
/// reproducible from `(seed, openings_prefix, side_configs, max_ply)`.
pub fn play_match_game<R: Rng + ?Sized>(
    sf: &mut StockfishProcess,
    rng: &mut R,
    opening_moves: &[String],
    a_color: Color,
    side_a: &SideConfig,
    side_b: &SideConfig,
    max_ply: u32,
) -> Result<MatchOutcome, GameError> {
    sf.new_game()?;

    let mut board = Chess::default();
    let mut moves: Vec<String> = Vec::with_capacity(opening_moves.len() + 128);
    let mut history: Vec<u64> = Vec::with_capacity(opening_moves.len() + 128);
    history.push(zobrist(&board));

    // Replay opening moves onto both shakmaty board and engine position.
    // After this, `board.turn()` is the side actually to move from this
    // opening, and the engine's cached position string is in sync.
    for uci in opening_moves {
        let parsed: UciMove = uci
            .parse()
            .map_err(|_| GameError::IllegalMoveFromStockfish { uci: uci.clone() })?;
        let m = parsed
            .to_move(&board)
            .map_err(|_| GameError::IllegalMoveFromStockfish { uci: uci.clone() })?;
        board.play_unchecked(m);
        history.push(zobrist(&board));
        sf.play_move(uci);
        moves.push(uci.clone());
    }

    let total_max_ply = max_ply.saturating_add(opening_moves.len() as u32);

    for ply in (opening_moves.len() as u32)..total_max_ply {
        let cur_hash = *history.last().unwrap();
        if let Some(reason) = detect_pre_eval_terminal(&board, &history, cur_hash, ply as usize) {
            return Ok(finalize(reason, ply as usize, a_color, moves));
        }

        let res = sf.candidates_after_play_moves()?;
        if res.terminal.is_some() {
            // Pre-eval check above already covered checkmate/stalemate; if
            // evallegal still says terminal here it means our pre-eval
            // detector disagreed with the engine — treat as bug.
            return Err(GameError::UnexpectedNoneBestmove);
        }
        if res.candidates.is_empty() {
            return Err(GameError::NoCandidates);
        }

        // Strategic 50-move claim. Best score (mover-POV) drives the decision;
        // matches play_game's behavior so both protocols agree on when a
        // tournament game should end at the 50-move boundary.
        let best_score_cp = res
            .candidates
            .iter()
            .map(|c| c.score_cp)
            .fold(f32::NEG_INFINITY, f32::max);
        if should_strategic_claim_50mv(board.halfmoves(), best_score_cp) {
            return Ok(finalize(
                OutcomeReason::FiftyMoveRule,
                ply as usize,
                a_color,
                moves,
            ));
        }

        let mover = board.turn();
        let side = if mover == a_color { side_a } else { side_b };
        let pick = softmax_sample(&res.candidates, side.sample_score, side.temperature, rng)
            .ok_or(GameError::NoCandidates)?;

        let parsed: UciMove = pick
            .uci
            .parse()
            .map_err(|_| GameError::IllegalMoveFromStockfish { uci: pick.uci.clone() })?;
        let m = parsed
            .to_move(&board)
            .map_err(|_| GameError::IllegalMoveFromStockfish { uci: pick.uci.clone() })?;
        board.play_unchecked(m);
        history.push(zobrist(&board));
        sf.play_move(&pick.uci);
        moves.push(pick.uci.clone());
    }

    // Hit the ply cap. Re-check pre-eval terminals on the final position so
    // a natural mate / stalemate / 75-move / 3-fold landing exactly on the
    // last allowed move isn't mislabeled as PlyLimit. Mirrors play_game.
    let final_hash = *history.last().unwrap();
    let n_plies = moves.len();
    let outcome = detect_pre_eval_terminal(&board, &history, final_hash, n_plies)
        .unwrap_or(OutcomeReason::PlyLimit);
    Ok(finalize(outcome, n_plies, a_color, moves))
}

fn finalize(
    reason: OutcomeReason,
    n_plies: usize,
    a_color: Color,
    moves: Vec<String>,
) -> MatchOutcome {
    let winner = match reason {
        OutcomeReason::WhiteCheckmate => Some(Color::White),
        OutcomeReason::BlackCheckmate => Some(Color::Black),
        _ => None,
    };
    MatchOutcome { reason, winner, n_plies, a_color, moves }
}

/// Generate one opening as a UCI move list by sampling `n_plies` legal
/// moves uniformly at random from the start position. Deterministic given
/// the seeded RNG. Returns the move list (consumable by `play_match_game`).
///
/// Uses pure shakmaty — no engine round-trip — so opening generation is
/// fast and side-effect-free. The same opening is used for both halves of
/// a pair (color-swap), guaranteeing both games start from byte-identical
/// state.
pub fn generate_opening<R: Rng + ?Sized>(rng: &mut R, n_plies: u32) -> Vec<String> {
    use shakmaty::Position;
    let mut board = Chess::default();
    let mut moves = Vec::with_capacity(n_plies as usize);
    for _ in 0..n_plies {
        let legals = board.legal_moves();
        if legals.is_empty() {
            // Pathological: 0 legal moves before opening completes. Rare
            // (would need 2-ply mates from startpos, which don't exist),
            // but bail out gracefully rather than panic.
            break;
        }
        let idx = rng.gen_range(0..legals.len());
        let m = &legals[idx];
        let uci = m.to_uci(board.castles().mode()).to_string();
        board.play_unchecked(m.clone());
        moves.push(uci);
    }
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn opening_generation_is_deterministic() {
        let mut rng_a = ChaCha8Rng::seed_from_u64(42);
        let mut rng_b = ChaCha8Rng::seed_from_u64(42);
        let a = generate_opening(&mut rng_a, 4);
        let b = generate_opening(&mut rng_b, 4);
        assert_eq!(a, b);
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn opening_generation_varies_with_seed() {
        let mut rng_a = ChaCha8Rng::seed_from_u64(1);
        let mut rng_b = ChaCha8Rng::seed_from_u64(2);
        let a = generate_opening(&mut rng_a, 4);
        let b = generate_opening(&mut rng_b, 4);
        assert_ne!(a, b);
    }

    #[test]
    fn finalize_attributes_winner_correctly() {
        let moves = vec!["e2e4".into()];
        let o = finalize(OutcomeReason::WhiteCheckmate, 1, Color::White, moves.clone());
        assert_eq!(o.winner, Some(Color::White));
        let o = finalize(OutcomeReason::BlackCheckmate, 1, Color::Black, moves.clone());
        assert_eq!(o.winner, Some(Color::Black));
        let o = finalize(OutcomeReason::Stalemate, 1, Color::White, moves.clone());
        assert_eq!(o.winner, None);
        let o = finalize(OutcomeReason::FiftyMoveRule, 1, Color::White, moves);
        assert_eq!(o.winner, None);
    }
}

