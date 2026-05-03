//! Outcome classification for self-play games.
//!
//! Token IDs are imported directly from `chess_engine::vocab` — single
//! source of truth, no drift risk. The constants are re-exported below
//! so callers don't have to depend on chess_engine's module path.

pub use chess_engine::vocab::{
    BLACK_CHECKMATES, DRAW_BY_RULE, PLY_LIMIT, STALEMATE, WHITE_CHECKMATES,
};

/// Why a game ended. Maps to `outcome_token` and `result` columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeReason {
    /// White delivered checkmate (the move whose ply count is odd).
    WhiteCheckmate,
    /// Black delivered checkmate (the move whose ply count is even).
    BlackCheckmate,
    /// Side-to-move has no legal moves and is not in check.
    Stalemate,
    /// K vs K, K+B vs K, K+N vs K, etc.
    InsufficientMaterial,
    /// Same position has occurred 3+ times. Claimable under FIDE; we treat
    /// it as automatic for self-play.
    ThreefoldRepetition,
    /// 50 full moves (100 plies) without pawn move or capture. Same.
    FiftyMoveRule,
    /// Hit the per-game ply cap before any natural termination. Should be
    /// rare with the other detectors active.
    PlyLimit,
}

impl OutcomeReason {
    pub fn token(self) -> u16 {
        match self {
            Self::WhiteCheckmate => WHITE_CHECKMATES,
            Self::BlackCheckmate => BLACK_CHECKMATES,
            Self::Stalemate => STALEMATE,
            Self::InsufficientMaterial
            | Self::ThreefoldRepetition
            | Self::FiftyMoveRule => DRAW_BY_RULE,
            Self::PlyLimit => PLY_LIMIT,
        }
    }

    pub fn result_str(self) -> &'static str {
        match self {
            Self::WhiteCheckmate => "1-0",
            Self::BlackCheckmate => "0-1",
            _ => "1/2-1/2",
        }
    }

    /// Pick the right side from a checkmate detected after `n_moves_played`
    /// plies. Convention: ply 1 is white's first move; if `n` is odd, white
    /// just played the mating move.
    pub fn from_checkmate(n_moves_played: usize) -> Self {
        if n_moves_played % 2 == 1 {
            Self::WhiteCheckmate
        } else {
            Self::BlackCheckmate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_mapping_covers_all_variants() {
        assert_eq!(OutcomeReason::WhiteCheckmate.token(), WHITE_CHECKMATES);
        assert_eq!(OutcomeReason::BlackCheckmate.token(), BLACK_CHECKMATES);
        assert_eq!(OutcomeReason::Stalemate.token(), STALEMATE);
        assert_eq!(OutcomeReason::InsufficientMaterial.token(), DRAW_BY_RULE);
        assert_eq!(OutcomeReason::ThreefoldRepetition.token(), DRAW_BY_RULE);
        assert_eq!(OutcomeReason::FiftyMoveRule.token(), DRAW_BY_RULE);
        assert_eq!(OutcomeReason::PlyLimit.token(), PLY_LIMIT);
    }

    #[test]
    fn result_str_matches_chess_convention() {
        assert_eq!(OutcomeReason::WhiteCheckmate.result_str(), "1-0");
        assert_eq!(OutcomeReason::BlackCheckmate.result_str(), "0-1");
        assert_eq!(OutcomeReason::Stalemate.result_str(), "1/2-1/2");
        assert_eq!(OutcomeReason::FiftyMoveRule.result_str(), "1/2-1/2");
    }

    #[test]
    fn from_checkmate_uses_ply_parity() {
        // Fool's mate: 4 moves total; black mates → BlackCheckmate.
        assert_eq!(OutcomeReason::from_checkmate(4), OutcomeReason::BlackCheckmate);
        // Scholar's mate: 7 moves; white mates → WhiteCheckmate.
        assert_eq!(OutcomeReason::from_checkmate(7), OutcomeReason::WhiteCheckmate);
    }
}
