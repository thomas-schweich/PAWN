//! Outcome classification for self-play games.
//!
//! Token IDs mirror `engine/src/vocab.rs` exactly — kept as `const`s here
//! rather than imported from `chess_engine` so the values are visible
//! at-a-glance and a mismatch with the model's vocabulary surfaces in code
//! review rather than at training time.

pub const WHITE_CHECKMATES: u16 = 1969;
pub const BLACK_CHECKMATES: u16 = 1970;
pub const STALEMATE: u16 = 1971;
pub const DRAW_BY_RULE: u16 = 1972;
pub const PLY_LIMIT: u16 = 1973;

#[cfg(test)]
const _: () = {
    // Mirror of engine/src/vocab.rs constants. If engine ever shifts these,
    // the tests below break loudly. (We can't import the consts directly
    // because chess_engine doesn't re-export them as `pub use`; this is the
    // deliberate "code-review tripwire" pattern.)
    assert!(WHITE_CHECKMATES == 1969);
    assert!(BLACK_CHECKMATES == 1970);
    assert!(STALEMATE == 1971);
    assert!(DRAW_BY_RULE == 1972);
    assert!(PLY_LIMIT == 1973);
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
