/// Termination reasons for a chess game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Termination {
    Checkmate = 0,
    Stalemate = 1,
    SeventyFiveMoveRule = 2,
    FivefoldRepetition = 3,
    InsufficientMaterial = 4,
    PlyLimit = 5,
}

impl Termination {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Promotion piece types in our vocabulary order: q=0, r=1, b=2, n=3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PromoType {
    Queen = 0,
    Rook = 1,
    Bishop = 2,
    Knight = 3,
}

impl PromoType {
    pub fn from_index(i: u8) -> Option<Self> {
        match i {
            0 => Some(Self::Queen),
            1 => Some(Self::Rook),
            2 => Some(Self::Bishop),
            3 => Some(Self::Knight),
            _ => None,
        }
    }
}
