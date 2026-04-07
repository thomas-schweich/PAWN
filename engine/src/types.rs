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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_termination_as_u8_values() {
        // Must match the comment in PyGameState.check_termination (0-5)
        assert_eq!(Termination::Checkmate.as_u8(), 0);
        assert_eq!(Termination::Stalemate.as_u8(), 1);
        assert_eq!(Termination::SeventyFiveMoveRule.as_u8(), 2);
        assert_eq!(Termination::FivefoldRepetition.as_u8(), 3);
        assert_eq!(Termination::InsufficientMaterial.as_u8(), 4);
        assert_eq!(Termination::PlyLimit.as_u8(), 5);
    }

    #[test]
    fn test_termination_discriminants_stable() {
        // Verify #[repr(u8)] assignments via explicit cast
        assert_eq!(Termination::Checkmate as u8, 0);
        assert_eq!(Termination::PlyLimit as u8, 5);
    }

    #[test]
    fn test_termination_equality_and_clone() {
        let a = Termination::Checkmate;
        let b = a;
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_ne!(Termination::Checkmate, Termination::Stalemate);
    }

    #[test]
    fn test_promo_type_from_index_valid() {
        assert_eq!(PromoType::from_index(0), Some(PromoType::Queen));
        assert_eq!(PromoType::from_index(1), Some(PromoType::Rook));
        assert_eq!(PromoType::from_index(2), Some(PromoType::Bishop));
        assert_eq!(PromoType::from_index(3), Some(PromoType::Knight));
    }

    #[test]
    fn test_promo_type_from_index_invalid() {
        assert_eq!(PromoType::from_index(4), None);
        assert_eq!(PromoType::from_index(99), None);
        assert_eq!(PromoType::from_index(u8::MAX), None);
    }

    #[test]
    fn test_promo_type_discriminants() {
        // Must match vocab's q=0, r=1, b=2, n=3 order
        assert_eq!(PromoType::Queen as u8, 0);
        assert_eq!(PromoType::Rook as u8, 1);
        assert_eq!(PromoType::Bishop as u8, 2);
        assert_eq!(PromoType::Knight as u8, 3);
    }

    #[test]
    fn test_promo_type_roundtrip() {
        for i in 0..4u8 {
            let pt = PromoType::from_index(i).unwrap();
            assert_eq!(pt as u8, i);
        }
    }

    #[test]
    fn test_promo_type_equality_and_copy() {
        let a = PromoType::Queen;
        let b = a;  // Copy
        assert_eq!(a, b);
        assert_ne!(PromoType::Queen, PromoType::Rook);
    }
}
