//! Game state wrapper around shakmaty, with our vocabulary mapping.

use rand::Rng;
use shakmaty::{
    Chess, Color, EnPassantMode, Move, MoveList, Piece, Position, Role, Square,
};
use shakmaty::fen::Fen;

use crate::types::Termination;
use crate::vocab;

/// Convert our square index (file-major: a1=0, b1=1, ..., h8=63) to shakmaty Square.
#[inline]
pub fn our_sq_to_shakmaty(sq: u8) -> Square {
    // Our indexing: file = sq % 8, rank = sq / 8
    // shakmaty Square::new(file, rank) expects File and Rank enums
    // but Square also has from_coords(file, rank)
    let file = sq % 8;
    let rank = sq / 8;
    Square::from_coords(
        shakmaty::File::new(file as u32),
        shakmaty::Rank::new(rank as u32),
    )
}

/// Convert shakmaty Square to our square index.
#[inline]
pub fn shakmaty_sq_to_ours(sq: Square) -> u8 {
    let file = sq.file() as u8;
    let rank = sq.rank() as u8;
    rank * 8 + file
}

/// Convert a shakmaty Move to our token index.
pub fn move_to_token(m: &Move) -> u16 {
    let (src, dst) = match m {
        Move::Normal { from, to, .. } => (*from, *to),
        Move::EnPassant { from, to } => (*from, *to),
        Move::Castle { king, rook } => {
            // UCI king-movement notation
            let king_sq = *king;
            let rook_sq = *rook;
            let dst = if rook_sq.file() > king_sq.file() {
                // Kingside: king goes to g-file
                Square::from_coords(shakmaty::File::G, king_sq.rank())
            } else {
                // Queenside: king goes to c-file
                Square::from_coords(shakmaty::File::C, king_sq.rank())
            };
            (king_sq, dst)
        }
        Move::Put { .. } => panic!("Put moves not supported in standard chess"),
    };

    let src_idx = shakmaty_sq_to_ours(src);
    let dst_idx = shakmaty_sq_to_ours(dst);

    // Check if this is a promotion
    if let Move::Normal { promotion: Some(role), .. } = m {
        let promo_type = match role {
            Role::Queen => 0,
            Role::Rook => 1,
            Role::Bishop => 2,
            Role::Knight => 3,
            _ => panic!("Invalid promotion role: {:?}", role),
        };
        vocab::promo_token(src_idx, dst_idx, promo_type)
            .expect("Promotion move should have a valid promo pair")
    } else {
        vocab::base_grid_token(src_idx, dst_idx)
    }
}

/// Convert our token index to a shakmaty Move, given the current position.
/// Finds the legal move matching the token's (src, dst, promo) decomposition.
pub fn token_to_move(pos: &Chess, token: u16) -> Option<Move> {
    // Validate the token is decomposable (not PAD/EOG)
    vocab::decompose_token(token)?;
    let legal = pos.legal_moves();

    for m in &legal {
        if move_to_token(m) == token {
            return Some(m.clone());
        }
    }

    None
}

/// Piece encoding for board state extraction.
/// 0=empty, 1-6=white P/N/B/R/Q/K, 7-12=black P/N/B/R/Q/K
pub fn piece_to_code(piece: Option<Piece>) -> i8 {
    match piece {
        None => 0,
        Some(p) => {
            let base = match p.role {
                Role::Pawn => 1,
                Role::Knight => 2,
                Role::Bishop => 3,
                Role::Rook => 4,
                Role::Queen => 5,
                Role::King => 6,
            };
            if p.color == Color::White { base } else { base + 6 }
        }
    }
}

/// Full game state for replaying and analysis.
#[derive(Clone)]
pub struct GameState {
    pos: Chess,
    move_history: Vec<u16>,  // tokens
    position_hashes: Vec<u64>,
    halfmove_clock: u32,
}

impl GameState {
    pub fn new() -> Self {
        let pos = Chess::default();
        let hash = Self::position_hash(&pos);
        Self {
            pos,
            move_history: Vec::new(),
            position_hashes: vec![hash],
            halfmove_clock: 0,
        }
    }

    /// Simple position hash for repetition detection.
    /// Uses the board layout + castling rights + ep square + side to move.
    fn position_hash(pos: &Chess) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // Hash piece placement
        for sq in Square::ALL {
            let piece = pos.board().piece_at(sq);
            piece.hash(&mut hasher);
        }

        // Hash side to move
        pos.turn().hash(&mut hasher);

        // Hash castling rights
        pos.castles().castling_rights().hash(&mut hasher);

        // Hash en passant square
        // shakmaty's legal_ep_square accounts for actual EP capture availability
        pos.legal_ep_square().hash(&mut hasher);

        hasher.finish()
    }

    pub fn position(&self) -> &Chess {
        &self.pos
    }

    pub fn turn(&self) -> Color {
        self.pos.turn()
    }

    pub fn is_white_to_move(&self) -> bool {
        self.pos.turn() == Color::White
    }

    pub fn ply(&self) -> usize {
        self.move_history.len()
    }

    pub fn move_history(&self) -> &[u16] {
        &self.move_history
    }

    pub fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }

    /// Get legal moves as token indices.
    pub fn legal_move_tokens(&self) -> Vec<u16> {
        let legal = self.pos.legal_moves();
        legal.iter().map(|m| move_to_token(m)).collect()
    }

    /// Get legal moves as shakmaty Move objects.
    pub fn legal_moves(&self) -> MoveList {
        self.pos.legal_moves()
    }

    /// Apply a move given as a token index.
    pub fn make_move(&mut self, token: u16) -> Result<(), String> {
        let m = token_to_move(&self.pos, token)
            .ok_or_else(|| format!("Token {} is not a legal move at ply {}", token, self.ply()))?;

        // Update halfmove clock
        let is_pawn = match &m {
            Move::Normal { role, .. } => *role == Role::Pawn,
            Move::EnPassant { .. } => true,
            Move::Castle { .. } => false,
            Move::Put { .. } => false,
        };
        let is_capture = m.is_capture();

        if is_pawn || is_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        self.pos.play_unchecked(m);
        self.move_history.push(token);
        let hash = Self::position_hash(&self.pos);
        self.position_hashes.push(hash);

        Ok(())
    }

    /// Check if the game is over. Returns the termination reason if so.
    pub fn check_termination(&self, max_ply: usize) -> Option<Termination> {
        let legal = self.pos.legal_moves();

        // Check terminal states (checkmate/stalemate) before ply limit so
        // that games ending in checkmate on the final ply get the correct
        // termination code rather than PlyLimit.
        if legal.is_empty() {
            if self.pos.is_check() {
                return Some(Termination::Checkmate);
            } else {
                return Some(Termination::Stalemate);
            }
        }

        if self.ply() >= max_ply {
            return Some(Termination::PlyLimit);
        }

        // 75-move rule: 150 halfmoves without capture or pawn push
        if self.halfmove_clock >= 150 {
            return Some(Termination::SeventyFiveMoveRule);
        }

        // Fivefold repetition
        if self.is_fivefold_repetition() {
            return Some(Termination::FivefoldRepetition);
        }

        // Insufficient material
        if self.pos.is_insufficient_material() {
            return Some(Termination::InsufficientMaterial);
        }

        None
    }

    pub fn is_fivefold_repetition(&self) -> bool {
        let current = self.position_hashes.last().unwrap();
        let count = self.position_hashes.iter().filter(|h| *h == current).count();
        count >= 5
    }

    /// Compute the legal move grid: [u64; 64] where bit d of grid[s] is set
    /// if a move from square s to square d is legal.
    pub fn legal_move_grid(&self) -> [u64; 64] {
        let mut grid = [0u64; 64];
        let legal = self.pos.legal_moves();

        for m in &legal {
            let token = move_to_token(m);
            if let Some((src, dst, _promo)) = vocab::decompose_token(token) {
                grid[src as usize] |= 1u64 << dst;
            }
        }

        grid
    }

    /// Compute the promotion mask: [[bool; 4]; 44] where mask[pair_idx][promo_type]
    /// is true if that specific promotion is legal.
    pub fn legal_promo_mask(&self) -> [[bool; 4]; 44] {
        let mut mask = [[false; 4]; 44];
        let legal = self.pos.legal_moves();

        for m in &legal {
            if let Move::Normal { from, to, promotion: Some(role), .. } = m {
                let src = shakmaty_sq_to_ours(*from);
                let dst = shakmaty_sq_to_ours(*to);
                if let Some(pair_idx) = vocab::promo_pair_index(src, dst) {
                    let promo_type = match role {
                        Role::Queen => 0,
                        Role::Rook => 1,
                        Role::Bishop => 2,
                        Role::Knight => 3,
                        _ => continue,
                    };
                    mask[pair_idx][promo_type] = true;
                }
            }
        }

        mask
    }

    /// Extract board state for probing.
    pub fn board_array(&self) -> [[i8; 8]; 8] {
        let mut board = [[0i8; 8]; 8];
        for rank in 0..8 {
            for file in 0..8 {
                let sq = Square::from_coords(
                    shakmaty::File::new(file as u32),
                    shakmaty::Rank::new(rank as u32),
                );
                board[rank][file] = piece_to_code(self.pos.board().piece_at(sq));
            }
        }
        board
    }

    /// Get castling rights as a 4-bit field: bit 0=K, 1=Q, 2=k, 3=q.
    pub fn castling_rights_bits(&self) -> u8 {
        let rights = self.pos.castles().castling_rights();
        let mut bits = 0u8;
        if rights.contains(Square::H1) { bits |= 1; }  // White kingside
        if rights.contains(Square::A1) { bits |= 2; }  // White queenside
        if rights.contains(Square::H8) { bits |= 4; }  // Black kingside
        if rights.contains(Square::A8) { bits |= 8; }  // Black queenside
        bits
    }

    /// Get en passant square as our index (0-63), or -1 if none.
    pub fn ep_square(&self) -> i8 {
        match self.pos.legal_ep_square() {
            Some(sq) => shakmaty_sq_to_ours(sq) as i8,
            None => -1,
        }
    }

    pub fn is_check(&self) -> bool {
        self.pos.is_check()
    }

    /// Get legal moves structured for RL move selection.
    ///
    /// Returns (grid_indices, promotions) where:
    /// - grid_indices: flat src*64+dst for every legal move (promotion pairs deduplicated)
    /// - promotions: Vec of (pair_idx, legal_promo_types) for each promotion-eligible square pair
    pub fn legal_moves_structured(&self) -> (Vec<u16>, Vec<(u16, Vec<u8>)>) {
        let legal = self.pos.legal_moves();
        let mut grid_indices: Vec<u16> = Vec::with_capacity(legal.len());
        let mut promo_map: Vec<(u16, Vec<u8>)> = Vec::new();
        let mut seen_promo_flat: u16 = u16::MAX; // track last seen promo flat_idx for dedup

        for m in &legal {
            let token = move_to_token(m);
            let (src, dst, promo) = vocab::decompose_token(token).unwrap();
            let flat_idx = (src as u16) * 64 + (dst as u16);

            if promo == 0 {
                grid_indices.push(flat_idx);
            } else {
                let pair_idx = vocab::promo_pair_index(src, dst).unwrap();
                let promo_type = promo - 1; // 1-indexed to 0-indexed

                if flat_idx != seen_promo_flat {
                    // New promotion pair — add grid index and start new entry
                    grid_indices.push(flat_idx);
                    promo_map.push((pair_idx as u16, vec![promo_type]));
                    seen_promo_flat = flat_idx;
                } else {
                    // Same pair, add promo type
                    promo_map.last_mut().unwrap().1.push(promo_type);
                }
            }
        }

        (grid_indices, promo_map)
    }

    /// Return a dense 4096-element mask: true if (src*64+dst) has a legal move.
    pub fn legal_moves_grid_mask(&self) -> [bool; 4096] {
        let legal = self.pos.legal_moves();
        let mut mask = [false; 4096];
        for m in &legal {
            let token = move_to_token(m);
            let (src, dst, _promo) = vocab::decompose_token(token).unwrap();
            let flat_idx = (src as usize) * 64 + (dst as usize);
            mask[flat_idx] = true;
        }
        mask
    }

    /// Get all legal move data in a single pass: structured moves + dense grid mask.
    ///
    /// Computes `legal_moves()` once and derives both structured data (for promo
    /// handling) and the dense 4096-bool mask (for softmax masking).
    pub fn legal_moves_full(&self) -> (Vec<u16>, Vec<(u16, Vec<u8>)>, [bool; 4096]) {
        let legal = self.pos.legal_moves();
        let mut grid_indices: Vec<u16> = Vec::with_capacity(legal.len());
        let mut promo_map: Vec<(u16, Vec<u8>)> = Vec::new();
        let mut seen_promo_flat: u16 = u16::MAX;
        let mut mask = [false; 4096];

        for m in &legal {
            let token = move_to_token(m);
            let (src, dst, promo) = vocab::decompose_token(token).unwrap();
            let flat_idx = (src as u16) * 64 + (dst as u16);

            mask[flat_idx as usize] = true;

            if promo == 0 {
                grid_indices.push(flat_idx);
            } else {
                let pair_idx = vocab::promo_pair_index(src, dst).unwrap();
                let promo_type = promo - 1;

                if flat_idx != seen_promo_flat {
                    grid_indices.push(flat_idx);
                    promo_map.push((pair_idx as u16, vec![promo_type]));
                    seen_promo_flat = flat_idx;
                } else {
                    promo_map.last_mut().unwrap().1.push(promo_type);
                }
            }
        }

        (grid_indices, promo_map, mask)
    }

    /// Apply a move and return its UCI string. Returns Err if illegal.
    pub fn make_move_uci(&mut self, token: u16) -> Result<String, String> {
        let uci = vocab::token_to_uci(token)
            .ok_or_else(|| format!("Token {} has no UCI representation", token))?;
        self.make_move(token)?;
        Ok(uci)
    }

    /// Get the UCI position string for engine communication.
    /// Returns "position startpos" or "position startpos moves e2e4 e7e5 ..."
    pub fn uci_position_string(&self) -> String {
        if self.move_history.is_empty() {
            return "position startpos".to_string();
        }
        let mut s = String::with_capacity(24 + self.move_history.len() * 6);
        s.push_str("position startpos moves");
        for &token in &self.move_history {
            s.push(' ');
            s.push_str(&vocab::token_to_uci(token).unwrap());
        }
        s
    }

    /// Get the FEN string for the current position.
    pub fn fen(&self) -> String {
        let setup = self.pos.to_setup(EnPassantMode::Legal);
        let fen = Fen::try_from(setup).expect("valid position should produce valid FEN");
        fen.to_string()
    }

    /// Pick a random legal move, apply it, and return the token.
    /// Returns None if no legal moves (game is over).
    pub fn make_random_move(&mut self, rng: &mut impl Rng) -> Option<u16> {
        let legal = self.pos.legal_moves();
        if legal.is_empty() {
            return None;
        }
        let idx = rng.gen_range(0..legal.len());
        let m = &legal[idx];
        let token = move_to_token(m);
        // We know the move is legal, so this should always succeed
        self.make_move(token).ok();
        Some(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_conversion_roundtrip() {
        for i in 0..64u8 {
            let sq = our_sq_to_shakmaty(i);
            assert_eq!(shakmaty_sq_to_ours(sq), i, "Roundtrip failed for {}", i);
        }
    }

    #[test]
    fn test_initial_legal_moves() {
        let state = GameState::new();
        let tokens = state.legal_move_tokens();
        // Starting position has 20 legal moves
        assert_eq!(tokens.len(), 20, "Starting position should have 20 legal moves");
    }

    #[test]
    fn test_make_move() {
        let mut state = GameState::new();
        // e2e4: src=e2=12, dst=e4=28
        let token = vocab::base_grid_token(12, 28);
        state.make_move(token).unwrap();
        assert_eq!(state.ply(), 1);
        assert_eq!(state.turn(), Color::Black);
    }

    #[test]
    fn test_legal_move_grid() {
        let state = GameState::new();
        let grid = state.legal_move_grid();
        // Count total legal moves from grid
        let total: u32 = grid.iter().map(|g| g.count_ones()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_castling_token() {
        // Verify e1g1 maps correctly
        let src = shakmaty_sq_to_ours(Square::E1);
        let dst = shakmaty_sq_to_ours(Square::G1);
        assert_eq!(src, 4);  // e1
        assert_eq!(dst, 6);  // g1
        let token = vocab::base_grid_token(src, dst);
        let uci = vocab::token_to_uci(token).unwrap();
        assert_eq!(uci, "e1g1");
    }
}
