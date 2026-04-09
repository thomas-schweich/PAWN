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

/// Convert a shakmaty Move to our token index (searchless_chess action ID).
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

    // Build UCI string on the stack (4-5 bytes, no heap allocation)
    let mut buf = [0u8; 5];
    let src_name = vocab::SQUARE_NAMES[src_idx as usize].as_bytes();
    let dst_name = vocab::SQUARE_NAMES[dst_idx as usize].as_bytes();
    buf[0] = src_name[0];
    buf[1] = src_name[1];
    buf[2] = dst_name[0];
    buf[3] = dst_name[1];

    let len = if let Move::Normal { promotion: Some(role), .. } = m {
        buf[4] = match role {
            Role::Queen => b'q',
            Role::Rook => b'r',
            Role::Bishop => b'b',
            Role::Knight => b'n',
            _ => panic!("Invalid promotion role: {:?}", role),
        };
        5
    } else {
        4
    };

    let uci = std::str::from_utf8(&buf[..len]).unwrap();
    vocab::uci_to_action(uci)
        .unwrap_or_else(|| panic!("Move {} not found in searchless vocabulary", uci))
}

/// Convert our token index to a shakmaty Move, given the current position.
/// Finds the legal move matching the token's (src, dst, promo) decomposition.
pub fn token_to_move(pos: &Chess, token: u16) -> Option<Move> {
    // Validate the token is decomposable (not PAD/outcome)
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

    /// Compute grid + promo mask from a single `legal_moves()` call.
    /// Use when both are needed (e.g. label replay) to avoid 2x move generation.
    pub fn legal_move_grid_and_promo(&self) -> ([u64; 64], [[bool; 4]; 44]) {
        let legal = self.pos.legal_moves();
        decompose_legal_moves(&legal)
    }

    /// Compute grid, promo mask, and token list from a single `legal_moves()` call.
    /// Use in game generation where all three are needed, avoiding 3x+ move generation.
    /// Also returns the raw MoveList for mate-in-1 checks.
    pub fn legal_move_all(&self) -> ([u64; 64], [[bool; 4]; 44], Vec<u16>, MoveList) {
        let legal = self.pos.legal_moves();
        let (grid, mask) = decompose_legal_moves(&legal);
        let tokens: Vec<u16> = legal.iter().map(|m| move_to_token(m)).collect();
        (grid, mask, tokens, legal)
    }

    /// Check termination using a pre-computed MoveList, avoiding a redundant
    /// `legal_moves()` call. The caller must pass the same legal moves that
    /// correspond to the current position.
    pub fn check_termination_with_legal(&self, max_ply: usize, legal: &MoveList) -> Option<Termination> {
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

        if self.halfmove_clock >= 150 {
            return Some(Termination::SeventyFiveMoveRule);
        }

        if self.is_fivefold_repetition() {
            return Some(Termination::FivefoldRepetition);
        }

        if self.pos.is_insufficient_material() {
            return Some(Termination::InsufficientMaterial);
        }

        None
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

    /// Create a GameState by replaying a sequence of move tokens from the starting position.
    /// Returns an error if any token is invalid or illegal.
    pub fn from_move_tokens(tokens: &[u16]) -> Result<Self, String> {
        let mut state = Self::new();
        for (i, &token) in tokens.iter().enumerate() {
            state.make_move(token).map_err(|e| format!("ply {}: {}", i, e))?;
        }
        Ok(state)
    }

    /// Play out a random game from the current position to completion.
    /// Returns the termination type.
    pub fn play_random_to_end(&mut self, rng: &mut impl Rng, max_ply: usize) -> Termination {
        loop {
            if let Some(term) = self.check_termination(max_ply) {
                return term;
            }
            if self.make_random_move(rng).is_none() {
                return Termination::Stalemate;
            }
        }
    }
}

/// Decompose a pre-computed MoveList into a grid and promo mask in a single pass.
pub fn decompose_legal_moves(legal: &MoveList) -> ([u64; 64], [[bool; 4]; 44]) {
    let mut grid = [0u64; 64];
    let mut mask = [[false; 4]; 44];

    for m in legal {
        let token = move_to_token(m);
        if let Some((src, dst, _promo)) = vocab::decompose_token(token) {
            grid[src as usize] |= 1u64 << dst;
        }
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

    (grid, mask)
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
        let token = vocab::uci_token("e2e4");
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
        let token = vocab::uci_token("e1g1");
        let uci = vocab::token_to_uci(token).unwrap();
        assert_eq!(uci, "e1g1");
    }

    // ==== New tests added by Agent A (Rust Core) ====

    // Helper: construct a GameState by playing a list of UCI strings.
    fn replay_ucis(ucis: &[&str]) -> GameState {
        let mut state = GameState::new();
        let (_, m2t) = vocab::build_vocab_maps();
        for uci in ucis {
            let token = *m2t.get(*uci).unwrap_or_else(|| panic!("unknown uci: {}", uci));
            state.make_move(token).unwrap_or_else(|e| panic!("Illegal {}: {}", uci, e));
        }
        state
    }

    #[test]
    fn test_square_specific_mappings() {
        // Explicit mappings - file-major layout
        assert_eq!(shakmaty_sq_to_ours(Square::A1), 0);
        assert_eq!(shakmaty_sq_to_ours(Square::H1), 7);
        assert_eq!(shakmaty_sq_to_ours(Square::A2), 8);
        assert_eq!(shakmaty_sq_to_ours(Square::E2), 12);
        assert_eq!(shakmaty_sq_to_ours(Square::E4), 28);
        assert_eq!(shakmaty_sq_to_ours(Square::E8), 60);
        assert_eq!(shakmaty_sq_to_ours(Square::H8), 63);
        // And inverses
        assert_eq!(our_sq_to_shakmaty(0), Square::A1);
        assert_eq!(our_sq_to_shakmaty(28), Square::E4);
        assert_eq!(our_sq_to_shakmaty(63), Square::H8);
    }

    #[test]
    fn test_piece_to_code_all_pieces() {
        // None -> 0
        assert_eq!(piece_to_code(None), 0);
        // White pieces: 1..=6 for P,N,B,R,Q,K
        assert_eq!(piece_to_code(Some(Piece { color: Color::White, role: Role::Pawn })), 1);
        assert_eq!(piece_to_code(Some(Piece { color: Color::White, role: Role::Knight })), 2);
        assert_eq!(piece_to_code(Some(Piece { color: Color::White, role: Role::Bishop })), 3);
        assert_eq!(piece_to_code(Some(Piece { color: Color::White, role: Role::Rook })), 4);
        assert_eq!(piece_to_code(Some(Piece { color: Color::White, role: Role::Queen })), 5);
        assert_eq!(piece_to_code(Some(Piece { color: Color::White, role: Role::King })), 6);
        // Black pieces: 7..=12
        assert_eq!(piece_to_code(Some(Piece { color: Color::Black, role: Role::Pawn })), 7);
        assert_eq!(piece_to_code(Some(Piece { color: Color::Black, role: Role::King })), 12);
    }

    #[test]
    fn test_initial_position_board_array() {
        let state = GameState::new();
        let board = state.board_array();
        // Rank 1 (index 0): white pieces
        assert_eq!(board[0][0], 4); // White rook a1
        assert_eq!(board[0][1], 2); // White knight b1
        assert_eq!(board[0][2], 3); // White bishop c1
        assert_eq!(board[0][3], 5); // White queen d1
        assert_eq!(board[0][4], 6); // White king e1
        // Rank 2 (index 1): all white pawns
        for file in 0..8 {
            assert_eq!(board[1][file], 1, "rank 2 file {} should be white pawn", file);
        }
        // Rank 3-6: empty
        for rank in 2..6 {
            for file in 0..8 {
                assert_eq!(board[rank][file], 0, "middle square r{} f{} should be empty", rank, file);
            }
        }
        // Rank 7 (index 6): all black pawns
        for file in 0..8 {
            assert_eq!(board[6][file], 7, "rank 7 file {} should be black pawn", file);
        }
        // Rank 8 (index 7): black pieces
        assert_eq!(board[7][0], 10); // Black rook a8
        assert_eq!(board[7][4], 12); // Black king e8
    }

    #[test]
    fn test_board_array_values_in_range() {
        // Play a few moves and ensure all board values stay in [0, 12]
        let state = replay_ucis(&["e2e4", "e7e5", "g1f3", "b8c6"]);
        let board = state.board_array();
        for rank in 0..8 {
            for file in 0..8 {
                let v = board[rank][file];
                assert!(v >= 0 && v <= 12, "out of range piece code {} at r{} f{}", v, rank, file);
            }
        }
    }

    #[test]
    fn test_initial_castling_rights_all_set() {
        let state = GameState::new();
        // All four rights should be set at startpos
        assert_eq!(state.castling_rights_bits(), 0b1111);
    }

    #[test]
    fn test_castling_rights_lost_when_king_moves() {
        // Move king after clearing: e2e4 e7e5 (king can't move yet); instead:
        // Play: e2e4 e7e5 e1e2 => white loses both castling rights
        let state = replay_ucis(&["e2e4", "e7e5", "e1e2"]);
        let bits = state.castling_rights_bits();
        // White kingside (bit 0) and queenside (bit 1) cleared
        assert_eq!(bits & 0b0011, 0, "White castling rights should be lost, got {:#b}", bits);
        // Black still has both
        assert_eq!(bits & 0b1100, 0b1100);
    }

    #[test]
    fn test_castling_rights_lost_when_rook_moves() {
        // Play: a2a4 e7e5 a1a3 => white loses queenside
        let state = replay_ucis(&["a2a4", "e7e5", "a1a3"]);
        let bits = state.castling_rights_bits();
        // Queenside (bit 1) cleared, kingside (bit 0) intact
        assert_eq!(bits & 0b0001, 0b0001, "White kingside intact");
        assert_eq!(bits & 0b0010, 0, "White queenside lost");
    }

    #[test]
    fn test_ep_square_no_ep_initially() {
        let state = GameState::new();
        assert_eq!(state.ep_square(), -1);
    }

    #[test]
    fn test_ep_square_after_pawn_double_push() {
        // e2e4 - sets ep square, but only legal_ep_square = Some iff capture is pseudolegal.
        // After e2e4, black can capture en passant only if a black pawn is adjacent to e4 -
        // which isn't true initially. So legal_ep_square should be None.
        let state = replay_ucis(&["e2e4"]);
        assert_eq!(state.ep_square(), -1, "No black pawn adjacent -> legal_ep_square is None");

        // After e4 d5, white's d2 pawn has no adjacency. After e4 f5, black's f-pawn at f5 adjacent to e4.
        // Play: e2e4, f7f5, e4e5, d7d5 => white's e5 pawn is adjacent to d5 (black's just-pushed pawn).
        // EP square should be d6 (square BEHIND the d7-d5 push).
        let state = replay_ucis(&["e2e4", "f7f5", "e4e5", "d7d5"]);
        let ep = state.ep_square();
        // d6 = file=3, rank=5 -> 5*8+3 = 43
        assert_eq!(ep, 43, "EP square after d7d5 with adjacent e5 pawn should be d6 (43)");
    }

    #[test]
    fn test_halfmove_clock_zero_initially() {
        let state = GameState::new();
        assert_eq!(state.halfmove_clock(), 0);
    }

    #[test]
    fn test_halfmove_clock_resets_on_pawn_move() {
        // Initial clock=0; play Nf3 (clock=1), then e7e5 (pawn, resets to 0)
        let state = replay_ucis(&["g1f3"]);
        assert_eq!(state.halfmove_clock(), 1);
        let state = replay_ucis(&["g1f3", "e7e5"]);
        assert_eq!(state.halfmove_clock(), 0);
    }

    #[test]
    fn test_halfmove_clock_increments_on_knight_move() {
        // Knight moves twice: g1f3, b8c6 -> clock=2
        let state = replay_ucis(&["g1f3", "b8c6"]);
        assert_eq!(state.halfmove_clock(), 2);
    }

    #[test]
    fn test_halfmove_clock_resets_on_capture() {
        // Four knight moves to build clock > 0, then Nxf7 captures the f7 pawn.
        let state = replay_ucis(&["g1f3", "b8c6", "f3g5", "c6d4"]);
        assert!(state.halfmove_clock() > 0,
            "Clock should be > 0 after four non-pawn non-capture moves, got {}",
            state.halfmove_clock());
        let before = state.halfmove_clock();
        // Now play Nxf7 — knight captures the f7 pawn
        let state = replay_ucis(&["g1f3", "b8c6", "f3g5", "c6d4", "g5f7"]);
        assert_eq!(state.halfmove_clock(), 0,
            "Halfmove clock should reset to 0 after capture (was {} before)", before);
        // Verify clock increments again after non-pawn non-capture move
        let state = replay_ucis(&["g1f3", "b8c6", "f3g5", "c6d4", "g5f7", "d4c6"]);
        assert_eq!(state.halfmove_clock(), 1,
            "Clock should increment after non-capture knight move");
    }

    #[test]
    fn test_is_check_startpos() {
        let state = GameState::new();
        assert!(!state.is_check());
    }

    #[test]
    fn test_is_check_after_checkgiving_move() {
        // Scholar's mate sequence partial: e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7#
        // After h5f7#, black is in check AND mated
        let state = replay_ucis(&["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]);
        assert!(state.is_check(), "Black should be in check after h5f7#");
    }

    #[test]
    fn test_scholars_mate_checkmate_detection() {
        // Classic Scholar's mate: 4-move checkmate delivered by white
        let state = replay_ucis(&["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]);
        // game_length = 7 (odd => white made last move)
        assert_eq!(state.ply(), 7);
        let term = state.check_termination(256);
        assert_eq!(term, Some(Termination::Checkmate));
        // No legal moves for black
        assert_eq!(state.legal_move_tokens().len(), 0);
    }

    #[test]
    fn test_fools_mate_checkmate() {
        // Fool's mate: f2f3 e7e5 g2g4 d8h4# (black wins in 4 ply)
        let state = replay_ucis(&["f2f3", "e7e5", "g2g4", "d8h4"]);
        assert_eq!(state.ply(), 4);
        let term = state.check_termination(256);
        assert_eq!(term, Some(Termination::Checkmate));
        // White has no legal moves
        assert_eq!(state.legal_move_tokens().len(), 0);
        assert!(state.is_check());
    }

    #[test]
    fn test_make_move_illegal_token() {
        let mut state = GameState::new();
        // e1e2 (king stepping forward) is not legal at startpos (pawn in the way)
        let token = vocab::uci_token("e1e2");
        let result = state.make_move(token);
        assert!(result.is_err());
        // State should not have advanced
        assert_eq!(state.ply(), 0);
    }

    #[test]
    fn test_make_move_invalid_token() {
        let mut state = GameState::new();
        // PAD token is never a legal move
        let result = state.make_move(vocab::PAD_TOKEN);
        assert!(result.is_err());
        // Outcome token
        let result = state.make_move(vocab::OUTCOME_BASE);
        assert!(result.is_err());
        assert_eq!(state.ply(), 0);
    }

    #[test]
    fn test_ply_increments_with_each_move() {
        let mut state = GameState::new();
        assert_eq!(state.ply(), 0);
        let (_, m2t) = vocab::build_vocab_maps();
        state.make_move(*m2t.get("e2e4").unwrap()).unwrap();
        assert_eq!(state.ply(), 1);
        state.make_move(*m2t.get("e7e5").unwrap()).unwrap();
        assert_eq!(state.ply(), 2);
    }

    #[test]
    fn test_turn_alternates() {
        let mut state = GameState::new();
        assert_eq!(state.turn(), Color::White);
        assert!(state.is_white_to_move());
        let (_, m2t) = vocab::build_vocab_maps();
        state.make_move(*m2t.get("e2e4").unwrap()).unwrap();
        assert_eq!(state.turn(), Color::Black);
        assert!(!state.is_white_to_move());
        state.make_move(*m2t.get("e7e5").unwrap()).unwrap();
        assert_eq!(state.turn(), Color::White);
    }

    #[test]
    fn test_move_history_appends() {
        let mut state = GameState::new();
        assert_eq!(state.move_history().len(), 0);
        let (_, m2t) = vocab::build_vocab_maps();
        let t = *m2t.get("e2e4").unwrap();
        state.make_move(t).unwrap();
        assert_eq!(state.move_history().len(), 1);
        assert_eq!(state.move_history()[0], t);
    }

    #[test]
    fn test_from_move_tokens_replay() {
        // Construct game from tokens and verify ply
        let (_, m2t) = vocab::build_vocab_maps();
        let tokens: Vec<u16> = ["e2e4", "e7e5", "g1f3", "b8c6"]
            .iter()
            .map(|u| *m2t.get(*u).unwrap())
            .collect();
        let state = GameState::from_move_tokens(&tokens).unwrap();
        assert_eq!(state.ply(), 4);
        assert_eq!(state.move_history(), &tokens[..]);
    }

    #[test]
    fn test_from_move_tokens_invalid_returns_error() {
        let (_, m2t) = vocab::build_vocab_maps();
        // Start legal, then include an illegal move
        let tokens: Vec<u16> = vec![
            *m2t.get("e2e4").unwrap(),
            *m2t.get("a1a8").unwrap(), // rook cannot jump to a8 with a2 pawn blocking
        ];
        let result = GameState::from_move_tokens(&tokens);
        match result {
            Ok(_) => panic!("Expected error, got Ok"),
            Err(e) => assert!(e.contains("ply 1"), "error message doesn't mention ply 1: {}", e),
        }
    }

    #[test]
    fn test_legal_moves_count_after_e4() {
        // After e4, black has 20 legal moves (same as white had at start)
        let state = replay_ucis(&["e2e4"]);
        let tokens = state.legal_move_tokens();
        assert_eq!(tokens.len(), 20);
    }

    #[test]
    fn test_legal_move_grid_matches_tokens_count() {
        let state = GameState::new();
        let tokens = state.legal_move_tokens();
        let grid = state.legal_move_grid();
        let total_bits: u32 = grid.iter().map(|g| g.count_ones()).sum();
        assert_eq!(total_bits as usize, tokens.len());
    }

    #[test]
    fn test_fen_startpos() {
        let state = GameState::new();
        let fen = state.fen();
        // Startpos FEN
        assert!(fen.starts_with("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            "got FEN: {}", fen);
    }

    #[test]
    fn test_uci_position_string_startpos() {
        let state = GameState::new();
        assert_eq!(state.uci_position_string(), "position startpos");
    }

    #[test]
    fn test_uci_position_string_with_moves() {
        let state = replay_ucis(&["e2e4", "e7e5"]);
        assert_eq!(state.uci_position_string(), "position startpos moves e2e4 e7e5");
    }

    #[test]
    fn test_legal_promo_mask_none_initially() {
        let state = GameState::new();
        let mask = state.legal_promo_mask();
        for pair in 0..44 {
            for t in 0..4 {
                assert!(!mask[pair][t], "No promotions initially");
            }
        }
    }

    #[test]
    fn test_legal_moves_full_matches_components() {
        let state = replay_ucis(&["e2e4", "e7e5", "g1f3"]);
        let (indices_f, promos_f, mask_f) = state.legal_moves_full();
        let (indices_s, promos_s) = state.legal_moves_structured();
        let mask_m = state.legal_moves_grid_mask();
        assert_eq!(indices_f, indices_s);
        assert_eq!(promos_f, promos_s);
        for i in 0..4096 {
            assert_eq!(mask_f[i], mask_m[i], "mask mismatch at {}", i);
        }
    }

    #[test]
    fn test_move_to_token_token_to_move_roundtrip() {
        // At startpos, every legal move round-trips through move_to_token / token_to_move
        let state = GameState::new();
        let legal = state.position().legal_moves();
        for m in &legal {
            let tok = move_to_token(m);
            let m2 = token_to_move(state.position(), tok);
            assert!(m2.is_some(), "roundtrip failed for move");
            // Same source and destination (castling moves are represented differently but compare by token)
            assert_eq!(move_to_token(&m2.unwrap()), tok);
        }
    }

    #[test]
    fn test_token_to_move_illegal_returns_none() {
        let state = GameState::new();
        // a1a8 is not a legal move at startpos (rook blocked)
        let token = vocab::uci_token("a1a8");
        assert!(token_to_move(state.position(), token).is_none());
    }

    #[test]
    fn test_token_to_move_invalid_token_none() {
        let state = GameState::new();
        // PAD token
        assert!(token_to_move(state.position(), vocab::PAD_TOKEN).is_none());
        // Outcome token
        assert!(token_to_move(state.position(), vocab::OUTCOME_BASE).is_none());
    }

    #[test]
    fn test_make_random_move_advances_state() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut state = GameState::new();
        let tok = state.make_random_move(&mut rng);
        assert!(tok.is_some());
        assert_eq!(state.ply(), 1);
    }

    #[test]
    fn test_fivefold_repetition_false_at_start() {
        let state = GameState::new();
        assert!(!state.is_fivefold_repetition());
    }

    #[test]
    fn test_play_random_to_end_finishes() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut state = GameState::new();
        let _term = state.play_random_to_end(&mut rng, 256);
        // Should terminate with ply <= 256
        assert!(state.ply() <= 256);
        // Should be terminated
        assert!(state.check_termination(256).is_some());
    }

    #[test]
    fn test_check_termination_none_at_start() {
        let state = GameState::new();
        assert!(state.check_termination(256).is_none());
    }

    #[test]
    fn test_check_termination_ply_limit() {
        // Play until ply limit hit (use small max_ply)
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut state = GameState::new();
        while state.check_termination(10).is_none() && state.ply() < 10 {
            if state.make_random_move(&mut rng).is_none() {
                break;
            }
        }
        // At ply=10 with max_ply=10, should return PlyLimit (unless game ended earlier)
        // seed=42 games don't end that fast, so expect PlyLimit
        if state.ply() == 10 {
            assert_eq!(state.check_termination(10), Some(Termination::PlyLimit));
        }
    }

    #[test]
    fn test_make_move_uci_returns_uci_string() {
        let mut state = GameState::new();
        let tok = vocab::uci_token("e2e4"); // e2e4
        let uci = state.make_move_uci(tok).unwrap();
        assert_eq!(uci, "e2e4");
        assert_eq!(state.ply(), 1);
    }

    #[test]
    fn test_castling_kingside_white() {
        // Set up castling: e4, e5, Nf3, Nc6, Bc4, Nf6, O-O
        let state = replay_ucis(&["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]);
        // After O-O, king is on g1
        let board = state.board_array();
        assert_eq!(board[0][6], 6, "White king should be on g1");
        assert_eq!(board[0][5], 4, "Rook should be on f1");
        assert_eq!(board[0][4], 0, "e1 should be empty");
        assert_eq!(board[0][7], 0, "h1 should be empty");
        // White has lost all castling rights
        assert_eq!(state.castling_rights_bits() & 0b0011, 0);
    }

    #[test]
    fn test_castling_queenside_black() {
        // Setup to allow black to O-O-O: 1.Nf3 Nc6 2.e3 d5 3.Be2 Bd7 4.O-O Qd6 5.d3 O-O-O
        // Shorter: we just need to clear B8, C8, D8 and ensure king/rook haven't moved
        // Play: 1.e4 d5 2.d4 c6 3.d5 Nd7 4.dxc6 Qc7 5.cxb7 Nb6 6.bxa8=Q Bb7 7.Qxb7 Qxb7 — too complex
        // 1. e4 b6 2. d4 Bb7 3. Nc3 Nc6 4. Bf4 Qc8 5. Qd2 Nb8 6. O-O-O
        let state = replay_ucis(&[
            "e2e4", "b7b6", "d2d4", "c8b7", "b1c3", "b8c6",
            "c1f4", "d8c8", "d1d2", "c6b8", "e1c1" // White O-O-O
        ]);
        let board = state.board_array();
        // After O-O-O, white king is on c1 (index 2)
        assert_eq!(board[0][2], 6, "White king should be on c1");
        assert_eq!(board[0][3], 4, "Rook should be on d1");
        assert_eq!(board[0][0], 0, "a1 should be empty");
    }

    #[test]
    fn test_en_passant_capture_execution() {
        // Set up: 1.e4 Nf6 2.e5 d5 — now e5 pawn can capture d6 en passant
        let state = replay_ucis(&["e2e4", "g8f6", "e4e5", "d7d5"]);
        // EP square should be d6 (index 43)
        assert_eq!(state.ep_square(), 43);
        // Execute e5d6 (en passant)
        let state2 = replay_ucis(&["e2e4", "g8f6", "e4e5", "d7d5", "e5d6"]);
        let board = state2.board_array();
        // d5 black pawn should be captured (empty)
        assert_eq!(board[4][3], 0, "d5 should be empty after ep");
        // d6 should have white pawn
        assert_eq!(board[5][3], 1, "d6 should have white pawn");
        // Halfmove clock resets to 0 (capture)
        assert_eq!(state2.halfmove_clock(), 0);
    }
}
