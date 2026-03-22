//! Per-ply edge case bitfield computation. Spec §7.7.
//!
//! Each bit describes the board state BEFORE the move at that ply is played,
//! from the perspective of the SIDE TO MOVE. All flags are color-agnostic.

use rayon::prelude::*;
use shakmaty::{Bitboard, Chess, Color, Move, MoveList, Position, Role, Square};
use std::collections::HashMap;

use crate::board::GameState;
use crate::vocab;

// Bit positions (spec §7.7.2)
pub const IN_CHECK: u64                     = 1 << 0;
pub const IN_DOUBLE_CHECK: u64              = 1 << 1;
pub const CHECKMATE: u64                    = 1 << 2;
pub const STALEMATE: u64                    = 1 << 3;
pub const PIN_RESTRICTS_MOVEMENT: u64       = 1 << 4;
pub const PAWN_CAPTURE_AVAILABLE: u64       = 1 << 5;
pub const PROMOTION_AVAILABLE: u64          = 1 << 6;
pub const EP_CAPTURE_AVAILABLE: u64         = 1 << 7;
pub const CASTLE_LEGAL_KINGSIDE: u64        = 1 << 8;
pub const CASTLE_LEGAL_QUEENSIDE: u64       = 1 << 9;
pub const CASTLE_BLOCKED_CHECK: u64         = 1 << 10;
pub const RIGHTS_LOST_KINGSIDE: u64         = 1 << 11;
pub const RIGHTS_LOST_QUEENSIDE: u64        = 1 << 12;
pub const RIGHTS_LOST_CAPTURE_KINGSIDE: u64 = 1 << 13;
pub const RIGHTS_LOST_CAPTURE_QUEENSIDE: u64= 1 << 14;
pub const PIECE_BLOCKS_B: u64               = 1 << 15;
pub const PIECE_BLOCKS_C: u64               = 1 << 16;
pub const PIECE_BLOCKS_D: u64               = 1 << 17;
pub const PIECE_BLOCKS_F: u64               = 1 << 18;
pub const PIECE_BLOCKS_G: u64               = 1 << 19;
pub const ATTACK_BLOCKS_C: u64              = 1 << 20;
pub const ATTACK_BLOCKS_D: u64              = 1 << 21;
pub const ATTACK_BLOCKS_F: u64              = 1 << 22;
pub const ATTACK_BLOCKS_G: u64              = 1 << 23;
pub const EP_FILE_A: u64                    = 1 << 24;
pub const EP_FILE_B: u64                    = 1 << 25;
pub const EP_FILE_C: u64                    = 1 << 26;
pub const EP_FILE_D: u64                    = 1 << 27;
pub const EP_FILE_E: u64                    = 1 << 28;
pub const EP_FILE_F: u64                    = 1 << 29;
pub const EP_FILE_G: u64                    = 1 << 30;
pub const EP_FILE_H: u64                    = 1 << 31;
pub const PROMO_FILE_A: u64                 = 1 << 32;
pub const PROMO_FILE_B: u64                 = 1 << 33;
pub const PROMO_FILE_C: u64                 = 1 << 34;
pub const PROMO_FILE_D: u64                 = 1 << 35;
pub const PROMO_FILE_E: u64                 = 1 << 36;
pub const PROMO_FILE_F: u64                 = 1 << 37;
pub const PROMO_FILE_G: u64                 = 1 << 38;
pub const PROMO_FILE_H: u64                 = 1 << 39;
pub const SEVENTY_FIVE_MOVE_RULE: u64       = 1 << 40;
pub const FIVEFOLD_REPETITION: u64          = 1 << 41;
pub const INSUFFICIENT_MATERIAL: u64        = 1 << 42;

/// Named bit map for Python export.
pub fn edge_case_bits() -> HashMap<String, u64> {
    let mut m = HashMap::new();
    m.insert("IN_CHECK".into(), IN_CHECK);
    m.insert("IN_DOUBLE_CHECK".into(), IN_DOUBLE_CHECK);
    m.insert("CHECKMATE".into(), CHECKMATE);
    m.insert("STALEMATE".into(), STALEMATE);
    m.insert("PIN_RESTRICTS_MOVEMENT".into(), PIN_RESTRICTS_MOVEMENT);
    m.insert("PAWN_CAPTURE_AVAILABLE".into(), PAWN_CAPTURE_AVAILABLE);
    m.insert("PROMOTION_AVAILABLE".into(), PROMOTION_AVAILABLE);
    m.insert("EP_CAPTURE_AVAILABLE".into(), EP_CAPTURE_AVAILABLE);
    m.insert("CASTLE_LEGAL_KINGSIDE".into(), CASTLE_LEGAL_KINGSIDE);
    m.insert("CASTLE_LEGAL_QUEENSIDE".into(), CASTLE_LEGAL_QUEENSIDE);
    m.insert("CASTLE_BLOCKED_CHECK".into(), CASTLE_BLOCKED_CHECK);
    m.insert("RIGHTS_LOST_KINGSIDE".into(), RIGHTS_LOST_KINGSIDE);
    m.insert("RIGHTS_LOST_QUEENSIDE".into(), RIGHTS_LOST_QUEENSIDE);
    m.insert("RIGHTS_LOST_CAPTURE_KINGSIDE".into(), RIGHTS_LOST_CAPTURE_KINGSIDE);
    m.insert("RIGHTS_LOST_CAPTURE_QUEENSIDE".into(), RIGHTS_LOST_CAPTURE_QUEENSIDE);
    m.insert("PIECE_BLOCKS_B".into(), PIECE_BLOCKS_B);
    m.insert("PIECE_BLOCKS_C".into(), PIECE_BLOCKS_C);
    m.insert("PIECE_BLOCKS_D".into(), PIECE_BLOCKS_D);
    m.insert("PIECE_BLOCKS_F".into(), PIECE_BLOCKS_F);
    m.insert("PIECE_BLOCKS_G".into(), PIECE_BLOCKS_G);
    m.insert("ATTACK_BLOCKS_C".into(), ATTACK_BLOCKS_C);
    m.insert("ATTACK_BLOCKS_D".into(), ATTACK_BLOCKS_D);
    m.insert("ATTACK_BLOCKS_F".into(), ATTACK_BLOCKS_F);
    m.insert("ATTACK_BLOCKS_G".into(), ATTACK_BLOCKS_G);
    m.insert("EP_FILE_A".into(), EP_FILE_A);
    m.insert("EP_FILE_B".into(), EP_FILE_B);
    m.insert("EP_FILE_C".into(), EP_FILE_C);
    m.insert("EP_FILE_D".into(), EP_FILE_D);
    m.insert("EP_FILE_E".into(), EP_FILE_E);
    m.insert("EP_FILE_F".into(), EP_FILE_F);
    m.insert("EP_FILE_G".into(), EP_FILE_G);
    m.insert("EP_FILE_H".into(), EP_FILE_H);
    m.insert("PROMO_FILE_A".into(), PROMO_FILE_A);
    m.insert("PROMO_FILE_B".into(), PROMO_FILE_B);
    m.insert("PROMO_FILE_C".into(), PROMO_FILE_C);
    m.insert("PROMO_FILE_D".into(), PROMO_FILE_D);
    m.insert("PROMO_FILE_E".into(), PROMO_FILE_E);
    m.insert("PROMO_FILE_F".into(), PROMO_FILE_F);
    m.insert("PROMO_FILE_G".into(), PROMO_FILE_G);
    m.insert("PROMO_FILE_H".into(), PROMO_FILE_H);
    m.insert("SEVENTY_FIVE_MOVE_RULE".into(), SEVENTY_FIVE_MOVE_RULE);
    m.insert("FIVEFOLD_REPETITION".into(), FIVEFOLD_REPETITION);
    m.insert("INSUFFICIENT_MATERIAL".into(), INSUFFICIENT_MATERIAL);
    m
}

/// Compute the per-ply edge case bitfield for a single position.
/// `prev_castling_*` are the castling rights from the previous position of the
/// SAME color (two ply ago), used for transition flags.
/// `prev_move` is the move the opponent just played (for rook-capture detection).
fn compute_ply_bits(
    pos: &Chess,
    legal: &MoveList,
    prev_own_ks: bool,
    prev_own_qs: bool,
    curr_own_ks: bool,
    curr_own_qs: bool,
    opponent_captured_ks_rook: bool,
    opponent_captured_qs_rook: bool,
    is_final_ply: bool,
    termination_bits: u64,
) -> u64 {
    let mut bits: u64 = 0;
    let turn = pos.turn();

    // --- Check and terminal states (bits 0-3) ---
    let in_check = pos.is_check();
    if in_check {
        bits |= IN_CHECK;

        // Double check: more than one checker
        let checkers = pos.checkers();
        if checkers.count() >= 2 {
            bits |= IN_DOUBLE_CHECK;
        }
    }

    if legal.is_empty() {
        if in_check {
            bits |= CHECKMATE;
        } else {
            bits |= STALEMATE;
        }
    }

    // --- Pins (bit 4) ---
    // A pin restricts movement if some piece has fewer legal moves than it would
    // without the pin. We detect this by checking if any legal move list is
    // restricted compared to pseudo-legal.
    // Simpler approach: check if any piece (not king) that is on a line between
    // a sliding attacker and the king has restricted moves.
    if has_restricting_pin(pos, legal) {
        bits |= PIN_RESTRICTS_MOVEMENT;
    }

    // --- Pawn mechanics (bits 5-6) ---
    let mut has_pawn_capture = false;
    let mut has_promotion = false;
    let mut promo_dst_files: u8 = 0; // bit per file

    for m in legal {
        match m {
            Move::Normal { role: Role::Pawn, from, to, promotion, .. } => {
                if from.file() != to.file() {
                    // Diagonal move = capture (or EP, handled separately)
                    has_pawn_capture = true;
                }
                if promotion.is_some() {
                    has_promotion = true;
                    promo_dst_files |= 1 << (to.file() as u8);
                }
            }
            Move::EnPassant { .. } => {
                // EP is also a pawn capture
                has_pawn_capture = true;
            }
            _ => {}
        }
    }

    if has_pawn_capture {
        bits |= PAWN_CAPTURE_AVAILABLE;
    }
    if has_promotion {
        bits |= PROMOTION_AVAILABLE;
        // Per-file promotion bits
        for file in 0u8..8 {
            if promo_dst_files & (1 << file) != 0 {
                bits |= 1u64 << (32 + file as u64);
            }
        }
    }

    // --- En passant (bits 7, 24-31) ---
    let has_ep = legal.iter().any(|m| matches!(m, Move::EnPassant { .. }));
    if has_ep {
        bits |= EP_CAPTURE_AVAILABLE;
        // Find EP file from the EP square
        if let Some(ep_sq) = pos.legal_ep_square() {
            let file = ep_sq.file() as u8;
            bits |= 1u64 << (24 + file as u64);
        }
    }

    // --- Castling (bits 8-23) ---
    let back_rank = if turn == Color::White {
        shakmaty::Rank::First
    } else {
        shakmaty::Rank::Eighth
    };

    // Check if castling moves are in the legal move list
    let has_castle_ks = legal.iter().any(|m| {
        matches!(m, Move::Castle { king, rook } if rook.file() > king.file())
    });
    let has_castle_qs = legal.iter().any(|m| {
        matches!(m, Move::Castle { king, rook } if rook.file() < king.file())
    });

    if has_castle_ks {
        bits |= CASTLE_LEGAL_KINGSIDE;
    }
    if has_castle_qs {
        bits |= CASTLE_LEGAL_QUEENSIDE;
    }

    // Castling blocked by check (bit 10)
    if in_check && (curr_own_ks || curr_own_qs) {
        bits |= CASTLE_BLOCKED_CHECK;
    }

    // Castling rights transitions (bits 11-14)
    if prev_own_ks && !curr_own_ks {
        bits |= RIGHTS_LOST_KINGSIDE;
        if opponent_captured_ks_rook {
            bits |= RIGHTS_LOST_CAPTURE_KINGSIDE;
        }
    }
    if prev_own_qs && !curr_own_qs {
        bits |= RIGHTS_LOST_QUEENSIDE;
        if opponent_captured_qs_rook {
            bits |= RIGHTS_LOST_CAPTURE_QUEENSIDE;
        }
    }

    // Castling blocked by piece (bits 15-19) and attack (bits 20-23)
    // Only relevant if the side has the corresponding castling rights
    let opp = !turn;
    if curr_own_qs {
        let b_sq = Square::from_coords(shakmaty::File::B, back_rank);
        let c_sq = Square::from_coords(shakmaty::File::C, back_rank);
        let d_sq = Square::from_coords(shakmaty::File::D, back_rank);

        let b_occupied = pos.board().piece_at(b_sq).is_some();
        let c_occupied = pos.board().piece_at(c_sq).is_some();
        let d_occupied = pos.board().piece_at(d_sq).is_some();

        if b_occupied { bits |= PIECE_BLOCKS_B; }
        if c_occupied {
            bits |= PIECE_BLOCKS_C;
        } else if is_attacked(pos, c_sq, opp) {
            bits |= ATTACK_BLOCKS_C;
        }
        if d_occupied {
            bits |= PIECE_BLOCKS_D;
        } else if is_attacked(pos, d_sq, opp) {
            bits |= ATTACK_BLOCKS_D;
        }
    }
    if curr_own_ks {
        let f_sq = Square::from_coords(shakmaty::File::F, back_rank);
        let g_sq = Square::from_coords(shakmaty::File::G, back_rank);

        let f_occupied = pos.board().piece_at(f_sq).is_some();
        let g_occupied = pos.board().piece_at(g_sq).is_some();

        if f_occupied {
            bits |= PIECE_BLOCKS_F;
        } else if is_attacked(pos, f_sq, opp) {
            bits |= ATTACK_BLOCKS_F;
        }
        if g_occupied {
            bits |= PIECE_BLOCKS_G;
        } else if is_attacked(pos, g_sq, opp) {
            bits |= ATTACK_BLOCKS_G;
        }
    }

    // --- Draw termination (bits 40-42) ---
    if is_final_ply {
        bits |= termination_bits;
    }

    bits
}

/// Check if a square is attacked by the given color.
fn is_attacked(pos: &Chess, sq: Square, by_color: Color) -> bool {
    let attackers = pos.board().attacks_to(sq, by_color, pos.board().occupied());
    !attackers.is_empty()
}

/// Detect whether any pin restricts a piece's movement.
/// A pin restricts movement if at least one piece has fewer legal moves than
/// it would have if we only considered its normal movement geometry.
/// We detect this by checking: for each non-king piece that is on a ray between
/// an enemy slider and our king, does the legal move list exclude some of its
/// pseudo-legal moves?
fn has_restricting_pin(pos: &Chess, legal: &MoveList) -> bool {
    let turn = pos.turn();
    let king_sq = pos.board().king_of(turn).expect("King must exist");

    // Build set of (from, to) pairs that are legal for non-king pieces
    let mut legal_from_to: std::collections::HashSet<(Square, Square)> =
        std::collections::HashSet::new();
    let mut piece_legal_count: HashMap<Square, usize> = HashMap::new();

    for m in legal {
        let from = match m {
            Move::Normal { from, to, role, .. } if *role != Role::King => {
                legal_from_to.insert((*from, *to));
                *from
            }
            Move::EnPassant { from, to } => {
                legal_from_to.insert((*from, *to));
                *from
            }
            _ => continue,
        };
        *piece_legal_count.entry(from).or_insert(0) += 1;
    }

    // For each non-king piece of our color, check if it's pinned by seeing
    // if removing it from the board would expose the king to a new attack.
    let our_pieces = pos.board().by_color(turn) & !Bitboard::from_square(king_sq);

    for sq in our_pieces {
        let piece = pos.board().piece_at(sq).unwrap();
        if piece.role == Role::King {
            continue;
        }

        // Quick check: is this piece between the king and an enemy slider?
        // Check all 8 directions from king
        let between_king_and_attacker = is_on_pin_ray(pos, sq, king_sq, turn);
        if !between_king_and_attacker {
            continue;
        }

        // This piece might be pinned. Check if it has ANY pseudo-legal moves
        // that are NOT in the legal move set. If a pseudo-legal move is missing
        // from legal moves, it's because of a pin.
        let pseudo_moves = pseudo_legal_moves_for_piece(pos, sq);
        for (_, to) in &pseudo_moves {
            if !legal_from_to.contains(&(sq, *to)) {
                return true;
            }
        }
    }

    false
}

/// Check if `piece_sq` is on a ray between `king_sq` and an enemy sliding piece.
fn is_on_pin_ray(pos: &Chess, piece_sq: Square, king_sq: Square, our_color: Color) -> bool {
    // Check if piece is on a file/rank/diagonal with the king
    let pf = piece_sq.file() as i8;
    let pr = piece_sq.rank() as i8;
    let kf = king_sq.file() as i8;
    let kr = king_sq.rank() as i8;

    let df = (pf - kf).signum();
    let dr = (pr - kr).signum();

    // Must be on a line (not the same square)
    if df == 0 && dr == 0 {
        return false;
    }
    let on_file = pf == kf;
    let on_rank = pr == kr;
    let on_diag = (pf - kf).abs() == (pr - kr).abs();

    if !on_file && !on_rank && !on_diag {
        return false;
    }

    // Walk from piece away from king to find an enemy slider
    let enemy = !our_color;
    let mut cf = pf + df;
    let mut cr = pr + dr;
    while cf >= 0 && cf < 8 && cr >= 0 && cr < 8 {
        let sq = Square::from_coords(
            shakmaty::File::new(cf as u32),
            shakmaty::Rank::new(cr as u32),
        );
        if let Some(p) = pos.board().piece_at(sq) {
            if p.color == enemy {
                // Check if this enemy piece attacks along this ray
                let is_slider = match p.role {
                    Role::Bishop => on_diag,
                    Role::Rook => on_file || on_rank,
                    Role::Queen => true,
                    _ => false,
                };
                return is_slider;
            } else {
                // Our own piece blocks the ray
                return false;
            }
        }
        cf += df;
        cr += dr;
    }
    false
}

/// Get pseudo-legal destination squares for a piece (ignoring pins/check).
/// This is a simplified version — we only need to know IF any move is restricted.
fn pseudo_legal_moves_for_piece(pos: &Chess, sq: Square) -> Vec<(Square, Square)> {
    let piece = match pos.board().piece_at(sq) {
        Some(p) => p,
        None => return vec![],
    };

    let mut moves = Vec::new();
    let occupied = pos.board().occupied();
    let our_pieces = pos.board().by_color(piece.color);

    let attacks = match piece.role {
        Role::Pawn => {
            // Generate pseudo-legal pawn moves so pin detection works
            let mut pawn_moves = Vec::new();
            let rank = sq.rank() as i8;
            let file = sq.file() as i8;
            let (dir, home_rank) = if piece.color == Color::White {
                (1i8, 1i8)
            } else {
                (-1i8, 6i8)
            };

            // Single push
            let fwd_rank = rank + dir;
            if fwd_rank >= 0 && fwd_rank < 8 {
                let fwd = Square::from_coords(
                    shakmaty::File::new(file as u32),
                    shakmaty::Rank::new(fwd_rank as u32),
                );
                if pos.board().piece_at(fwd).is_none() {
                    pawn_moves.push((sq, fwd));

                    // Double push from home rank
                    if rank == home_rank {
                        let dbl_rank = rank + 2 * dir;
                        if dbl_rank >= 0 && dbl_rank < 8 {
                            let dbl = Square::from_coords(
                                shakmaty::File::new(file as u32),
                                shakmaty::Rank::new(dbl_rank as u32),
                            );
                            if pos.board().piece_at(dbl).is_none() {
                                pawn_moves.push((sq, dbl));
                            }
                        }
                    }
                }
            }

            // Captures (including en passant square)
            for cap_df in [-1i8, 1i8] {
                let cap_file = file + cap_df;
                if cap_file >= 0 && cap_file < 8 && fwd_rank >= 0 && fwd_rank < 8 {
                    let cap_sq = Square::from_coords(
                        shakmaty::File::new(cap_file as u32),
                        shakmaty::Rank::new(fwd_rank as u32),
                    );
                    let is_enemy = pos.board().piece_at(cap_sq)
                        .map_or(false, |p| p.color != piece.color);
                    let is_ep = pos.legal_ep_square() == Some(cap_sq);
                    if is_enemy || is_ep {
                        pawn_moves.push((sq, cap_sq));
                    }
                }
            }

            return pawn_moves;
        }
        Role::Knight => shakmaty::attacks::knight_attacks(sq),
        Role::Bishop => shakmaty::attacks::bishop_attacks(sq, occupied),
        Role::Rook => shakmaty::attacks::rook_attacks(sq, occupied),
        Role::Queen => shakmaty::attacks::queen_attacks(sq, occupied),
        Role::King => shakmaty::attacks::king_attacks(sq),
    };

    // Destinations: any square not occupied by our own pieces
    let dests = attacks & !our_pieces;
    for dst in dests {
        moves.push((sq, dst));
    }
    moves
}

/// Get the castling rights for a specific color.
/// Returns (has_kingside, has_queenside).
fn color_castling_rights(pos: &Chess, color: Color) -> (bool, bool) {
    let rights = pos.castles().castling_rights();
    match color {
        Color::White => (
            rights.contains(Square::H1),
            rights.contains(Square::A1),
        ),
        Color::Black => (
            rights.contains(Square::H8),
            rights.contains(Square::A8),
        ),
    }
}

/// Compute per-ply edge stats for a batch of games. Spec §7.7.1.
pub fn compute_edge_stats_per_ply(
    move_ids: &[i16],
    game_lengths: &[i16],
    max_ply: usize,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let batch = game_lengths.len();

    let results: Vec<(Vec<u64>, u64, u64)> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let length = game_lengths[b] as usize;
            let mut state = GameState::new();
            // +1 to hold the terminal position (after the last move)
            let mut ply_bits = vec![0u64; length + 1];
            let mut white_acc: u64 = 0;
            let mut black_acc: u64 = 0;

            // Track castling rights for transition detection
            // prev_own_ks/qs: the side-to-move's castling rights as of their PREVIOUS turn
            let mut white_prev_ks = true; // initial white rights
            let mut white_prev_qs = true;
            let mut black_prev_ks = true;
            let mut black_prev_qs = true;
            let mut white_first_turn = true;
            let mut black_first_turn = true;

            // Track if opponent captured our rook (for RIGHTS_LOST_CAPTURE detection)
            let mut opponent_captured_white_ks_rook = false;
            let mut opponent_captured_white_qs_rook = false;
            let mut opponent_captured_black_ks_rook = false;
            let mut opponent_captured_black_qs_rook = false;

            for t in 0..length {
                let pos = state.position();
                let turn = pos.turn();
                let legal = pos.legal_moves();

                let (curr_own_ks, curr_own_qs) = color_castling_rights(pos, turn);

                let (prev_ks, prev_qs) = match turn {
                    Color::White => {
                        if white_first_turn {
                            white_first_turn = false;
                            (true, true) // No transition on first turn
                        } else {
                            (white_prev_ks, white_prev_qs)
                        }
                    }
                    Color::Black => {
                        if black_first_turn {
                            black_first_turn = false;
                            (true, true)
                        } else {
                            (black_prev_ks, black_prev_qs)
                        }
                    }
                };

                let (opp_cap_ks, opp_cap_qs) = match turn {
                    Color::White => (opponent_captured_white_ks_rook, opponent_captured_white_qs_rook),
                    Color::Black => (opponent_captured_black_ks_rook, opponent_captured_black_qs_rook),
                };

                let bits = compute_ply_bits(
                    pos, &legal,
                    prev_ks, prev_qs,
                    curr_own_ks, curr_own_qs,
                    opp_cap_ks, opp_cap_qs,
                    false, 0,
                );

                ply_bits[t] = bits;
                match turn {
                    Color::White => white_acc |= bits,
                    Color::Black => black_acc |= bits,
                }

                // Save current rights as prev for next time this color moves
                match turn {
                    Color::White => {
                        white_prev_ks = curr_own_ks;
                        white_prev_qs = curr_own_qs;
                        // Reset capture flags
                        opponent_captured_white_ks_rook = false;
                        opponent_captured_white_qs_rook = false;
                    }
                    Color::Black => {
                        black_prev_ks = curr_own_ks;
                        black_prev_qs = curr_own_qs;
                        opponent_captured_black_ks_rook = false;
                        opponent_captured_black_qs_rook = false;
                    }
                }

                // Apply the move and detect rook captures for the opponent
                let token = move_ids[b * max_ply + t] as u16;

                // Before making the move, check if it captures a rook on its home square
                if let Some((_src, dst, _)) = vocab::decompose_token(token) {
                    let dst_sq = crate::board::our_sq_to_shakmaty(dst);
                    if let Some(captured) = pos.board().piece_at(dst_sq) {
                        if captured.role == Role::Rook {
                            match captured.color {
                                Color::White => {
                                    if dst_sq == Square::H1 { opponent_captured_white_ks_rook = true; }
                                    if dst_sq == Square::A1 { opponent_captured_white_qs_rook = true; }
                                }
                                Color::Black => {
                                    if dst_sq == Square::H8 { opponent_captured_black_ks_rook = true; }
                                    if dst_sq == Square::A8 { opponent_captured_black_qs_rook = true; }
                                }
                            }
                        }
                    }
                }

                state.make_move(token).expect("Move should be legal");
            }

            // Examine the terminal position (after the last move).
            // This is where checkmate/stalemate/draw states live.
            {
                let pos = state.position();
                let turn = pos.turn();
                let legal = pos.legal_moves();

                let (curr_own_ks, curr_own_qs) = color_castling_rights(pos, turn);
                let (prev_ks, prev_qs) = match turn {
                    Color::White => (white_prev_ks, white_prev_qs),
                    Color::Black => (black_prev_ks, black_prev_qs),
                };
                let (opp_cap_ks, opp_cap_qs) = match turn {
                    Color::White => (opponent_captured_white_ks_rook, opponent_captured_white_qs_rook),
                    Color::Black => (opponent_captured_black_ks_rook, opponent_captured_black_qs_rook),
                };

                let mut term_bits = 0u64;
                if state.halfmove_clock() >= 150 {
                    term_bits |= SEVENTY_FIVE_MOVE_RULE;
                }
                if state.is_fivefold_repetition() {
                    term_bits |= FIVEFOLD_REPETITION;
                }
                if pos.is_insufficient_material() {
                    term_bits |= INSUFFICIENT_MATERIAL;
                }

                let bits = compute_ply_bits(
                    pos, &legal,
                    prev_ks, prev_qs,
                    curr_own_ks, curr_own_qs,
                    opp_cap_ks, opp_cap_qs,
                    true, term_bits,
                );

                ply_bits[length] = bits;
                match turn {
                    Color::White => white_acc |= bits,
                    Color::Black => black_acc |= bits,
                }
            }

            (ply_bits, white_acc, black_acc)
        })
        .collect();

    // Pack into flat arrays
    // Note: per_ply is sized batch * max_ply. The terminal ply at index `length`
    // fits only if length < max_ply (which is the common case since games rarely
    // reach exactly max_ply moves).
    let mut per_ply = vec![0u64; batch * max_ply];
    let mut white = Vec::with_capacity(batch);
    let mut black = Vec::with_capacity(batch);

    for (b, (ply_bits, w, bl)) in results.into_iter().enumerate() {
        let length = game_lengths[b] as usize;
        // Copy ply bits including terminal position at index `length` if it fits
        let copy_len = std::cmp::min(length + 1, max_ply);
        per_ply[b * max_ply..b * max_ply + copy_len]
            .copy_from_slice(&ply_bits[..copy_len]);
        white.push(w);
        black.push(bl);
    }

    (per_ply, white, black)
}

/// Compute per-game accumulators only (no per-ply storage). Spec §7.7.3.
pub fn compute_edge_stats_per_game(
    move_ids: &[i16],
    game_lengths: &[i16],
    max_ply: usize,
) -> (Vec<u64>, Vec<u64>) {
    let (_, white, black) = compute_edge_stats_per_ply(move_ids, game_lengths, max_ply);
    (white, black)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::generate_random_games;

    #[test]
    fn test_initial_position_bits() {
        // Generate a short game and check initial ply
        let batch = generate_random_games(1, 256, 42);
        let (per_ply, white, black) = compute_edge_stats_per_ply(
            &batch.move_ids, &batch.game_lengths, 256,
        );

        // Initial position: white to move, not in check, has no EP, no promotion
        let bits = per_ply[0];
        assert_eq!(bits & IN_CHECK, 0);
        assert_eq!(bits & CHECKMATE, 0);
        assert_eq!(bits & EP_CAPTURE_AVAILABLE, 0);
        assert_eq!(bits & PROMOTION_AVAILABLE, 0);
        // Should have pawn captures = false (no captures available from starting pos)
        assert_eq!(bits & PAWN_CAPTURE_AVAILABLE, 0);
    }

    #[test]
    fn test_edge_stats_accumulators() {
        let batch = generate_random_games(10, 256, 42);
        let (_, white, black) = compute_edge_stats_per_ply(
            &batch.move_ids, &batch.game_lengths, 256,
        );
        let (white2, black2) = compute_edge_stats_per_game(
            &batch.move_ids, &batch.game_lengths, 256,
        );
        assert_eq!(white, white2);
        assert_eq!(black, black2);
    }
}
