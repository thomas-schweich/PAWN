//! Batch RL environment — owns all game state in Rust.
//!
//! Supports random opponents (pure Rust) and UCI engine opponents
//! (via the `ruci` crate with `engine-sync` feature).

use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use crate::board::GameState;
use crate::types::Termination;

/// Per-game metadata tracked alongside the Rust GameState.
struct GameMeta {
    agent_is_white: bool,
    terminated: bool,
    forfeited: bool,
    outcome_reward: f32,
    agent_plies: u32,
    termination_code: i8, // -1 if not terminated
}

impl GameMeta {
    fn new(agent_is_white: bool) -> Self {
        Self {
            agent_is_white,
            terminated: false,
            forfeited: false,
            outcome_reward: 0.0,
            agent_plies: 0,
            termination_code: -1,
        }
    }
}

/// Batch RL environment owning N concurrent chess games.
pub struct BatchRLEnv {
    games: Vec<GameState>,
    meta: Vec<GameMeta>,
    n_games: usize,
    max_ply: usize,
    rng: ChaCha8Rng,
}

impl BatchRLEnv {
    pub fn new(n_games: usize, max_ply: usize, seed: u64) -> Self {
        Self {
            games: Vec::with_capacity(n_games),
            meta: Vec::with_capacity(n_games),
            n_games,
            max_ply,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Reset all games. First half: agent=white, second half: agent=black.
    pub fn reset(&mut self) {
        self.games.clear();
        self.meta.clear();
        let n_white = self.n_games / 2;
        for i in 0..self.n_games {
            self.games.push(GameState::new());
            self.meta.push(GameMeta::new(i < n_white));
        }
    }

    pub fn n_games(&self) -> usize {
        self.n_games
    }

    pub fn all_terminated(&self) -> bool {
        self.meta.iter().all(|m| m.terminated)
    }

    /// Game indices where it's the agent's turn and game is not over.
    pub fn active_agent_games(&self) -> Vec<u32> {
        (0..self.n_games)
            .filter(|&i| {
                let m = &self.meta[i];
                if m.terminated {
                    return false;
                }
                let white_to_move = self.games[i].is_white_to_move();
                white_to_move == m.agent_is_white
            })
            .map(|i| i as u32)
            .collect()
    }

    /// Game indices where it's the opponent's turn and game is not over.
    pub fn active_opponent_games(&self) -> Vec<u32> {
        (0..self.n_games)
            .filter(|&i| {
                let m = &self.meta[i];
                if m.terminated {
                    return false;
                }
                let white_to_move = self.games[i].is_white_to_move();
                white_to_move != m.agent_is_white
            })
            .map(|i| i as u32)
            .collect()
    }

    // ------------------------------------------------------------------
    // Finalization
    // ------------------------------------------------------------------

    fn finalize(&mut self, gi: usize) {
        let m = &mut self.meta[gi];
        m.terminated = true;

        if let Some(term) = self.games[gi].check_termination(self.max_ply) {
            m.termination_code = term.as_u8() as i8;
            match term {
                Termination::Checkmate => {
                    // The side to move is checkmated (lost).
                    let loser_is_white = self.games[gi].is_white_to_move();
                    if loser_is_white == m.agent_is_white {
                        m.outcome_reward = -1.0; // agent lost
                    } else {
                        m.outcome_reward = 1.0; // agent won
                    }
                }
                // All other terminations are draws
                _ => {
                    m.outcome_reward = 0.0;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Side-agnostic moves (for autoregressive generation)
    // ------------------------------------------------------------------

    /// Apply moves to the specified games regardless of whose turn it is.
    /// Returns (legality_flags, termination_codes).
    /// Termination code is -1 if the game continues after the move.
    pub fn apply_moves(&mut self, game_indices: &[u32], tokens: &[u16]) -> (Vec<bool>, Vec<i8>) {
        let mut flags = Vec::with_capacity(game_indices.len());
        let mut term_codes = Vec::with_capacity(game_indices.len());
        for (&gi, &token) in game_indices.iter().zip(tokens.iter()) {
            let gi = gi as usize;
            if self.meta[gi].terminated {
                flags.push(false);
                term_codes.push(self.meta[gi].termination_code);
                continue;
            }

            if self.games[gi].make_move(token).is_err() {
                self.meta[gi].terminated = true;
                self.meta[gi].forfeited = true;
                self.meta[gi].termination_code = -3; // forfeit
                flags.push(false);
                term_codes.push(-3);
                continue;
            }

            flags.push(true);

            if let Some(term) = self.games[gi].check_termination(self.max_ply) {
                self.meta[gi].terminated = true;
                self.meta[gi].termination_code = term.as_u8() as i8;
                term_codes.push(term.as_u8() as i8);
            } else {
                term_codes.push(-1);
            }
        }
        (flags, term_codes)
    }

    /// Load prefix move sequences into games. Each game replays the given
    /// moves from the start position. Returns per-game termination codes
    /// (-1 if still going, >=0 if terminated during prefix).
    pub fn load_prefixes(&mut self, move_ids: &[u16], lengths: &[u32], n_games: usize, max_ply: usize) -> Vec<i8> {
        let mut term_codes = Vec::with_capacity(n_games);
        for gi in 0..n_games {
            let len = lengths[gi] as usize;
            let mut tc: i8 = -1;
            for t in 0..len {
                let token = move_ids[gi * max_ply + t];
                if self.games[gi].make_move(token).is_err() {
                    self.meta[gi].terminated = true;
                    self.meta[gi].forfeited = true;
                    self.meta[gi].termination_code = -3;
                    tc = -3;
                    break;
                }
                if let Some(term) = self.games[gi].check_termination(self.max_ply) {
                    self.meta[gi].terminated = true;
                    self.meta[gi].termination_code = term.as_u8() as i8;
                    tc = term.as_u8() as i8;
                    break;
                }
            }
            term_codes.push(tc);
        }
        term_codes
    }

    // ------------------------------------------------------------------
    // Agent moves
    // ------------------------------------------------------------------

    /// Apply agent moves to the specified games. Returns legality flags.
    pub fn apply_agent_moves(&mut self, game_indices: &[u32], tokens: &[u16]) -> Vec<bool> {
        let mut flags = Vec::with_capacity(game_indices.len());
        for (&gi, &token) in game_indices.iter().zip(tokens.iter()) {
            let gi = gi as usize;
            if self.meta[gi].terminated {
                flags.push(false);
                continue;
            }

            if self.games[gi].make_move(token).is_err() {
                self.meta[gi].terminated = true;
                self.meta[gi].forfeited = true;
                self.meta[gi].outcome_reward = -1.0;
                flags.push(false);
                continue;
            }

            flags.push(true);
            self.meta[gi].agent_plies += 1;

            if self.games[gi].check_termination(self.max_ply).is_some() {
                self.finalize(gi);
            }
        }
        flags
    }

    // ------------------------------------------------------------------
    // Opponent moves (random)
    // ------------------------------------------------------------------

    /// Make random legal moves for all games where it's the opponent's turn.
    /// Returns indices of games that were acted upon.
    pub fn apply_random_opponent_moves(&mut self) -> Vec<u32> {
        let opp_games = self.active_opponent_games();
        for &gi in &opp_games {
            let gi = gi as usize;
            self.games[gi].make_random_move(&mut self.rng);
            if self.games[gi].check_termination(self.max_ply).is_some() {
                self.finalize(gi);
            }
        }
        opp_games
    }

    // ------------------------------------------------------------------
    // Opponent moves (UCI engine token from Python)
    // ------------------------------------------------------------------

    /// Apply opponent moves received from an external UCI engine.
    /// Does NOT increment agent_plies. Returns legality flags.
    pub fn apply_opponent_moves(&mut self, game_indices: &[u32], tokens: &[u16]) -> Vec<bool> {
        let mut flags = Vec::with_capacity(game_indices.len());
        for (&gi, &token) in game_indices.iter().zip(tokens.iter()) {
            let gi = gi as usize;
            if self.meta[gi].terminated {
                flags.push(false);
                continue;
            }

            if self.games[gi].make_move(token).is_err() {
                // Engine move was illegal — shouldn't happen. Terminate as draw.
                self.meta[gi].terminated = true;
                self.meta[gi].outcome_reward = 0.0;
                flags.push(false);
                continue;
            }

            flags.push(true);

            if self.games[gi].check_termination(self.max_ply).is_some() {
                self.finalize(gi);
            }
        }
        flags
    }

    // ------------------------------------------------------------------
    // Bulk data extraction
    // ------------------------------------------------------------------

    /// Get legal move token masks for a batch of games.
    /// Returns flat bool vec of len(game_indices) * vocab_size.
    pub fn get_legal_token_masks_batch(
        &self,
        game_indices: &[u32],
        vocab_size: usize,
    ) -> Vec<bool> {
        let b = game_indices.len();
        let mut masks = vec![false; b * vocab_size];
        for (bi, &gi) in game_indices.iter().enumerate() {
            for tok in self.games[gi as usize].legal_move_tokens() {
                let idx = bi * vocab_size + tok as usize;
                if idx < masks.len() {
                    masks[idx] = true;
                }
            }
        }
        masks
    }

    /// Get legal moves for a batch of games: structured data + dense masks.
    /// Returns (per_game_structured, flat_dense_masks).
    /// flat_dense_masks is a flat vec of len(game_indices) * 4096 bools.
    pub fn get_legal_moves_batch(
        &self,
        game_indices: &[u32],
    ) -> (Vec<(Vec<u16>, Vec<(u16, Vec<u8>)>)>, Vec<bool>) {
        let b = game_indices.len();
        let mut structured = Vec::with_capacity(b);
        let mut flat_masks = vec![false; b * 4096];

        for (bi, &gi) in game_indices.iter().enumerate() {
            let (indices, promos, mask) = self.games[gi as usize].legal_moves_full();
            structured.push((indices, promos));
            flat_masks[bi * 4096..(bi + 1) * 4096].copy_from_slice(&mask);
        }

        (structured, flat_masks)
    }

    /// Get move histories as a flat i64 vec of shape (B, max_ply) + lengths.
    pub fn get_move_histories(&self, game_indices: &[u32]) -> (Vec<i64>, Vec<i32>) {
        let b = game_indices.len();
        let mut flat = vec![0i64; b * self.max_ply];
        let mut lengths = Vec::with_capacity(b);

        for (bi, &gi) in game_indices.iter().enumerate() {
            let hist = self.games[gi as usize].move_history();
            let len = hist.len().min(self.max_ply);
            for t in 0..len {
                flat[bi * self.max_ply + t] = hist[t] as i64;
            }
            lengths.push(len as i32);
        }

        (flat, lengths)
    }

    /// Get sentinel tokens (first legal move token) for specified games.
    pub fn get_sentinel_tokens(&self, game_indices: &[u32]) -> Vec<u16> {
        game_indices
            .iter()
            .map(|&gi| {
                let tokens = self.games[gi as usize].legal_move_tokens();
                if tokens.is_empty() { 1 } else { tokens[0] }
            })
            .collect()
    }

    /// Get FEN strings for specified games (for Stockfish eval).
    pub fn get_fens(&self, game_indices: &[u32]) -> Vec<String> {
        game_indices
            .iter()
            .map(|&gi| self.games[gi as usize].fen())
            .collect()
    }

    /// Get UCI position strings for specified games (for UCI engine communication).
    pub fn get_uci_positions(&self, game_indices: &[u32]) -> Vec<String> {
        game_indices
            .iter()
            .map(|&gi| self.games[gi as usize].uci_position_string())
            .collect()
    }

    /// Get ply counts for specified games.
    pub fn get_plies(&self, game_indices: &[u32]) -> Vec<u32> {
        game_indices
            .iter()
            .map(|&gi| self.games[gi as usize].ply() as u32)
            .collect()
    }

    /// Per-game outcome data for all N games.
    /// Returns (terminated, forfeited, outcome_reward, agent_plies,
    ///          termination_codes, agent_is_white).
    pub fn get_outcomes(&self) -> (Vec<bool>, Vec<bool>, Vec<f32>, Vec<u32>, Vec<i8>, Vec<bool>) {
        let mut terminated = Vec::with_capacity(self.n_games);
        let mut forfeited = Vec::with_capacity(self.n_games);
        let mut rewards = Vec::with_capacity(self.n_games);
        let mut plies = Vec::with_capacity(self.n_games);
        let mut codes = Vec::with_capacity(self.n_games);
        let mut colors = Vec::with_capacity(self.n_games);

        for m in &self.meta {
            terminated.push(m.terminated);
            forfeited.push(m.forfeited);
            rewards.push(m.outcome_reward);
            plies.push(m.agent_plies);
            codes.push(m.termination_code);
            colors.push(m.agent_is_white);
        }

        (terminated, forfeited, rewards, plies, codes, colors)
    }

    /// Access a single game's state (for debugging/testing).
    pub fn game(&self, gi: usize) -> &GameState {
        &self.games[gi]
    }

    pub fn meta(&self, gi: usize) -> (bool, bool, bool, f32, u32, i8) {
        let m = &self.meta[gi];
        (
            m.agent_is_white,
            m.terminated,
            m.forfeited,
            m.outcome_reward,
            m.agent_plies,
            m.termination_code,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset_and_active_games() {
        let mut env = BatchRLEnv::new(4, 256, 42);
        env.reset();
        assert_eq!(env.n_games(), 4);
        assert!(!env.all_terminated());

        // First half (0,1) = agent white, should be active (white moves first)
        // Second half (2,3) = agent black, opponent moves first
        let agent = env.active_agent_games();
        let opp = env.active_opponent_games();
        assert_eq!(agent, vec![0, 1]);
        assert_eq!(opp, vec![2, 3]);
    }

    #[test]
    fn test_random_opponent_moves() {
        let mut env = BatchRLEnv::new(4, 256, 42);
        env.reset();

        // Apply random opponent opening moves for black-agent games
        let acted = env.apply_random_opponent_moves();
        assert_eq!(acted, vec![2, 3]); // only the games where opponent goes first

        // Now all 4 games should be active for the agent
        let agent = env.active_agent_games();
        assert_eq!(agent, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_fen_export() {
        let mut env = BatchRLEnv::new(1, 256, 42);
        env.reset();
        let fens = env.get_fens(&[0]);
        assert!(fens[0].contains("rnbqkbnr")); // starting position
    }

    #[test]
    fn test_legal_moves_batch() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        let (structured, masks) = env.get_legal_moves_batch(&[0, 1]);
        assert_eq!(structured.len(), 2);
        assert_eq!(masks.len(), 2 * 4096);
        // Starting position has 20 legal moves
        assert_eq!(structured[0].0.len(), 20);
    }

    #[test]
    fn test_apply_moves_legal() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        // e2e4 is legal for both games
        let e2e4 = crate::vocab::base_grid_token(12, 28);
        let (flags, codes) = env.apply_moves(&[0, 1], &[e2e4, e2e4]);
        assert_eq!(flags, vec![true, true]);
        assert_eq!(codes, vec![-1, -1]); // still going
    }

    #[test]
    fn test_apply_moves_illegal_forfeits() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        // PAD token (0) is never legal
        let (flags, codes) = env.apply_moves(&[0, 1], &[0u16, 0u16]);
        assert_eq!(flags, vec![false, false]);
        assert_eq!(codes, vec![-3, -3]); // forfeit
    }

    #[test]
    fn test_all_terminated_after_forfeit() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        let _ = env.apply_moves(&[0, 1], &[0u16, 0u16]);
        assert!(env.all_terminated());
    }

    #[test]
    fn test_get_move_histories_startpos() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        let (flat, lengths) = env.get_move_histories(&[0, 1]);
        assert_eq!(flat.len(), 2 * 256);
        assert_eq!(lengths, vec![0i32, 0i32]);
        // all entries should be 0 since no moves played
        assert!(flat.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_get_move_histories_after_move() {
        let mut env = BatchRLEnv::new(1, 256, 42);
        env.reset();
        let e2e4 = crate::vocab::base_grid_token(12, 28);
        let _ = env.apply_moves(&[0], &[e2e4]);
        let (flat, lengths) = env.get_move_histories(&[0]);
        assert_eq!(lengths, vec![1i32]);
        assert_eq!(flat[0], e2e4 as i64);
    }

    #[test]
    fn test_get_legal_token_masks_batch_size() {
        let mut env = BatchRLEnv::new(3, 256, 42);
        env.reset();
        let vocab_size = crate::vocab::VOCAB_SIZE;
        let masks = env.get_legal_token_masks_batch(&[0, 1, 2], vocab_size);
        assert_eq!(masks.len(), 3 * vocab_size);
        // Each game has 20 legal moves at startpos
        for bi in 0..3 {
            let count: usize = (0..vocab_size)
                .filter(|&v| masks[bi * vocab_size + v])
                .count();
            assert_eq!(count, 20, "game {} should have 20 legal moves", bi);
        }
    }

    #[test]
    fn test_get_plies() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        let plies = env.get_plies(&[0, 1]);
        assert_eq!(plies, vec![0u32, 0u32]);

        // Apply a move and check ply goes up
        let e2e4 = crate::vocab::base_grid_token(12, 28);
        let _ = env.apply_moves(&[0], &[e2e4]);
        let plies = env.get_plies(&[0, 1]);
        assert_eq!(plies, vec![1u32, 0u32]);
    }

    #[test]
    fn test_get_fens_startpos() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        let fens = env.get_fens(&[0, 1]);
        assert_eq!(fens.len(), 2);
        // Starting FEN has "rnbqkbnr" on rank 8
        assert!(fens[0].contains("rnbqkbnr"));
        assert!(fens[1].contains("rnbqkbnr"));
        // Starting white to move: "w"
        assert!(fens[0].contains(" w "));
    }

    #[test]
    fn test_get_outcomes_initial() {
        let mut env = BatchRLEnv::new(4, 256, 42);
        env.reset();
        let (term, forf, rew, plies, codes, colors) = env.get_outcomes();
        assert_eq!(term, vec![false; 4]);
        assert_eq!(forf, vec![false; 4]);
        assert_eq!(rew, vec![0.0f32; 4]);
        assert_eq!(plies, vec![0u32; 4]);
        assert_eq!(codes, vec![-1i8; 4]);
        // First half white, second half black
        assert_eq!(colors, vec![true, true, false, false]);
    }

    #[test]
    fn test_apply_moves_to_terminated_game_noop() {
        let mut env = BatchRLEnv::new(1, 256, 42);
        env.reset();
        // Force termination via invalid move
        let _ = env.apply_moves(&[0], &[0u16]);
        assert!(env.all_terminated());
        // Try another move — should be rejected
        let (flags, codes) = env.apply_moves(&[0], &[0u16]);
        assert_eq!(flags, vec![false]);
        assert_eq!(codes[0], -3); // still the original termination code
    }

    #[test]
    fn test_load_prefixes_basic() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        let e2e4 = crate::vocab::base_grid_token(12, 28);
        let e7e5 = crate::vocab::base_grid_token(52, 36);
        let max_ply = 8;
        let move_ids = vec![e2e4, e7e5, 0, 0, 0, 0, 0, 0,  // game 0: 2 moves
                            e2e4, 0, 0, 0, 0, 0, 0, 0];   // game 1: 1 move
        let lengths = vec![2u32, 1u32];
        let tc = env.load_prefixes(&move_ids, &lengths, 2, max_ply);
        assert_eq!(tc, vec![-1i8, -1i8]);
        let plies = env.get_plies(&[0, 1]);
        assert_eq!(plies, vec![2u32, 1u32]);
    }

    #[test]
    fn test_active_games_disjoint() {
        // agent_games and opponent_games together must cover all non-terminated games
        let mut env = BatchRLEnv::new(4, 256, 42);
        env.reset();
        let agent = env.active_agent_games();
        let opp = env.active_opponent_games();

        // No overlap
        for a in &agent {
            assert!(!opp.contains(a), "game {} should not be in both", a);
        }
        // Union is all games (none terminated)
        let mut all: Vec<u32> = agent.iter().chain(opp.iter()).copied().collect();
        all.sort();
        assert_eq!(all, vec![0u32, 1u32, 2u32, 3u32]);
    }

    #[test]
    fn test_meta_reports_agent_color() {
        let mut env = BatchRLEnv::new(4, 256, 42);
        env.reset();
        // First half = white, second half = black
        let (a_w, _, _, _, _, _) = env.meta(0);
        assert!(a_w);
        let (a_w, _, _, _, _, _) = env.meta(1);
        assert!(a_w);
        let (a_w, _, _, _, _, _) = env.meta(2);
        assert!(!a_w);
        let (a_w, _, _, _, _, _) = env.meta(3);
        assert!(!a_w);
    }

    #[test]
    fn test_get_sentinel_tokens() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        let sent = env.get_sentinel_tokens(&[0, 1]);
        assert_eq!(sent.len(), 2);
        // Each sentinel should be a legal move token (not PAD)
        assert_ne!(sent[0], crate::vocab::PAD_TOKEN);
        assert_ne!(sent[1], crate::vocab::PAD_TOKEN);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        let e2e4 = crate::vocab::base_grid_token(12, 28);
        let _ = env.apply_moves(&[0], &[e2e4]);
        let plies = env.get_plies(&[0]);
        assert_eq!(plies[0], 1);

        env.reset();
        let plies = env.get_plies(&[0]);
        assert_eq!(plies[0], 0, "after reset, ply count should be zero");
    }

    #[test]
    fn test_legal_token_masks_change_with_position() {
        // Starting from different positions, masks should differ.
        let mut env = BatchRLEnv::new(2, 256, 42);
        env.reset();
        let vocab_size = crate::vocab::VOCAB_SIZE;
        let masks_before = env.get_legal_token_masks_batch(&[0, 1], vocab_size);
        // They should be identical at start
        assert_eq!(
            &masks_before[..vocab_size],
            &masks_before[vocab_size..2 * vocab_size]
        );

        // Apply e2e4 to game 0 only
        let e2e4 = crate::vocab::base_grid_token(12, 28);
        let _ = env.apply_moves(&[0], &[e2e4]);
        let masks_after = env.get_legal_token_masks_batch(&[0, 1], vocab_size);
        // Game 0 now has black to move (different masks), game 1 still startpos
        assert_ne!(
            &masks_after[..vocab_size],
            &masks_after[vocab_size..2 * vocab_size],
            "masks should differ when games diverge"
        );
    }

    #[test]
    fn test_get_uci_positions_startpos() {
        let mut env = BatchRLEnv::new(1, 256, 42);
        env.reset();
        let pos = env.get_uci_positions(&[0]);
        assert_eq!(pos.len(), 1);
        // UCI position string should start with "startpos" or "position startpos"
        assert!(pos[0].contains("startpos"), "got: {}", pos[0]);
    }

    #[test]
    fn test_n_games_accessor() {
        let env = BatchRLEnv::new(7, 256, 42);
        assert_eq!(env.n_games(), 7);
    }
}
