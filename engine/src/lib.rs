pub mod types;
pub mod searchless_vocab;
pub mod vocab;
pub mod board;
pub mod random;
pub mod pgn;
pub mod uci;
pub mod engine_gen;
pub mod rl_batch;
pub mod batch;
pub mod labels;
pub mod edgestats;
pub mod diagnostic;
pub mod validate;
pub mod extract;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyUntypedArrayMethods};

/// Export the move vocabulary. Spec §7.1.
#[pyfunction]
fn export_move_vocabulary(py: Python<'_>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    let sq_names: Vec<&str> = vocab::SQUARE_NAMES.to_vec();
    dict.set_item("square_names", sq_names)?;

    let promo_pieces: Vec<&str> = vocab::PROMO_PIECES.to_vec();
    dict.set_item("promo_pieces", promo_pieces)?;

    let pairs = vocab::promo_pairs();
    let promo_pairs_list = PyList::empty(py);
    for &(s, d) in pairs.iter() {
        promo_pairs_list.append((s, d))?;
    }
    dict.set_item("promo_pairs", promo_pairs_list)?;

    let (t2m, m2t) = vocab::build_vocab_maps();
    let t2m_dict = PyDict::new(py);
    for (k, v) in &t2m {
        t2m_dict.set_item(*k, v.as_str())?;
    }
    dict.set_item("token_to_move", t2m_dict)?;

    let m2t_dict = PyDict::new(py);
    for (k, v) in &m2t {
        m2t_dict.set_item(k.as_str(), *v)?;
    }
    dict.set_item("move_to_token", m2t_dict)?;

    Ok(dict.into())
}

/// Generate a training batch. Spec §7.2.
#[pyfunction]
#[pyo3(signature = (batch_size, max_ply=256, seed=42))]
fn generate_training_batch<'py>(
    py: Python<'py>,
    batch_size: usize,
    max_ply: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<i16>>,
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray3<u64>>,
    Bound<'py, PyArray4<bool>>,
    Bound<'py, PyArray1<u8>>,
)> {
    let result = py.allow_threads(|| {
        batch::generate_training_batch(batch_size, max_ply, seed)
    });

    let move_ids = numpy::PyArray::from_vec(py, result.move_ids)
        .reshape([batch_size, max_ply])?;
    let game_lengths = numpy::PyArray::from_vec(py, result.game_lengths);
    let legal_move_grid = numpy::PyArray::from_vec(py, result.legal_move_grid)
        .reshape([batch_size, max_ply, 64])?;
    let legal_promo_mask = numpy::PyArray::from_vec(py, result.legal_promo_mask)
        .reshape([batch_size, max_ply, 44, 4])?;
    let termination_codes = numpy::PyArray::from_vec(py, result.termination_codes);

    Ok((move_ids, game_lengths, legal_move_grid, legal_promo_mask, termination_codes))
}

/// Generate checkmate-only games with balanced white/black wins.
#[pyfunction]
#[pyo3(signature = (n_white_wins, n_black_wins, max_ply=256, seed=42))]
fn generate_checkmate_games<'py>(
    py: Python<'py>,
    n_white_wins: usize,
    n_black_wins: usize,
    max_ply: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<i16>>,
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray1<u8>>,
    usize,  // total_generated
)> {
    let (result, total_generated) = py.allow_threads(|| {
        batch::generate_checkmate_games(n_white_wins, n_black_wins, max_ply, seed)
    });

    let n_games = result.n_games;
    let move_ids = numpy::PyArray::from_vec(py, result.move_ids)
        .reshape([n_games, max_ply])?;
    let game_lengths = numpy::PyArray::from_vec(py, result.game_lengths);
    let termination_codes = numpy::PyArray::from_vec(py, result.termination_codes);

    Ok((move_ids, game_lengths, termination_codes, total_generated))
}

/// Generate checkmate training batch with multi-hot mating move targets.
#[pyfunction]
#[pyo3(signature = (n_games, max_ply=256, seed=42))]
fn generate_checkmate_training_batch<'py>(
    py: Python<'py>,
    n_games: usize,
    max_ply: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<i16>>,   // move_ids (n, max_ply)
    Bound<'py, PyArray1<i16>>,   // game_lengths (n,)
    Bound<'py, PyArray2<u64>>,   // checkmate_targets (n, 64) bit-packed
    Bound<'py, PyArray2<u64>>,   // legal_grids (n, 64) bit-packed
    usize,                        // total_generated
)> {
    let result = py.allow_threads(|| {
        batch::generate_checkmate_training_batch(n_games, max_ply, seed)
    });

    let n = result.n_games;
    let move_ids = numpy::PyArray::from_vec(py, result.move_ids)
        .reshape([n, max_ply])?;
    let game_lengths = numpy::PyArray::from_vec(py, result.game_lengths);
    let checkmate_targets = numpy::PyArray::from_vec(py, result.checkmate_targets)
        .reshape([n, 64])?;
    let legal_grids = numpy::PyArray::from_vec(py, result.legal_grids)
        .reshape([n, 64])?;

    Ok((move_ids, game_lengths, checkmate_targets, legal_grids, result.total_generated))
}

/// Generate random games without labels.
#[pyfunction]
#[pyo3(signature = (n_games, max_ply=256, seed=42, discard_ply_limit=false, mate_boost=0.0))]
fn generate_random_games<'py>(
    py: Python<'py>,
    n_games: usize,
    max_ply: usize,
    seed: u64,
    discard_ply_limit: bool,
    mate_boost: f64,
) -> PyResult<(
    Bound<'py, PyArray2<i16>>,
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray1<u8>>,
)> {
    let result = py.allow_threads(|| {
        batch::generate_random_games(n_games, max_ply, seed, mate_boost, discard_ply_limit)
    });

    let move_ids = numpy::PyArray::from_vec(py, result.move_ids)
        .reshape([n_games, max_ply])?;
    let game_lengths = numpy::PyArray::from_vec(py, result.game_lengths);
    let termination_codes = numpy::PyArray::from_vec(py, result.termination_codes);

    Ok((move_ids, game_lengths, termination_codes))
}

/// Generate a CLM training batch with model-ready tensors.
///
/// Returns (input_ids, targets, loss_mask, move_ids, game_lengths, term_codes).
/// input_ids = [outcome, ply_1, ..., ply_N, PAD, ...] (seq_len per row).
/// move_ids are the raw moves (seq_len-1 per row) for replay operations.
#[pyfunction]
#[pyo3(signature = (batch_size, seq_len=256, seed=42, discard_ply_limit=false, mate_boost=0.0))]
fn generate_clm_batch<'py>(
    py: Python<'py>,
    batch_size: usize,
    seq_len: usize,
    seed: u64,
    discard_ply_limit: bool,
    mate_boost: f64,
) -> PyResult<(
    Bound<'py, PyArray2<i16>>,   // input_ids (B, seq_len)
    Bound<'py, PyArray2<i16>>,   // targets (B, seq_len)
    Bound<'py, PyArray2<bool>>,  // loss_mask (B, seq_len)
    Bound<'py, PyArray2<i16>>,   // move_ids (B, seq_len-1)
    Bound<'py, PyArray1<i16>>,   // game_lengths (B,)
    Bound<'py, PyArray1<u8>>,    // termination_codes (B,)
)> {
    let result = py.allow_threads(|| {
        batch::generate_clm_batch(batch_size, seq_len, seed, discard_ply_limit, mate_boost)
    });

    let max_ply = seq_len - 1;
    let input_ids = numpy::PyArray::from_vec(py, result.input_ids)
        .reshape([batch_size, seq_len])?;
    let targets = numpy::PyArray::from_vec(py, result.targets)
        .reshape([batch_size, seq_len])?;
    let loss_mask = numpy::PyArray::from_vec(py, result.loss_mask)
        .reshape([batch_size, seq_len])?;
    let move_ids = numpy::PyArray::from_vec(py, result.move_ids)
        .reshape([batch_size, max_ply])?;
    let game_lengths = numpy::PyArray::from_vec(py, result.game_lengths);
    let termination_codes = numpy::PyArray::from_vec(py, result.termination_codes);

    Ok((input_ids, targets, loss_mask, move_ids, game_lengths, termination_codes))
}

/// Compute legal move masks by replaying games. Spec §7.4.
#[pyfunction]
fn compute_legal_move_masks<'py>(
    py: Python<'py>,
    move_ids: numpy::PyReadonlyArray2<'py, i16>,
    game_lengths: numpy::PyReadonlyArray1<'py, i16>,
) -> PyResult<(
    Bound<'py, PyArray3<u64>>,
    Bound<'py, PyArray4<bool>>,
)> {
    let move_ids_slice = move_ids.as_slice()?;
    let game_lengths_slice = game_lengths.as_slice()?;
    let dims = move_ids.shape();
    let batch_size = dims[0];
    let max_ply = dims[1];

    if game_lengths_slice.len() != batch_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("game_lengths has {} elements but move_ids has {} rows", game_lengths_slice.len(), batch_size),
        ));
    }
    for (i, &gl) in game_lengths_slice.iter().enumerate() {
        if gl < 0 || (gl as usize) > max_ply {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("game_lengths[{}] = {} is out of range [0, {}]", i, gl, max_ply),
            ));
        }
    }

    let (grids, promos) = py.allow_threads(|| {
        labels::compute_legal_move_masks(move_ids_slice, game_lengths_slice, max_ply)
    });

    let grids_arr = numpy::PyArray::from_vec(py, grids)
        .reshape([batch_size, max_ply, 64])?;
    let promos_arr = numpy::PyArray::from_vec(py, promos)
        .reshape([batch_size, max_ply, 44, 4])?;

    Ok((grids_arr, promos_arr))
}

/// Replay games and return a dense (batch, max_ply, vocab_size) bool token mask.
/// Fuses replay with token mask construction — no intermediate bitboard grid.
#[pyfunction]
fn compute_legal_token_masks<'py>(
    py: Python<'py>,
    move_ids: numpy::PyReadonlyArray2<'py, i16>,
    game_lengths: numpy::PyReadonlyArray1<'py, i16>,
    vocab_size: usize,
) -> PyResult<Bound<'py, PyArray3<bool>>> {
    let move_ids_slice = move_ids.as_slice()?;
    let game_lengths_slice = game_lengths.as_slice()?;
    let dims = move_ids.shape();
    let batch_size = dims[0];
    let max_ply = dims[1];

    if game_lengths_slice.len() != batch_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("game_lengths has {} elements but move_ids has {} rows",
                    game_lengths_slice.len(), batch_size),
        ));
    }
    for (i, &gl) in game_lengths_slice.iter().enumerate() {
        if gl < 0 || (gl as usize) > max_ply {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("game_lengths[{}] = {} is out of range [0, {}]", i, gl, max_ply),
            ));
        }
    }

    let masks = py.allow_threads(|| {
        labels::compute_legal_token_masks(
            move_ids_slice, game_lengths_slice, max_ply, vocab_size,
        )
    });

    let arr = numpy::PyArray::from_vec(py, masks)
        .reshape([batch_size, max_ply, vocab_size])?;
    Ok(arr)
}

/// Sparse legal token mask: returns flat i64 indices for GPU scatter.
#[pyfunction]
fn compute_legal_token_masks_sparse<'py>(
    py: Python<'py>,
    move_ids: numpy::PyReadonlyArray2<'py, i16>,
    game_lengths: numpy::PyReadonlyArray1<'py, i16>,
    seq_len: usize,
    vocab_size: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let move_ids_slice = move_ids.as_slice()?;
    let game_lengths_slice = game_lengths.as_slice()?;
    let dims = move_ids.shape();
    let batch_size = dims[0];
    let max_ply = dims[1];

    if game_lengths_slice.len() != batch_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("game_lengths has {} elements but move_ids has {} rows",
                    game_lengths_slice.len(), batch_size),
        ));
    }
    for (i, &gl) in game_lengths_slice.iter().enumerate() {
        if gl < 0 || (gl as usize) > max_ply {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("game_lengths[{}] = {} is out of range [0, {}]", i, gl, max_ply),
            ));
        }
    }

    let indices = py.allow_threads(|| {
        labels::compute_legal_token_masks_sparse(
            move_ids_slice, game_lengths_slice, max_ply, seq_len, vocab_size,
        )
    });

    Ok(numpy::PyArray::from_vec(py, indices))
}

/// Extract board states. Spec §7.5.
#[pyfunction]
fn extract_board_states<'py>(
    py: Python<'py>,
    move_ids: numpy::PyReadonlyArray2<'py, i16>,
    game_lengths: numpy::PyReadonlyArray1<'py, i16>,
) -> PyResult<(
    Bound<'py, PyArray4<i8>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray2<i8>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<u8>>,
)> {
    let move_ids_slice = move_ids.as_slice()?;
    let game_lengths_slice = game_lengths.as_slice()?;
    let dims = move_ids.shape();
    let batch_size = dims[0];
    let max_ply = dims[1];

    if game_lengths_slice.len() != batch_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("game_lengths has {} elements but move_ids has {} rows", game_lengths_slice.len(), batch_size),
        ));
    }
    for (i, &gl) in game_lengths_slice.iter().enumerate() {
        if gl < 0 || (gl as usize) > max_ply {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("game_lengths[{}] = {} is out of range [0, {}]", i, gl, max_ply),
            ));
        }
    }

    let states = py.allow_threads(|| {
        extract::extract_board_states(move_ids_slice, game_lengths_slice, max_ply)
    });

    let boards = numpy::PyArray::from_vec(py, states.boards)
        .reshape([batch_size, max_ply, 8, 8])?;
    let side_to_move = numpy::PyArray::from_vec(py, states.side_to_move)
        .reshape([batch_size, max_ply])?;
    let castling_rights = numpy::PyArray::from_vec(py, states.castling_rights)
        .reshape([batch_size, max_ply])?;
    let ep_square = numpy::PyArray::from_vec(py, states.ep_square)
        .reshape([batch_size, max_ply])?;
    let is_check = numpy::PyArray::from_vec(py, states.is_check)
        .reshape([batch_size, max_ply])?;
    let halfmove_clock = numpy::PyArray::from_vec(py, states.halfmove_clock)
        .reshape([batch_size, max_ply])?;

    Ok((boards, side_to_move, castling_rights, ep_square, is_check, halfmove_clock))
}

/// Validate games. Spec §7.6.
#[pyfunction]
fn validate_games_py<'py>(
    py: Python<'py>,
    move_ids: numpy::PyReadonlyArray2<'py, i16>,
    game_lengths: numpy::PyReadonlyArray1<'py, i16>,
) -> PyResult<(
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<i16>>,
)> {
    let move_ids_slice = move_ids.as_slice()?;
    let game_lengths_slice = game_lengths.as_slice()?;
    let batch_size = move_ids.shape()[0];
    let max_ply = move_ids.shape()[1];

    if game_lengths_slice.len() != batch_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("game_lengths has {} elements but move_ids has {} rows", game_lengths_slice.len(), batch_size),
        ));
    }

    let (is_valid, first_illegal) = py.allow_threads(|| {
        validate::validate_games(move_ids_slice, game_lengths_slice, max_ply)
    });

    let is_valid_arr = numpy::PyArray::from_vec(py, is_valid);
    let first_illegal_arr = numpy::PyArray::from_vec(py, first_illegal);

    Ok((is_valid_arr, first_illegal_arr))
}

/// Compute per-ply edge case statistics. Spec §7.7.1.
#[pyfunction]
fn compute_edge_stats_per_ply_py<'py>(
    py: Python<'py>,
    move_ids: numpy::PyReadonlyArray2<'py, i16>,
    game_lengths: numpy::PyReadonlyArray1<'py, i16>,
) -> PyResult<(
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    let move_ids_slice = move_ids.as_slice()?;
    let game_lengths_slice = game_lengths.as_slice()?;
    let dims = move_ids.shape();
    let batch_size = dims[0];
    let max_ply = dims[1];

    if game_lengths_slice.len() != batch_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("game_lengths has {} elements but move_ids has {} rows", game_lengths_slice.len(), batch_size),
        ));
    }

    let (per_ply, white, black) = py.allow_threads(|| {
        edgestats::compute_edge_stats_per_ply(move_ids_slice, game_lengths_slice, max_ply)
    });

    let per_ply_arr = numpy::PyArray::from_vec(py, per_ply)
        .reshape([batch_size, max_ply])?;
    let white_arr = numpy::PyArray::from_vec(py, white);
    let black_arr = numpy::PyArray::from_vec(py, black);

    Ok((per_ply_arr, white_arr, black_arr))
}

/// Compute per-game edge case statistics. Spec §7.7.3.
#[pyfunction]
fn compute_edge_stats_per_game_py<'py>(
    py: Python<'py>,
    move_ids: numpy::PyReadonlyArray2<'py, i16>,
    game_lengths: numpy::PyReadonlyArray1<'py, i16>,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    let move_ids_slice = move_ids.as_slice()?;
    let game_lengths_slice = game_lengths.as_slice()?;
    let max_ply = move_ids.shape()[1];

    let (white, black) = py.allow_threads(|| {
        edgestats::compute_edge_stats_per_game(move_ids_slice, game_lengths_slice, max_ply)
    });

    let white_arr = numpy::PyArray::from_vec(py, white);
    let black_arr = numpy::PyArray::from_vec(py, black);

    Ok((white_arr, black_arr))
}

/// Generate diagnostic sets. Spec §7.8.
#[pyfunction]
#[pyo3(signature = (quotas_white, quotas_black, total_games, max_ply=256, seed=42, max_simulated_factor=100.0))]
fn generate_diagnostic_sets_py<'py>(
    py: Python<'py>,
    quotas_white: numpy::PyReadonlyArray1<'py, i32>,
    quotas_black: numpy::PyReadonlyArray1<'py, i32>,
    total_games: usize,
    max_ply: usize,
    seed: u64,
    max_simulated_factor: f64,
) -> PyResult<(
    Bound<'py, PyArray2<i16>>,
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
)> {
    let qw_slice = quotas_white.as_slice()?;
    let qb_slice = quotas_black.as_slice()?;

    if qw_slice.len() < 64 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("quotas_white has {} elements, expected at least 64", qw_slice.len()),
        ));
    }
    if qb_slice.len() < 64 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("quotas_black has {} elements, expected at least 64", qb_slice.len()),
        ));
    }

    let mut qw = [0i32; 64];
    let mut qb = [0i32; 64];
    qw.copy_from_slice(&qw_slice[..64]);
    qb.copy_from_slice(&qb_slice[..64]);

    let output = py.allow_threads(|| {
        diagnostic::generate_diagnostic_sets(&qw, &qb, total_games, max_ply, seed, max_simulated_factor)
    });

    let n = output.n_games;
    let move_ids = numpy::PyArray::from_vec(py, output.move_ids)
        .reshape([n, max_ply])?;
    let game_lengths = numpy::PyArray::from_vec(py, output.game_lengths);
    let termination_codes = numpy::PyArray::from_vec(py, output.termination_codes);
    let per_ply_stats = numpy::PyArray::from_vec(py, output.per_ply_stats)
        .reshape([n, max_ply])?;
    let white = numpy::PyArray::from_vec(py, output.white);
    let black = numpy::PyArray::from_vec(py, output.black);
    let qa_white = numpy::PyArray::from_vec(py, output.quota_assignment_white);
    let qa_black = numpy::PyArray::from_vec(py, output.quota_assignment_black);
    let qf_white = numpy::PyArray::from_vec(py, output.quotas_filled_white);
    let qf_black = numpy::PyArray::from_vec(py, output.quotas_filled_black);

    Ok((move_ids, game_lengths, termination_codes, per_ply_stats,
        white, black, qa_white, qa_black, qf_white, qf_black))
}

/// Export edge case bit constants.
#[pyfunction]
fn edge_case_bits(py: Python<'_>) -> PyResult<PyObject> {
    let bits = edgestats::edge_case_bits();
    let dict = PyDict::new(py);
    for (k, v) in &bits {
        dict.set_item(k.as_str(), *v)?;
    }
    Ok(dict.into())
}

/// Simple health check.
#[pyfunction]
fn hello() -> &'static str {
    "pawn-engine ready"
}

// ---------------------------------------------------------------------------
// Interactive GameState for RL
// ---------------------------------------------------------------------------

/// Python-visible interactive game state for RL training.
#[pyclass]
struct PyGameState {
    inner: board::GameState,
    max_ply: usize,
}

#[pymethods]
impl PyGameState {
    #[new]
    #[pyo3(signature = (max_ply=256))]
    fn new(max_ply: usize) -> Self {
        Self {
            inner: board::GameState::new(),
            max_ply,
        }
    }

    /// Apply a move given as a token index. Returns True if legal, False otherwise.
    fn make_move(&mut self, token: u16) -> bool {
        self.inner.make_move(token).is_ok()
    }

    /// Get all legal moves as a list of token indices.
    fn legal_move_tokens(&self) -> Vec<u16> {
        self.inner.legal_move_tokens()
    }

    /// Number of plies played so far.
    fn ply(&self) -> usize {
        self.inner.ply()
    }

    /// True if white to move.
    fn is_white_to_move(&self) -> bool {
        self.inner.is_white_to_move()
    }

    /// Check if game is over. Returns termination code (0-5) or -1 if not over.
    /// 0=Checkmate, 1=Stalemate, 2=SeventyFiveMoveRule,
    /// 3=FivefoldRepetition, 4=InsufficientMaterial, 5=PlyLimit
    fn check_termination(&self) -> i8 {
        match self.inner.check_termination(self.max_ply) {
            Some(t) => t as i8,
            None => -1,
        }
    }

    /// True if the game is over for any reason.
    fn is_game_over(&self) -> bool {
        self.inner.check_termination(self.max_ply).is_some()
    }

    /// True if current side is in checkmate (game over, current side lost).
    fn is_checkmate(&self) -> bool {
        self.inner.check_termination(self.max_ply) == Some(types::Termination::Checkmate)
    }

    /// True if current side is in check.
    fn is_check(&self) -> bool {
        self.inner.is_check()
    }

    /// Get the move history as a list of tokens.
    fn move_history(&self) -> Vec<u16> {
        self.inner.move_history().to_vec()
    }

    /// Get legal moves structured for RL: (grid_indices, promotions).
    /// grid_indices: flat src*64+dst for all legal moves (promo pairs deduplicated).
    /// promotions: list of (pair_idx, [promo_types]) for promotion moves.
    fn legal_moves_structured(&self) -> (Vec<u16>, Vec<(u16, Vec<u8>)>) {
        self.inner.legal_moves_structured()
    }

    /// Return dense 4096-element bool mask of legal grid cells.
    fn legal_moves_grid_mask(&self) -> Vec<bool> {
        self.inner.legal_moves_grid_mask().to_vec()
    }

    /// Get all legal move data in one call: (grid_indices, promotions, dense_mask).
    /// Computes legal_moves() once internally, avoiding redundant work.
    fn legal_moves_full(&self) -> (Vec<u16>, Vec<(u16, Vec<u8>)>, Vec<bool>) {
        let (indices, promos, mask) = self.inner.legal_moves_full();
        (indices, promos, mask.to_vec())
    }

    /// Apply a move and return its UCI string. Returns None if illegal.
    fn make_move_uci(&mut self, token: u16) -> Option<String> {
        self.inner.make_move_uci(token).ok()
    }

    /// Get UCI position string for engine communication.
    /// Returns "position startpos" or "position startpos moves e2e4 e7e5 ..."
    fn uci_position(&self) -> String {
        self.inner.uci_position_string()
    }
}

// ---------------------------------------------------------------------------
// PGN → token conversion
// ---------------------------------------------------------------------------

/// Read a PGN file, parse games, convert to tokens — all in Rust.
/// Returns (move_ids: (N, max_ply) i16, lengths: (N,) i16, n_parsed: int).
#[pyfunction]
#[pyo3(signature = (path, max_ply=128, max_games=1000000, min_ply=10))]
fn parse_pgn_file<'py>(
    py: Python<'py>,
    path: &str,
    max_ply: usize,
    max_games: usize,
    min_ply: usize,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<i16>>,
    Bound<'py, numpy::PyArray1<i16>>,
    usize,
)> {
    let (flat, lengths, n_parsed) = py.allow_threads(|| {
        pgn::pgn_file_to_tokens(path, max_ply, max_games, min_ply)
    });

    let n = lengths.len();
    let move_ids = numpy::PyArray::from_vec(py, flat)
        .reshape([n, max_ply])?;
    let lengths_arr = numpy::PyArray::from_vec(py, lengths);

    Ok((move_ids, lengths_arr, n_parsed))
}

/// Convert a batch of PGN games (each as a list of SAN move strings) to
/// token sequences. Returns (move_ids: (N, max_ply) i16, lengths: (N,) i16).
#[pyfunction]
#[pyo3(signature = (games, max_ply=256))]
fn pgn_to_tokens<'py>(
    py: Python<'py>,
    games: Vec<Vec<String>>,
    max_ply: usize,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<i16>>,
    Bound<'py, numpy::PyArray1<i16>>,
)> {
    let n = games.len();
    let (flat, lengths) = py.allow_threads(|| {
        let refs: Vec<Vec<&str>> = games
            .iter()
            .map(|g| g.iter().map(|s| s.as_str()).collect())
            .collect();
        pgn::batch_san_to_tokens(&refs, max_ply)
    });

    let move_ids = numpy::PyArray::from_vec(py, flat)
        .reshape([n, max_ply])?;
    let lengths_arr = numpy::PyArray::from_vec(py, lengths);

    Ok((move_ids, lengths_arr))
}

// ---------------------------------------------------------------------------
// UCI parsing
// ---------------------------------------------------------------------------

/// Parse a file of UCI game lines into token sequences.
/// File format: one game per line, space-separated UCI moves, optional result
/// marker (1-0, 0-1, 1/2-1/2, *).
///
/// Returns (move_ids: (N, max_ply) i16, lengths: (N,) i16, n_parsed: int).
#[pyfunction]
#[pyo3(signature = (path, max_ply=256, max_games=1_000_000, min_ply=10))]
fn parse_uci_file<'py>(
    py: Python<'py>,
    path: &str,
    max_ply: usize,
    max_games: usize,
    min_ply: usize,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<i16>>,
    Bound<'py, numpy::PyArray1<i16>>,
    usize,
)> {
    let (flat, lengths, n_parsed) = py.allow_threads(|| {
        uci::uci_file_to_tokens(path, max_ply, max_games, min_ply)
    });

    let n = lengths.len();
    let move_ids = numpy::PyArray::from_vec(py, flat)
        .reshape([n, max_ply])?;
    let lengths_arr = numpy::PyArray::from_vec(py, lengths);

    Ok((move_ids, lengths_arr, n_parsed))
}

/// Convert a batch of UCI game strings to token sequences.
/// Each game is a list of UCI move strings (e.g., ["e2e4", "e7e5"]).
///
/// Returns (move_ids: (N, max_ply) i16, lengths: (N,) i16).
#[pyfunction]
#[pyo3(signature = (games, max_ply=256))]
fn uci_to_tokens<'py>(
    py: Python<'py>,
    games: Vec<Vec<String>>,
    max_ply: usize,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<i16>>,
    Bound<'py, numpy::PyArray1<i16>>,
)> {
    let n = games.len();
    let (flat, lengths) = py.allow_threads(|| {
        let mut flat = vec![0i16; n * max_ply];
        let mut lengths = Vec::with_capacity(n);
        // Convert in parallel
        let results: Vec<(Vec<u16>, usize)> = games
            .iter()
            .map(|g| {
                let refs: Vec<&str> = g.iter().map(|s| s.as_str()).collect();
                uci::uci_moves_to_tokens(&refs, max_ply)
            })
            .collect();
        for (gi, (tokens, n_valid)) in results.iter().enumerate() {
            for (t, &tok) in tokens.iter().enumerate() {
                flat[gi * max_ply + t] = tok as i16;
            }
            lengths.push(*n_valid as i16);
        }
        (flat, lengths)
    });

    let move_ids = numpy::PyArray::from_vec(py, flat)
        .reshape([n, max_ply])?;
    let lengths_arr = numpy::PyArray::from_vec(py, lengths);

    Ok((move_ids, lengths_arr))
}

/// Convert a batch of PGN games (SAN moves) to UCI move strings.
/// Each game is a list of SAN strings (e.g., ["e4", "e5", "Nf3"]).
///
/// Returns a list of lists of UCI strings.
#[pyfunction]
fn pgn_to_uci(py: Python<'_>, games: Vec<Vec<String>>) -> PyResult<Vec<Vec<String>>> {
    let results = py.allow_threads(|| {
        let refs: Vec<Vec<&str>> = games
            .iter()
            .map(|g| g.iter().map(|s| s.as_str()).collect())
            .collect();
        uci::batch_san_to_uci(&refs)
    });

    Ok(results.into_iter().map(|(uci, _)| uci).collect())
}

// ---------------------------------------------------------------------------
// Enriched PGN parsing (for dataset construction)
// ---------------------------------------------------------------------------

/// Parse PGN text with full annotation extraction for dataset building.
///
/// Extracts move tokens, clock annotations, eval annotations, and all PGN
/// headers in a single pass. Designed for streaming: Python passes chunks of
/// PGN text (containing complete games), Rust returns structured columns.
///
/// Returns a dict with:
///   tokens: ndarray[i16, (N, max_ply)]   — PAWN token IDs, 0-padded
///   clocks: ndarray[u16, (N, max_ply)]   — seconds remaining, 0-padded
///   evals: ndarray[i16, (N, max_ply)]    — centipawns, 0-padded (i16::MIN = no annotation)
///   game_lengths: ndarray[u16, (N,)]     — number of plies per game
///   white_elo: ndarray[u16, (N,)]        — white Elo (0 if missing)
///   black_elo: ndarray[u16, (N,)]        — black Elo (0 if missing)
///   white_rating_diff: ndarray[i16, (N,)] — white rating change (0 if missing)
///   black_rating_diff: ndarray[i16, (N,)] — black rating change (0 if missing)
///   result: list[str]                     — "1-0", "0-1", "1/2-1/2", or ""
///   white: list[str]                      — white player name
///   black: list[str]                      — black player name
///   eco: list[str]                        — ECO code
///   opening: list[str]                    — opening name
///   time_control: list[str]               — time control string
///   termination: list[str]                — termination reason
///   date_time: list[str]                  — "YYYY.MM.DD HH:MM:SS" UTC
///   site: list[str]                       — game URL
#[pyfunction]
#[pyo3(signature = (content, max_ply=255, max_games=1_000_000, min_ply=1))]
fn parse_pgn_enriched<'py>(
    py: Python<'py>,
    content: &str,
    max_ply: usize,
    max_games: usize,
    min_ply: usize,
) -> PyResult<PyObject> {
    let games = py.allow_threads(|| {
        pgn::parse_pgn_enriched(content, max_ply, max_games, min_ply)
    });

    let n = games.len();
    let dict = PyDict::new(py);

    // Flat 0-padded arrays for tokens, clocks, evals (N * max_ply)
    let mut flat_tokens = vec![0i16; n * max_ply];
    let mut flat_clocks = vec![0u16; n * max_ply];
    let mut flat_evals = vec![0i16; n * max_ply];

    // Scalar arrays
    let mut lengths_out = Vec::with_capacity(n);
    let mut white_elo_out = Vec::with_capacity(n);
    let mut black_elo_out = Vec::with_capacity(n);
    let mut white_rd_out = Vec::with_capacity(n);
    let mut black_rd_out = Vec::with_capacity(n);

    // String lists
    let mut result_out = Vec::with_capacity(n);
    let mut white_out = Vec::with_capacity(n);
    let mut black_out = Vec::with_capacity(n);
    let mut eco_out = Vec::with_capacity(n);
    let mut opening_out = Vec::with_capacity(n);
    let mut tc_out = Vec::with_capacity(n);
    let mut term_out = Vec::with_capacity(n);
    let mut datetime_out = Vec::with_capacity(n);
    let mut site_out = Vec::with_capacity(n);

    for (gi, g) in games.iter().enumerate() {
        let offset = gi * max_ply;
        let len = g.game_length.min(max_ply);
        for t in 0..len {
            flat_tokens[offset + t] = g.tokens[t] as i16;
            flat_clocks[offset + t] = g.clocks[t];
            flat_evals[offset + t] = g.evals[t];
        }

        lengths_out.push(g.game_length as u16);

        let h = &g.headers;
        white_elo_out.push(h.get("WhiteElo").and_then(|s| s.parse::<u16>().ok()).unwrap_or(0));
        black_elo_out.push(h.get("BlackElo").and_then(|s| s.parse::<u16>().ok()).unwrap_or(0));
        white_rd_out.push(h.get("WhiteRatingDiff").and_then(|s| s.parse::<i16>().ok()).unwrap_or(0));
        black_rd_out.push(h.get("BlackRatingDiff").and_then(|s| s.parse::<i16>().ok()).unwrap_or(0));

        result_out.push(h.get("Result").cloned().unwrap_or_default());
        white_out.push(h.get("White").cloned().unwrap_or_default());
        black_out.push(h.get("Black").cloned().unwrap_or_default());
        eco_out.push(h.get("ECO").cloned().unwrap_or_default());
        opening_out.push(h.get("Opening").cloned().unwrap_or_default());
        tc_out.push(h.get("TimeControl").cloned().unwrap_or_default());
        term_out.push(h.get("Termination").cloned().unwrap_or_default());
        site_out.push(h.get("Site").cloned().unwrap_or_default());

        let date = h.get("UTCDate").cloned().unwrap_or_default();
        let time = h.get("UTCTime").cloned().unwrap_or_default();
        if !date.is_empty() && !time.is_empty() {
            datetime_out.push(format!("{} {}", date, time));
        } else {
            datetime_out.push(date);
        }
    }

    // 2D numpy arrays: (N, max_ply)
    let tokens_arr = numpy::PyArray::from_vec(py, flat_tokens).reshape([n, max_ply])?;
    let clocks_arr = numpy::PyArray::from_vec(py, flat_clocks).reshape([n, max_ply])?;
    let evals_arr = numpy::PyArray::from_vec(py, flat_evals).reshape([n, max_ply])?;

    // 1D numpy arrays
    let lengths_arr = numpy::PyArray::from_vec(py, lengths_out);
    let white_elo_arr = numpy::PyArray::from_vec(py, white_elo_out);
    let black_elo_arr = numpy::PyArray::from_vec(py, black_elo_out);
    let white_rd_arr = numpy::PyArray::from_vec(py, white_rd_out);
    let black_rd_arr = numpy::PyArray::from_vec(py, black_rd_out);

    dict.set_item("tokens", tokens_arr)?;
    dict.set_item("clocks", clocks_arr)?;
    dict.set_item("evals", evals_arr)?;
    dict.set_item("game_lengths", lengths_arr)?;
    dict.set_item("white_elo", white_elo_arr)?;
    dict.set_item("black_elo", black_elo_arr)?;
    dict.set_item("white_rating_diff", white_rd_arr)?;
    dict.set_item("black_rating_diff", black_rd_arr)?;
    dict.set_item("result", result_out)?;
    dict.set_item("white", white_out)?;
    dict.set_item("black", black_out)?;
    dict.set_item("eco", eco_out)?;
    dict.set_item("opening", opening_out)?;
    dict.set_item("time_control", tc_out)?;
    dict.set_item("termination", term_out)?;
    dict.set_item("date_time", datetime_out)?;
    dict.set_item("site", site_out)?;

    Ok(dict.into())
}

/// Parse Lichess PGN with outcome tokens prepended, no eval column.
///
/// Returns a dict with:
///   tokens: ndarray[i16, (N, max_ply+1)]  — [outcome, ply_1, ..., ply_N, 0, ...]
///   clocks: ndarray[u16, (N, max_ply)]    — clock per move ply (no outcome slot)
///   game_lengths: ndarray[u16, (N,)]      — move count (excl. outcome token)
///   original_lengths: ndarray[u32, (N,)]  — full game length before truncation
///   outcome_tokens: ndarray[u16, (N,)]    — outcome token ID per game
///   + all string metadata columns (same as parse_pgn_enriched)
#[pyfunction]
#[pyo3(signature = (content, max_ply=255, max_games=1_000_000, min_ply=1))]
fn parse_pgn_lichess<'py>(
    py: Python<'py>,
    content: &str,
    max_ply: usize,
    max_games: usize,
    min_ply: usize,
) -> PyResult<PyObject> {
    let games = py.allow_threads(|| {
        pgn::parse_pgn_lichess(content, max_ply, max_games, min_ply)
    });

    let n = games.len();
    let seq_len = max_ply + 1; // outcome token + max_ply moves
    let dict = PyDict::new(py);

    // Tokens: (N, seq_len) — outcome token at position 0, then moves, then 0-padding
    let mut flat_tokens = vec![0i16; n * seq_len];
    // Clocks: (N, max_ply) — parallel to moves only, no outcome slot
    let mut flat_clocks = vec![0u16; n * max_ply];
    let mut lengths_out = Vec::with_capacity(n);
    let mut orig_lengths_out = Vec::with_capacity(n);
    let mut outcome_tokens_out = Vec::with_capacity(n);

    let mut result_out = Vec::with_capacity(n);
    let mut white_out = Vec::with_capacity(n);
    let mut black_out = Vec::with_capacity(n);
    let mut white_elo_out = Vec::with_capacity(n);
    let mut black_elo_out = Vec::with_capacity(n);
    let mut white_rd_out = Vec::with_capacity(n);
    let mut black_rd_out = Vec::with_capacity(n);
    let mut eco_out = Vec::with_capacity(n);
    let mut opening_out = Vec::with_capacity(n);
    let mut tc_out = Vec::with_capacity(n);
    let mut term_out = Vec::with_capacity(n);
    let mut datetime_out = Vec::with_capacity(n);
    let mut site_out = Vec::with_capacity(n);

    for (gi, g) in games.iter().enumerate() {
        // tokens includes outcome at [0], moves at [1..=game_length]
        let tok_offset = gi * seq_len;
        for (t, &tok) in g.tokens.iter().enumerate() {
            if t < seq_len {
                flat_tokens[tok_offset + t] = tok as i16;
            }
        }

        // clocks parallel to moves only
        let clk_offset = gi * max_ply;
        for (t, &clk) in g.clocks.iter().enumerate() {
            if t < max_ply {
                flat_clocks[clk_offset + t] = clk;
            }
        }

        lengths_out.push(g.game_length as u16);
        orig_lengths_out.push(g.original_length as u32);
        outcome_tokens_out.push(g.outcome_token);

        let h = &g.headers;
        white_elo_out.push(h.get("WhiteElo").and_then(|s| s.parse::<u16>().ok()).unwrap_or(0));
        black_elo_out.push(h.get("BlackElo").and_then(|s| s.parse::<u16>().ok()).unwrap_or(0));
        white_rd_out.push(h.get("WhiteRatingDiff").and_then(|s| s.parse::<i16>().ok()).unwrap_or(0));
        black_rd_out.push(h.get("BlackRatingDiff").and_then(|s| s.parse::<i16>().ok()).unwrap_or(0));
        result_out.push(h.get("Result").cloned().unwrap_or_default());
        white_out.push(h.get("White").cloned().unwrap_or_default());
        black_out.push(h.get("Black").cloned().unwrap_or_default());
        eco_out.push(h.get("ECO").cloned().unwrap_or_default());
        opening_out.push(h.get("Opening").cloned().unwrap_or_default());
        tc_out.push(h.get("TimeControl").cloned().unwrap_or_default());
        term_out.push(h.get("Termination").cloned().unwrap_or_default());
        site_out.push(h.get("Site").cloned().unwrap_or_default());
        let date = h.get("UTCDate").cloned().unwrap_or_default();
        let time = h.get("UTCTime").cloned().unwrap_or_default();
        if !date.is_empty() && !time.is_empty() {
            datetime_out.push(format!("{} {}", date, time));
        } else {
            datetime_out.push(date);
        }
    }

    let tokens_arr = numpy::PyArray::from_vec(py, flat_tokens).reshape([n, seq_len])?;
    let clocks_arr = numpy::PyArray::from_vec(py, flat_clocks).reshape([n, max_ply])?;
    let lengths_arr = numpy::PyArray::from_vec(py, lengths_out);
    let orig_lengths_arr = numpy::PyArray::from_vec(py, orig_lengths_out);
    let outcome_tokens_arr = numpy::PyArray::from_vec(py, outcome_tokens_out);
    let white_elo_arr = numpy::PyArray::from_vec(py, white_elo_out);
    let black_elo_arr = numpy::PyArray::from_vec(py, black_elo_out);
    let white_rd_arr = numpy::PyArray::from_vec(py, white_rd_out);
    let black_rd_arr = numpy::PyArray::from_vec(py, black_rd_out);

    dict.set_item("tokens", tokens_arr)?;
    dict.set_item("clocks", clocks_arr)?;
    dict.set_item("game_lengths", lengths_arr)?;
    dict.set_item("original_lengths", orig_lengths_arr)?;
    dict.set_item("outcome_tokens", outcome_tokens_arr)?;
    dict.set_item("white_elo", white_elo_arr)?;
    dict.set_item("black_elo", black_elo_arr)?;
    dict.set_item("white_rating_diff", white_rd_arr)?;
    dict.set_item("black_rating_diff", black_rd_arr)?;
    dict.set_item("result", result_out)?;
    dict.set_item("white", white_out)?;
    dict.set_item("black", black_out)?;
    dict.set_item("eco", eco_out)?;
    dict.set_item("opening", opening_out)?;
    dict.set_item("time_control", tc_out)?;
    dict.set_item("termination", term_out)?;
    dict.set_item("date_time", datetime_out)?;
    dict.set_item("site", site_out)?;

    Ok(dict.into())
}

/// Count games in a PGN string whose UTCDate falls within [date_start, date_end].
/// Header-only scan — no tokenization. Very fast.
#[pyfunction]
fn count_pgn_games_in_date_range(
    py: Python<'_>,
    content: &str,
    date_start: &str,
    date_end: &str,
) -> PyResult<usize> {
    let count = py.allow_threads(|| {
        pgn::count_games_in_date_range(content, date_start, date_end)
    });
    Ok(count)
}

/// Parse only specific games (by index within a date range) from a PGN string.
///
/// Used for uniform random sampling: call count_pgn_games_in_date_range first
/// to get the total, generate random indices in Python, then call this to
/// parse only those games. `game_offset` is the cumulative count of
/// date-matching games from previous chunks.
///
/// Returns the same dict format as parse_pgn_enriched.
#[pyfunction]
#[pyo3(signature = (content, indices, date_start, date_end, game_offset=0, max_ply=255, min_ply=1))]
fn parse_pgn_sampled<'py>(
    py: Python<'py>,
    content: &str,
    indices: Vec<usize>,
    date_start: &str,
    date_end: &str,
    game_offset: usize,
    max_ply: usize,
    min_ply: usize,
) -> PyResult<PyObject> {
    let index_set: std::collections::HashSet<usize> = indices.into_iter().collect();

    let games = py.allow_threads(|| {
        pgn::parse_pgn_enriched_sampled(
            content, max_ply, min_ply, date_start, date_end, &index_set, game_offset,
        )
    });

    // Reuse the same dict-building logic as parse_pgn_enriched
    let n = games.len();
    let dict = PyDict::new(py);

    let mut flat_tokens = vec![0i16; n * max_ply];
    let mut flat_clocks = vec![0u16; n * max_ply];
    let mut flat_evals = vec![0i16; n * max_ply];
    let mut lengths_out = Vec::with_capacity(n);
    let mut white_elo_out = Vec::with_capacity(n);
    let mut black_elo_out = Vec::with_capacity(n);
    let mut white_rd_out = Vec::with_capacity(n);
    let mut black_rd_out = Vec::with_capacity(n);
    let mut result_out = Vec::with_capacity(n);
    let mut white_out = Vec::with_capacity(n);
    let mut black_out = Vec::with_capacity(n);
    let mut eco_out = Vec::with_capacity(n);
    let mut opening_out = Vec::with_capacity(n);
    let mut tc_out = Vec::with_capacity(n);
    let mut term_out = Vec::with_capacity(n);
    let mut datetime_out = Vec::with_capacity(n);
    let mut site_out = Vec::with_capacity(n);

    for (gi, g) in games.iter().enumerate() {
        let offset = gi * max_ply;
        let len = g.game_length.min(max_ply);
        for t in 0..len {
            flat_tokens[offset + t] = g.tokens[t] as i16;
            flat_clocks[offset + t] = g.clocks[t];
            flat_evals[offset + t] = g.evals[t];
        }
        lengths_out.push(g.game_length as u16);
        let h = &g.headers;
        white_elo_out.push(h.get("WhiteElo").and_then(|s| s.parse::<u16>().ok()).unwrap_or(0));
        black_elo_out.push(h.get("BlackElo").and_then(|s| s.parse::<u16>().ok()).unwrap_or(0));
        white_rd_out.push(h.get("WhiteRatingDiff").and_then(|s| s.parse::<i16>().ok()).unwrap_or(0));
        black_rd_out.push(h.get("BlackRatingDiff").and_then(|s| s.parse::<i16>().ok()).unwrap_or(0));
        result_out.push(h.get("Result").cloned().unwrap_or_default());
        white_out.push(h.get("White").cloned().unwrap_or_default());
        black_out.push(h.get("Black").cloned().unwrap_or_default());
        eco_out.push(h.get("ECO").cloned().unwrap_or_default());
        opening_out.push(h.get("Opening").cloned().unwrap_or_default());
        tc_out.push(h.get("TimeControl").cloned().unwrap_or_default());
        term_out.push(h.get("Termination").cloned().unwrap_or_default());
        site_out.push(h.get("Site").cloned().unwrap_or_default());
        let date = h.get("UTCDate").cloned().unwrap_or_default();
        let time = h.get("UTCTime").cloned().unwrap_or_default();
        if !date.is_empty() && !time.is_empty() {
            datetime_out.push(format!("{} {}", date, time));
        } else {
            datetime_out.push(date);
        }
    }

    let tokens_arr = numpy::PyArray::from_vec(py, flat_tokens).reshape([n, max_ply])?;
    let clocks_arr = numpy::PyArray::from_vec(py, flat_clocks).reshape([n, max_ply])?;
    let evals_arr = numpy::PyArray::from_vec(py, flat_evals).reshape([n, max_ply])?;
    let lengths_arr = numpy::PyArray::from_vec(py, lengths_out);
    let white_elo_arr = numpy::PyArray::from_vec(py, white_elo_out);
    let black_elo_arr = numpy::PyArray::from_vec(py, black_elo_out);
    let white_rd_arr = numpy::PyArray::from_vec(py, white_rd_out);
    let black_rd_arr = numpy::PyArray::from_vec(py, black_rd_out);

    dict.set_item("tokens", tokens_arr)?;
    dict.set_item("clocks", clocks_arr)?;
    dict.set_item("evals", evals_arr)?;
    dict.set_item("game_lengths", lengths_arr)?;
    dict.set_item("white_elo", white_elo_arr)?;
    dict.set_item("black_elo", black_elo_arr)?;
    dict.set_item("white_rating_diff", white_rd_arr)?;
    dict.set_item("black_rating_diff", black_rd_arr)?;
    dict.set_item("result", result_out)?;
    dict.set_item("white", white_out)?;
    dict.set_item("black", black_out)?;
    dict.set_item("eco", eco_out)?;
    dict.set_item("opening", opening_out)?;
    dict.set_item("time_control", tc_out)?;
    dict.set_item("termination", term_out)?;
    dict.set_item("date_time", datetime_out)?;
    dict.set_item("site", site_out)?;

    Ok(dict.into())
}

// ---------------------------------------------------------------------------
// UCI engine self-play generation
// ---------------------------------------------------------------------------

/// Generate self-play games using an external UCI engine (Stockfish, Lc0, etc).
///
/// Each worker spawns its own engine subprocess. Returns a dict with columns:
///   uci: list[str]        — space-separated UCI moves per game
///   result: list[str]     — "1-0", "0-1", "1/2-1/2"
///   n_ply: list[int]      — half-move count per game
///   worker_id: list[int]  — worker index
///   seed: list[int]       — worker RNG seed
#[pyfunction]
#[pyo3(signature = (
    engine_path,
    nodes = 1,
    total_games = 100_000,
    n_workers = 8,
    base_seed = 10_000,
    temperature = 1.0,
    multi_pv = 5,
    sample_plies = 999,
    hash_mb = 16,
    max_ply = 500,
))]
fn generate_engine_games_py(
    py: Python<'_>,
    engine_path: &str,
    nodes: u32,
    total_games: u32,
    n_workers: u32,
    base_seed: u64,
    temperature: f64,
    multi_pv: u32,
    sample_plies: u32,
    hash_mb: u32,
    max_ply: u32,
) -> PyResult<PyObject> {
    let results = py.allow_threads(|| {
        engine_gen::generate_engine_games(
            engine_path, nodes, total_games, n_workers, base_seed,
            temperature, multi_pv, sample_plies, hash_mb, max_ply,
        )
    });

    let dict = PyDict::new(py);

    let uci: Vec<String> = results.iter().map(|r| r.uci.clone()).collect();
    let result: Vec<String> = results.iter().map(|r| r.result.clone()).collect();
    let n_ply: Vec<u16> = results.iter().map(|r| r.n_ply).collect();

    // Reconstruct worker_id and seed from the generation order
    let base = total_games / n_workers;
    let remainder = total_games % n_workers;
    let mut worker_ids: Vec<u16> = Vec::with_capacity(results.len());
    let mut seeds: Vec<u64> = Vec::with_capacity(results.len());
    for i in 0..n_workers {
        let games = base + if i < remainder { 1 } else { 0 };
        for _ in 0..games {
            worker_ids.push(i as u16);
            seeds.push(base_seed + i as u64);
        }
    }

    dict.set_item("uci", uci)?;
    dict.set_item("result", result)?;
    dict.set_item("n_ply", n_ply)?;
    dict.set_item("worker_id", worker_ids)?;
    dict.set_item("seed", seeds)?;
    dict.set_item("nodes", vec![nodes as i32; results.len()])?;

    Ok(dict.into())
}

// ---------------------------------------------------------------------------
// Batch RL environment (all game state in Rust)
// ---------------------------------------------------------------------------

#[pyclass]
struct PyBatchRLEnv {
    inner: rl_batch::BatchRLEnv,
}

#[pymethods]
impl PyBatchRLEnv {
    #[new]
    #[pyo3(signature = (n_games, max_ply=256, seed=42))]
    fn new(n_games: usize, max_ply: usize, seed: u64) -> Self {
        Self {
            inner: rl_batch::BatchRLEnv::new(n_games, max_ply, seed),
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn n_games(&self) -> usize {
        self.inner.n_games()
    }

    fn all_terminated(&self) -> bool {
        self.inner.all_terminated()
    }

    fn active_agent_games(&self) -> Vec<u32> {
        self.inner.active_agent_games()
    }

    fn active_opponent_games(&self) -> Vec<u32> {
        self.inner.active_opponent_games()
    }

    /// Apply moves regardless of side. Returns (legality_bools, term_codes).
    fn apply_moves<'py>(
        &mut self,
        py: Python<'py>,
        game_indices: Vec<u32>,
        tokens: Vec<u16>,
    ) -> (
        Bound<'py, numpy::PyArray1<bool>>,
        Bound<'py, numpy::PyArray1<i8>>,
    ) {
        let (flags, codes) = self.inner.apply_moves(&game_indices, &tokens);
        (
            numpy::PyArray::from_vec(py, flags),
            numpy::PyArray::from_vec(py, codes),
        )
    }

    /// Load prefix move sequences. Returns per-game term codes (-1 = still going).
    fn load_prefixes<'py>(
        &mut self,
        py: Python<'py>,
        move_ids: numpy::PyReadonlyArray2<'py, u16>,
        lengths: numpy::PyReadonlyArray1<'py, u32>,
    ) -> Bound<'py, numpy::PyArray1<i8>> {
        let ids = move_ids.as_slice().unwrap();
        let lens = lengths.as_slice().unwrap();
        let n_games = lens.len();
        let max_ply = if n_games > 0 { ids.len() / n_games } else { 0 };
        let codes = self.inner.load_prefixes(ids, lens, n_games, max_ply);
        numpy::PyArray::from_vec(py, codes)
    }

    /// Apply agent moves. Returns list of legality bools.
    fn apply_agent_moves(&mut self, game_indices: Vec<u32>, tokens: Vec<u16>) -> Vec<bool> {
        self.inner.apply_agent_moves(&game_indices, &tokens)
    }

    /// Random opponent moves for all pending opponent games. Returns acted game indices.
    fn apply_random_opponent_moves(&mut self) -> Vec<u32> {
        self.inner.apply_random_opponent_moves()
    }

    /// Apply UCI engine opponent moves. Returns legality bools.
    fn apply_opponent_moves(&mut self, game_indices: Vec<u32>, tokens: Vec<u16>) -> Vec<bool> {
        self.inner.apply_opponent_moves(&game_indices, &tokens)
    }

    /// Legal move token masks: numpy bool (B, vocab_size).
    /// Includes promotion tokens — correct for autoregressive generation.
    #[pyo3(signature = (game_indices, vocab_size=4284))]
    fn get_legal_token_masks_batch<'py>(
        &self,
        py: Python<'py>,
        game_indices: Vec<u32>,
        vocab_size: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray2<bool>>> {
        let b = game_indices.len();
        let flat = self.inner.get_legal_token_masks_batch(&game_indices, vocab_size);
        let arr = numpy::PyArray::from_vec(py, flat)
            .reshape([b, vocab_size])?;
        Ok(arr)
    }

    /// Bulk legal move data: returns (structured_list, dense_mask_array).
    /// structured_list: list of (grid_indices, promotions) per game.
    /// dense_mask_array: numpy bool (B, 4096).
    fn get_legal_moves_batch<'py>(
        &self,
        py: Python<'py>,
        game_indices: Vec<u32>,
    ) -> PyResult<(
        Vec<(Vec<u16>, Vec<(u16, Vec<u8>)>)>,
        Bound<'py, numpy::PyArray2<bool>>,
    )> {
        let b = game_indices.len();
        let (structured, flat_masks) = self.inner.get_legal_moves_batch(&game_indices);
        let mask_array = numpy::PyArray::from_vec(py, flat_masks)
            .reshape([b, 4096])?;
        Ok((structured, mask_array))
    }

    /// Move histories as numpy arrays: (move_ids (B, max_ply) i64, lengths (B,) i32).
    fn get_move_histories<'py>(
        &self,
        py: Python<'py>,
        game_indices: Vec<u32>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray2<i64>>,
        Bound<'py, numpy::PyArray1<i32>>,
    )> {
        let b = game_indices.len();
        let (flat, lengths) = self.inner.get_move_histories(&game_indices);
        let cols = if b > 0 { flat.len() / b } else { 0 };
        let move_ids = numpy::PyArray::from_vec(py, flat)
            .reshape([b, cols])?;
        let lengths_arr = numpy::PyArray::from_vec(py, lengths);
        Ok((move_ids, lengths_arr))
    }

    /// Sentinel tokens for specified games.
    fn get_sentinel_tokens(&self, game_indices: Vec<u32>) -> Vec<u16> {
        self.inner.get_sentinel_tokens(&game_indices)
    }

    /// FEN strings for specified games (for Stockfish eval).
    fn get_fens(&self, game_indices: Vec<u32>) -> Vec<String> {
        self.inner.get_fens(&game_indices)
    }

    /// UCI position strings for specified games (for UCI engine communication).
    fn get_uci_positions(&self, game_indices: Vec<u32>) -> Vec<String> {
        self.inner.get_uci_positions(&game_indices)
    }

    /// Ply counts for specified games.
    fn get_plies(&self, game_indices: Vec<u32>) -> Vec<u32> {
        self.inner.get_plies(&game_indices)
    }

    /// Per-game outcomes as numpy arrays:
    /// (terminated, forfeited, outcome_reward, agent_plies, term_codes, agent_is_white)
    fn get_outcomes<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, numpy::PyArray1<bool>>,
        Bound<'py, numpy::PyArray1<bool>>,
        Bound<'py, numpy::PyArray1<f32>>,
        Bound<'py, numpy::PyArray1<u32>>,
        Bound<'py, numpy::PyArray1<i8>>,
        Bound<'py, numpy::PyArray1<bool>>,
    ) {
        let (terminated, forfeited, rewards, plies, codes, colors) = self.inner.get_outcomes();
        (
            numpy::PyArray::from_vec(py, terminated),
            numpy::PyArray::from_vec(py, forfeited),
            numpy::PyArray::from_vec(py, rewards),
            numpy::PyArray::from_vec(py, plies),
            numpy::PyArray::from_vec(py, codes),
            numpy::PyArray::from_vec(py, colors),
        )
    }

    /// Single game info: (agent_is_white, terminated, forfeited, outcome_reward, agent_plies, term_code)
    fn get_game_info(&self, game_idx: u32) -> (bool, bool, bool, f32, u32, i8) {
        self.inner.meta(game_idx as usize)
    }
}

/// Compute theoretical accuracy ceiling via Monte Carlo rollouts.
///
/// For a sample of positions from random games, estimates:
/// - Unconditional ceiling: E[1/N_legal]
/// - Naive conditional ceiling (0-depth): prune moves that immediately
///   terminate with the wrong outcome, then E[1/N_remaining]
/// - MC conditional ceiling: E[max_m P(m|outcome, history)] via rollouts
///
/// Outcomes are side-aware: WhiteCheckmated and BlackCheckmated are
/// separate buckets (fixing a bug where all checkmates were conflated).
///
/// Returns dict with overall ceilings and per-position data.
#[pyfunction]
#[pyo3(signature = (n_games=1000, max_ply=255, n_rollouts=32, sample_rate=0.01, seed=77777))]
fn compute_accuracy_ceiling_py(
    py: Python<'_>,
    n_games: usize,
    max_ply: usize,
    n_rollouts: usize,
    sample_rate: f64,
    seed: u64,
) -> PyResult<PyObject> {
    let results = py.allow_threads(|| {
        random::compute_accuracy_ceiling(n_games, max_ply, n_rollouts, sample_rate, seed)
    });

    let n = results.len();
    let mut uncond_sum = 0.0f64;
    let mut cond_sum = 0.0f64;
    let mut naive_cond_sum = 0.0f64;
    let mut corrected_cond_sum = 0.0f64;

    // Build per-position arrays
    let mut plies = Vec::with_capacity(n);
    let mut game_lengths = Vec::with_capacity(n);
    let mut n_legals = Vec::with_capacity(n);
    let mut unconditionals = Vec::with_capacity(n);
    let mut conditionals = Vec::with_capacity(n);
    let mut naive_conditionals = Vec::with_capacity(n);
    let mut corrected_conditionals = Vec::with_capacity(n);
    let mut outcomes = Vec::with_capacity(n);
    let mut game_indices = Vec::with_capacity(n);

    for r in &results {
        uncond_sum += r.unconditional;
        cond_sum += r.conditional;
        naive_cond_sum += r.naive_conditional;
        corrected_cond_sum += r.conditional_corrected;
        plies.push(r.ply);
        game_lengths.push(r.game_length);
        n_legals.push(r.n_legal);
        unconditionals.push(r.unconditional as f32);
        conditionals.push(r.conditional as f32);
        naive_conditionals.push(r.naive_conditional as f32);
        corrected_conditionals.push(r.conditional_corrected as f32);
        outcomes.push(r.actual_outcome);
        game_indices.push(r.game_idx);
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("n_positions", n)?;
    dict.set_item("n_games", n_games)?;
    dict.set_item("n_rollouts", n_rollouts)?;
    dict.set_item("sample_rate", sample_rate)?;
    dict.set_item("unconditional_ceiling", if n > 0 { uncond_sum / n as f64 } else { 0.0 })?;
    dict.set_item("naive_conditional_ceiling", if n > 0 { naive_cond_sum / n as f64 } else { 0.0 })?;
    dict.set_item("conditional_ceiling", if n > 0 { cond_sum / n as f64 } else { 0.0 })?;
    dict.set_item("conditional_corrected_ceiling", if n > 0 { corrected_cond_sum / n as f64 } else { 0.0 })?;

    // Return numpy arrays for per-position data
    let np = py.import("numpy")?;
    dict.set_item("ply", np.call_method1("array", (plies,))?)?;
    dict.set_item("game_length", np.call_method1("array", (game_lengths,))?)?;
    dict.set_item("n_legal", np.call_method1("array", (n_legals,))?)?;
    dict.set_item("unconditional", np.call_method1("array", (unconditionals,))?)?;
    dict.set_item("naive_conditional", np.call_method1("array", (naive_conditionals,))?)?;
    dict.set_item("conditional", np.call_method1("array", (conditionals,))?)?;
    dict.set_item("conditional_corrected", np.call_method1("array", (corrected_conditionals,))?)?;
    let outcomes_u16: Vec<u16> = outcomes.iter().map(|&x| x as u16).collect();
    dict.set_item("outcome", np.call_method1("array", (outcomes_u16,))?)?;
    dict.set_item("game_idx", np.call_method1("array", (game_indices,))?)?;

    Ok(dict.into())
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(export_move_vocabulary, m)?)?;
    m.add_function(wrap_pyfunction!(generate_training_batch, m)?)?;
    m.add_function(wrap_pyfunction!(generate_random_games, m)?)?;
    m.add_function(wrap_pyfunction!(generate_clm_batch, m)?)?;
    m.add_function(wrap_pyfunction!(generate_checkmate_games, m)?)?;
    m.add_function(wrap_pyfunction!(generate_checkmate_training_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute_legal_move_masks, m)?)?;
    m.add_function(wrap_pyfunction!(compute_legal_token_masks, m)?)?;
    m.add_function(wrap_pyfunction!(compute_legal_token_masks_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(extract_board_states, m)?)?;
    m.add_function(wrap_pyfunction!(validate_games_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_edge_stats_per_ply_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_edge_stats_per_game_py, m)?)?;
    m.add_function(wrap_pyfunction!(generate_diagnostic_sets_py, m)?)?;
    m.add_function(wrap_pyfunction!(edge_case_bits, m)?)?;
    m.add_class::<PyGameState>()?;
    m.add_class::<PyBatchRLEnv>()?;
    m.add_function(wrap_pyfunction!(parse_pgn_file, m)?)?;
    m.add_function(wrap_pyfunction!(pgn_to_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(parse_uci_file, m)?)?;
    m.add_function(wrap_pyfunction!(uci_to_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(pgn_to_uci, m)?)?;
    m.add_function(wrap_pyfunction!(parse_pgn_enriched, m)?)?;
    m.add_function(wrap_pyfunction!(parse_pgn_lichess, m)?)?;
    m.add_function(wrap_pyfunction!(count_pgn_games_in_date_range, m)?)?;
    m.add_function(wrap_pyfunction!(parse_pgn_sampled, m)?)?;
    m.add_function(wrap_pyfunction!(generate_engine_games_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_accuracy_ceiling_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Tests for lib.rs: PyO3 glue, vocab constants visibility, and
    //! cross-module dispatch invariants.
    //!
    //! PyO3-wrapped functions cannot be called directly in unit tests (they
    //! require a Python interpreter), so these tests exercise the underlying
    //! Rust functions and verify the module layout.

    use super::*;

    #[test]
    fn test_decompose_token_dispatch_pad_range() {
        // Dispatch: PAD_TOKEN returns None
        assert!(vocab::decompose_token(vocab::PAD_TOKEN).is_none());
    }

    #[test]
    fn test_decompose_token_action_range() {
        // All 1968 actions should decompose
        for action in 0..vocab::NUM_ACTIONS as u16 {
            assert!(vocab::decompose_token(action).is_some(),
                "action {} should decompose", action);
        }
    }

    #[test]
    fn test_decompose_token_promo_actions() {
        // Promotion actions should have promo >= 1
        let a7a8q = vocab::uci_token("a7a8q");
        let (_, _, promo) = vocab::decompose_token(a7a8q).unwrap();
        assert!(promo >= 1 && promo <= 4, "promo action should have 1<=promo<=4");
    }

    #[test]
    fn test_decompose_token_dispatch_outcome_range() {
        // Outcome tokens return None
        for t in vocab::OUTCOME_BASE..=vocab::DRAW_BY_TIME {
            assert!(vocab::decompose_token(t).is_none(),
                "outcome token {} should not decompose", t);
        }
    }

    #[test]
    fn test_decompose_token_boundary_values() {
        // Boundaries: actions [0, 1967], PAD [1968], outcomes [1969, 1979]
        assert!(vocab::decompose_token(0).is_some());    // first action
        assert!(vocab::decompose_token(1967).is_some());  // last action
        assert!(vocab::decompose_token(1968).is_none());  // PAD
        assert!(vocab::decompose_token(1969).is_none());  // first outcome
        assert!(vocab::decompose_token(1979).is_none());  // last outcome
        assert!(vocab::decompose_token(1980).is_none());  // beyond vocab
    }

    #[test]
    fn test_module_exports_hello() {
        // Pure Rust test of the hello() function that's exported to Python
        assert_eq!(hello(), "pawn-engine ready");
    }

    #[test]
    fn test_square_names_accessible_from_lib() {
        // PyO3 export_move_vocabulary accesses SQUARE_NAMES
        assert_eq!(vocab::SQUARE_NAMES.len(), 64);
        assert_eq!(vocab::SQUARE_NAMES[0], "a1");
        assert_eq!(vocab::SQUARE_NAMES[63], "h8");
    }

    #[test]
    fn test_promo_pieces_accessible_from_lib() {
        assert_eq!(vocab::PROMO_PIECES.len(), 4);
        assert_eq!(vocab::PROMO_PIECES, ["q", "r", "b", "n"]);
    }
}
