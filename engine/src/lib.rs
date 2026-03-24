pub mod types;
pub mod vocab;
pub mod board;
pub mod random;
pub mod pgn;
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

/// Generate random games without labels. Spec §7.3.
#[pyfunction]
#[pyo3(signature = (n_games, max_ply=256, seed=42, discard_ply_limit=false))]
fn generate_random_games<'py>(
    py: Python<'py>,
    n_games: usize,
    max_ply: usize,
    seed: u64,
    discard_ply_limit: bool,
) -> PyResult<(
    Bound<'py, PyArray2<i16>>,
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray1<u8>>,
)> {
    let result = py.allow_threads(|| {
        if discard_ply_limit {
            batch::generate_completed_games(n_games, max_ply, seed)
        } else {
            batch::generate_random_games(n_games, max_ply, seed)
        }
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
#[pyo3(signature = (batch_size, seq_len=256, seed=42, discard_ply_limit=false))]
fn generate_clm_batch<'py>(
    py: Python<'py>,
    batch_size: usize,
    seq_len: usize,
    seed: u64,
    discard_ply_limit: bool,
) -> PyResult<(
    Bound<'py, PyArray2<i16>>,   // input_ids (B, seq_len)
    Bound<'py, PyArray2<i16>>,   // targets (B, seq_len)
    Bound<'py, PyArray2<bool>>,  // loss_mask (B, seq_len)
    Bound<'py, PyArray2<i16>>,   // move_ids (B, seq_len-1)
    Bound<'py, PyArray1<i16>>,   // game_lengths (B,)
    Bound<'py, PyArray1<u8>>,    // termination_codes (B,)
)> {
    let result = py.allow_threads(|| {
        batch::generate_clm_batch(batch_size, seq_len, seed, discard_ply_limit)
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
    #[pyo3(signature = (game_indices, vocab_size=4278))]
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
    Ok(())
}
