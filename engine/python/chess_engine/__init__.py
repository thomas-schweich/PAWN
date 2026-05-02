"""ml-chess-engine-rs: High-performance chess engine for ML training pipelines.

All chess state management, move generation, and game simulation is handled
by the Rust engine. This package provides Python bindings via PyO3.
"""

from chess_engine._engine import (
    # Core game generation
    generate_training_batch,
    generate_random_games,
    generate_clm_batch,
    generate_checkmate_games,
    generate_checkmate_training_batch,
    # Diagnostic sets
    generate_diagnostic_sets_py as generate_diagnostic_sets,
    # Edge case statistics
    compute_edge_stats_per_ply_py as compute_edge_stats_per_ply,
    compute_edge_stats_per_game_py as compute_edge_stats_per_game,
    edge_case_bits,
    # Game validation & analysis
    compute_legal_move_masks,
    compute_legal_token_masks,
    compute_legal_token_masks_sparse,
    extract_board_states,
    validate_games_py as validate_games,
    # PGN parsing
    parse_pgn_file,
    pgn_to_tokens,
    parse_pgn_enriched,
    parse_pgn_lichess,
    count_pgn_games_in_date_range,
    parse_pgn_sampled,
    # UCI parsing
    parse_uci_file,
    uci_to_tokens,
    pgn_to_uci,
    uci_to_san,
    # Engine self-play generation
    generate_engine_games_py as generate_engine_games,
    # Vocabulary
    export_move_vocabulary,
    # Interactive game state (for RL)
    PyGameState,
    PyBatchRLEnv,
    # Accuracy ceiling
    compute_accuracy_ceiling_py as compute_accuracy_ceiling,
    # Utilities
    hello,
)

__all__ = [
    "generate_training_batch",
    "generate_random_games",
    "generate_clm_batch",
    "generate_checkmate_games",
    "generate_checkmate_training_batch",
    "generate_diagnostic_sets",
    "compute_edge_stats_per_ply",
    "compute_edge_stats_per_game",
    "edge_case_bits",
    "compute_legal_move_masks",
    "compute_legal_token_masks",
    "compute_legal_token_masks_sparse",
    "extract_board_states",
    "validate_games",
    "parse_pgn_file",
    "pgn_to_tokens",
    "parse_pgn_enriched",
    "parse_pgn_lichess",
    "count_pgn_games_in_date_range",
    "parse_pgn_sampled",
    "parse_uci_file",
    "uci_to_tokens",
    "pgn_to_uci",
    "uci_to_san",
    "generate_engine_games",
    "export_move_vocabulary",
    "PyGameState",
    "PyBatchRLEnv",
    "compute_accuracy_ceiling",
    "hello",
]
