"""PAWN Post-Training Evaluation Suite.

Implements the full validation spec: corpus generation, theoretical bounds,
representation probes, edge-case diagnostics, outcome token signal tests,
and Lichess generalization analysis.
"""

from .corpus import generate_corpus, load_corpus, sanity_checks, summary_stats
from .bounds import compute_theoretical_bounds
from .probes import PROBES, extract_probe_data, train_all_probes, train_single_probe
from .generation import (
    autoregressive_generate,
    outcome_signal_test,
    prefix_continuation_test,
    poisoned_prefix_test,
    impossible_task_test,
    improbable_task_test,
)
from .diagnostics import (
    generate_diagnostic_corpus,
    extract_diagnostic_positions,
    evaluate_diagnostic_positions,
)
from .lichess import prepare_lichess_corpus, evaluate_on_lichess
from .worker import (
    run_in_worker,
    run_probes,
    run_outcome_signal_test,
    run_prefix_continuation_test,
    run_poisoned_prefix_test,
    run_impossible_task_test,
    run_improbable_task_test,
    run_diagnostic_eval,
)
from . import viz
