#!/usr/bin/env python3
"""Run probes and diagnostics on one or more backbone PAWN checkpoints.

Each positional ``models`` argument is a model spec of the form
``[name=]<path>``, where ``<path>`` is either a HuggingFace model repo ID
(e.g. ``thomas-schweich/pawn-base``) or a local checkpoint directory.
``load_backbone_weights`` handles both transparently. The optional ``name=``
prefix overrides the auto-derived label used for result filenames; if
omitted, the label is taken from the last non-``checkpoints``/``step_*``
component of the path.

Results for each model land in ``<output_dir>/<name>/eval_results.json``.
"""

import argparse
import json
from pathlib import Path

import torch

from pawn.config import CLMConfig
from pawn.model import PAWNCLM
from pawn.checkpoint import load_backbone_weights
from pawn.gpu import configure_gpu
from pawn.eval_suite.probes import extract_probe_data, train_all_probes
from pawn.eval_suite.diagnostics import (
    generate_diagnostic_corpus,
    extract_diagnostic_positions,
    evaluate_diagnostic_positions,
)


def derive_name(path: str) -> str:
    """Best-effort label derivation from a model path.

    Walks the path components from right to left and returns the first
    component that isn't a checkpoint-flavoured boilerplate segment
    (``checkpoints``, ``best``, ``step_*``). Falls back to the basename
    or ``"model"`` when everything is boilerplate.
    """
    parts = [p for p in path.rstrip("/").split("/") if p]
    skip_literals = {"checkpoints", "best"}
    for p in reversed(parts):
        if p in skip_literals:
            continue
        if p.startswith("step_") or p.startswith("step-"):
            continue
        return p
    return parts[-1] if parts else "model"


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse a ``[name=]path`` model spec into ``(name, path)``.

    Only splits on the first ``=`` so paths containing ``=`` still work
    when an explicit name is given. Refuses empty names/paths.
    """
    if "=" in spec:
        name, _, path = spec.partition("=")
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise argparse.ArgumentTypeError(
                f"model spec {spec!r} must be of the form 'name=path' with "
                "both sides non-empty."
            )
        return name, path
    path = spec.strip()
    if not path:
        raise argparse.ArgumentTypeError("empty model spec")
    return derive_name(path), path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run probes and diagnostics on one or more trained PAWN backbones.",
    )
    parser.add_argument(
        "models", nargs="+",
        help=(
            "Model spec(s) as '[name=]path'. Path is a HuggingFace model "
            "repo ID (e.g. 'thomas-schweich/pawn-base') or a local "
            "checkpoint directory. The optional 'name=' prefix sets the "
            "output label; otherwise it's auto-derived from the path."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/evals"),
        help="Directory to write per-model eval_results.json (default: data/evals)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-probe-train", type=int, default=2048)
    parser.add_argument("--n-probe-val", type=int, default=512)
    parser.add_argument("--n-per-category", type=int, default=10_000)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument(
        "--prepend-outcome", action="store_true",
        help="Set when evaluating a model trained with prepend_outcome=True "
             "(pre-flip outcome-prefixed sequence layout).",
    )
    parser.add_argument(
        "--top-layer-only", action="store_true",
        help="Only train probes on the final hidden layer. Much faster than "
             "the default per-layer sweep, at the cost of the depth curve.",
    )
    args = parser.parse_args()

    specs: list[tuple[str, str]] = [parse_model_spec(s) for s in args.models]
    seen_names: set[str] = set()
    for name, _ in specs:
        if name in seen_names:
            parser.error(
                f"duplicate model name {name!r}; disambiguate with explicit "
                "'name=path' prefixes."
            )
        seen_names.add(name)

    device = args.device
    if device == "cuda":
        gpu_cfg = configure_gpu()
        import pawn.model as model_module
        if gpu_cfg.get("sdpa_backend"):
            model_module.SDPA_BACKEND = gpu_cfg["sdpa_backend"]

    print("Generating diagnostic corpus...", flush=True)
    corpus = generate_diagnostic_corpus(n_per_category=args.n_per_category)
    positions = extract_diagnostic_positions(corpus, max_per_category=args.n_per_category)

    # Probe data is sized per model from its ``max_seq_len`` so the
    # pipeline works on 256-ctx and 512-ctx checkpoints alike.

    for name, path in specs:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"EVALUATING {name}  ({path})")
        print(sep, flush=True)

        state_dict, model_config = load_backbone_weights(path)
        cfg = CLMConfig(**model_config) if model_config else CLMConfig()
        model = PAWNCLM(cfg).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

        probe_max_ply = model.cfg.max_seq_len
        print(f"Generating probe data (max_ply={probe_max_ply})...", flush=True)
        train_data = extract_probe_data(
            args.n_probe_train, probe_max_ply, seed=12345,
            prepend_outcome=args.prepend_outcome,
        )
        val_data = extract_probe_data(
            args.n_probe_val, probe_max_ply, seed=54321,
            prepend_outcome=args.prepend_outcome,
        )

        results: dict = {}

        probe_label = "top layer only" if args.top_layer_only else "per layer"
        print(f"\nRunning probes ({probe_label})...", flush=True)
        probe_results = train_all_probes(
            model, train_data, val_data, device=device,
            per_layer=not args.top_layer_only, n_epochs=args.n_epochs,
            verbose=True, prepend_outcome=args.prepend_outcome,
        )
        results["probes"] = probe_results

        print("\nRunning diagnostics...", flush=True)
        diag_results = evaluate_diagnostic_positions(
            model, positions, corpus, device=device,
            prepend_outcome=args.prepend_outcome,
        )
        results["diagnostics"] = diag_results

        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved: {out_path}", flush=True)

        del model, state_dict, train_data, val_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nALL EVALS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
