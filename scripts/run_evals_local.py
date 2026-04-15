#!/usr/bin/env python3
"""Run probes and diagnostics on all three trained models locally."""

import argparse
import json
import torch

from pawn.config import CLMConfig
from pawn.model import PAWNCLM
from pawn.checkpoint import load_backbone_weights
from pawn.gpu import configure_gpu, apply_gpu_config
from pawn.eval_suite.probes import extract_probe_data, train_all_probes
from pawn.eval_suite.diagnostics import (
    generate_diagnostic_corpus,
    extract_diagnostic_positions, evaluate_diagnostic_positions,
)


def main():
    parser = argparse.ArgumentParser(description="Run probes and diagnostics on trained PAWN models.")
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
    args = parser.parse_args()

    device = args.device
    if device == "cuda":
        gpu_cfg = configure_gpu()
        import pawn.model as model_module
        if gpu_cfg.get("sdpa_backend"):
            model_module.SDPA_BACKEND = gpu_cfg["sdpa_backend"]

    variants = {
        "small": {"path": "data/eval_small/checkpoints/step_00100000", "cfg": CLMConfig.small()},
        "base":  {"path": "data/eval_base/checkpoints/step_00100000",  "cfg": CLMConfig.base()},
        "large": {"path": "data/eval_large/checkpoints/step_00100000", "cfg": CLMConfig.large()},
    }

    print("Generating diagnostic corpus...", flush=True)
    corpus = generate_diagnostic_corpus(n_per_category=args.n_per_category)
    positions = extract_diagnostic_positions(corpus, max_per_category=args.n_per_category)

    # Probe data is sized per variant from the loaded model's
    # ``max_seq_len`` so it works on 256-ctx and 512-ctx checkpoints
    # alike.

    for name, info in variants.items():
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"EVALUATING {name}")
        print(sep, flush=True)

        state_dict, _ = load_backbone_weights(info["path"])
        model = PAWNCLM(info["cfg"]).to(device)
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

        results = {}

        print("\nRunning probes...", flush=True)
        probe_results = train_all_probes(
            model, train_data, val_data, device=device,
            per_layer=True, n_epochs=args.n_epochs, verbose=True,
            prepend_outcome=args.prepend_outcome,
        )
        results["probes"] = probe_results

        print("\nRunning diagnostics...", flush=True)
        diag_results = evaluate_diagnostic_positions(
            model, positions, corpus, device=device,
            prepend_outcome=args.prepend_outcome,
        )
        results["diagnostics"] = diag_results

        out_path = f"data/eval_{name}/eval_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved: {out_path}", flush=True)

        del model, state_dict, train_data, val_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nALL EVALS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
