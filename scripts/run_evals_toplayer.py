#!/usr/bin/env python3
"""Run top-layer probes and diagnostics on all three trained models. Fast version."""

import json
import torch

from pawn.config import CLMConfig
from pawn.model import PAWNCLM
from pawn.checkpoint import load_backbone_weights
from pawn.gpu import configure_gpu
from pawn.eval_suite.probes import extract_probe_data, train_all_probes
from pawn.eval_suite.diagnostics import (
    generate_diagnostic_corpus,
    extract_diagnostic_positions, evaluate_diagnostic_positions,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
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

print("Generating probe data...", flush=True)
train_data = extract_probe_data(2048, 256, seed=12345)
val_data = extract_probe_data(512, 256, seed=54321)
print("Done.", flush=True)

# Generate diagnostic corpus (quota-controlled for rare edge cases)
print("Generating diagnostic corpus...", flush=True)
corpus = generate_diagnostic_corpus(n_per_category=10_000)
positions = extract_diagnostic_positions(corpus, max_per_category=10_000)
print("Done.", flush=True)

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

    results = {}

    # Top-layer probes only (per_layer=False)
    print("\nRunning probes (top layer only)...", flush=True)
    probe_results = train_all_probes(
        model, train_data, val_data, device=device,
        per_layer=False, n_epochs=20, verbose=True,
    )
    results["probes"] = probe_results

    # Diagnostics
    print("\nRunning diagnostics...", flush=True)
    diag_results = evaluate_diagnostic_positions(model, positions, corpus, device=device)
    results["diagnostics"] = diag_results

    out_path = f"data/eval_{name}/eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out_path}", flush=True)

    del model, state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nALL EVALS COMPLETE", flush=True)
