#!/usr/bin/env python3
"""Generate HuggingFace model cards for PAWN variants.

Fetches eval_results.json and metrics.jsonl from each HuggingFace model repo,
renders the Jinja2 template, and optionally uploads the result.

This script intentionally fails loudly if any metrics are missing to ensure bad numbers don't
wind up getting posted due to e.g. a connection error. Do not add fallback values or default to
zero. Just use direct subscripting, division, etc. so that an exception is thrown if anything is
suspect. Better not to update the model card in such cases.

Usage:
    # Preview locally
    python scripts/generate_model_cards.py

    # Generate and upload to HuggingFace
    python scripts/generate_model_cards.py --push

    # Single variant
    python scripts/generate_model_cards.py --variants base --push
"""

import argparse
import json
from pathlib import Path

import jinja2


VARIANTS = {
    "small": {
        "repo": "thomas-schweich/pawn-small",
        "variant_name": "Small",
        "variant_label": "small",
        "variant_factory": "small",
    },
    "base": {
        "repo": "thomas-schweich/pawn-base",
        "variant_name": "Base",
        "variant_label": "base (default)",
        "variant_factory": "base",
    },
    "large": {
        "repo": "thomas-schweich/pawn-large",
        "variant_name": "Large",
        "variant_label": "large",
        "variant_factory": "large",
    },
}


def fetch_config(repo: str) -> dict:
    """Download config.json from a HuggingFace model repo."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "config.json")
    with open(path) as f:
        return json.load(f)


def params_str(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1_000_000:
        return f"~{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"~{n / 1_000:.0f}K"
    return str(n)


def count_params_from_weights(repo: str) -> int:
    """Count exact parameters from the safetensors weights on HuggingFace."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(repo, "model.safetensors")
    total = 0
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            total += f.get_tensor(key).numel()
    return total

CEILING_PATH = Path("data/theoretical_ceiling.json")


def load_ceilings() -> dict:
    """Load accuracy ceilings from the canonical JSON artifact."""
    if not CEILING_PATH.exists():
        raise FileNotFoundError(
            f"{CEILING_PATH} not found. Run scripts/compute_theoretical_ceiling.py first."
        )
    with open(CEILING_PATH) as f:
        data = json.load(f)
    return {
        "uncond": data["unconditional_ceiling"] * 100,
        "naive": data["naive_conditional_ceiling"] * 100,
        "mc_naive": data["conditional_ceiling"] * 100,
        "mc_corrected": data["conditional_corrected_ceiling"] * 100,
        "n_rollouts": data["n_rollouts"],
    }

PROBE_DESCRIPTIONS = {
    "piece_type": "Per-square piece type (13 classes x 64 squares)",
    "side_to_move": "Whose turn it is",
    "is_check": "Whether the side to move is in check",
    "castling_rights": "KQkq castling availability",
    "ep_square": "En passant target square (64 + none)",
    "material_count": "Piece counts per type per color",
    "legal_move_count": "Number of legal moves available",
    "halfmove_clock": "Plies since last capture or pawn move",
    "game_phase": "Opening / middlegame / endgame",
}

PROBE_NAMES = {
    "piece_type": "Piece type",
    "side_to_move": "Side to move",
    "is_check": "Is check",
    "castling_rights": "Castling rights",
    "ep_square": "En passant square",
    "material_count": "Material count",
    "legal_move_count": "Legal move count",
    "halfmove_clock": "Halfmove clock",
    "game_phase": "Game phase",
}

DIAGNOSTIC_NAMES = {
    "in_check": "In check",
    "double_check": "Double check",
    "pin_restricts": "Pin restricts movement",
    "ep_available": "En passant available",
    "castle_legal_k": "Castling legal (kingside)",
    "castle_legal_q": "Castling legal (queenside)",
    "castle_blocked_check": "Castling blocked by check",
    "promotion_available": "Promotion available",
    "checkmate": "Checkmate (terminal)",
    "stalemate": "Stalemate (terminal)",
}


def fetch_eval_results(repo: str) -> dict:
    """Download eval_results.json from a HuggingFace model repo."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "eval_results.json")
    with open(path) as f:
        return json.load(f)


def fetch_best_metrics(repo: str) -> dict:
    """Download metrics.jsonl and extract best val metrics.

    Returns the best val record by loss. If that record is missing extended
    fields (top5_accuracy, legal_move_rate), merges them from the best val
    record that does have them. This handles the case where val was logged
    every 500 steps but checkpoints (with backfilled extended metrics) only
    exist every 5K steps.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "metrics.jsonl")
    best_loss = float("inf")
    best = None
    best_extended_loss = float("inf")
    best_extended = None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "val":
                loss = r.get("val/loss", float("inf"))
                if loss < best_loss:
                    best_loss = loss
                    best = r
                if "val/top5_accuracy" in r and loss < best_extended_loss:
                    best_extended_loss = loss
                    best_extended = r
    if best is None:
        raise ValueError(f"No val records found in metrics.jsonl from {repo}")
    # Merge extended fields from the best record that has them
    if best_extended is not None:
        for key in ("val/top5_accuracy", "val/perplexity", "val/legal_move_rate"):
            if key not in best and key in best_extended:
                best[key] = best_extended[key]
    return best


def format_probe(eval_results: dict, probe_name: str) -> str:
    """Format a probe result."""
    probe = eval_results["probes"][probe_name]
    # Use the highest layer
    layer_key = sorted(probe.keys(), reverse=True)[0]
    data = probe[layer_key]
    acc = data.get("best_accuracy", data["accuracy"])
    mae = data.get("mae")
    if mae is not None:
        return f"{acc:.1%} (MAE {mae:.1f})"
    return f"{acc:.1%}"


def format_diagnostic(eval_results: dict, diag_name: str) -> tuple[str, str]:
    """Format a diagnostic result as (n_positions, value)."""
    diag = eval_results["diagnostics"][diag_name]
    n = diag["n_positions"]
    if diag_name in ("checkmate", "stalemate"):
        val = diag["mean_pad_prob"]
    else:
        val = diag["mean_legal_rate"]
    return str(n), f"{val:.1%}"


def build_context(variant_key: str, variant: dict) -> dict:
    """Build the full Jinja template context for a variant."""
    repo = variant["repo"]
    print(f"  Fetching config and metrics from {repo}...")

    ctx = dict(variant)
    ctx["variant_key"] = variant_key

    # Fetch model architecture from config.json
    config = fetch_config(repo)
    mc = config["model_config"]
    ctx["d_model"] = mc["d_model"]
    ctx["n_layers"] = mc["n_layers"]
    ctx["n_heads"] = mc["n_heads"]
    ctx["d_ff"] = mc["d_ff"]
    ctx["head_dim"] = ctx["d_model"] // ctx["n_heads"]
    ctx["params_num"] = count_params_from_weights(repo)
    ctx["params"] = params_str(ctx["params_num"])

    # Fetch training metrics
    best = fetch_best_metrics(repo)
    if not best:
        raise RuntimeError(f"Could not fetch metrics.jsonl from {repo}")
    ctx["top1"] = best["val/accuracy"] * 100
    ctx["val_loss"] = best["val/loss"]
    # These fields were added in later training runs — warn if missing
    if "val/top5_accuracy" not in best:
        print(f"  WARNING: val/top5_accuracy missing from {repo} metrics")
    if "val/legal_move_rate" not in best:
        print(f"  WARNING: val/legal_move_rate missing from {repo} metrics")
    ctx["top5"] = best.get("val/top5_accuracy", None)
    if ctx["top5"] is not None:
        ctx["top5"] *= 100
    ctx["legal_rate"] = best.get("val/legal_move_rate", None)
    if ctx["legal_rate"] is not None:
        ctx["legal_rate"] *= 100

    # Accuracy ratios
    ceil = load_ceilings()
    ctx["uncond_ceiling"] = ceil["uncond"]
    ctx["mc_naive_ceiling"] = ceil["mc_naive"]
    ctx["mc_corrected_ceiling"] = ceil["mc_corrected"]
    ctx["n_rollouts"] = ceil["n_rollouts"]
    ctx["uncond_ratio"] = round(ctx["top1"] / ceil["uncond"] * 100)
    ctx["mc_naive_ratio"] = round(ctx["top1"] / ceil["mc_naive"] * 100)
    ctx["mc_corrected_ratio"] = round(ctx["top1"] / ceil["mc_corrected"] * 100)

    # Fetch eval results for probes and diagnostics
    eval_results = fetch_eval_results(repo)
    if not eval_results:
        raise RuntimeError(f"Could not fetch eval_results.json from {repo}")

    ctx["probes"] = [
        {
            "name": PROBE_NAMES[k],
            "result": format_probe(eval_results, k),
            "description": PROBE_DESCRIPTIONS[k],
        }
        for k in PROBE_DESCRIPTIONS
        if k in eval_results.get("probes", {})
    ]

    ctx["diagnostics"] = []
    for k, name in DIAGNOSTIC_NAMES.items():
        if k in eval_results.get("diagnostics", {}):
            n, val = format_diagnostic(eval_results, k)
            ctx["diagnostics"].append({"name": name, "n": n, "value": val})

    return ctx


def main():
    parser = argparse.ArgumentParser(description="Generate HuggingFace model cards")
    parser.add_argument("--push", action="store_true", help="Upload cards to HuggingFace")
    parser.add_argument("--template", type=Path, default=Path("cards/hf_model_card.md.j2"))
    parser.add_argument("--output-dir", type=Path, default=Path("cards/model"))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    args = parser.parse_args()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(args.template.parent)),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    template = env.get_template(args.template.name)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for variant_key in args.variants:
        if variant_key not in VARIANTS:
            print(f"Unknown variant: {variant_key}")
            continue

        print(f"\n=== {VARIANTS[variant_key]['variant_name']} ===")
        ctx = build_context(variant_key, VARIANTS[variant_key])
        card = template.render(**ctx)

        output_path = args.output_dir / f"pawn-{variant_key}.md"
        output_path.write_text(card)
        print(f"  Written to {output_path}")

        if args.push:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo="README.md",
                repo_id=ctx["repo"],
                repo_type="model",
            )
            print(f"  Uploaded to {ctx['repo']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
