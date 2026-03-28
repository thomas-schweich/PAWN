#!/usr/bin/env python3
"""Generate HuggingFace model cards for PAWN variants.

Fetches eval_results.json and metrics.jsonl from each HuggingFace model repo,
renders the Jinja2 template, and optionally uploads the result.

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
        "variant_key": "small",
        "params": "~9.5M",
        "params_num": 9524224,
        "d_model": 256,
        "n_layers": 8,
        "n_heads": 4,
        "d_ff": 1024,
        "head_dim": 64,
    },
    "base": {
        "repo": "thomas-schweich/pawn-base",
        "variant_name": "Base",
        "variant_label": "base (default)",
        "variant_factory": "base",
        "variant_key": "base",
        "params": "~35.8M",
        "params_num": 35824640,
        "d_model": 512,
        "n_layers": 8,
        "n_heads": 8,
        "d_ff": 2048,
        "head_dim": 64,
    },
    "large": {
        "repo": "thomas-schweich/pawn-large",
        "variant_name": "Large",
        "variant_label": "large",
        "variant_factory": "large",
        "variant_key": "large",
        "params": "~68.4M",
        "params_num": 68370432,
        "d_model": 640,
        "n_layers": 10,
        "n_heads": 8,
        "d_ff": 2560,
        "head_dim": 80,
    },
}

# Accuracy ceiling constants
UNCOND_CEILING = 6.43
NAIVE_CEILING = 6.44
MCTS_CEILING = 7.92

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


def fetch_eval_results(repo: str) -> dict | None:
    """Download eval_results.json from a HuggingFace model repo."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo, "eval_results.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not fetch eval_results.json from {repo}: {e}")
        return None


def fetch_best_metrics(repo: str) -> dict | None:
    """Download metrics.jsonl and extract best val metrics."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo, "metrics.jsonl")
        best_loss = float("inf")
        best = None
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") == "val":
                    loss = r.get("val/loss", float("inf"))
                    if loss < best_loss:
                        best_loss = loss
                        best = r
        return best
    except Exception as e:
        print(f"  Warning: could not fetch metrics.jsonl from {repo}: {e}")
        return None


def format_probe(eval_results: dict, probe_name: str) -> str:
    """Format a probe result."""
    probes = eval_results.get("probes", {})
    probe = probes.get(probe_name, {})
    for layer_key in sorted(probe.keys(), reverse=True):
        data = probe[layer_key]
        acc = data.get("best_accuracy", data.get("accuracy", 0))
        mae = data.get("mae")
        if mae is not None:
            return f"{acc:.1%} (MAE {mae:.1f})"
        return f"{acc:.1%}"
    return "N/A"


def format_diagnostic(eval_results: dict, diag_name: str) -> tuple[str, str]:
    """Format a diagnostic result as (n_positions, value)."""
    diags = eval_results.get("diagnostics", {})
    diag = diags.get(diag_name, {})
    n = diag.get("n_positions", 0)
    if diag_name in ("checkmate", "stalemate"):
        val = diag.get("mean_pad_prob", 0)
    else:
        val = diag.get("mean_legal_rate", 0)
    return str(n), f"{val:.1%}"


def build_context(variant_key: str, variant: dict) -> dict:
    """Build the full Jinja template context for a variant."""
    repo = variant["repo"]
    print(f"  Fetching metrics from {repo}...")

    ctx = dict(variant)

    # Fetch training metrics
    best = fetch_best_metrics(repo)
    if best:
        ctx["top1"] = best.get("val/accuracy", 0) * 100
        ctx["top5"] = best.get("val/top5_accuracy", 0) * 100
        ctx["val_loss"] = best.get("val/loss", 0)
        ctx["legal_rate"] = best.get("val/legal_move_rate", 0) * 100
    else:
        # Fallback hardcoded values from postmortem
        fallback = {
            "small": {"top1": 6.73, "top5": 27.44, "val_loss": 3.160, "legal_rate": 99.29},
            "base": {"top1": 6.86, "top5": 27.76, "val_loss": 3.096, "legal_rate": 99.97},
            "large": {"top1": 6.94, "top5": 27.78, "val_loss": 3.092, "legal_rate": 99.98},
        }
        ctx.update(fallback.get(variant_key, {}))

    # Accuracy ratios
    ctx["uncond_ratio"] = round(ctx["top1"] / UNCOND_CEILING * 100)
    ctx["naive_ratio"] = round(ctx["top1"] / NAIVE_CEILING * 100)
    ctx["mcts_ratio"] = round(ctx["top1"] / MCTS_CEILING * 100)

    # Fetch eval results for probes and diagnostics
    eval_results = fetch_eval_results(repo)

    if eval_results:
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
    else:
        ctx["probes"] = []
        ctx["diagnostics"] = []

    return ctx


def main():
    parser = argparse.ArgumentParser(description="Generate HuggingFace model cards")
    parser.add_argument("--push", action="store_true", help="Upload cards to HuggingFace")
    parser.add_argument("--template", type=Path, default=Path("templates/hf_model_card.md.j2"))
    parser.add_argument("--output-dir", type=Path, default=Path("templates/generated"))
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

        output_path = args.output_dir / f"pawn-{variant_key}-README.md"
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
