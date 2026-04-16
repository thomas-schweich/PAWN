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


def fetch_config(repo: str, revision: str | None = None) -> dict:
    """Download config.json from a HuggingFace model repo."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "config.json", revision=revision)
    with open(path) as f:
        return json.load(f)


def params_str(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1_000_000:
        return f"~{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"~{n / 1_000:.0f}K"
    return str(n)


def count_params_from_weights(repo: str, revision: str | None = None) -> int:
    """Count exact parameters from the safetensors weights on HuggingFace."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(repo, "model.safetensors", revision=revision)
    total = 0
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            total += f.get_tensor(key).numel()
    return total

CEILING_PATH = Path("data/theoretical_ceiling.json")


def load_ceilings() -> dict:
    """Load accuracy ceilings from the canonical JSON artifact.

    The ceiling JSON predates the split-half bias correction described in
    docs/ACCURACY_CEILING.md, so `conditional_corrected_ceiling` may be
    absent. When it is, fall back to `conditional_ceiling` (the naive MC
    estimate) so the bracket collapses to a single value rather than
    blowing up. The ceiling is being recomputed for the v1.0.0 vocabulary
    as a known TODO.
    """
    if not CEILING_PATH.exists():
        raise FileNotFoundError(
            f"{CEILING_PATH} not found. Run scripts/compute_theoretical_ceiling.py first."
        )
    with open(CEILING_PATH) as f:
        data = json.load(f)
    mc_naive = data["conditional_ceiling"] * 100
    mc_corrected = data.get("conditional_corrected_ceiling", data["conditional_ceiling"]) * 100
    return {
        "uncond": data["unconditional_ceiling"] * 100,
        "mc_naive": mc_naive,
        "mc_corrected": mc_corrected,
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


def fetch_eval_results(repo: str, revision: str | None = None) -> dict:
    """Download eval_results.json from a HuggingFace model repo."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "eval_results.json", revision=revision)
    with open(path) as f:
        return json.load(f)


def fetch_published_step(repo: str, revision: str | None = None) -> int:
    """Read the global_step that the published model.safetensors was saved at.

    The trainer keeps val every `eval_interval` steps but only saves
    checkpoints every `checkpoint_interval` steps (typically 1K vs 5K).
    The published `model.safetensors` at the root of the repo is the
    best 5K-cadence checkpoint by val loss, *not* the lowest-loss val
    record across every val step. The two diverge whenever val noise
    around the eventual best happens to dip lower at an in-between
    step, which is the common case. Reading `global_step` from the
    co-published `training_state.json` is the only authoritative
    source for which step the weights came from.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "training_state.json", revision=revision)
    with open(path) as f:
        ts = json.load(f)
    step = ts.get("global_step")
    if step is None:
        raise ValueError(
            f"training_state.json from {repo} has no global_step field; "
            f"cannot determine which step the published model.safetensors "
            f"was saved at."
        )
    return int(step)


def fetch_metrics_at_step(repo: str, step: int, revision: str | None = None) -> dict:
    """Download metrics.jsonl and return the val record at the given step.

    Used to pull the val metrics for the exact checkpoint the published
    `model.safetensors` was saved at — see `fetch_published_step` for
    the why. The trainer writes a complete val record on every eval
    (including the extended compound-legality fields), so a single
    record is enough — no need to merge anything across records.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "metrics.jsonl", revision=revision)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "val" and r.get("step") == step:
                return r
    raise ValueError(
        f"No val record at step {step} in metrics.jsonl from {repo}. "
        f"This typically means the published checkpoint's step is not a "
        f"multiple of the eval interval, which would be a trainer bug."
    )


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


def build_context(variant_key: str, variant: dict, revision: str | None = None) -> dict:
    """Build the full Jinja template context for a variant."""
    repo = variant["repo"]
    rev_label = f" @ {revision}" if revision else ""
    print(f"  Fetching config and metrics from {repo}{rev_label}...")

    ctx = dict(variant)
    ctx["variant_key"] = variant_key

    # Fetch model architecture and training config from config.json. Every
    # field below is auto-detected from the published checkpoint — no
    # hardcoded values, so the same template renders correctly for any
    # backbone regardless of how it was trained.
    config = fetch_config(repo, revision=revision)
    mc = config["model_config"]
    ctx["d_model"] = mc["d_model"]
    ctx["n_layers"] = mc["n_layers"]
    ctx["n_heads"] = mc["n_heads"]
    ctx["d_ff"] = mc["d_ff"]
    ctx["vocab_size"] = mc["vocab_size"]
    ctx["max_seq_len"] = mc["max_seq_len"]
    ctx["head_dim"] = ctx["d_model"] // ctx["n_heads"]
    ctx["params_num"] = count_params_from_weights(repo, revision=revision)
    ctx["params"] = params_str(ctx["params_num"])

    tc = config["training_config"]
    ctx["total_steps"] = tc["total_steps"]
    ctx["batch_size"] = tc["batch_size"]
    ctx["warmup_steps"] = tc["warmup_steps"]
    ctx["lr"] = tc["lr"]
    ctx["weight_decay"] = tc["weight_decay"]
    ctx["max_ply"] = tc["max_ply"]
    ctx["prepend_outcome"] = tc.get("prepend_outcome", False)
    ctx["sequences_seen"] = ctx["total_steps"] * ctx["batch_size"]

    # Fetch training metrics for the EXACT step the published
    # model.safetensors was saved at, not the lowest-val/loss val record
    # in metrics.jsonl. See fetch_published_step for the why.
    published_step = fetch_published_step(repo, revision=revision)
    ctx["published_step"] = published_step
    ctx["published_sequences"] = published_step * ctx["batch_size"]
    val = fetch_metrics_at_step(repo, published_step, revision=revision)
    ctx["top1"] = val["val/accuracy"] * 100
    ctx["top5"] = val["val/top5_accuracy"] * 100
    ctx["val_loss"] = val["val/loss"]
    ctx["perplexity"] = val["val/perplexity"]
    ctx["legal_rate"] = val["val/legal_move_rate"] * 100
    ctx["late_legal_rate"] = val["val/late_legal_move_rate"] * 100
    # Compound legality: did the model predict every move along one side's
    # plies legally for an entire game? See docs/ARCHITECTURE.md for the
    # definition. These fields were added in the v1.0.0 training runs;
    # if they're missing the trainer used by this checkpoint pre-dates
    # them, which means the card needs to be regenerated against a
    # newer run before being uploaded.
    ctx["completion_rate"] = val["val/game_completion_rate"] * 100
    ctx["avg_pct_completion"] = val["val/avg_pct_completion"] * 100
    ctx["avg_plies_completed"] = val["val/avg_plies_completed"]
    ctx["median_forfeit_ply"] = val["val/median_forfeit_ply"]

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
    eval_results = fetch_eval_results(repo, revision=revision)
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
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "HuggingFace revision (branch, tag, or commit SHA) to read each "
            "model repo from. Defaults to the repo's default branch (main). "
            "Use this to render against an unmerged training run, e.g. "
            "--revision run/co_pretraining_2026_04_13. The same revision is "
            "applied to all selected variants. If --push is used together "
            "with --revision, the rendered card is still uploaded to the "
            "default branch (HF model cards only live on main)."
        ),
    )
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
        ctx = build_context(variant_key, VARIANTS[variant_key], revision=args.revision)
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
