#!/usr/bin/env python3
"""Supernet-vs-canonical quality-parity harness entry point (Phase-B B3).

Loads a **supernet** checkpoint and a **canonical** (distilled or
independently-trained) checkpoint, slices the supernet down to the
canonical's width (the contrast arm), and emits a JSON gap report with three
signed ``supernet - canonical`` deltas:

  * per-phase move-accuracy delta (overall / opening / midgame / endgame),
  * linear-probe decodability delta (next-move source-square probe), and
  * reference-LoRA held-out val-loss delta.

This is a measurement tool: the cross-model comparison is later analysis,
not a training gate. See ``pawn/parity.py`` for the harness mechanics.

Example::

    uv run --extra rocm python scripts/eval_parity.py \\
        --supernet-checkpoint <supernet-ckpt> \\
        --canonical-checkpoint <canonical-ckpt> \\
        --output parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pawn.checkpoint import load_model, resolve_checkpoint_source, require_uniform
from pawn.corpus import conditioning_from_run_block, generate_corpus
from pawn.parity import ReferenceLoRASpec, run_parity_harness


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_parity")
    ap.add_argument(
        "--supernet-checkpoint", required=True,
        help="v2 HF repo ID or local dir for the SUPERNET checkpoint "
             "(the contrast arm — sliced to the canonical's width).",
    )
    ap.add_argument(
        "--canonical-checkpoint", required=True,
        help="v2 HF repo ID or local dir for the CANONICAL (distilled / "
             "independently-trained) checkpoint at the comparison width.",
    )
    ap.add_argument("--n-games", type=int, default=512,
                    help="games in the shared eval corpus (accuracy + probe).")
    ap.add_argument("--max-ply", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ref-lora-train-games", type=int, default=256,
                    help="games for the reference-LoRA finetune corpus.")
    ap.add_argument("--ref-lora-val-games", type=int, default=128,
                    help="games for the reference-LoRA held-out corpus.")
    ap.add_argument("--ref-lora-steps", type=int, default=20)
    ap.add_argument("--ref-lora-rank", type=int, default=4)
    ap.add_argument("--probe-max-positions", type=int, default=2048)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for corpus generation + probe + reference LoRA.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    supernet_loaded, sup_run = load_model(
        resolve_checkpoint_source(args.supernet_checkpoint)
    )
    supernet = require_uniform(supernet_loaded, "the parity harness")
    canonical_loaded, can_run = load_model(
        resolve_checkpoint_source(args.canonical_checkpoint)
    )
    canonical = require_uniform(canonical_loaded, "the parity harness")

    # Both models must share a conditioning layout so the move positions land
    # at the same absolute offset; the canonical's layout is authoritative
    # (it is the comparison target). A mismatch would smear the per-phase
    # binning and probe positions across the two arms, so surface it loudly.
    sup_cond = conditioning_from_run_block(sup_run)
    can_cond = conditioning_from_run_block(can_run)
    if sup_cond != can_cond:
        raise SystemExit(
            f"conditioning mismatch: supernet={sup_cond!r} vs "
            f"canonical={can_cond!r}; the parity harness compares models at "
            f"the same sequence layout. Re-export one checkpoint or supply "
            f"matching conditioning."
        )
    conditioning = can_cond

    eval_corpus = generate_corpus(
        n_games=args.n_games, max_ply=args.max_ply, seq_len=args.seq_len,
        seed=args.seed, conditioning=conditioning,
    )
    train_corpus = generate_corpus(
        n_games=args.ref_lora_train_games, max_ply=args.max_ply,
        seq_len=args.seq_len, seed=args.seed + 1, conditioning=conditioning,
    )
    val_corpus = generate_corpus(
        n_games=args.ref_lora_val_games, max_ply=args.max_ply,
        seq_len=args.seq_len, seed=args.seed + 2, conditioning=conditioning,
    )

    spec = ReferenceLoRASpec(
        rank=args.ref_lora_rank, steps=args.ref_lora_steps, seed=args.seed,
    )
    report = run_parity_harness(
        supernet, canonical,
        eval_corpus=eval_corpus,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        batch_size=args.batch_size,
        probe_max_positions=args.probe_max_positions,
        probe_epochs=args.probe_epochs,
        ref_lora_spec=spec,
    )

    payload = {
        "supernet_checkpoint": args.supernet_checkpoint,
        "canonical_checkpoint": args.canonical_checkpoint,
        "n_games": args.n_games,
        **report.to_dict(),
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
