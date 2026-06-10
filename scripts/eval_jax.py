#!/usr/bin/env python3
"""Move-accuracy evaluation entry point.

Reports the full v1 metric set — overall + per-phase top-1, top-5, CE
loss, perplexity, and legal-move rate — with the MAIA opening-skip
(``--min-eval-ply``, default 10) and an optional per-ply breakdown.

Eval data is either freshly-generated random self-play (the default) or
a held-out Lichess slice (``--pgn``), the latter optionally filtered to a
within-distribution Elo band (``--elo-min`` / ``--elo-max``) so an
adapter trained on one band is scored on that band. Adapter checkpoints
(bottleneck / pure-FiLM / hybrid) are loaded via :func:`load_eval_model`,
which re-applies the typed sidecar so the *adapted* model is evaluated,
not the bare backbone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pawn.checkpoint import load_eval_model, resolve_checkpoint_source
from pawn.corpus import (
    Corpus,
    conditioning_from_run_block,
    generate_corpus,
    to_v1_contract,
)
from pawn.factored_model import FactoredPAWNModel
from pawn.eval import (
    compute_compound_legality,
    compute_per_ply_accuracy,
    compute_val_metrics,
)
from pawn.jax_setup import setup_jax_caching


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_jax")
    ap.add_argument("--checkpoint", required=True,
                    help="v2 HF repo ID or local checkpoint dir. "
                         "(v1 PyTorch repos require `git checkout v1.0.0`.)")
    ap.add_argument("--n-games", type=int, default=512)
    ap.add_argument("--max-ply", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    # Real / published Lichess eval data (not random-only). When omitted,
    # the eval runs on freshly-generated random self-play.
    ap.add_argument("--pgn", default=None,
                    help="Lichess parquet file or HF dataset repo ID to "
                         "evaluate on. Default: freshly-generated random "
                         "self-play games.")
    ap.add_argument("--split", default="validation",
                    help="dataset split for --pgn (default: validation, "
                         "the held-out source). Carving val out of train "
                         "silently leaks; only override for single-file "
                         "local sources without a split structure.")
    # Within-distribution Elo filter (v1 eval_accuracy.py --elo-min/--elo-max):
    # restrict the held-out slice to the band the adapter was trained on.
    ap.add_argument("--elo-min", type=int, default=None,
                    help="filter --pgn games to both players' Elo >= this.")
    ap.add_argument("--elo-max", type=int, default=None,
                    help="filter --pgn games to both players' Elo < this.")
    # MAIA opening-skip: drop the book-ish opening plies from the headline
    # metrics (v1 eval_accuracy.py default 10). The per-phase breakdown
    # still reports opening accuracy from ply 0.
    ap.add_argument("--min-eval-ply", type=int, default=10,
                    help="skip the first N ply for the overall metrics "
                         "(MAIA methodology, default 10). Per-phase "
                         "accuracy is unaffected (always from ply 0).")
    ap.add_argument("--compound-legality", action="store_true",
                    help="also report the teacher-forced game-completion rate "
                         "(v1's 'game completion rate' = fraction of games "
                         "whose every argmax move prediction is legal).")
    ap.add_argument("--per-ply", action="store_true",
                    help="also report a per-ply top-1 accuracy breakdown.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    # Enable the persistent compilation cache before any eval forward jit
    # compiles — the eval forward shares the model HLO with training, so the
    # cache hits across eval / pretrain processes at matching shape/dtype.
    cache_path = setup_jax_caching()
    if cache_path is not None:
        print(f"JAX compilation cache: {cache_path}")

    ckpt = args.checkpoint
    ckpt_path = resolve_checkpoint_source(ckpt)
    # load_eval_model re-applies bottleneck / pure-FiLM / hybrid sidecars so
    # adapter checkpoints eval the ADAPTED model, not the bare backbone.
    model, run_block = load_eval_model(ckpt_path)
    # Build the corpus with the checkpoint's own conditioning so the move
    # positions land at the same absolute offset the model was trained
    # under (plan §8.1). A checkpoint written without a run block predates
    # the conditioning prefix → the BOS-only ``[]`` layout (C=1).
    conditioning = conditioning_from_run_block(run_block)

    if args.pgn is not None:
        from pawn.lichess_data import load_lichess_corpus

        corpus: Corpus = load_lichess_corpus(
            args.pgn,
            split=args.split,
            elo_min=args.elo_min, elo_max=args.elo_max,
            seq_len=args.seq_len, max_games=args.n_games,
            conditioning=conditioning,
        )
    else:
        corpus = generate_corpus(
            n_games=args.n_games, max_ply=args.max_ply, seq_len=args.seq_len,
            seed=0, conditioning=conditioning,
        )

    # Factored (v1-architecture) checkpoints eval under v1's bare-moves
    # contract: BOS=1980 is out-of-vocab for them — the packed corpus's
    # slot-0 BOS would trip the factored _embed's loud OOV guard before any
    # metric is produced (round-3 review, codex P2). Apply the same
    # validated transform the trainer / probes use. `to_v1_contract`
    # requires C=1, which is guaranteed here: factored checkpoints can only
    # be trained with conditioning=[] (PretrainConfig validator), and the
    # corpus above was built from the checkpoint's own conditioning.
    if isinstance(model, FactoredPAWNModel):
        corpus = to_v1_contract(corpus)

    vm = compute_val_metrics(
        model, corpus,
        batch_size=args.batch_size,
        min_eval_ply=args.min_eval_ply,
    )
    payload: dict[str, object] = {
        "checkpoint": ckpt,
        "n_games": corpus.n_games,
        "min_eval_ply": args.min_eval_ply,
        "overall_accuracy": vm.top1,
        "top5_accuracy": vm.top5,
        "loss": vm.val_loss,
        "perplexity": vm.perplexity,
        "legal_move_rate": vm.legal_move_rate,
        "late_legal_move_rate": vm.late_legal_move_rate,
        "opening_accuracy": vm.phases.opening,
        "midgame_accuracy": vm.phases.midgame,
        "endgame_accuracy": vm.phases.endgame,
        "n_supervised": vm.phases.n_total,
    }
    if args.compound_legality:
        # Game-completion is a no-opening-skip metric — v1 defined it over
        # EVERY supervised ply. Always pass min_eval_ply=0 (NOT
        # args.min_eval_ply, whose default 10 is the MAIA accuracy skip): an
        # opening skip would let an illegal opening move still "complete" the
        # game and would count games shorter than the skip as vacuously legal,
        # inflating the v1-comparable rate.
        cl = compute_compound_legality(
            model, corpus, batch_size=args.batch_size, min_eval_ply=0,
        )
        payload["game_completion_rate"] = cl.game_completion_rate
        payload["compound_per_move_legal_rate"] = cl.per_move_legal_rate
    if args.per_ply:
        per_ply = compute_per_ply_accuracy(
            model, corpus, batch_size=args.batch_size,
        )
        payload["per_ply"] = {
            str(ply): {"top1_accuracy": per_ply.accuracy[ply], "n": per_ply.n[ply]}
            for ply in sorted(per_ply.accuracy)
        }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
