"""Teacher-forced legality eval for a converted v1 PyTorch checkpoint.

Loads a v1 checkpoint as the in-memory factored JAX model
(:func:`pawn._legacy.legacy.load_v1_factored_model` — no un-factoring, no disk
round-trip) and runs per-move legal rate + compound (game-completion) legality
under v1's NATIVE sequence contract.

v1 (``prepend_outcome=False``) trained on bare ``[m_1, m_2, …]`` sequences — no
BOS, no prefix (C=0). We pack that contract directly via
:func:`pawn.corpus.generate_corpus(bare_moves=True)` (no BOS, strictly
right-padded, v1 loss mask: first move unsupervised, end-of-game PAD
supervised) — the same contract the factored arch now trains under.

Validated: converted ``thomas-schweich/pawn-large`` reproduces its published
99.9990% per-move legal / 99.76% game-completion under this contract.
"""
from __future__ import annotations

import argparse
import json

from pawn._legacy.legacy import load_v1_factored_model
from pawn.corpus import generate_corpus
from pawn.eval import compute_compound_legality, compute_val_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="v1 HF repo id (e.g. thomas-schweich/pawn-large) or "
                         "local v1 checkpoint dir.")
    ap.add_argument("--n-games", type=int, default=2048)
    ap.add_argument("--max-ply", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--min-eval-ply", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    model, cfg = load_v1_factored_model(args.checkpoint)
    # v1's native C=0 bare-moves contract, packed directly (no BOS).
    v1c = generate_corpus(
        n_games=args.n_games, max_ply=args.max_ply, seq_len=args.seq_len,
        seed=args.seed, conditioning=(), bare_moves=True,
    )
    vm = compute_val_metrics(
        model, v1c, batch_size=args.batch_size, min_eval_ply=args.min_eval_ply,
    )
    cl = compute_compound_legality(
        model, v1c, batch_size=args.batch_size, min_eval_ply=0,
    )
    payload = {
        "checkpoint": args.checkpoint,
        "dims": {
            "d_model": cfg.d_model, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "head_dim": cfg.head_dim,
            "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
        },
        "n_games": args.n_games,
        "per_move_legal_rate": vm.legal_move_rate,
        "teacher_forced_game_completion": cl.game_completion_rate,
        "top1": vm.top1, "top5": vm.top5, "perplexity": vm.perplexity,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
