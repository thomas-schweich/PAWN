"""True autoregressive compound legality — AR game-completion rate.

v1 (and the v2 ``--compound-legality`` eval) only ever measured the
TEACHER-FORCED game-completion rate: the model predicts each move given the
ground-truth history, so a mistake at ply ``p`` does not corrupt the history
seen at ply ``p+1``. This measures the AUTOREGRESSIVE version — the model
generates each move from its OWN prior moves (``mask_illegal=False``), so the
first illegal move forfeits the game and would have corrupted every later ply.

``ar_game_completion_rate`` = fraction of games that reach a terminal position
with NO illegal move (``forfeit_ply == -1``). This is the honest "can it
actually play a whole game" number, and is expected to be LOWER than the
teacher-forced rate (errors compound) — though a model biased toward its
training distribution can also stay more on-distribution as it generates, so
the sign of the gap is genuinely an open question (it's a known v1 TODO).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pawn.checkpoint import load_model, resolve_checkpoint_source
from pawn.config import NUM_ACTIONS
from pawn.corpus import conditioning_from_run_block
from pawn.generation import autoregressive_generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-games", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 = sample from the model's distribution (matches "
                         "how it would actually generate). Use 0 for greedy.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    model, run_block = load_model(resolve_checkpoint_source(args.checkpoint))
    # The checkpoint's own conditioning layout (this teacher trained with
    # conditioning=[] — no outcome prefix); pass it through so the AR decode
    # uses the [BOS][cond...] prefix the model was trained under.
    conditioning = conditioning_from_run_block(run_block)

    gen = autoregressive_generate(
        model,
        # ``outcome_token`` is only inserted for the "outcome" conditioning
        # kind; with conditioning=[] it is unused. NUM_ACTIONS+1 is the first
        # outcome id (a valid token either way).
        outcome_token=NUM_ACTIONS + 1,
        n_games=args.n_games,
        mask_illegal=False,  # let the model forfeit on its first illegal move
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        seed=args.seed,
        batch_size=args.batch_size,
        conditioning=conditioning,
    )

    forfeit_ply = np.asarray(gen["forfeit_ply"])
    game_lengths = np.asarray(gen["game_lengths"])
    term_codes = np.asarray(gen["term_codes"])
    n = int(forfeit_ply.shape[0])

    completed = forfeit_ply == -1  # no illegal move at any ply
    forfeited = ~completed
    payload: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "n_games": n,
        "temperature": args.temperature,
        "conditioning": list(conditioning),
        "ar_game_completion_rate": float(completed.mean()),
        "ar_forfeit_rate": float(forfeited.mean()),
        "mean_game_length": float(game_lengths.mean()),
        "mean_forfeit_ply": (
            float(forfeit_ply[forfeited].mean()) if bool(forfeited.any()) else None
        ),
        "forfeit_ply_pctiles": (
            {
                str(p): float(np.percentile(forfeit_ply[forfeited], p))
                for p in (10, 50, 90)
            }
            if bool(forfeited.any())
            else None
        ),
        "term_code_counts": {
            str(int(c)): int((term_codes == c).sum())
            for c in np.unique(term_codes)
        },
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
