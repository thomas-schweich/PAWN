"""True autoregressive game-completion for a converted v1 checkpoint.

The companion ``eval_ar_legality.py`` measures AR completion for v2 models
(``[BOS][cond…]`` prefix → the model generates ``m_1`` itself). v1 can't use
that harness: v1's vocab is 1980 (ids 0..1979), so ``BOS_TOKEN=1980`` is
out-of-vocab and the factored ``_embed`` would index ``decomp_table[1980]``
out of bounds. v1 also never *supervised* the first move — it trained on bare
``[m_1, m_2, …]`` sequences where position ``i`` predicts move ``i+1``, so
position ``-1`` (which would predict ``m_1``) does not exist. v1 therefore
cannot generate ``m_1`` and must be **seeded** with a legal first move.

This script runs a self-contained batched AR loop under v1's native contract:

    1. ``env.reset()`` — ``n_games`` boards at the opening.
    2. Seed ``m_1`` with a uniformly-random *legal* first move per game, apply
       it to the env, and write it to slot 0 (bare moves, no BOS/prefix).
    3. Decode: full-forward over ``[m_1..m_k]`` (all-real attention mask),
       read logits at the last position, sample ``m_{k+1}``
       (``temperature=1.0`` ⇒ sample from the model's distribution, matching
       how it would actually play; ``0`` ⇒ greedy), check legality against the
       engine, and either advance the board or forfeit (first illegal move).

``ar_game_completion_rate`` = fraction of games that reach a terminal position
(or the ply limit) with **no** illegal move at any ply (``forfeit_ply == -1``)
— the identical definition ``eval_ar_legality.py`` uses for v2, so the two
numbers are directly comparable. The one asymmetry: v2 *generates* ``m_1`` (and
can forfeit on it), whereas v1 is *seeded* with a guaranteed-legal ``m_1``.
That slightly favours v1, but ply-1 forfeits are vanishingly rare for a trained
model, so the gap is immaterial. ``--use-bos`` flips the loop to v2's contract
(BOS prefill, model generates ``m_1``); running a v2 checkpoint through it
reproduces ``eval_ar_legality.py``'s number and validates this loop's env /
forfeit / termination logic against the proven path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import chess_engine as engine
from pawn.config import BOS_TOKEN, NUM_ACTIONS, PAD_TOKEN
from pawn.factored_model import FactoredPAWNModel
from pawn.generation import _map_term_code_to_outcome_name
from pawn.model import PAWNModel


def _ar_complete(
    model: PAWNModel | FactoredPAWNModel,
    *,
    n_games: int,
    max_seq_len: int,
    temperature: float,
    seed: int,
    use_bos: bool,
) -> dict[str, np.ndarray]:
    """Batched AR decode with engine-tracked legality / termination.

    ``use_bos=False`` (v1): bare ``[m_1, m_2, …]`` contract, ``m_1`` seeded
    with a random legal move. ``use_bos=True`` (v2): ``[BOS]`` prefill at slot
    0, model generates ``m_1`` from BOS — the v2 ``autoregressive_generate``
    contract, used to validate this loop against the known v2 number.

    Env-step body (PAD / forfeit / terminating-move bookkeeping) is a faithful
    copy of :func:`pawn.generation.autoregressive_generate._sample_and_step_env`.
    """
    C = 1 if use_bos else 0  # noqa: N806 — prefix width (BOS only, or none)
    sequences = np.full((n_games, max_seq_len), PAD_TOKEN, dtype=np.int32)
    if use_bos:
        sequences[:, 0] = BOS_TOKEN

    max_move_positions = max_seq_len - C
    env = engine.PyBatchRLEnv(n_games, max_ply=max_move_positions, seed=seed)
    env.reset()

    terminated = np.zeros(n_games, dtype=bool)
    terminated_at = np.full(n_games, -1, dtype=np.int32)
    forfeit_ply = np.full(n_games, -1, dtype=np.int32)
    term_codes = np.full(n_games, -1, dtype=np.int8)
    all_indices = np.arange(n_games, dtype=np.uint32)
    rng = np.random.default_rng(seed)

    out_vocab = int(model.cfg.vocab_size)

    def _sample_and_step_env(pos: int, next_logits_arr: np.ndarray) -> None:
        active = ~terminated
        if temperature != 1.0:
            next_logits_arr = next_logits_arr / temperature
        gumbel = -np.log(-np.log(rng.uniform(1e-10, 1.0, size=next_logits_arr.shape)))
        sampled = (next_logits_arr + gumbel).argmax(axis=-1).astype(np.int32)
        sequences[:, pos] = sampled
        ply = pos - C
        pad_mask = active & (sampled == PAD_TOKEN)
        if pad_mask.any():
            terminated[pad_mask] = True
            terminated_at[pad_mask] = ply
            term_codes[pad_mask] = -2
        move_mask = active & ~pad_mask & ~terminated
        if move_mask.any():
            mv_idx = all_indices[move_mask]
            mv_tok = sampled[move_mask].astype(np.uint16)
            legality, step_tc = env.apply_moves(mv_idx, mv_tok)
            legality = np.asarray(legality)
            step_tc = np.asarray(step_tc)
            illegal = ~legality
            if illegal.any():
                forfeit_global = mv_idx[illegal]
                forfeit_ply[forfeit_global] = ply
                terminated[forfeit_global] = True
                terminated_at[forfeit_global] = ply
                term_codes[forfeit_global] = -3
            termed = legality & (step_tc >= 0)
            if termed.any():
                tg = mv_idx[termed]
                terminated[tg] = True
                terminated_at[tg] = ply + 1
                term_codes[tg] = step_tc[termed]

    @eqx.filter_jit
    def _forward(t: jax.Array, a: jax.Array) -> jax.Array:
        return model(t, a, compute_dtype=None)

    seq_positions = np.arange(max_seq_len, dtype=np.int32)

    def _logits_for_slot(real_len: int) -> np.ndarray:
        """Forward the full fixed-width buffer with an attention mask True over
        slots ``[0, real_len)`` (PAD elsewhere, masked out), and return the
        logits at the last real slot ``real_len - 1`` — the prediction for slot
        ``real_len``. Constant input shape ⇒ a single XLA compile across the
        whole decode (vs. recompiling per growing length)."""
        mask = jnp.asarray(seq_positions[None, :] < real_len)
        mask = jnp.broadcast_to(mask, (n_games, max_seq_len))
        logits = _forward(jnp.asarray(sequences), mask)
        return np.asarray(logits[:, real_len - 1, :])

    # --- Seed m_1 for the bare-moves (v1) contract ------------------------
    prefill_len = C
    if not use_bos:
        # Uniformly-random legal first move per game (restricted to the
        # action sub-vocab). All games start from the same opening, so every
        # row has the standard 20 legal first moves.
        raw = np.asarray(env.get_legal_token_masks_batch(all_indices, out_vocab))
        legal_actions = raw[:, :NUM_ACTIONS].astype(bool)
        first = np.empty(n_games, dtype=np.int32)
        for i in range(n_games):
            cands = np.flatnonzero(legal_actions[i])
            first[i] = int(rng.choice(cands))
        legality, step_tc = env.apply_moves(all_indices, first.astype(np.uint16))
        legality = np.asarray(legality)
        if not legality.all():  # impossible (random legal move) — guard anyway
            bad = all_indices[~legality]
            forfeit_ply[bad] = 0
            terminated[bad] = True
            terminated_at[bad] = 0
            term_codes[bad] = -3
        step_tc = np.asarray(step_tc)
        termed = legality & (step_tc >= 0)
        if termed.any():
            tg = all_indices[termed]
            terminated[tg] = True
            terminated_at[tg] = 1
            term_codes[tg] = step_tc[termed]
        sequences[:, 0] = first
        prefill_len = 1

    # --- Decode loop (fixed-width full-forward, single compile) -----------
    # Slots [0, prefill_len) are written; predict slot prefill_len next.
    next_logits = _logits_for_slot(prefill_len)
    for pos in range(prefill_len, max_seq_len):
        if not (~terminated).any():
            break
        _sample_and_step_env(pos, next_logits)
        if pos < max_seq_len - 1 and not terminated.all():
            next_logits = _logits_for_slot(pos + 1)

    still_going = ~terminated
    terminated_at[still_going] = max_move_positions
    term_codes[still_going] = 5
    return {
        "term_codes": term_codes,
        "game_lengths": terminated_at.astype(np.int32),
        "forfeit_ply": forfeit_ply,
    }


def _run_batched(
    model: PAWNModel | FactoredPAWNModel, *,
    n_games: int, batch_size: int, max_seq_len: int,
    temperature: float, seed: int, use_bos: bool,
) -> dict[str, np.ndarray]:
    """Chunk ``n_games`` into ``batch_size`` decodes (bounds the live logits /
    env footprint) and concatenate. Each chunk uses ``seed + chunk_index``."""
    chunks: list[dict[str, np.ndarray]] = []
    for ci, start in enumerate(range(0, n_games, batch_size)):
        end = min(start + batch_size, n_games)
        chunks.append(_ar_complete(
            model, n_games=end - start, max_seq_len=max_seq_len,
            temperature=temperature, seed=seed + ci, use_bos=use_bos,
        ))
    return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}


def _is_v2_checkpoint_dir(p: Path) -> bool:
    """True when ``p`` looks like a v2 checkpoint dir (config.json with the
    v2 ``{"version", "model"}`` schema), complete or not.

    v1 torch checkpoints carry ``{"format_version", "model_config"}``
    instead, so the two formats are disambiguated by schema rather than by
    the ``.complete`` sentinel — a partially-saved v2 dir then still routes
    to the v2 loader, whose sentinel verification raises the precise
    ``IncompleteCheckpointError`` instead of the v1 loader's confusing
    key-mismatch failure.
    """
    cfg_file = p / "config.json"
    if not cfg_file.is_file():
        return False
    try:
        raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and "version" in raw and "model" in raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="v1 HF repo id (e.g. thomas-schweich/pawn-large), a "
                         "v2-format factored checkpoint dir (an --arch "
                         "factored-v1 run's step_NNNN), or — with --use-bos — "
                         "a uniform v2 checkpoint for loop validation.")
    ap.add_argument("--n-games", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use-bos", action="store_true",
                    help="v2 contract (BOS prefill, model generates m_1). "
                         "Loads via pawn.checkpoint.load_model. Used to "
                         "validate this loop against the known v2 AR number.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    model: PAWNModel | FactoredPAWNModel
    if args.use_bos:
        from pawn.checkpoint import (
            load_model,
            require_uniform,
            resolve_checkpoint_source,
        )
        loaded, _ = load_model(resolve_checkpoint_source(args.checkpoint))
        # BOS=1980 is out-of-vocab for factored models — their `_embed`
        # would silently map it to the last outcome embedding and the
        # number would be quietly wrong (review round-1, type lane).
        model = require_uniform(loaded, "--use-bos (v2 BOS contract)")
        cfg = model.cfg
    elif _is_v2_checkpoint_dir(Path(args.checkpoint)):
        # A v2-format checkpoint dir (e.g. an `--arch factored-v1` run's
        # step_NNNN). Routed on the v2 config.json schema, NOT on the
        # `.complete` sentinel — an incomplete v2 save must hit the v2
        # loader's precise IncompleteCheckpointError, not fall through to
        # the v1 torch loader and die on a baffling key mismatch (review
        # round-2, test-risk). The bare-moves contract below requires the
        # factored architecture.
        from pawn.checkpoint import load_model
        loaded, _ = load_model(Path(args.checkpoint))
        if not isinstance(loaded, FactoredPAWNModel):
            raise SystemExit(
                "the bare-moves (no-BOS) contract requires a factored "
                f"(v1-architecture) checkpoint; {args.checkpoint} holds a "
                f"{type(loaded).__name__}. For uniform v2 checkpoints use "
                "--use-bos (their native contract) instead."
            )
        model = loaded
        cfg = model.cfg
    else:
        from pawn._legacy.legacy import load_v1_factored_model
        model, cfg = load_v1_factored_model(args.checkpoint)

    gen = _run_batched(
        model, n_games=args.n_games, batch_size=args.batch_size,
        max_seq_len=args.max_seq_len, temperature=args.temperature,
        seed=args.seed, use_bos=args.use_bos,
    )

    forfeit_ply = gen["forfeit_ply"]
    game_lengths = gen["game_lengths"]
    term_codes = gen["term_codes"]
    n = int(forfeit_ply.shape[0])
    completed = forfeit_ply == -1
    forfeited = ~completed

    outcome_dist: dict[str, int] = {}
    for i in range(n):
        name = _map_term_code_to_outcome_name(
            int(term_codes[i]), int(game_lengths[i])
        )
        outcome_dist[name] = outcome_dist.get(name, 0) + 1

    payload: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "contract": "v2-bos" if args.use_bos else "v1-bare-seeded-m1",
        "dims": {
            "d_model": cfg.d_model, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "head_dim": cfg.head_dim,
            "vocab_size": cfg.vocab_size,
        },
        "n_games": n,
        "temperature": args.temperature,
        "ar_game_completion_rate": float(completed.mean()),
        "ar_forfeit_rate": float(forfeited.mean()),
        "mean_game_length": float(game_lengths.mean()),
        "mean_forfeit_ply": (
            float(forfeit_ply[forfeited].mean()) if bool(forfeited.any()) else None
        ),
        "forfeit_ply_pctiles": (
            {str(p): float(np.percentile(forfeit_ply[forfeited], p))
             for p in (10, 50, 90)}
            if bool(forfeited.any()) else None
        ),
        "outcome_distribution": {k: v / max(1, n) for k, v in outcome_dist.items()},
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
