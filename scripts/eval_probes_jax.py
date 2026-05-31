#!/usr/bin/env python3
"""Linear probes on FROZEN per-layer hidden states (H6).

Forwards the frozen backbone on engine games, extracts each layer's
residual-stream hidden state, and fits a held-out linear probe per layer
against an engine-derived board feature
(:func:`chess_engine.extract_board_states`). Emits per-layer held-out
probe accuracy — real signal, not synthetic noise.

The probe-feature suite mirrors v1 ``eval_suite.probes.PROBES``: the
carried-over classification probes (side_to_move, occupancy, piece_type,
piece_type_all, is_check, castling_rights, ep_square, game_phase) plus the
MSE-regression probes (material_count, legal_move_count, halfmove_clock),
whose held-out score is R² and which additionally report MAE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess_engine as engine

from pawn.checkpoint import load_model, resolve_checkpoint_source
from pawn.corpus import conditioning_from_run_block
from pawn.probes import PROBE_FEATURES, ProbeResult, run_layer_probes


def _layer_name(layer: int) -> str:
    """Map a probe layer index to its v1-parity output key.

    ``run_layer_probes`` returns results keyed by ``0..n_layers`` where index
    ``0`` is the embedding output and index ``i`` (i>0) is the output of
    transformer block ``i``. v1 names these ``embed`` then ``layer_0`` ..
    ``layer_{n_layers-1}`` (the v1 ``layer_{i}`` is the output of block ``i``,
    so a v1 ``layer_0`` corresponds to index ``1`` here). Any downstream parser
    keyed on v1 layer names must see ``embed, layer_0, ..., layer_{n-1}``.
    """
    return "embed" if layer == 0 else f"layer_{layer - 1}"


def _layer_payload(r: ProbeResult, loss_type: str) -> dict[str, object]:
    """Per-layer metrics block, matching the v1 probe schema keys.

    ``accuracy`` is fraction-correct for classification and held-out R² for
    ``mse``; ``best_accuracy`` is the across-epoch best; ``loss`` is the
    held-out loss; ``mae`` is present only for regression probes.
    """
    payload: dict[str, object] = {
        # `val_accuracy` is the v2 held-out headline; the v1-parity alias
        # `accuracy` carries the same number so the v1 schema is honoured.
        "val_accuracy": r.accuracy,
        "accuracy": r.accuracy,
        "best_accuracy": r.best_accuracy,
        "loss": r.loss,
        "train_accuracy": r.train_accuracy,
        "n_train": r.n_train,
        "n_val": r.n_val,
    }
    if loss_type == "mse" and r.mae is not None:
        payload["mae"] = r.mae
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_probes_jax")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--n-games", type=int, default=256,
        help="games for the probe train+val pool (seed --seed)",
    )
    ap.add_argument(
        "--n-val-games", type=int, default=0,
        help="if >0, generate a SEPARATE pool of val games (seed --val-seed) "
             "and probe on it instead of carving val from the train pool",
    )
    ap.add_argument("--max-ply", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--val-seed", type=int, default=None,
        help="seed for the separate val-game pool (default: --seed + 1)",
    )
    ap.add_argument(
        "--feature", choices=sorted(PROBE_FEATURES), default="side_to_move",
        help="board feature to probe for (default: side_to_move). Use "
             "'all' to run the full feature suite.",
    )
    ap.add_argument(
        "--all-features", action="store_true",
        help="probe every feature in the suite (overrides --feature)",
    )
    ap.add_argument(
        "--probe-square", type=int, default=28,
        help="board square 0..63 (rank-major) for occupancy/piece_type",
    )
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--top-layer-only", action="store_true",
        help="report only the top transformer layer's probe (skip the "
             "per-layer sweep in the summary headline)",
    )
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument(
        "--prepend-outcome", dest="prepend_outcome", action="store_true",
        default=None,
        help="force outcome-prefixed probe layout (overrides the "
             "checkpoint's saved conditioning)",
    )
    grp.add_argument(
        "--pure-moves", dest="prepend_outcome", action="store_false",
        help="force the pure-moves probe layout (no outcome prefix)",
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    ckpt = args.checkpoint
    ckpt_path = resolve_checkpoint_source(ckpt)
    model, run_block = load_model(ckpt_path)
    # Rebuild the checkpoint's own conditioning layout so probe states use
    # the same absolute-RoPE prefix the model trained under (plan §8.1).
    conditioning = tuple(conditioning_from_run_block(run_block))

    # --prepend-outcome / --pure-moves force an outcome-prefixed layout even
    # for a checkpoint whose saved conditioning omits it (e.g. ambiguous
    # pre-flag checkpoints). When forced on and the checkpoint already
    # conditions on outcome we leave it; when forced on and it doesn't, we
    # prepend it; when forced off we strip it.
    if args.prepend_outcome is True and "outcome" not in conditioning:
        conditioning = ("outcome", *conditioning)
    elif args.prepend_outcome is False and "outcome" in conditioning:
        conditioning = tuple(k for k in conditioning if k != "outcome")

    features = sorted(PROBE_FEATURES) if args.all_features else [args.feature]
    val_seed = args.val_seed if args.val_seed is not None else args.seed + 1

    # Engine self-play games — the frozen forward + engine labels run on
    # these. The probe never trains the backbone. With --n-val-games we draw
    # a SEPARATE held-out game pool (distinct seed) so train/val come from
    # different games, not a within-game split.
    train_ids, train_lengths, _term = engine.generate_random_games(
        args.n_games, args.max_ply, args.seed
    )
    separate_val = args.n_val_games > 0
    val_ids = None
    val_lengths = None
    if separate_val:
        val_ids, val_lengths, _vterm = engine.generate_random_games(
            args.n_val_games, args.max_ply, val_seed
        )

    probes_payload: dict[str, object] = {}
    feature_best: dict[str, float] = {}
    for feat in features:
        spec = PROBE_FEATURES[feat]
        labeler = spec.make_labeler(args.probe_square)
        per_layer = run_layer_probes(
            model, train_ids, train_lengths,
            n_classes=spec.n_outputs, labeler=labeler,
            loss_type=spec.loss_type,
            conditioning=conditioning,
            n_epochs=args.n_epochs,
            val_frac=args.val_frac,
            batch_size=args.batch_size,
            needs_legal_counts=spec.needs_legal_counts,
            # With a separate val pool, train and val come from different
            # games (distinct seeds); otherwise val_frac carves within-pool.
            val_move_ids=val_ids if separate_val else None,
            val_game_lengths=val_lengths if separate_val else None,
            key=args.seed,
        )
        layers_payload = {
            _layer_name(layer): _layer_payload(r, spec.loss_type)
            for layer, r in sorted(per_layer.items())
        }
        best_idx, best = max(per_layer.items(), key=lambda kv: kv[1].best_accuracy)
        best_layer = _layer_name(best_idx)
        if args.top_layer_only:
            top = max(per_layer)
            top_name = _layer_name(top)
            layers_payload = {top_name: layers_payload[top_name]}
            best = per_layer[top]
            best_layer = top_name
        probes_payload[feat] = {
            "loss_type": spec.loss_type,
            "n_outputs": spec.n_outputs,
            "probe_square": (
                args.probe_square if feat in ("occupancy", "piece_type") else None
            ),
            "layers": layers_payload,
            "best_layer": best_layer,
            "best_val_accuracy": best.best_accuracy,
        }
        feature_best[feat] = best.best_accuracy

    # Headline back-compat field (run_evals_backbone / dashboards): the best
    # held-out per-layer probe accuracy across whatever was run.
    headline = max(feature_best.values()) if feature_best else 0.0

    variant = (
        f"{model.cfg.d_model}d/{model.cfg.n_layers}L"
    )
    payload: dict[str, object] = {
        # v1-parity top-level keys: run/step/variant/model_config + probes.
        "run": ckpt,
        "checkpoint": ckpt,
        "step": run_block.get("step") if isinstance(run_block, dict) else None,
        "variant": variant,
        "prepend_outcome": "outcome" in conditioning,
        "conditioning": list(conditioning),
        "model_config": {
            "d_model": int(model.cfg.d_model),
            "n_layers": int(model.cfg.n_layers),
            "n_heads": int(model.cfg.n_heads),
        },
        "n_games": args.n_games,
        "n_val_games": args.n_val_games if separate_val else None,
        "probes": probes_payload,
        "probe_accuracy": headline,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
