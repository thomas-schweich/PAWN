"""Optional Weights & Biases metric mirror.

Every v2 training entry point funnels through `init_wandb` /
`log_metrics` / `finish_wandb` for consistent run naming + lifecycle.
The module is **torch-free** (wandb is the only dependency, and it's
behind the `wandb` extra per plan §10 S1).

Reproducibility metadata (slug, git hash, hostname, platform) is
embedded in the W&B config so a sweep over checkpoints stays
auditable. The `PAWN_WANDB_MODE` env var (`online` / `offline` /
`disabled`) is honoured at init time.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
from collections.abc import Mapping
from typing import Any, Literal

__all__ = [
    "init_wandb",
    "log_metrics",
    "finish_wandb",
]


def init_wandb(
    *,
    project: str,
    slug: str,
    run_config: Mapping[str, Any],
    git_hash: str | None = None,
    enabled: bool = True,
) -> Any:
    """Initialise a W&B run for this training session.

    Returns the W&B ``run`` object (or ``None`` when disabled). Tags
    include ``git:<hash>`` for cross-resume grouping; the run name is
    the slug. Config carries the resolved run config + host metadata.
    """
    if not enabled:
        return None
    mode_env = os.environ.get("PAWN_WANDB_MODE", "online")
    mode: Literal["online", "offline", "disabled", "shared"]
    if mode_env == "online":
        mode = "online"
    elif mode_env == "offline":
        mode = "offline"
    elif mode_env == "disabled":
        return None
    elif mode_env == "shared":
        mode = "shared"
    else:
        # Unrecognised PAWN_WANDB_MODE: fall back to "online" rather than
        # crashing — the env var is intended as an operator-facing knob,
        # not a typed enum.
        mode = "online"
    try:
        import wandb
    except ImportError:
        return None

    cfg: dict[str, Any] = {
        "slug": slug,
        "git_hash": git_hash,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    cfg.update(run_config)
    tags = []
    if git_hash:
        tags.append(f"git:{git_hash[:8]}")
    return wandb.init(
        project=project,
        name=slug,
        config=cfg,
        tags=tags,
        group=slug,
        mode=mode,
    )


def log_metrics(
    run: Any, metrics: Mapping[str, Any], *, step: int | None = None
) -> None:
    if run is None:
        return
    if step is not None:
        run.log(dict(metrics), step=step)
    else:
        run.log(dict(metrics))


def finish_wandb(run: Any) -> None:
    if run is None:
        return
    run.finish()
