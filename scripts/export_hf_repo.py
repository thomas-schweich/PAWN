#!/usr/bin/env python3
"""Compatibility shim for the v1 post-hoc HF export tool.

The v1 ``scripts/export_hf_repo.py`` packaged a finished training run
into HF-repo layout (best checkpoint by val loss at root,
other checkpoints under ``checkpoints/``, README, metrics). The v2
training stack pushes to HF *during* training via the ``--hf-repo``
flag on ``scripts/train_jax.py`` / ``scripts/train_jax_adapter.py`` —
the trainer's lifecycle helper drains the upload queue at the SIGTERM
path so no separate packaging step is needed.

This wrapper exits with a clear migration message rather than
silently producing nothing. If a real need for the v1 post-hoc
packaging surface emerges (e.g., re-packaging an old local-only run),
that's a separate v2 follow-up; it's currently not wired up.
"""

from __future__ import annotations

import sys


_MIGRATION_NOTE = """\

v1 `scripts/export_hf_repo.py` packaged a finished local training run
into HF-repo layout. v2 pushes to HF during training via:

  python scripts/train_jax.py        --hf-repo USER/repo ...
  python scripts/train_jax_adapter.py --hf-repo USER/repo ...

The trainer's `HFPushTracker` uploads every `step_<N>/` directory as
it's written and drains the queue on graceful shutdown. No separate
post-hoc packaging step is needed for new runs.

If you have a legacy local-only run to package, that path is not yet
ported in the v2 stack. Track the gap in the project's open issues
and contribute a v2 port if needed.
"""


def main() -> int:
    print(
        "scripts/export_hf_repo.py: this tool was a v1-only post-hoc "
        "HF packager. The v2 stack pushes to HF during training via "
        "the `--hf-repo` flag.\n"
        f"{_MIGRATION_NOTE}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
