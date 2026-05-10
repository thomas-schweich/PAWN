# stockfish-datagen

Standalone Stockfish self-play data generator. Produces zstd-compressed
parquet shards in the same column shape as `extract_lichess_parquet.py`'s
output (the shared columns), so the model's data pipeline doesn't need to
care which source the data came from.

## Usage

The binary takes a subcommand (`run`, `dry-run`, or `tournament`) and a
`--config PATH` flag.

```bash
# Validate a config without running anything.
cargo run --release -p stockfish-datagen -- dry-run \
  --config stockfish-datagen/examples/smoke.json

# Tiny smoke run (64 games, ~2s).
cargo run --release -p stockfish-datagen -- run \
  --config stockfish-datagen/examples/smoke.json

# Production: 100M games across 5 tiers (20M per tier).
# *Requires the patched binary* — every tier sets store_legal_move_evals=true
# and net_selection=large, both of which trigger the patched-binary preflight
# check (`evallegal` UCI command + `NetSelection` option). Vanilla SF18 will
# fail the preflight before any worker spawns. Build via
# `bash stockfish-datagen/scripts/build_patched_stockfish.sh`.
cargo run --release -p stockfish-datagen -- run \
  --config stockfish-datagen/examples/stockfish_100m.json

# Play two SampleScore × temperature configs against each other and
# report W/D/L + Elo with a Wilson 95% CI. Always uses the patched
# binary's `evallegal` protocol — point `stockfish_path` at the binary
# built by `stockfish-datagen/scripts/build_patched_stockfish.sh`. A
# tournament-side preflight aborts before any worker spawns if the
# binary doesn't recognize `evallegal`.
cargo run --release -p stockfish-datagen -- tournament \
  --config stockfish-datagen/examples/tournament_cp_vs_v_T0.json
```

## Reproducibility

Every row in the output carries a `game_seed` (`UInt64`). Combined with
`stockfish_version` and the per-row tier config (`nodes`, `multi_pv`,
`opening_multi_pv`, `opening_plies`, `sample_plies`, `temperature`,
`sample_score`, `net_selection`), that seed is sufficient to
deterministically replay the exact game.

For **searchless tiers** (`searchless: true`), the per-row search-budget
fields above are all `null`. The two searchless-relevant knobs —
`sample_score` (`"cp"` / `"v"`) and `net_selection` (`"auto"` / `"small"`
/ `"large"`) — are persisted per-row as nullable string columns
(see the schema table below), so a moved shard remains fully
attributable without the run-config JSON. `net_selection` is also
populated for non-searchless tiers that pinned a network; only
`sample_score` is searchless-only.

The version pin is enforced at startup: the config's
`stockfish_version` is matched at a word boundary against the
binary's `id name` line — i.e. either an exact equal or a prefix
ending at a space, so `"Stockfish 18"` accepts
`"Stockfish 18 by ..."` but does NOT accept `"Stockfish 180"`. Any
mismatch aborts before any parquet is written. Different Stockfish
releases ship different NNUE nets and would silently produce
different games from the same seed.

The `game_seed` column is stored as `UInt64` (Arrow + Parquet both
support it natively), so polars / pyarrow / pandas readers see the
actual u64 value with no bit-reinterpretation step required.

The seed hierarchy is:

```
master_seed (config)
  → tier_seed (per tier, derived from sha256(tier.name)[..8])
    → game_seed (per game, derived from the global_game_index)
```

Each step is splitmix64 mixing — pure functional, no PRNG state to
thread around. Resume can recompute any game's `game_seed[i]` from
the global index without replaying anything.

Two properties fall out of this seeding model and are pinned by the
per-tier `tier_fingerprint`:

- **Reorder-safe**: tier seeds depend on `tier.name`, not the tier's
  position in the `tiers:` list. Reordering / inserting / removing
  *other* tiers in the config doesn't change this tier's data.
- **n_workers-invariant**: game seeds depend only on the global game
  index within a tier, not on which worker thread generates the game.
  `n_workers` is purely operational — change it freely between runs.

The trade-off: renaming a tier *does* change its data (the new name
hashes to a different `tier_seed`). Treat rename as creating a new
tier. See "Dataset extension recipes" below for the supported
operations and their resume-safety properties.

## Output layout

```
<output_dir>/
  <tier_name>/
    _manifest.json                              # written last, signals tier-complete
    shard-s000000-r000834.parquet               # one shard per global shard id; -r<N> is the row count
    shard-s000001-r000833.parquet
    ...
```

Shards are numbered by global `shard_id`. Shard `s` owns the contiguous
global-game-index range `[s * shard_size_games, min((s+1) * shard_size_games, n_games))`,
so a tier with 100M games and `shard_size_games = 2000` has shards
`s000000` through `s049999` (50K shards total, last one possibly
smaller). Workers pull shard ids from a shared atomic counter; no
worker owns a fixed slice of work, so `n_workers` is purely a
parallelism knob.

The trailing `-r<NNNNNN>` field encodes the row count at six zero-padded
digits (so any `shard_size_games <= 999_999` fits). Resume reads this
straight from the directory listing rather than opening parquet metadata,
which lets a remote-sync tool drop zero-byte placeholder files locally
that the resume logic treats identically to real shards. See the
HuggingFace sync section below.

Under **multi-pod cooperation** (`stockfish-datagen run --shard-id-range A:B`,
see "Multi-pod runs" below), each pod's `_tier_state.json` and
`_manifest.json` get a `-s<A>-s<B>.json` suffix so multiple pods can
coexist in the same tier directory without colliding. Run
`scripts/datagen_reconcile_tier.py` after all pods finish to merge the
per-pod manifests into a unified `_manifest.json`.

Shards are zstd-compressed parquet (level 19, 16 MB pages). They're
written atomically: `shard-…parquet.tmp` is fsynced, then renamed to
`shard-…parquet`. A crash mid-shard leaves an orphan `.parquet.tmp`
rather than a half-valid finished file. The same fsync-then-rename
pattern is used for `_manifest.json`, so a power failure can never
leave a zero-byte manifest that incorrectly signals "tier complete".

The manifest is the tier-complete signal. Re-running no-ops any
tier whose manifest's per-tier fingerprint matches. The fingerprint
covers only the inputs that affect this tier's *generated bytes*:
`sha256(tier.name)`, the tier-config body (excluding `n_games`), the
master seed, the Stockfish version, the *effective* `stockfish_hash_mb`
(per-tier override or top-level default), `shard_size_games`, `max_ply`,
and the `SHARD_SCHEMA_VERSION` constant from `src/shard.rs`.

Explicitly **excluded** from the fingerprint (operational only):
`n_workers`, `stockfish_path`, `output_dir`, and `tier.n_games`. The
last is what makes tier extension safe: bumping `n_games` upward adds
shards `[total_shards_old, total_shards_new)` but leaves existing
shards bit-identical (game seeds depend only on the global game index).
See "Dataset extension recipes" further down for the full table of
extension operations.

Adding or modifying *other* tiers in the run config does NOT
invalidate prior tiers' manifests.

`SHARD_SCHEMA_VERSION` is the parquet-schema discriminator: bump it
in source whenever the schema gains/loses/changes a column in a way
that would mix incompatibly with prior shards in the same tier
directory. Resume across such an upgrade then fails the fingerprint
check rather than silently writing new-schema shards into a directory
of old-schema ones (which would produce a tier dir that strict polars
reads can't open and that pyarrow silently degrades). Current value
is documented in source; see also the "Schema compatibility with
pre-existing shards" subsection further down.

If the manifest is present but its fingerprint does NOT match the
current config (i.e. you've changed something on the per-tier
fingerprint list above for an already-completed tier, OR you've
upgraded the binary across a `SHARD_SCHEMA_VERSION` bump), `run_tier`
fails fast with a contextual error rather than silently re-running.
Either restore the original config / binary for that tier or
manually delete the tier's `_manifest.json` AND `_tier_state.json` to
force a regenerate from scratch.

A second sentinel `_tier_state.json` is written *before* any games
are generated and carries the same per-tier fingerprint. This
catches the "manifest is missing because we crashed before writing
it, AND someone changed the config before retrying" case: the
tier-state would mismatch and the run fails fast rather than
silently mixing bytes from two configs into one tier output.

Deleting a manifest (but not the tier-state) causes a re-scan: any
`.parquet` shards still on disk count toward `games_done` and each
worker resumes from its next chunk index, validated against the
tier-state's fingerprint.

## Schema

| column              | type            | notes                                   |
|---------------------|-----------------|-----------------------------------------|
| `tokens`            | `List<Int16>`   | searchless_chess action tokens (0..1967) |
| `san`               | `List<String>`  | SAN with check/mate suffixes            |
| `uci`               | `List<String>`  | UCI strings as Stockfish emitted them   |
| `game_length`       | `UInt16`        | == len of tokens / san / uci            |
| `outcome_token`     | `UInt16`        | 1969..=1973 (mirrors engine vocab.rs)   |
| `result`            | `String`        | "1-0", "0-1", or "1/2-1/2"              |
| `nodes`             | `Int32?`        | search-mode-only; null on searchless tiers |
| `multi_pv`          | `Int32?`        | search-mode-only; null on searchless tiers |
| `opening_multi_pv`  | `Int32?`        | search-mode-only; null on searchless tiers |
| `opening_plies`     | `Int32?`        | search-mode-only; null on searchless tiers |
| `sample_plies`      | `Int32?`        | search-mode-only; null on searchless tiers |
| `sample_score`      | `Utf8?`         | searchless-only: `"cp"` or `"v"`; null otherwise |
| `net_selection`     | `Utf8?`         | either tier mode: `"auto"` / `"small"` / `"large"`; null when the tier left the engine on its default |
| `temperature`       | `Float32`       | per-row, from tier config               |
| `worker_id`         | `Int16`         | which worker produced the row           |
| `game_seed`         | `UInt64`        | per-game seed (reproduction key)        |
| `stockfish_version` | `String`        | from Stockfish's `id name` line         |
| `legal_move_evals`  | `List<List<Struct{move_idx: Int16, score_cp: Int16, score_eval_v: Int16?, score_psqt: Int16?, score_positional: Int16?}>>` | per-ply per-legal-move payload from the tier's **selection** engine call when `store_legal_move_evals: true`; empty outer list when not. Semantics depend on tier mode: searchless tiers populate every legal move with all five fields (evallegal source); search-mode tiers populate the multipv top-K with `score_cp` only and the three nullable fields are `None`. Always non-null at the row level — readers find an empty `[]` rather than a row-level null on tiers that didn't opt in. |
| `static_legal_move_evals` | (same Struct as `legal_move_evals`) | per-ply per-legal-move **canonical NNUE static eval** captured by a separate `evallegal` call after each ply's selection. Same shape and same struct fields as `legal_move_evals`, but always full-evallegal-sourced (every legal move, all four score fields — `score_cp`, `score_eval_v`, `score_psqt`, `score_positional` — populated alongside `move_idx`). Populated only when `store_legal_move_evals: true` AND the tier is non-searchless — on searchless tiers this is **null** (the same data already lives in `legal_move_evals`, so duplicating would double tier-0 storage). Convention for downstream: read the canonical static eval as `static_legal_move_evals if not None else legal_move_evals`. The two columns describe the same plies but with different move sets and orderings; join via `move_idx` if pairing is needed. |

### Schema compatibility with pre-existing shards

The `static_legal_move_evals` column was added in the same release that
moved the writer to zstd-19 + 16 MB pages, and `SHARD_SCHEMA_VERSION`
was bumped from v1 to v2. Shards written by earlier versions don't
have the column on disk. Compatibility:

- Reading an old shard *standalone* with the new code — polars and
  pyarrow both work; downstream code sees the schema as it was at
  write time.
- **Resuming a tier directory built by the previous schema version
  fails the fingerprint check (intentional).** The bumped
  `SHARD_SCHEMA_VERSION` is now part of the per-tier fingerprint, so
  `_tier_state.json` from a v1 partial run no longer matches the v2
  fingerprint a new binary computes. `run_tier` aborts loudly with a
  contextual error before ever scanning the directory's existing
  shards. Recovery: either delete the tier's `_manifest.json` AND
  `_tier_state.json` to force a regenerate from scratch, or revert
  the binary to a v1-compatible build and finish that run first.
  Without this guard, the new binary would silently mix v1 shards
  (missing the column) with v2 shards (with the column null on
  searchless tiers, populated on search tiers) in the same directory
  — the failure mode the next bullet describes.
- Reading a *mix* of old and new shards in one query — polars (strict)
  fails with `SchemaError`; pyarrow silently drops the new column
  (returns schema intersection). For training pipelines that need the
  static column, either filter to new-schema-only shards or pass
  `extra_columns='ignore'` to polars and check for null. We don't
  recommend mixing — regenerate the dataset on the new code if you
  want consistent labels everywhere.

### Network selection and label uniformity

`net_selection` applies globally to the spawned Stockfish process — the
same setting governs both `go nodes N` (selection) and `evallegal` (the
per-ply teacher signal in `static_legal_move_evals`). There's no way to
use one network for selection and a different one for labels on the same
process.

Without it set (i.e. `auto`), `Eval::evaluate` picks per-position:
small net when `|simple_eval(pos)| > 962` (heavy material imbalance),
big net otherwise, with potential re-eval when the small-net result is
within ±277. **For distillation this means a single dataset's
`legal_move_evals` / `static_legal_move_evals` columns end up as a
mixture of small-net and big-net evaluations, switching per-position
based on material imbalance.** Set `net_selection: "large"` to force
uniform big-net labels everywhere — the canonical pick for any tier
that sets `store_legal_move_evals: true`.

## Tuning `n_workers` (CPU pinning is on by default)

Each worker thread + its Stockfish child are pinned to the same logical
cpu (`worker_id % n_logical`) on Linux. The pair is intrinsically
serialized over a pipe, so they want the same core's L1/L2 cache. This
shifts the optimal worker count *down* compared to the unpinned baseline.

Local sweep on a 16-thread / 8-physical-core / 2-SMT box (5k games,
nodes=1, sample_plies=12):

| workers | no-pin g/s | pinned g/s | delta |
|---|---|---|---|
| 8 | 179.3 | 244.1 | **+36%** |
| 12 | 219.2 | 301.5 | **+38%** |
| 13 | — | 312.0 | (peak − 3%) |
| **14** | 261.4 | **321.4** | +23% (pinned peak) |
| 15 | 284.6 | 309.9 | +9% |
| 16 | **300.6** | 301.3 | ≈0 (no-pin peak) |
| 20 | 298.3 | 231.7 | -22% |
| 24 | 303.9 | 260.0 | -14% |

### Rule of thumb: `n_workers = total_threads − threads_per_core`

Equivalent reading: **occupy every physical core except one, and leave
that one core entirely free for the kernel + the watcher thread + HF
upload networking + parquet I/O**.

- 16-thread / 8-physical / 2-SMT (this dev box): 16 − 2 = **14** (verified)
- 128-thread / 64-physical / 2-SMT (typical vast.ai): 128 − 2 = **126**
- Same chip with SMT off: 64 − 1 = 63
- 4-way SMT (POWER): 128 − 4 = 124

Why this works (under packed Linux topology, the modern default):
`core_affinity::get_core_ids()` returns logical cpus in topology order
(cpus 0,1 are SMT siblings on physical 0; cpus 2,3 on physical 1; …).
With `worker_id % n_logical` pinning and the rule above, the LAST
physical core's siblings are entirely out of the rotation, so that
core is uncontested for everything else the pod is doing.

Caveats:
- **Empirically verified at SMT=2 / 16-thread only.** The 128-thread
  case is theoretical extrapolation. A 30-second 5k-game sweep at
  {n−4, n−2, n} workers on the actual pod is cheap insurance.
- **Assumes packed topology.** On scattered topology (older systems),
  `worker_id % n_logical` doesn't leave a fully-free physical core.
- **NUMA:** the rule generalizes to multi-socket boxes without change.
  What the free core absorbs (kernel networking, the watcher thread,
  parquet write-back, orchestrator overhead) doesn't scale with
  worker count or socket count, so one global free physical core
  suffices regardless of NUMA layout. The realistic cost of NUMA-
  remote NIC softirqs preempting worker pairs is ~1% throughput on
  the affected socket — not worth burning a second physical core
  per node to avoid.

The bundled `examples/stockfish_100m.json` defaults to `n_workers: 126`
assuming a 128-thread / 64-physical / 2-SMT vast.ai box. On a different
topology, override per the rule above.

### Validating on the actual pod (~1-3 minutes)

`scripts/sweep_pod_workers.sh` runs a quick `n_workers` sweep on the
pod's hardware. Useful before launching the full fire-and-forget run,
since `n_workers` is part of the per-tier fingerprint — changing it
mid-run would invalidate the partial state and waste pod hours.

```bash
# Auto-detect topology, sweep ±4 around the rule's prediction:
bash /opt/datagen/scripts/sweep_pod_workers.sh

# Or test specific values:
bash /opt/datagen/scripts/sweep_pod_workers.sh 122 124 126 128
```

Each point is 5000 nodes=1 games (~5–15s of pod time), full pipeline
including pinning. Output is sorted by rate descending; pick the
winner and put it in your run config before launching the production
HF-sync job. The script prints to stdout only — nothing is uploaded.

The `dev` Docker target (see `Dockerfile.datagen`) is purpose-built
for this workflow: it adds sshd + tmux + htop + jq + numactl on top
of the runtime image and keeps the pod alive indefinitely so you can
SSH in, run the sweep, edit configs, then launch the production run
manually inside `tmux` for detach-survival. See the Dockerfile header
for the build/run incantations.

CPU pinning is Linux-only (`core_affinity::set_for_current` for the
worker thread, `libc::sched_setaffinity` for the Stockfish child PID);
on macOS both calls no-op. See `stockfish-datagen/src/affinity.rs`
for the full module docstring.

### NUMA placement on dual-socket pods

`stockfish-datagen` calls `set_mempolicy(MPOL_INTERLEAVE)` at startup
(Linux only; see `src/numa.rs`). The kernel inherits mempolicy across
`execve`, so every spawned stockfish child first-touches its NNUE
page-cache fills under interleave policy — spreading the ~50 MB of
weight pages evenly across NUMA nodes. On dual-socket EPYC pods that
reclaims roughly the half-of-workers-pay-3.2× cross-socket penalty
identified in `ANALYSIS.md`. On single-socket pods the policy
collapses to local placement (no-op).

If you'd rather avoid the topology question entirely: prefer
single-socket EPYC 9654 pods (96 cores / 192 threads, ~$0.27/h on
vast.ai). They produce similar aggregate g/s to the dual-socket box
(~$0.83/h) and have no first-touch / cross-socket cost. Cost-of-data
is comparable; CPU is used more efficiently.

### Multi-pod runs (fanning a tier across machines)

For long-pole tiers (e.g. `nodes_1024` at 20M games), parallelize
across multiple pods using `--shard-id-range`:

```bash
# Pod A — first half of nodes_1024:
stockfish-datagen run --config stockfish_100m.json \
    --tiers 4 --shard-id-range 0:5000

# Pod B — second half:
stockfish-datagen run --config stockfish_100m.json \
    --tiers 4 --shard-id-range 5000:10000

# After both pods finish: merge their per-pod manifests into the
# canonical `_manifest.json` (also runnable with --dry-run first).
python scripts/datagen_reconcile_tier.py \
    --config stockfish_100m.json \
    --repo-id <repo> --tier nodes_1024
```

Each pod writes per-pod sentinels (`_tier_state-s<A>-s<B>.json`,
`_manifest-s<A>-s<B>.json`) so the pods don't fight over the
canonical names. Both `cfg.n_workers` and the per-tier `n_games` /
`shard_size_games` MUST match across pods — they determine the
total-shards count and therefore the global shard-id range. Anything
else (`stockfish_path`, `output_dir`, `n_workers`) can differ per pod.

### Dataset extension recipes

After the shard-id partitioning refactor, the following operations are
supported on an existing dataset without invalidating its shards:

| Operation | Safe? | Mechanism |
|-----------|-------|-----------|
| Add a new tier | ✓ | Append to `tiers:` with a unique `name`. New tier gets its own directory + fingerprint; existing tiers untouched. |
| Grow `tier.n_games` | ✓ | Bump `n_games` upward. Resume adds shards `[total_shards_old, total_shards_new)`. If the previous last shard was partial (`n_games_old % shard_size_games != 0`), it's regenerated to the full row count — existing-prefix games stay bit-identical because game seeds depend only on global index. |
| Reorder `tiers:` | ✓ | Tier identity is the `name` (sha256-hashed into the seed), not list position. |
| Change `n_workers` | ✓ | Operational only — no impact on game contents. |
| Change `stockfish_path` / `output_dir` | ✓ | Operational only. |
| Move dataset to a new HF repo | ✓ | Just push; fingerprints don't include repo. |
| Rename a tier | ✗ | New name hashes to a different `tier_seed`. Treat as creating a new tier; the old shards remain valid under the old name. |
| Change `master_seed`, `stockfish_version`, `max_ply`, `shard_size_games`, `stockfish_hash_mb`, or any other tier-config field | ✗ | All are dataset-affecting per the fingerprint contract. Operator must move existing shards aside before resume. |
| Shrink `tier.n_games` | partial | New shards `[total_shards_new, total_shards_old)` become orphans on disk / HF. Resume itself doesn't touch them; clean up manually if you care. |

## Strategic 50-move-rule claim

The 50-move rule is *claimable* in FIDE rules, not automatic, and a
strong player has no incentive to claim if they're winning. Our
generator models this: at the moment the 50-move threshold is reached
(halfmove 100), the side about to move uses Stockfish's eval —
specifically the top-candidate `score_cp` (which is documented as
side-to-move POV). If they're losing (`score_cp < 0`), they claim →
game ends as `FiftyMoveRule`. If winning or even, they keep playing.

In practice this becomes a "50-or-51-or-… move rule": claims
fluctuate as evaluations swing, until either someone resets the
halfmove clock with a capture / pawn move or the FIDE 75-move
*automatic* rule fires at halfmove 150 (the hard upper bound).

**What this gives the model**: a learnable signal for *when* to claim.
The dataset contains games that ended in 50-move-rule draws at varying
halfmove offsets in [100, 150], correlated with the position's eval at
that point. With unconditional claim at halfmove 100, every such game
would terminate at the same halfmove and the model would just learn
"halfmove 100 → game over."

3-fold repetition stays unconditional rather than eval-strategic —
making it strategic risks both sides perpetually shuffling in a drawn
position, which the 50-move clock would eventually reset anyway.

## Operator notes

- **SIGINT / Ctrl-C is safe.** Hitting Ctrl-C mid-shard leaves a
  `.parquet.tmp` orphan that the next run cleans up on
  `ShardWriter::create`. Hitting Ctrl-C between the last shard write
  and the manifest write leaves all shards complete but no
  `_manifest.json` — the next run sees the matching `_tier_state.json`,
  scans the existing shards, finds every worker already at its target,
  writes the manifest, and exits cleanly without re-running anything.
  (The `live_resume_all_workers_done_no_manifest` test pins this.)
- **Long multi-tier jobs** are checkpointed at tier granularity. A
  crash in tier N doesn't affect manifests for tiers 0..N-1, and the
  rerun resumes tier N from its last completed shard per worker.

## Fire-and-forget pod runs (HuggingFace sync)

`scripts/datagen_with_hf_sync.py` wraps the rust binary so a pod can
generate to a HuggingFace dataset without an operator in the loop, and
resume after a crash without re-downloading the existing shards.

```bash
HF_TOKEN=... python scripts/datagen_with_hf_sync.py \
    --config stockfish-datagen/examples/stockfish_100m.json \
    --repo-id thomas-schweich/pawn-stockfish-100m \
    --prune-local
```

Three phases run in one process:

1. **Primer.** Lists the dataset repo. Downloads the per-tier sentinels
   (`_manifest.json`, `_tier_state.json`) into the local output dir.
   For every remote shard file, creates a *zero-byte placeholder* at the
   matching local path. Because the rust binary's resume logic recovers
   `(worker_id, chunk_idx, n_rows)` from the filename alone, a directory
   full of placeholders looks identical to a directory full of real
   shards — no parquet metadata is read on resume, so no actual data
   needs to be downloaded.
2. **Subprocess.** Spawns `stockfish-datagen run --config <cfg>`.
   SIGINT/SIGTERM are forwarded so the rust binary's graceful-shutdown
   semantics are preserved.
3. **Watcher.** A daemon thread polls the output dir every
   `--poll-interval` seconds (default 30 s) and uploads any new
   completed shards (and updated sentinels) via `HfApi.upload_file`.
   Zero-byte placeholders are skipped — those are already remote.
   With `--prune-local`, the local file is replaced with a zero-byte
   placeholder after a successful upload, so disk usage stays flat
   regardless of the run's total size.

CLI flags worth knowing:

- `--prune-local` — replace each local shard with a zero-byte
  placeholder after a successful upload. Use on disk-constrained pods.
- `--no-primer` — skip the primer phase (no `list_repo_files` call,
  no sentinel downloads, no placeholder creation). The cheap repo
  existence/creation check (`HfApi.repo_info` or `create_repo`) still
  runs; uploads can't create repos themselves so this guarantees the
  watcher's first upload won't 404. Useful for local testing or
  restarting against a known-empty repo.
- `--poll-interval <seconds>` — watcher poll cadence. Default 30 s.
- `--max-consecutive-failures <N>` — watcher gives up + SIGTERMs the
  rust child after N consecutive cycle failures. Default 10
  (~5 minutes at the default poll). Tune up for runs that may span
  longer documented HF outages.
- `--watcher-drain-timeout-hours <hours>` — after the rust binary exits,
  wait up to this many hours for the watcher to flush any pending shard
  uploads before exiting. Default 4 hours. If the timeout fires with
  uploads still pending, exit code is `4` (or the rust binary's non-zero
  `rc` if that also failed; rc takes precedence). Local shards stay on
  disk and the next run's primer picks up where this one left off.
- `--log-level {DEBUG,INFO,WARNING,ERROR}` — default INFO.

Exit codes: `0` clean; non-zero `rc` from the rust binary if it
failed; `3` watcher gave up after consecutive failures (auth/quota/
permanent 4xx — the rust child is also SIGTERM'd, so partial output
remains and is recoverable on the next run via the primer); `4`
final drain failed (or was skipped because the watcher was still
mid-upload at join timeout — local shards stay on disk and resume
picks them up).

Pod recipe (Docker image's default entrypoint dispatches to this script
when both env vars are set). Use the single `:datagen` tag — the
patched stockfish binary is JIT-built at first launch for the host
CPU and cached on the persistent volume (see
`scripts/build_stockfish_for_host.sh`), so one image covers every
supported microarch (Zen 4 vnni512, Skylake-X avx512, Haswell+ avx2,
modern fallback):

```bash
docker run --rm -d \
    -e HF_TOKEN=... \
    -e DATAGEN_HF_REPO=thomas-schweich/pawn-stockfish-100m \
    -e DATAGEN_CONFIG=/opt/datagen/examples/stockfish_100m.json \
    -e DATAGEN_PRUNE_LOCAL=1 \
    -v /workspace/sf:/workspace/sf \
    thomasschweich/pawn:datagen
```

First launch takes an extra ~30-60 s on a 16-core box (or ~5-10 s on
a 96-core EPYC) for the stockfish build; subsequent launches on the
same persistent volume are cache hits (~10 ms).

Recognized env vars:

| Env var                | Effect                                                        |
|------------------------|---------------------------------------------------------------|
| `HF_TOKEN`             | HuggingFace auth (mandatory). Aliases `HUGGING_FACE_HUB_TOKEN` and `HUGGINGFACE_HUB_TOKEN` are also accepted, as is the on-disk credential store written by `hf auth login`. |
| `DATAGEN_HF_REPO`      | Dataset repo, e.g. `org/name`. Required for the entrypoint.   |
| `DATAGEN_CONFIG`       | Path to the run config inside the image.                      |
| `DATAGEN_PRUNE_LOCAL`  | Any non-empty value enables `--prune-local`.                  |
| `DATAGEN_POLL_INTERVAL`| Watcher poll seconds (overrides default 30).                  |
| `DATAGEN_EXTRA_ARGS`   | Extra flags appended to the orchestrator command line (e.g. `--max-consecutive-failures 30 --log-level DEBUG`). |

Caveats:

- Don't run two pods writing to the same dataset repo. There's no
  cross-process locking; concurrent writers will race on commits.
- The rust binary's tier-state fingerprint check still applies after
  primer download — if the local config doesn't match the remote
  `_tier_state.json`, the run aborts before generating anything.

## Failure model

There is no "drop and continue" path. If `play_game` returns an
empty move list (impossible from the starting position) or the
engine can't re-tokenize Stockfish's UCI output, the worker fails
the whole tier with a contextual error. This avoids the silent-
desync that a "drop and skip" path would create on resume — every
written row corresponds to a contiguous prefix of game indices,
so resume from `(rows_on_disk, next_chunk_idx)` is always correct.
