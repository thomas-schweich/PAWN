# Pareto Frontier: Adapter Parameter Count vs Accuracy

You are a solo ML researcher with 18 hours on 2x A100 SXM GPUs (160GB VRAM total, 234GB RAM, 32 vCPUs). Your job is to map the **Pareto front of val_loss vs trainable parameter count** across all available adaptation strategies for the PAWN chess transformer. The goal: understand which strategies give the best accuracy at every parameter budget, from 10K to 10M parameters.

**Why this matters:** We want to enable a hypernetwork to generate adapter weights that produce different "player styles." Lower parameter counts mean smaller hypernetwork outputs, so finding strategies that are accurate *and* compact is critical. But highest absolute accuracy also matters — we're approaching MAIA-1 performance with a CLM that tracks state across tokens rather than seeing the full board.

You have the Optuna MCP tool for trial management (multi-objective ask/tell, visualization). You are running in a pod where **only `/workspace` is persistent**.

**As your very first action**, start a 15-minute check-in loop: `/loop 15m date`. Use each check-in to: review running trials, report results to Optuna, launch replacements for finished trials, and update lab notes. When all GPU slots are full, wait for a trial to finish — but when one completes, report it and launch the next promptly.

**IMPORTANT RULES:**
- **Never enter planning mode.** No one is available to approve plans. Just execute.
- **Maintain lab notes** at `/workspace/lab_notes.md`. These are your persistent memory across context compactions.
- **Use Optuna ask/tell for adaptive sampling.** Seed with existing results, then let the sampler drive. Do not manually grid-search.
- **Use `--no-compile` for trials under 20K steps.** `torch.compile` takes 15-30 min and doesn't amortize on short runs. Enable it for 50K+ step runs only.
- **Before your context gets large**, proactively update lab notes with everything a fresh agent would need to continue.

## Lab Notes

Maintain `/workspace/lab_notes.md` as a running log. A fresh agent reading only the lab notes + this prompt should be able to pick up exactly where you left off.

The lab notes must always contain:
1. **Current phase** and what's running now (trial numbers, PIDs)
2. **All completed trial results** — table with trial number, strategy, key params, param_count, val_loss, top1, status
3. **Current Pareto front** — the non-dominated set of (param_count, val_loss) points
4. **Decisions made** and key findings
5. **What to do next** — specific instructions for the incoming agent
6. **Wall clock** — sweep start time, current time, estimated remaining

Update at every phase boundary and after every batch of trials.

## First Steps

```bash
# Symlink results and logs to persistent storage
mkdir -p /workspace/sweep_results /workspace/logs /workspace/plots /workspace/optuna-storage
ln -sfn /workspace/sweep_results /opt/pawn/local/optuna_results
ln -sfn /workspace/logs /opt/pawn/logs

# Append sweep instructions to CLAUDE.md for context compaction recovery
cat >> /opt/pawn/CLAUDE.md << 'SWEEP_EOF'

## Active Sweep (appended at runtime)

A Pareto frontier sweep is in progress. On every new context (including after compaction):
1. Read `/workspace/lab_notes.md` — has all trial results and current state
2. Read `prompts/pareto_sweep.md` — has the full sweep procedure
3. Resume from where the lab notes say you left off
4. Never enter planning mode
SWEEP_EOF

# CRITICAL: Verify GPUs are available. Training on CPU wastes the entire budget.
python3 -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB each')"

# CUDA MPS should already be running (started by setup-workspace.sh).
# Verify it's active:
ps aux | grep nvidia-cuda-mps | grep -v grep || echo "WARNING: MPS not running. Ask the user to run setup-workspace.sh as root."
```

Then create the Optuna study:
- Study name: `pareto-sweep`
- **Directions: `["minimize", "minimize"]`** — minimize val_loss AND param_count (bi-objective)
- Metric names: `["val_loss", "param_count"]`

Launch the Optuna dashboard on port 58080.

Initialize lab notes at `/workspace/lab_notes.md`.

**GPU verification is not optional.** If the check fails, STOP. Do not train on CPU.

## Backup to HuggingFace

Periodically sync all results to the HF bucket. Run this at every phase boundary and every ~2 hours:

```bash
hf sync /workspace hf://buckets/thomas-schweich/pawn-pareto-sweep-03-30-2026
```

This uploads everything: sweep results, checkpoints, Optuna DB, lab notes, logs, and plots. If the pod dies, all work is recoverable from the bucket.

## Known Infrastructure Issues

These were discovered during the previous sweep. Do not waste time rediscovering them:

1. **`uv run` is broken** on pods — workspace member resolution fails. Use `python3` directly.
2. **Always pass `--log-dir /workspace/logs`** — the default path lacks write permissions.
3. **Process management**: Use `nohup python3 ... > /workspace/sweep_results/trial_N.log 2>&1 &` for background trials. Do NOT use `setsid` (creates duplicates).
4. **Optuna MCP storage**: 4 slashes for absolute path: `sqlite:////workspace/optuna-storage/study.db`
5. **AMP float16 NaN**: Occurs at step 25-40K with lr >= 7.07e-4. Use `--amp-dtype bfloat16` for long runs, or drop lr to 5e-4 with float16.

## What You're Optimizing

The unified training script `scripts/train_adapter.py` supports 8 adaptation strategies on a frozen PAWN-Base transformer (d_model=512, 8 layers, 35.8M frozen params). All predict 1800-1900 Elo Lichess moves.

### Strategies

| Strategy | Description | Param range | Notes |
|----------|-------------|-------------|-------|
| `bottleneck` | Residual MLP after attn/FFN | 1K-10M | Proven winner at 10M. Underexplored below 100K. |
| `lora` | Low-rank attention/FFN adaptation | 8K-500K | Barely tested on Lichess. Strong hypernetwork candidate. |
| `film` | Per-channel affine modulation | ~17K-33K | Tested once (30.3%). Ultra-low param niche. |
| `sparse` | Random binary mask weight perturbations | 10K-2.7M | Underperforms bottleneck at all scales tested. |
| `rosa` | LoRA warmup + gradient masks + joint training | 50K-10M | 3-phase. Co-trained sparse may differ from retro. |
| `hybrid` | LoRA + FiLM combined | 30K-500K | Tested once at 65K (34.1%). Underexplored. |
| `specialized_clm` | From-scratch standalone CLM (no backbone) | 50K-5M | Only 1 data point (529K=30.9%). Key comparison. |
| `unfreeze` | Fine-tune top N backbone layers | 2.6M-5.2M | Untested. Qualitatively different from adapters. |

### How to Run Trials

```bash
python3 scripts/train_adapter.py --strategy <STRATEGY> \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full \
    --elo-min 1800 --elo-max 1900 \
    --log-dir /workspace/logs \
    --local-checkpoints \
    [strategy-specific args]
```

The script writes `config.json` (full normalized config with nulls for irrelevant params) alongside `metrics.jsonl` in the run directory.

### Strategy-Specific CLI Args

**Placement** (bottleneck, lora, sparse, rosa, unfreeze):
- `--adapter-layers 4,5,6,7` — which layers (default: all)

**Bottleneck** (bottleneck, rosa retro-bottleneck):
- `--bottleneck-dim N` — hidden dimension
- `--no-adapt-attn` / `--no-adapt-ffn` — skip positions

**Low-rank** (lora, rosa, hybrid):
- `--lora-rank N` — rank
- `--lora-targets qkvo|qv|qkv`
- `--lora-ffn` — also adapt FFN

**Sparse** (sparse, rosa):
- `--density F` — mask density
- `--sparse-targets qkvo|qv|qkv`
- `--sparse-ffn`

**Mask generation** (rosa only):
- `--rosa-mode rosa|retro-sparse|retro-bottleneck`
- `--rosa-warmup-steps N` — LoRA warmup steps
- `--mask-samples N` — batches for gradient accumulation
- `--grad-alpha 1|2` — gradient exponent

**FiLM** (film, hybrid):
- `--use-output-film`

**From-scratch** (specialized_clm only):
- `--d-model N --n-layers N --n-heads N`
- No `--checkpoint` needed (ignored)

**Unfreeze**:
- `--unfreeze-layers 6,7` — which layers to fine-tune

**Training** (all strategies):
- `--lr F --batch-size N --total-steps N --eval-interval N`
- `--warmup-frac F --weight-decay F --max-grad-norm F`
- `--amp-dtype float16|bfloat16`
- `--no-compile` — disable torch.compile (USE THIS for trials < 20K steps)
- `--max-games N` — training data size
- `--val-games N` — validation size (fixed at 50000)

### Parameter Count Formulas

```
Bottleneck: n_positions × n_layers × 2 × 512 × bottleneck_dim
LoRA:       n_targets × n_layers × 2 × 512 × rank  (+ FFN if enabled)
FiLM:       n_layers × 2 × 512 (+ vocab_size if output_film)
Sparse:     density × n_targets × n_layers × 512 × 512
Specialized CLM: ~d_model² × n_layers × 12  (rough estimate)
Unfreeze:   ~2.6M per layer (full transformer block)
```

### Tiny Model Canonical Sizes

For the `specialized_clm` strategy, use these predetermined architectures:

| Budget | d_model | n_layers | n_heads | ~Params |
|--------|---------|----------|---------|---------|
| ~50K | 32 | 2 | 2 | ~52K |
| ~130K | 48 | 2 | 4 | ~127K |
| ~500K | 84 | 2 | 4 | ~524K |
| ~2M | 128 | 4 | 4 | ~2.1M |
| ~5M | 192 | 4 | 8 | ~4.8M |

## Existing Results to Seed

You have ~75 existing data points from the previous bottleneck sweep and README experiments. **Seed these into the Optuna study using `add_trial`** before starting new exploration. This gives the sampler a strong prior.

### From Bottleneck HP Sweep (March 2026)

| Strategy | Params | val_loss | top1 | Config summary |
|----------|--------|----------|------|----------------|
| bottleneck | 10M | 1.5634 | 50.51% | dim=1220, layers 4-7, lr=5e-4, bs=256, 100K steps, 3.5M games |
| bottleneck | 10M | 1.6380 | 48.75% | dim=1220, layers 4-7, lr=7.07e-4, bs=128, 50K steps |
| bottleneck | 10M | 1.7007 | 47.25% | dim=1220, layers 4-7, lr=1e-3, bs=256, 10K steps, 1M games |
| bottleneck | 10M | 1.7137 | 46.88% | dim=1220, layers 4-7, lr=5e-4, bs=256, 10K steps, 1M games |
| bottleneck | 7.8M | 1.7759 | 45.38% | dim=478, layers 4-7, 10K steps |
| bottleneck | 10M | 1.8060 | 44.78% | dim=610, all 8 layers, lr=5e-4, bs=64, 10K steps |

### From README Experiments (1800-1900 Elo only)

| Strategy | Params | val_loss | top1 | Notes |
|----------|--------|----------|------|-------|
| bottleneck | 1.0M | — | 43.5% | dim=64 |
| bottleneck | 524K | — | 41.7% | dim=32 |
| sparse | 2.7M | — | 44.7% | density=0.081, qkvo+FFN |
| specialized_clm | 529K | — | 30.9% | d=84, 2 layers, from scratch |

**Note:** Only seed results from the 1800-1900 Elo band. Results from other Elo ranges are not comparable.

### From Architecture Comparison (March 2026 sweep)

| Strategy | Params | val_loss | top1 | Notes |
|----------|--------|----------|------|-------|
| retro-bottleneck | 10M+10K | 1.7492 | 46.15% | Sparse adds zero value |
| retro-sparse (density=1.0) | 8.4M | 1.8786 | 43.26% | Sparse ceiling |
| retro-sparse (density=0.119) | 1M | 2.1205 | 37.69% | |

## Concurrency

All trials share the 2 A100s via CUDA MPS. MPS handles GPU placement automatically — no `CUDA_VISIBLE_DEVICES` needed.

**You must determine concurrency dynamically.** Different strategies use wildly different amounts of VRAM (a 10K-param FiLM trial vs a 10M-param bottleneck trial). After launching your first few trials, check VRAM usage:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

Use this to estimate how many more trials of similar size you can fit. Keep both GPUs saturated but stay well below 150GB total to avoid OOM. As trials finish, launch new ones to fill the freed capacity.

32 vCPUs are available. Adjust `--num-workers` based on how many concurrent trials are running (total workers across all trials should stay under ~30).

## The Sweep: 3 Phases

### Phase 1: Pareto Exploration (Target: hours 0-5)

**Purpose:** Map the val_loss vs param_count Pareto front across all 8 strategies. Use Optuna's bi-objective sampler to efficiently explore.

**Training config:** `--max-games 1000000 --val-games 50000 --total-steps 10000 --eval-interval 1000 --elo-min 1800 --elo-max 1900 --no-compile --num-workers 2 --log-dir /workspace/logs --local-checkpoints`

**Search space (register with Optuna):**

| Parameter | Type | Range |
|-----------|------|-------|
| `strategy` | Categorical | all 8 |
| `log_param_budget` | Float | [3.7, 7.0] (5K to 10M) |
| `lr` | Float (log) | [1e-5, 2e-3] |
| `batch_size` | Categorical | [64, 128, 256] |
| `adapter_layers` | Categorical | ["all", "4,5,6,7", "2,3,4,5,6,7"] |

Strategy-specific params are conditional on strategy choice. Handle the conditional logic after each `ask`:

- **bottleneck**: derive `bottleneck_dim` from param budget. Choose `adapt_attn`/`adapt_ffn`.
- **lora**: derive `lora_rank` from budget. Choose `lora_targets`, `lora_ffn`.
- **film**: fixed ~17-33K params (budget only controls `use_output_film`).
- **sparse**: derive `density` from budget. Choose `sparse_targets`, `sparse_ffn`.
- **rosa**: derive `density` + optionally `bottleneck_dim` from budget. Choose `rosa_mode`, `lora_rank`.
- **hybrid**: split budget between LoRA rank and FiLM.
- **specialized_clm**: snap to nearest canonical size from the table above.
- **unfreeze**: `unfreeze_layers` from budget (1 layer = ~2.6M, 2 layers = ~5.2M).

**Procedure:**
1. Seed Optuna with all existing results from the tables above using `add_trial`.
2. Loop: `ask` → translate to CLI args → launch trial → on completion, read `config.json` for param_count, `metrics.jsonl` for best val_loss → `tell` with `[val_loss, param_count]`.
3. Keep GPUs saturated — determine concurrency from observed VRAM usage.
4. Target: 60-80 new trials in Phase 1.
5. After Phase 1, plot the Pareto front. Identify which strategies dominate at which scales.

**Kill policy:** If train_loss is NaN or increasing after step 2000, kill and tell Optuna with val_loss=999.

### Phase 2: Deep Dive on Interesting Regimes (Target: hours 5-10)

**Purpose:** For the most promising strategies and parameter budgets from Phase 1, run longer trials (50K steps) with more data to get reliable rankings.

**Training config:** `--max-games 3500000 --val-games 50000 --total-steps 50000 --eval-interval 2500 --batch-size 256 --amp-dtype bfloat16 --num-workers 4 --log-dir /workspace/logs --local-checkpoints`

Use `--no-compile` for these runs only if compile overhead is still problematic. At 50K steps, compile usually pays for itself.

**Procedure:**
1. From Phase 1 Pareto front, identify 3-4 parameter budgets where multiple strategies are competitive (e.g., 50K, 200K, 1M, 5M).
2. For each budget, run the top 2-3 strategies head-to-head.
3. Also run strategies that were underexplored in Phase 1 (especially: lora at various ranks, hybrid, unfreeze).
4. Target 20-30 trials. Adjust concurrency based on VRAM usage.

**Key questions to answer:**
- Does LoRA beat bottleneck at low param counts?
- Can specialized CLMs (from-scratch) match adapted models at any scale?
- Does unfreeze (direct fine-tuning) outperform structured adapters?
- At what param count does bottleneck become the clear winner?

### Phase 3: Final Runs + Validation (Target: hours 10-18)

**Purpose:** Train the Pareto-optimal configs to convergence and validate.

**Training config:** `--max-games 3500000 --val-games 50000 --total-steps 100000 --eval-interval 5000 --batch-size 256 --amp-dtype bfloat16 --num-workers 4 --log-dir /workspace/logs --local-checkpoints`

Enable `torch.compile` for these runs (remove `--no-compile`).

**Procedure:**
1. Select 5-8 Pareto-optimal configs spanning the param_count range.
2. Train each to convergence (100K steps or until early stopping).
3. Run each config twice (different data shuffle) for reproducibility — results should be within 0.01 val_loss.
4. Run `scripts/eval_accuracy.py` on the best checkpoints for per-ply accuracy curves.
5. Plot the final Pareto front.

**Deliverables:**
- Pareto front plot (val_loss vs param_count, colored by strategy)
- Best checkpoint per budget tier
- Reproducibility evidence
- Per-ply accuracy curves for Pareto-optimal configs

## Status Updates

Write `/workspace/sweep_status.md` at each phase boundary. At the end, write `/workspace/sweep_report.md` with:
- Full Pareto front (all trials, dominated and non-dominated)
- Winning strategy per budget tier
- Key scientific findings
- Recommendations for hypernetwork design (which strategies are most compact + accurate)

Save Optuna plots (`plot_pareto_front`, `plot_param_importances`, `plot_optimization_history`) to `/workspace/plots/` between phases.

## Time Budget

| Phase | Hours | Trials | Steps | Concurrency |
|-------|-------|--------|-------|-------------|
| 1: Pareto exploration | 0-5 | 60-80 | 10K | dynamic (maximize) |
| 2: Deep dive | 5-10 | 20-30 | 50K | dynamic (moderate) |
| 3: Final runs | 10-18 | 10-16 | 100K | dynamic (few, large) |

These are targets, not hard deadlines. The priority order is: **map the Pareto front > validate winners > maximize coverage**.

## Red Flags

- **NaN loss:** Kill immediately, tell Optuna as FAIL. Use `--amp-dtype bfloat16` for long runs.
- **VRAM approaching 150GB:** Reduce concurrency. MPS doesn't prevent OOM.
- **MPS daemon crash:** Check with `ps aux | grep mps`. If gone, ask the user to re-run `setup-workspace.sh` as root.
- **Step time regression:** Compare against Phase 1 baselines. If a trial is 3x slower than expected, check compile status.
- **Train/val gap > 0.3:** Overfitting. The model needs more data or regularization.

## Acceptable Outcomes (in order of preference)

1. **Clear Pareto front** across 10K-10M params with 5+ strategies represented, reproducibility evidence, and per-ply accuracy curves for winners
2. **Pareto front** with a surprise finding (e.g., LoRA dominates bottleneck below 100K, or specialized CLM catches up at 2M params)
3. **Strategy ranking** at 3+ budget tiers with evidence for the best hypernetwork-friendly architecture at each tier
