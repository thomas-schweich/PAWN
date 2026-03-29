#!/usr/bin/env bash
# scripts/run_optuna_trial.sh — Run a single bottleneck adapter trial and write the result
#
# Usage: bash scripts/run_optuna_trial.sh <trial_number> <lr> <warmup_frac> <weight_decay> \
#            <max_grad_norm> <batch_size> [bottleneck_dim] [extra_args...]
#
# Extra args are passed through to train_bottleneck.py (e.g. --max-games, --adapter-layers, etc.)
set -euo pipefail

TRIAL=${1:?trial_number required}
LR=${2:?lr required}
WARMUP=${3:?warmup_frac required}
WD=${4:?weight_decay required}
GRAD_NORM=${5:?max_grad_norm required}
BS=${6:?batch_size required}
DIM=${7:-610}
shift 7 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="$SCRIPT_DIR/local/optuna_results"
mkdir -p "$RESULT_DIR"
RESULT_FILE="$RESULT_DIR/trial_${TRIAL}.json"
LOG_FILE="$RESULT_DIR/trial_${TRIAL}.log"

echo "=== Trial $TRIAL: lr=$LR warmup=$WARMUP wd=$WD grad_norm=$GRAD_NORM bs=$BS dim=$DIM ${EXTRA_ARGS[*]:-} ==="

START_TIME=$(date +%s)

uv run python scripts/train_bottleneck.py \
    --checkpoint thomas-schweich/pawn-base \
    --pgn thomas-schweich/pawn-lichess-full \
    --bottleneck-dim "$DIM" \
    --lr "$LR" \
    --warmup-frac "$WARMUP" \
    --weight-decay "$WD" \
    --max-grad-norm "$GRAD_NORM" \
    --batch-size "$BS" \
    --local-checkpoints \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

END_TIME=$(date +%s)
WALL_SECS=$(( END_TIME - START_TIME ))

# Extract best val_loss from the "Done. Best val_loss=X.XXXX" line
BEST=$(grep -oP 'Best val_loss=\K[0-9.]+' "$LOG_FILE" || echo "")

if [ -z "$BEST" ]; then
    cat > "$RESULT_FILE" <<EOF
{
  "trial": $TRIAL,
  "status": "failed",
  "params": {"lr": $LR, "warmup_frac": $WARMUP, "weight_decay": $WD, "max_grad_norm": $GRAD_NORM, "batch_size": $BS, "bottleneck_dim": $DIM, "extra_args": "${EXTRA_ARGS[*]:-}"},
  "val_loss": null,
  "val_top1": null,
  "val_top5": null,
  "wall_seconds": $WALL_SECS
}
EOF
    echo "Trial $TRIAL FAILED — no val_loss found (${WALL_SECS}s)"
else
    # Find the epoch line matching best val_loss and extract metrics
    BEST_LINE=$(grep "val_loss=$BEST" "$LOG_FILE" | head -1)
    TOP1=$(echo "$BEST_LINE" | grep -oP 'val_top1=\K[0-9.]+' || echo "")
    TOP5=$(echo "$BEST_LINE" | grep -oP 'val_top5=\K[0-9.]+' || echo "")
    BEST_EPOCH=$(echo "$BEST_LINE" | grep -oP 'Epoch\s+\K[0-9]+' || echo "")

    # Training dynamics
    STOP_EPOCH=$(grep -oP 'Early stopping at epoch \K[0-9]+' "$LOG_FILE" || echo "")
    if [ -z "$STOP_EPOCH" ]; then
        STOP_EPOCH=$(grep -oP 'Epoch\s+\K[0-9]+' "$LOG_FILE" | tail -1 || echo "")
    fi

    # Final train loss (last epoch line)
    LAST_EPOCH_LINE=$(grep "Epoch" "$LOG_FILE" | tail -1)
    TRAIN_LOSS=$(echo "$LAST_EPOCH_LINE" | grep -oP 'train_loss=\K[0-9.]+' || echo "")
    TRAIN_TOP1=$(echo "$LAST_EPOCH_LINE" | grep -oP 'train_top1=\K[0-9.]+' || echo "")

    # Overfit gap
    if [ -n "$TRAIN_LOSS" ] && [ -n "$BEST" ]; then
        OVERFIT_GAP=$(python3 -c "print(round($BEST - ${TRAIN_LOSS}, 4))" 2>/dev/null || echo "")
    else
        OVERFIT_GAP=""
    fi

    # Step time (median from logged steps, skip first 100 for compile warmup)
    STEP_TIME=$(grep -oP '^\s+step\s+\d+.*\|\s+\K[0-9.]+(?=s)' "$LOG_FILE" 2>/dev/null | tail -20 | python3 -c "
import sys
vals = [float(l) for l in sys.stdin if l.strip()]
print(round(sorted(vals)[len(vals)//2], 3)) if vals else print('')
" 2>/dev/null || echo "")

    # Run directory
    RUN_DIR=$(grep -oP 'Checkpoints saved to \K.*' "$LOG_FILE" || echo "")

    cat > "$RESULT_FILE" <<EOF
{
  "trial": $TRIAL,
  "status": "complete",
  "params": {"lr": $LR, "warmup_frac": $WARMUP, "weight_decay": $WD, "max_grad_norm": $GRAD_NORM, "batch_size": $BS, "bottleneck_dim": $DIM, "extra_args": "${EXTRA_ARGS[*]:-}"},
  "val_loss": $BEST,
  "val_top1": ${TOP1:-null},
  "val_top5": ${TOP5:-null},
  "best_epoch": ${BEST_EPOCH:-null},
  "stop_epoch": ${STOP_EPOCH:-null},
  "train_loss": ${TRAIN_LOSS:-null},
  "train_top1": ${TRAIN_TOP1:-null},
  "overfit_gap": ${OVERFIT_GAP:-null},
  "wall_seconds": $WALL_SECS,
  "step_time_median": ${STEP_TIME:-null},
  "run_dir": "${RUN_DIR:-}"
}
EOF
    echo "Trial $TRIAL DONE — val_loss=$BEST top1=${TOP1:-?}% top5=${TOP5:-?}% (epoch ${BEST_EPOCH:-?}/${STOP_EPOCH:-?}, overfit_gap=${OVERFIT_GAP:-?}, ${WALL_SECS}s)"
fi
