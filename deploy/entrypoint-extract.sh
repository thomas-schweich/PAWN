#!/bin/bash
# Entrypoint for Lichess extraction container.
# Downloads, filters by Elo, writes Parquet, and uploads to HuggingFace.
#
# Required env vars:
#   HF_TOKEN   — HuggingFace write token
#   HF_REPO    — Target dataset repo (e.g. thomas-schweich/lichess-1800-1900)
#
# Optional env vars:
#   MONTHS     — Comma-separated year-months (default: "2025-01,2025-02")
#   MIN_ELO    — Minimum Elo for both players (default: 1800)
#   MAX_ELO    — Maximum Elo for both players (default: 1900)
set -euo pipefail

MONTHS="${MONTHS:-2025-01,2025-02}"
MIN_ELO="${MIN_ELO:-1800}"
MAX_ELO="${MAX_ELO:-1900}"
OUTPUT_DIR="/dev/shm/lichess"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is required"
    exit 1
fi
if [ -z "${HF_REPO:-}" ]; then
    echo "ERROR: HF_REPO is required"
    exit 1
fi

# Persist HF token
mkdir -p /root/.cache/huggingface
echo -n "$HF_TOKEN" > /root/.cache/huggingface/token

mkdir -p "$OUTPUT_DIR"

echo "=== Lichess Extraction ==="
echo "  Months: $MONTHS"
echo "  Elo range: $MIN_ELO - $MAX_ELO"
echo "  HF repo: $HF_REPO"
echo ""

# Launch one extraction per month in parallel
IFS=',' read -ra MONTH_LIST <<< "$MONTHS"
pids=()
for month in "${MONTH_LIST[@]}"; do
    month=$(echo "$month" | xargs)  # trim whitespace
    bash /opt/pawn/scripts/extract_lichess.sh "$month" "$MIN_ELO" "$MAX_ELO" "$OUTPUT_DIR" &
    pids+=($!)
done

# Wait for all extractions
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Extraction process $pid failed"
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "One or more extractions failed. Aborting."
    exit 1
fi

echo ""
echo "=== Extraction Complete ==="
ls -lh "$OUTPUT_DIR"/*.parquet

# Merge per-month Parquet files and upload
echo ""
echo "=== Merging and Uploading ==="
python3 -c "
import os
import glob
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

output_dir = '$OUTPUT_DIR'
hf_repo = '$HF_REPO'
min_elo = $MIN_ELO
max_elo = $MAX_ELO
months = '$MONTHS'

api = HfApi()

# Create dataset repo
try:
    create_repo(hf_repo, repo_type='dataset', exist_ok=True)
except Exception as e:
    print(f'Repo creation note: {e}')

# Find per-month files
parquet_files = sorted(glob.glob(os.path.join(output_dir, '*.parquet')))
print(f'Found {len(parquet_files)} parquet files')

# Merge into single file
merged_path = os.path.join(output_dir, f'lichess_elo{min_elo}_{max_elo}.parquet')
tables = [pq.read_table(f) for f in parquet_files]
import pyarrow as pa
merged = pa.concat_tables(tables)
pq.write_parquet(merged, merged_path, compression='zstd')

n_games = len(merged)
size_mb = os.path.getsize(merged_path) / 1024 / 1024
print(f'Merged: {n_games:,} games, {size_mb:.1f} MB')

# Upload merged file
api.upload_file(
    path_or_fileobj=merged_path,
    path_in_repo=f'data/lichess_elo{min_elo}_{max_elo}.parquet',
    repo_id=hf_repo,
    repo_type='dataset',
)
print(f'Uploaded data/lichess_elo{min_elo}_{max_elo}.parquet')

# Upload per-month files
for f in parquet_files:
    name = os.path.basename(f)
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f'data/monthly/{name}',
        repo_id=hf_repo,
        repo_type='dataset',
    )
    print(f'Uploaded data/monthly/{name}')

# Per-column stats for README
elo_mean = (merged.column('white_elo').to_pylist()
            + merged.column('black_elo').to_pylist())
avg_elo = sum(elo_mean) / len(elo_mean)
results = merged.column('result').to_pylist()
white_wins = results.count('1-0')
black_wins = results.count('0-1')
draws = results.count('1/2-1/2')

# Write README
readme = f'''---
license: cc0-1.0
task_categories:
  - other
tags:
  - chess
  - lichess
size_categories:
  - 1M<n<10M
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/lichess_elo{min_elo}_{max_elo}.parquet
---

# Lichess {min_elo}-{max_elo} Elo Games

Filtered subset of the [Lichess open database](https://database.lichess.org/) where
**both** players are rated {min_elo}-{max_elo}.

## Usage

```python
from datasets import load_dataset

ds = load_dataset(\"{hf_repo}\")
game = ds[\"train\"][0]
print(game[\"pgn\"])        # move text
print(game[\"white_elo\"])  # 1847
print(game[\"headers\"])    # full PGN headers (for parser compatibility)
```

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `pgn` | string | Move text (SAN) |
| `headers` | string | Full PGN header block |
| `white_elo` | int16 | White player Elo |
| `black_elo` | int16 | Black player Elo |
| `result` | string | Game result (1-0, 0-1, 1/2-1/2) |
| `time_control` | string | Time control (e.g. 600+0) |
| `opening` | string | Opening name |
| `date` | string | Game date (YYYY.MM.DD) |
| `month` | string | Source month (YYYY-MM) |

## Stats

- **Total games:** {n_games:,}
- **Average Elo:** {avg_elo:.0f}
- **White wins:** {white_wins:,} ({100*white_wins/n_games:.1f}%)
- **Black wins:** {black_wins:,} ({100*black_wins/n_games:.1f}%)
- **Draws:** {draws:,} ({100*draws/n_games:.1f}%)
- **Source months:** {months}

## License

[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), consistent with the
[Lichess database license](https://database.lichess.org/).

## Attribution

Data sourced from the [Lichess open database](https://database.lichess.org/).
Lichess is a free, open-source chess server: [lichess.org](https://lichess.org).
'''

api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo='README.md',
    repo_id=hf_repo,
    repo_type='dataset',
)
print('Uploaded README.md')
print()
print(f'Dataset: https://huggingface.co/datasets/{hf_repo}')
"

echo ""
echo "=== Done ==="
