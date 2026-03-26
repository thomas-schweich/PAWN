#!/bin/bash
# Entrypoint for Lc0 data generation container.
# Generates self-play games, writes Parquet to /dev/shm, uploads to HuggingFace.
#
# Required env vars:
#   HF_TOKEN   — HuggingFace write token
#   HF_REPO    — Target dataset repo (e.g. thomas-schweich/lc0-nodes1)
#
# Optional env vars:
#   GAMES      — Number of games to generate (default: 1000000)
#   WORKERS    — Parallel engines (default: 1, GPU contention with >1)
#   NODES      — Nodes per move (default: 1)
#   TIER       — Tier name override (default: nodes_0001)
set -euo pipefail

GAMES="${GAMES:-1000000}"
WORKERS="${WORKERS:-1}"
NODES="${NODES:-1}"
TIER="${TIER:-nodes_0001}"
OUTPUT_DIR="/dev/shm/lc0"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is required"
    exit 1
fi
if [ -z "${HF_REPO:-}" ]; then
    echo "ERROR: HF_REPO is required"
    exit 1
fi

mkdir -p /root/.cache/huggingface
echo -n "$HF_TOKEN" > /root/.cache/huggingface/token
mkdir -p "$OUTPUT_DIR"

echo "=== Lc0 Data Generation ==="
echo "  Games: $GAMES"
echo "  Workers: $WORKERS"
echo "  Nodes: $NODES"
echo "  HF repo: $HF_REPO"
echo ""

# Verify GPU + Lc0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Checking libraries ==="
ldd /usr/local/bin/lc0 | grep -E "not found|cuda|cublas|cudnn" || echo "All libs OK"
echo ""

echo "=== Verifying Lc0 ==="
/usr/local/bin/lc0-wrap --help 2>&1 | head -3 || echo "lc0-wrap --help failed: $?"
echo ""

echo "=== Lc0 UCI test (30s timeout) ==="
timeout 30 bash -c 'echo -e "uci\nquit" | /usr/local/bin/lc0-wrap 2>&1'
echo "Exit: $?"
echo ""

# Generate via Rust engine + write Parquet in Python
python3 -c "
import chess_engine
import pyarrow as pa
import pyarrow.parquet as pq
import time, os

t0 = time.time()
result = chess_engine.generate_engine_games(
    engine_path='/usr/local/bin/lc0-wrap',
    nodes=$NODES,
    total_games=$GAMES,
    n_workers=$WORKERS,
    base_seed=60000,
    temperature=1.0,
    multi_pv=5,
    sample_plies=999,
    hash_mb=16,
    max_ply=500,
)
dt = time.time() - t0
n = len(result['uci'])
print(f'Generated {n:,} games in {dt/60:.1f}m ({n/dt:.1f} games/s)')

schema = pa.schema([
    ('uci', pa.string()),
    ('result', pa.string()),
    ('n_ply', pa.int16()),
    ('nodes', pa.int32()),
    ('worker_id', pa.int16()),
    ('seed', pa.int32()),
])
table = pa.table({
    'uci': result['uci'],
    'result': result['result'],
    'n_ply': result['n_ply'],
    'nodes': result['nodes'],
    'worker_id': result['worker_id'],
    'seed': [int(s) for s in result['seed']],
}, schema=schema)

out = '$OUTPUT_DIR/lc0_nodes${NODES}.parquet'
pq.write_table(table, out, compression='zstd')
size = os.path.getsize(out) / 1e6
print(f'Wrote {out} ({size:.1f} MB, {len(table):,} rows)')
"

echo ""
echo "=== Generation Complete ==="
ls -lh "$OUTPUT_DIR"/*.parquet

# Upload to HuggingFace
echo ""
echo "=== Uploading to HuggingFace ==="
python3 -c "
import os
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

output_dir = '$OUTPUT_DIR'
hf_repo = '$HF_REPO'
games = $GAMES
nodes = $NODES

api = HfApi()
try:
    create_repo(hf_repo, repo_type='dataset', exist_ok=True)
except Exception as e:
    print(f'Repo creation note: {e}')

# Find and upload parquet file
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.parquet'):
        path = os.path.join(output_dir, f)
        table = pq.read_table(path)
        n_games = len(table)
        size_mb = os.path.getsize(path) / 1024 / 1024

        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f'data/{f}',
            repo_id=hf_repo,
            repo_type='dataset',
        )
        print(f'Uploaded data/{f} ({n_games:,} games, {size_mb:.1f} MB)')

# Stats for README
results = table.column('result').to_pylist()
plies = table.column('n_ply').to_pylist()
white_wins = results.count('1-0')
black_wins = results.count('0-1')
draws = results.count('1/2-1/2')
avg_ply = sum(plies) / len(plies)

readme = f'''---
license: mit
task_categories:
  - other
tags:
  - chess
  - lc0
  - leela
  - self-play
  - uci
size_categories:
  - 1M<n<10M
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/*.parquet
---

# Lc0 Self-Play (nodes={nodes})

{n_games:,} games of Lc0 (Leela Chess Zero) self-play at {nodes} node(s) per move — raw policy network output with no MCTS search. The policy head is trained via millions of games of MCTS self-play, so even at nodes=1 it encodes deep positional knowledge distilled from search.

## Usage

```python
from datasets import load_dataset

ds = load_dataset(\"{hf_repo}\")
game = ds[\"train\"][0]
print(game[\"uci\"])       # \"e2e4 e7e5 g1f3 b8c6 ...\"
print(game[\"result\"])    # \"1-0\"
print(game[\"n_ply\"])     # 87
```

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `uci` | string | Space-separated UCI moves |
| `result` | string | Game result (`1-0`, `0-1`, `1/2-1/2`, `*`) |
| `n_ply` | int16 | Number of half-moves |
| `nodes` | int32 | Node budget per move (always {nodes}) |
| `worker_id` | int16 | Worker index for reproducibility |
| `seed` | int32 | Worker RNG seed for reproducibility |

## Stats

- **Total games:** {n_games:,}
- **Average ply:** {avg_ply:.0f}
- **White wins:** {white_wins:,} ({100*white_wins/n_games:.1f}%)
- **Black wins:** {black_wins:,} ({100*black_wins/n_games:.1f}%)
- **Draws:** {draws:,} ({100*draws/n_games:.1f}%)

## Generation

Games generated with [Lc0 v0.31.2](https://lczero.org/) using the T2 768x15x24h network (swa-7464000).
MultiPV=5 with softmax temperature sampling for the first 15 plies, then top-1.

At nodes=1, Lc0 uses only its policy network — no Monte Carlo Tree Search. The policy
network is a transformer trained on MCTS self-play data, so it captures strategic patterns
that require no explicit search to express.

## License

MIT. Synthetic self-play data generated from open-source engines and networks.
'''

api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo='README.md',
    repo_id=hf_repo,
    repo_type='dataset',
)
print('Uploaded README.md')
print(f'Dataset: https://huggingface.co/datasets/{hf_repo}')
"

echo ""
echo "=== Done ==="
