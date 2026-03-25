#!/usr/bin/env bash
# Download and filter Lichess games by Elo range, output as Parquet.
# Streams zstd-compressed PGN, filters and parses headers, writes
# compressed Parquet incrementally (one row group per flush batch).
#
# Usage:
#   bash scripts/extract_lichess.sh <year-month> <min_elo> <max_elo> <output_dir>
#
# Example:
#   bash scripts/extract_lichess.sh 2025-01 1800 1900 /dev/shm/lichess
#
# Downloads from https://database.lichess.org/standard/
# Requires: zstd, python3, pyarrow
set -euo pipefail

YEAR_MONTH="${1:?Usage: $0 <year-month> <min_elo> <max_elo> <output_dir>}"
MIN_ELO="${2:?}"
MAX_ELO="${3:?}"
OUTPUT_DIR="${4:?}"

URL="https://database.lichess.org/standard/lichess_db_standard_rated_${YEAR_MONTH}.pgn.zst"
OUTPUT_FILE="${OUTPUT_DIR}/lichess_${YEAR_MONTH}_elo${MIN_ELO}_${MAX_ELO}.parquet"

mkdir -p "$OUTPUT_DIR"

echo "[${YEAR_MONTH}] Downloading and filtering: Elo ${MIN_ELO}-${MAX_ELO}"
echo "[${YEAR_MONTH}] Source: $URL"
echo "[${YEAR_MONTH}] Output: $OUTPUT_FILE"

# Stream: download -> decompress -> parse/filter/write Parquet
curl -sL "$URL" | zstd -d | python3 -c "
import sys
import pyarrow as pa
import pyarrow.parquet as pq

min_elo = ${MIN_ELO}
max_elo = ${MAX_ELO}
month = '${YEAR_MONTH}'
output = '${OUTPUT_FILE}'
FLUSH_EVERY = 50_000  # rows per row group

schema = pa.schema([
    ('pgn', pa.string()),
    ('headers', pa.string()),
    ('white_elo', pa.int16()),
    ('black_elo', pa.int16()),
    ('result', pa.string()),
    ('time_control', pa.string()),
    ('opening', pa.string()),
    ('date', pa.string()),
    ('month', pa.string()),
])

writer = pq.ParquetWriter(output, schema, compression='zstd')

# Accumulator for current batch
batch = {name: [] for name in schema.names}
n_scanned = 0
n_matched = 0

# Current game state
header_lines = []
move_lines = []
headers = {}
in_moves = False

def parse_header(line):
    # [Key \"Value\"] -> (key, value)
    try:
        key = line[1:line.index(' ')]
        val = line.split('\"')[1]
        return key, val
    except (IndexError, ValueError):
        return None, None

def flush_game():
    global n_matched
    white_elo = headers.get('WhiteElo')
    black_elo = headers.get('BlackElo')
    if white_elo is None or black_elo is None:
        return
    try:
        we = int(white_elo)
        be = int(black_elo)
    except ValueError:
        return
    if not (min_elo <= we <= max_elo and min_elo <= be <= max_elo):
        return

    moves = ' '.join(line.strip() for line in move_lines if line.strip())
    header_block = ''.join(header_lines)
    n_ply = moves.count('.')  # approximate

    batch['pgn'].append(moves)
    batch['headers'].append(header_block)
    batch['white_elo'].append(we)
    batch['black_elo'].append(be)
    batch['result'].append(headers.get('Result', ''))
    batch['time_control'].append(headers.get('TimeControl', ''))
    batch['opening'].append(headers.get('Opening', ''))
    batch['date'].append(headers.get('UTCDate', headers.get('Date', '')))
    batch['month'].append(month)
    n_matched += 1

def flush_batch():
    if not batch['pgn']:
        return
    table = pa.table(batch, schema=schema)
    writer.write_table(table)
    for k in batch:
        batch[k] = []

for line in sys.stdin:
    if line.startswith('[Event '):
        # New game — flush previous
        if header_lines or move_lines:
            flush_game()
            n_scanned += 1
            if n_scanned % 1_000_000 == 0:
                print(f'[{month}] Scanned {n_scanned:,} games, matched {n_matched:,}',
                      file=sys.stderr)
            if len(batch['pgn']) >= FLUSH_EVERY:
                flush_batch()
        header_lines = [line]
        move_lines = []
        headers = {}
        in_moves = False
        key, val = parse_header(line)
        if key:
            headers[key] = val
    elif line.startswith('['):
        header_lines.append(line)
        key, val = parse_header(line)
        if key:
            headers[key] = val
        in_moves = False
    elif line.strip() == '':
        if header_lines and not in_moves:
            in_moves = True
    else:
        in_moves = True
        move_lines.append(line)

# Flush final game and remaining batch
if header_lines or move_lines:
    flush_game()
    n_scanned += 1
flush_batch()
writer.close()

print(f'[{month}] Done: scanned {n_scanned:,} games, matched {n_matched:,}', file=sys.stderr)
"

file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
echo "[${YEAR_MONTH}] Output: ${file_size} (${OUTPUT_FILE})"
