//! One parquet shard per worker per chunk. Atomic on disk: written to
//! `shard-w<NNN>-c<NNNN>.parquet.tmp` and renamed on close to
//! `shard-w<NNN>-c<NNNN>-r<NNNNNN>.parquet`, where the trailing `r` field
//! is the row count. Encoding the row count in the filename lets resume
//! determine `(worker, chunk, n_rows)` from a directory listing alone —
//! no parquet metadata reads required. That in turn lets a remote sync
//! tool (e.g. an HF dataset uploader) drop zero-byte placeholder files
//! locally that the resume code can read just like real shards, so we
//! can resume on a fresh pod without re-downloading any actual data.
//!
//! A crash mid-shard leaves a `.tmp` orphan rather than a half-valid
//! `.parquet`. Resume scans only `.parquet` files.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use arrow::array::{
    ArrayBuilder, ArrayRef, Float32Builder, Int16Builder, Int32Builder, ListBuilder,
    RecordBatch, StringBuilder, StructBuilder, UInt16Builder, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use once_cell::sync::Lazy;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

/// One candidate move's distillation payload. Stored packed as
/// `Struct{move_idx: i16, score_cp: i16, score_v: i16?}` to keep parquet
/// rows compact while remaining trivially loadable from polars / pyarrow.
///
/// - `move_idx`: searchless_chess action vocab index (0..1968).
/// - `score_cp`: normalized centipawns, mover-POV. 100 cp ≈ "1 pawn equivalent".
/// - `score_v`: raw internal NNUE Value, mover-POV. `None` when the tier
///   sourced its candidates from `info ... multipv` (only the patched
///   binary's `evallegal` command surfaces raw values). Distillation
///   targets should use this when present — it's what the network actually
///   produces, before win-rate normalization shrinks magnitudes by `a/100`
///   ≈ 2–3.5×.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegalMoveEval {
    pub move_idx: i16,
    pub score_cp: i16,
    pub score_v: Option<i16>,
}

/// Fields of the `legal_move_evals` Struct element. Hoisted so the
/// schema and the StructBuilder agree on order/nullability.
fn legal_move_eval_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("move_idx", DataType::Int16, false),
        Field::new("score_cp", DataType::Int16, false),
        // Nullable: only the `evallegal` protocol provides this. Multipv-derived
        // candidates leave it null.
        Field::new("score_v", DataType::Int16, true),
    ])
}

fn legal_move_eval_struct_builders() -> Vec<Box<dyn ArrayBuilder>> {
    vec![
        Box::new(Int16Builder::new()),
        Box::new(Int16Builder::new()),
        Box::new(Int16Builder::new()),
    ]
}

/// One row to be written. Constructed by the worker after a game finishes.
#[derive(Debug, Clone)]
pub struct GameRow {
    pub tokens: Vec<i16>,
    pub san: Vec<String>,
    pub uci: Vec<String>,
    pub game_length: u16,
    pub outcome_token: u16,
    pub result: String,
    /// Search-mode-only metadata. `None` for searchless tiers — those
    /// fields don't apply to evallegal-driven generation, so we persist
    /// SQL-style nulls rather than misleading sentinel values.
    pub nodes: Option<i32>,
    pub multi_pv: Option<i32>,
    pub opening_multi_pv: Option<i32>,
    pub opening_plies: Option<i32>,
    pub sample_plies: Option<i32>,
    pub temperature: f32,
    pub worker_id: i16,
    pub game_seed: u64,
    pub stockfish_version: String,
    /// Per-ply per-legal-move NNUE eval payload. `None` when the tier did
    /// not request distillation data (non-storing tiers leave the parquet
    /// column as an empty list, paying only a few bytes of offsets per
    /// row). `Some(Vec)` outer length equals `game_length`; inner lengths
    /// are the number of legal moves at each ply (capped by the tier's
    /// MultiPV). See `TierConfig::store_legal_move_evals`.
    pub legal_move_evals: Option<Vec<Vec<LegalMoveEval>>>,
}

pub static SCHEMA: Lazy<SchemaRef> = Lazy::new(|| {
    // Inner fields nullable: arrow's default builders produce nullable
    // inner types, and downstream readers (polars, pyarrow) accept either
    // way. The outer List itself is non-null because every game has tokens.
    let list_i16 = DataType::List(Arc::new(Field::new("item", DataType::Int16, true)));
    let list_utf8 = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));

    // legal_move_evals: List<List<Struct{move_idx: i16, score_cp: i16}>>.
    // Outer list = per-ply; inner list = per-legal-move at that ply.
    // Always present in the schema for forward compat. Tiers that don't
    // request distillation data leave it as an empty outer list per row
    // (~8 bytes of offsets, negligible vs the rest of the row).
    let eval_struct_dt = DataType::Struct(legal_move_eval_struct_fields());
    let eval_inner_list_dt =
        DataType::List(Arc::new(Field::new("item", eval_struct_dt, true)));
    let eval_outer_list_dt =
        DataType::List(Arc::new(Field::new("item", eval_inner_list_dt, true)));

    Arc::new(Schema::new(vec![
        Field::new("tokens", list_i16, false),
        Field::new("san", list_utf8.clone(), false),
        Field::new("uci", list_utf8, false),
        Field::new("game_length", DataType::UInt16, false),
        Field::new("outcome_token", DataType::UInt16, false),
        Field::new("result", DataType::Utf8, false),
        // Search-mode-only metadata; nullable so searchless rows write
        // SQL-style nulls rather than carrying misleading values.
        Field::new("nodes", DataType::Int32, true),
        Field::new("multi_pv", DataType::Int32, true),
        Field::new("opening_multi_pv", DataType::Int32, true),
        Field::new("opening_plies", DataType::Int32, true),
        Field::new("sample_plies", DataType::Int32, true),
        Field::new("temperature", DataType::Float32, false),
        Field::new("worker_id", DataType::Int16, false),
        // game_seed is the per-game RNG seed, derived via splitmix64 — a
        // full u64. Stored as `UInt64` (Arrow + Parquet support it
        // natively) so any reader sees the actual value without
        // bit-pattern reinterpretation. Polars / pyarrow / pandas all
        // handle this correctly.
        Field::new("game_seed", DataType::UInt64, false),
        Field::new("stockfish_version", DataType::Utf8, false),
        Field::new("legal_move_evals", eval_outer_list_dt, false),
    ]))
});

/// Buffer rows in column builders, write once at close. Memory cap is the
/// shard size (~10k games × ~150 plies × few bytes/move = ~15MB), so we
/// don't need streaming writes within a shard.
pub struct ShardWriter {
    tier_dir: PathBuf,
    shard_worker_id: u32,
    shard_chunk_idx: u32,
    tmp_path: PathBuf,
    tokens: ListBuilder<Int16Builder>,
    san: ListBuilder<StringBuilder>,
    uci: ListBuilder<StringBuilder>,
    game_length: UInt16Builder,
    outcome_token: UInt16Builder,
    result: StringBuilder,
    nodes: Int32Builder,
    multi_pv: Int32Builder,
    opening_multi_pv: Int32Builder,
    opening_plies: Int32Builder,
    sample_plies: Int32Builder,
    temperature: Float32Builder,
    worker_id: Int16Builder,
    game_seed: UInt64Builder,
    stockfish_version: StringBuilder,
    legal_move_evals: ListBuilder<ListBuilder<StructBuilder>>,
    n_rows: usize,
}

impl ShardWriter {
    /// Open a writer for `(worker_id, chunk_idx)` under `tier_dir`. The
    /// final filename — which includes the row count — is determined at
    /// `close()` time, so writes go to a row-count-free `.parquet.tmp`
    /// path and the rename happens at the end.
    pub fn create(tier_dir: PathBuf, worker_id: u32, chunk_idx: u32) -> anyhow::Result<Self> {
        fs::create_dir_all(&tier_dir)
            .with_context(|| format!("creating shard dir {}", tier_dir.display()))?;
        let tmp_path = shard_tmp_path(&tier_dir, worker_id, chunk_idx);
        // Best effort: clean up any leftover .tmp from a prior crash before
        // we start writing this shard.
        let _ = fs::remove_file(&tmp_path);

        Ok(Self {
            tier_dir,
            shard_worker_id: worker_id,
            shard_chunk_idx: chunk_idx,
            tmp_path,
            tokens: ListBuilder::new(Int16Builder::new()),
            san: ListBuilder::new(StringBuilder::new()),
            uci: ListBuilder::new(StringBuilder::new()),
            game_length: UInt16Builder::new(),
            outcome_token: UInt16Builder::new(),
            result: StringBuilder::new(),
            nodes: Int32Builder::new(),
            multi_pv: Int32Builder::new(),
            opening_multi_pv: Int32Builder::new(),
            opening_plies: Int32Builder::new(),
            sample_plies: Int32Builder::new(),
            temperature: Float32Builder::new(),
            worker_id: Int16Builder::new(),
            game_seed: UInt64Builder::new(),
            stockfish_version: StringBuilder::new(),
            legal_move_evals: ListBuilder::new(ListBuilder::new(StructBuilder::new(
                legal_move_eval_struct_fields(),
                legal_move_eval_struct_builders(),
            ))),
            n_rows: 0,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn append(&mut self, row: &GameRow) {
        for &t in &row.tokens {
            self.tokens.values().append_value(t);
        }
        self.tokens.append(true);

        for s in &row.san {
            self.san.values().append_value(s);
        }
        self.san.append(true);

        for s in &row.uci {
            self.uci.values().append_value(s);
        }
        self.uci.append(true);

        self.game_length.append_value(row.game_length);
        self.outcome_token.append_value(row.outcome_token);
        self.result.append_value(&row.result);
        self.nodes.append_option(row.nodes);
        self.multi_pv.append_option(row.multi_pv);
        self.opening_multi_pv.append_option(row.opening_multi_pv);
        self.opening_plies.append_option(row.opening_plies);
        self.sample_plies.append_option(row.sample_plies);
        self.temperature.append_value(row.temperature);
        self.worker_id.append_value(row.worker_id);
        self.game_seed.append_value(row.game_seed);
        self.stockfish_version.append_value(&row.stockfish_version);

        // legal_move_evals: outer list = per-ply, inner list = per-move.
        // Tiers that don't store distillation data leave the outer list
        // empty (still non-null at row level — schema declares the column
        // non-null and downstream readers find an empty `[]` rather than
        // having to handle row-level nulls).
        if let Some(plies) = &row.legal_move_evals {
            let inner_list = self.legal_move_evals.values();
            for ply_evals in plies {
                let struct_b = inner_list.values();
                for ev in ply_evals {
                    struct_b
                        .field_builder::<Int16Builder>(0)
                        .expect("move_idx field 0")
                        .append_value(ev.move_idx);
                    struct_b
                        .field_builder::<Int16Builder>(1)
                        .expect("score_cp field 1")
                        .append_value(ev.score_cp);
                    let v_b = struct_b
                        .field_builder::<Int16Builder>(2)
                        .expect("score_v field 2");
                    match ev.score_v {
                        Some(v) => v_b.append_value(v),
                        None => v_b.append_null(),
                    }
                    struct_b.append(true);
                }
                inner_list.append(true);
            }
        }
        self.legal_move_evals.append(true);

        self.n_rows += 1;
    }

    /// Materialize the buffered rows, write the parquet, and atomically
    /// rename `.tmp` → final path (which includes the row count). Consumes self.
    pub fn close(mut self) -> anyhow::Result<PathBuf> {
        if self.n_rows == 0 {
            // Nothing to write — clean up the (never-created) tmp and return.
            let _ = fs::remove_file(&self.tmp_path);
            anyhow::bail!(
                "ShardWriter::close called with zero rows for w{:03}-c{:04} in {}",
                self.shard_worker_id, self.shard_chunk_idx, self.tier_dir.display(),
            );
        }
        let final_path = shard_final_path(
            &self.tier_dir, self.shard_worker_id, self.shard_chunk_idx, self.n_rows as u64,
        );

        let columns: Vec<ArrayRef> = vec![
            Arc::new(self.tokens.finish()),
            Arc::new(self.san.finish()),
            Arc::new(self.uci.finish()),
            Arc::new(self.game_length.finish()),
            Arc::new(self.outcome_token.finish()),
            Arc::new(self.result.finish()),
            Arc::new(self.nodes.finish()),
            Arc::new(self.multi_pv.finish()),
            Arc::new(self.opening_multi_pv.finish()),
            Arc::new(self.opening_plies.finish()),
            Arc::new(self.sample_plies.finish()),
            Arc::new(self.temperature.finish()),
            Arc::new(self.worker_id.finish()),
            Arc::new(self.game_seed.finish()),
            Arc::new(self.stockfish_version.finish()),
            Arc::new(self.legal_move_evals.finish()),
        ];
        let batch = RecordBatch::try_new(SCHEMA.clone(), columns)
            .context("building record batch")?;

        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();

        {
            let file = fs::File::create(&self.tmp_path)
                .with_context(|| format!("creating {}", self.tmp_path.display()))?;
            let mut writer = ArrowWriter::try_new(file, SCHEMA.clone(), Some(props))
                .context("opening parquet writer")?;
            writer.write(&batch).context("writing record batch")?;
            // ArrowWriter::close() returns the wrapped File via into_inner.
            // We need access to it to issue an fsync before the rename so
            // the data is durable across power failures (otherwise the
            // rename can succeed while the file's contents are still in
            // the page cache, leaving a corrupt parquet on disk).
            let file = writer.into_inner().context("flushing parquet writer")?;
            file.sync_all().context("fsyncing shard")?;
        }

        fs::rename(&self.tmp_path, &final_path)
            .with_context(|| format!("renaming {} -> {}", self.tmp_path.display(), final_path.display()))?;
        Ok(final_path)
    }
}

/// In-progress path: `<tier_dir>/shard-w<NNN>-c<NNNN>.parquet.tmp`.
/// The row count is unknown until close, so the tmp filename omits it.
pub fn shard_tmp_path(tier_dir: &Path, worker_id: u32, chunk_idx: u32) -> PathBuf {
    tier_dir.join(format!("shard-w{worker_id:03}-c{chunk_idx:04}.parquet.tmp"))
}

/// Final, post-rename path: `<tier_dir>/shard-w<NNN>-c<NNNN>-r<NNNNNN>.parquet`.
/// Encoding the row count in the filename lets resume — and remote-sync
/// placeholder files — recover `(worker, chunk, n_rows)` from a directory
/// listing alone. `n_rows` is zero-padded to a *minimum* of 6 digits;
/// `shard_size_games > 999_999` simply produces a wider field, which
/// `parse_shard_filename` and the Python regex `\d{6,}` both accept.
pub fn shard_final_path(tier_dir: &Path, worker_id: u32, chunk_idx: u32, n_rows: u64) -> PathBuf {
    tier_dir.join(format!("shard-w{worker_id:03}-c{chunk_idx:04}-r{n_rows:06}.parquet"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;
    use tempfile::tempdir;

    fn fake_row(seed: u64) -> GameRow {
        GameRow {
            tokens: vec![100, 200, 300],
            san: vec!["e4".into(), "e5".into(), "Nf3".into()],
            uci: vec!["e2e4".into(), "e7e5".into(), "g1f3".into()],
            game_length: 3,
            outcome_token: 1969,
            result: "1-0".into(),
            nodes: Some(1),
            multi_pv: Some(5),
            opening_multi_pv: Some(20),
            opening_plies: Some(2),
            sample_plies: Some(12),
            temperature: 1.0,
            worker_id: 0,
            game_seed: seed,
            stockfish_version: "Stockfish 18 by ...".into(),
            legal_move_evals: None,
        }
    }

    #[test]
    fn shard_paths_are_zero_padded() {
        let tmp = shard_tmp_path(Path::new("/tmp/x"), 3, 17);
        assert_eq!(tmp, PathBuf::from("/tmp/x/shard-w003-c0017.parquet.tmp"));
        let final_p = shard_final_path(Path::new("/tmp/x"), 3, 17, 834);
        assert_eq!(final_p, PathBuf::from("/tmp/x/shard-w003-c0017-r000834.parquet"));
    }

    #[test]
    fn write_then_read_round_trips() {
        let dir = tempdir().unwrap();
        let mut w = ShardWriter::create(dir.path().to_path_buf(), 0, 0).unwrap();
        for i in 0..5 {
            w.append(&fake_row(1000 + i));
        }
        let written = w.close().unwrap();
        assert_eq!(written, shard_final_path(dir.path(), 0, 0, 5));

        // Re-read via parquet's own reader and verify shape + a couple of cells.
        let file = File::open(&written).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), SCHEMA.fields().len());

        let game_seed_col = batch
            .column_by_name("game_seed")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        assert_eq!(game_seed_col.value(0), 1000);
        assert_eq!(game_seed_col.value(4), 1004);
    }

    #[test]
    fn legal_move_evals_round_trip() {
        use arrow::array::{Array, Int16Array, ListArray, StructArray};
        let dir = tempdir().unwrap();
        let mut w = ShardWriter::create(dir.path().to_path_buf(), 0, 0).unwrap();

        // Game 0: distillation tier — populated. Two plies, varying legal-move counts.
        // First ply mixes Some / None score_v to exercise the nullable column path.
        let mut row0 = fake_row(1);
        row0.legal_move_evals = Some(vec![
            vec![
                LegalMoveEval { move_idx: 100, score_cp: 50, score_v: Some(125) },
                LegalMoveEval { move_idx: 200, score_cp: -25, score_v: None },
            ],
            vec![LegalMoveEval { move_idx: 300, score_cp: 10, score_v: Some(28) }],
        ]);
        w.append(&row0);

        // Game 1: non-storing tier — None / empty outer list. Same shard
        // can mix both (in practice a tier is one mode or the other, but
        // the schema must support either per row).
        w.append(&fake_row(2));

        let path = w.close().unwrap();
        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        let batch = &batches[0];
        let outer = batch
            .column_by_name("legal_move_evals")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        // Row 0: two plies.
        let plies0 = outer.value(0);
        let plies0 = plies0.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(plies0.len(), 2);

        let ply0_0 = plies0.value(0);
        let ply0_0 = ply0_0.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(ply0_0.len(), 2);
        let move_idx = ply0_0.column(0).as_any().downcast_ref::<Int16Array>().unwrap();
        let score_cp = ply0_0.column(1).as_any().downcast_ref::<Int16Array>().unwrap();
        let score_v = ply0_0.column(2).as_any().downcast_ref::<Int16Array>().unwrap();
        assert_eq!(move_idx.value(0), 100);
        assert_eq!(score_cp.value(0), 50);
        assert_eq!(score_v.value(0), 125);
        assert!(score_v.is_valid(0));
        assert_eq!(move_idx.value(1), 200);
        assert_eq!(score_cp.value(1), -25);
        assert!(!score_v.is_valid(1)); // None → null in parquet

        // Row 1: empty outer list (no per-ply data captured).
        let plies1 = outer.value(1);
        let plies1 = plies1.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(plies1.len(), 0);
    }

    /// Regression test for the round-7 `game_seed: i64 -> UInt64` change.
    /// A high-bit seed (>= 2^63) MUST round-trip as itself, not flip to a
    /// negative integer. With the prior Int64 schema this would store a
    /// negative value and require explicit bit-reinterpretation in every
    /// reader.
    #[test]
    fn high_bit_game_seed_round_trips() {
        let dir = tempdir().unwrap();
        let mut w = ShardWriter::create(dir.path().to_path_buf(), 0, 0).unwrap();
        let big = 0xFFFF_FFFF_FFFF_FFFFu64; // u64::MAX — sign bit set
        w.append(&fake_row(big));
        let path = w.close().unwrap();

        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        let col = batches[0]
            .column_by_name("game_seed")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        assert_eq!(col.value(0), big, "high-bit seed must round-trip as u64::MAX");
    }

    #[test]
    fn close_with_no_rows_errors() {
        let dir = tempdir().unwrap();
        let w = ShardWriter::create(dir.path().to_path_buf(), 0, 0).unwrap();
        assert!(w.close().is_err());
    }

    #[test]
    fn tmp_orphan_is_cleaned_up_on_create() {
        let dir = tempdir().unwrap();
        let tmp = shard_tmp_path(dir.path(), 0, 0);
        // Plant a fake .tmp orphan.
        std::fs::write(&tmp, b"garbage").unwrap();
        assert!(tmp.exists());
        let mut w = ShardWriter::create(dir.path().to_path_buf(), 0, 0).unwrap();
        assert!(!tmp.exists(), "create() should have cleaned up the orphan tmp");
        w.append(&fake_row(0));
        let final_path = w.close().unwrap();
        assert!(final_path.exists());
        assert_eq!(final_path, shard_final_path(dir.path(), 0, 0, 1));
    }
}
