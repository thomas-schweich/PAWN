//! One parquet shard per worker per chunk. Atomic on disk: written to
//! `<path>.tmp` and renamed on close, so a crash mid-shard leaves a
//! `.tmp` orphan rather than a half-valid `.parquet`. Resume scans only
//! `.parquet` files.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use arrow::array::{
    ArrayRef, Float32Builder, Int16Builder, Int32Builder, Int64Builder, ListBuilder,
    RecordBatch, StringBuilder, UInt16Builder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use once_cell::sync::Lazy;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

/// One row to be written. Constructed by the worker after a game finishes.
#[derive(Debug, Clone)]
pub struct GameRow {
    pub tokens: Vec<i16>,
    pub san: Vec<String>,
    pub uci: Vec<String>,
    pub game_length: u16,
    pub outcome_token: u16,
    pub result: String,
    pub nodes: i32,
    pub multi_pv: i32,
    pub opening_multi_pv: i32,
    pub opening_plies: i32,
    pub sample_plies: i32,
    pub temperature: f32,
    pub worker_id: i16,
    pub game_seed: i64,
    pub stockfish_version: String,
}

pub static SCHEMA: Lazy<SchemaRef> = Lazy::new(|| {
    // Inner fields nullable: arrow's default builders produce nullable
    // inner types, and downstream readers (polars, pyarrow) accept either
    // way. The outer List itself is non-null because every game has tokens.
    let list_i16 = DataType::List(Arc::new(Field::new("item", DataType::Int16, true)));
    let list_utf8 = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    Arc::new(Schema::new(vec![
        Field::new("tokens", list_i16, false),
        Field::new("san", list_utf8.clone(), false),
        Field::new("uci", list_utf8, false),
        Field::new("game_length", DataType::UInt16, false),
        Field::new("outcome_token", DataType::UInt16, false),
        Field::new("result", DataType::Utf8, false),
        Field::new("nodes", DataType::Int32, false),
        Field::new("multi_pv", DataType::Int32, false),
        Field::new("opening_multi_pv", DataType::Int32, false),
        Field::new("opening_plies", DataType::Int32, false),
        Field::new("sample_plies", DataType::Int32, false),
        Field::new("temperature", DataType::Float32, false),
        Field::new("worker_id", DataType::Int16, false),
        Field::new("game_seed", DataType::Int64, false),
        Field::new("stockfish_version", DataType::Utf8, false),
    ]))
});

/// Buffer rows in column builders, write once at close. Memory cap is the
/// shard size (~10k games × ~150 plies × few bytes/move = ~15MB), so we
/// don't need streaming writes within a shard.
pub struct ShardWriter {
    final_path: PathBuf,
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
    game_seed: Int64Builder,
    stockfish_version: StringBuilder,
    n_rows: usize,
}

impl ShardWriter {
    pub fn create(final_path: PathBuf) -> anyhow::Result<Self> {
        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating shard dir {}", parent.display()))?;
        }
        let tmp_path = final_path.with_extension("parquet.tmp");
        // Best effort: clean up any leftover .tmp from a prior crash before
        // we start writing this shard.
        let _ = fs::remove_file(&tmp_path);

        Ok(Self {
            final_path,
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
            game_seed: Int64Builder::new(),
            stockfish_version: StringBuilder::new(),
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
        self.nodes.append_value(row.nodes);
        self.multi_pv.append_value(row.multi_pv);
        self.opening_multi_pv.append_value(row.opening_multi_pv);
        self.opening_plies.append_value(row.opening_plies);
        self.sample_plies.append_value(row.sample_plies);
        self.temperature.append_value(row.temperature);
        self.worker_id.append_value(row.worker_id);
        self.game_seed.append_value(row.game_seed);
        self.stockfish_version.append_value(&row.stockfish_version);

        self.n_rows += 1;
    }

    /// Materialize the buffered rows, write the parquet, and atomically
    /// rename `.tmp` → final path. Consumes self.
    pub fn close(mut self) -> anyhow::Result<PathBuf> {
        if self.n_rows == 0 {
            // Nothing to write — clean up the (never-created) tmp and return.
            let _ = fs::remove_file(&self.tmp_path);
            anyhow::bail!("ShardWriter::close called with zero rows for {}", self.final_path.display());
        }

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
            writer.close().context("closing parquet writer")?;
        }

        fs::rename(&self.tmp_path, &self.final_path)
            .with_context(|| format!("renaming {} -> {}", self.tmp_path.display(), self.final_path.display()))?;
        Ok(self.final_path)
    }
}

/// Standard naming: `<tier_dir>/shard-w<worker:03>-c<chunk:04>.parquet`.
/// Caller is responsible for ensuring `tier_dir` is created.
pub fn shard_path(tier_dir: &Path, worker_id: u32, chunk_idx: u32) -> PathBuf {
    tier_dir.join(format!("shard-w{worker_id:03}-c{chunk_idx:04}.parquet"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;
    use tempfile::tempdir;

    fn fake_row(seed: i64) -> GameRow {
        GameRow {
            tokens: vec![100, 200, 300],
            san: vec!["e4".into(), "e5".into(), "Nf3".into()],
            uci: vec!["e2e4".into(), "e7e5".into(), "g1f3".into()],
            game_length: 3,
            outcome_token: 1969,
            result: "1-0".into(),
            nodes: 1,
            multi_pv: 5,
            opening_multi_pv: 20,
            opening_plies: 2,
            sample_plies: 12,
            temperature: 1.0,
            worker_id: 0,
            game_seed: seed,
            stockfish_version: "Stockfish 18 by ...".into(),
        }
    }

    #[test]
    fn shard_path_is_zero_padded() {
        let p = shard_path(Path::new("/tmp/x"), 3, 17);
        assert_eq!(p, PathBuf::from("/tmp/x/shard-w003-c0017.parquet"));
    }

    #[test]
    fn write_then_read_round_trips() {
        let dir = tempdir().unwrap();
        let path = shard_path(dir.path(), 0, 0);
        let mut w = ShardWriter::create(path.clone()).unwrap();
        for i in 0..5 {
            w.append(&fake_row(1000 + i));
        }
        let written = w.close().unwrap();
        assert_eq!(written, path);

        // Re-read via parquet's own reader and verify shape + a couple of cells.
        let file = File::open(&path).unwrap();
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
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(game_seed_col.value(0), 1000);
        assert_eq!(game_seed_col.value(4), 1004);
    }

    #[test]
    fn close_with_no_rows_errors() {
        let dir = tempdir().unwrap();
        let path = shard_path(dir.path(), 0, 0);
        let w = ShardWriter::create(path).unwrap();
        assert!(w.close().is_err());
    }

    #[test]
    fn tmp_orphan_is_cleaned_up_on_create() {
        let dir = tempdir().unwrap();
        let path = shard_path(dir.path(), 0, 0);
        let tmp = path.with_extension("parquet.tmp");
        // Plant a fake .tmp orphan.
        std::fs::write(&tmp, b"garbage").unwrap();
        assert!(tmp.exists());
        let mut w = ShardWriter::create(path.clone()).unwrap();
        assert!(!tmp.exists(), "create() should have cleaned up the orphan tmp");
        w.append(&fake_row(0));
        let final_path = w.close().unwrap();
        assert!(final_path.exists());
    }
}
