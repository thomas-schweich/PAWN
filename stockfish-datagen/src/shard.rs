//! One parquet shard per **global shard id**. Atomic on disk: written to
//! `shard-s<NNNNNN>.parquet.tmp` and renamed on close to
//! `shard-s<NNNNNN>-r<NNNNNN>.parquet`, where the trailing `r` field is
//! the row count. Shard `s` owns the contiguous global-game-index range
//! `[s * shard_size_games, min((s+1) * shard_size_games, n_games))`;
//! workers claim shard ids from a shared atomic counter and are otherwise
//! interchangeable (`n_workers` is purely operational).
//!
//! Encoding the row count in the filename lets resume determine
//! `(shard_id, n_rows)` from a directory listing alone — no parquet
//! metadata reads required. That in turn lets a remote sync tool
//! (e.g. an HF dataset uploader) drop zero-byte placeholder files
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

/// Parquet schema version. Bump whenever the schema changes in a way
/// that would mix incompatibly with prior shards in the same tier
/// directory (added columns, removed columns, changed types).
///
/// `tier_fingerprint` (in `config.rs`) includes this in its hash so a
/// resumed run across a schema upgrade fails loudly with a fingerprint
/// mismatch rather than silently writing new-schema shards into a
/// directory of old-schema ones.
///
/// History:
///   v1 — original schema (tokens/san/uci + legal_move_evals only)
///   v2 — adds nullable `static_legal_move_evals` column for canonical
///        per-position NNUE labels on non-searchless tiers
///   v3 — shard-id partitioning refactor: drops `worker_id` (no longer
///        a stable per-game identifier), adds `global_game_index` (the
///        canonical per-tier global index — independent of n_workers
///        and the partition). game_seed = mix(tier_seed,
///        global_game_index); see `seed.rs`.
pub const SHARD_SCHEMA_VERSION: u32 = 3;

/// One candidate move's distillation payload. Stored packed as
/// `Struct{move_idx: i16, score_cp: i16, score_eval_v: i16?, score_psqt: i16?,
/// score_positional: i16?}` to keep parquet rows compact while remaining
/// trivially loadable from polars / pyarrow.
///
/// - `move_idx`: searchless_chess action vocab index (0..1968).
/// - `score_cp`: normalized centipawns, mover-POV. 100 cp ≈ "1 pawn equivalent".
/// - `score_eval_v`: `Eval::evaluate`'s post-processed Value, mover-POV.
///   What Stockfish plays with (head-blend + complexity damp + material/optimism
///   mix + 50-move shuffling damp + TB-clamp). Right target for play-policy
///   distillation. `Some` iff the row's source list was the patched binary's
///   `evallegal` protocol; `None` for multipv-sourced rows (the multipv
///   parser only surfaces normalized cp). The Some/None distinction is
///   per-source-list, not per-tier-mode — search-mode tiers populate
///   `legal_move_evals` from multipv (so `score_eval_v` is `None` there)
///   but ALSO populate `static_legal_move_evals` from a separate evallegal
///   call (where `score_eval_v` IS `Some`). See `GameRow` field docs.
/// - `score_psqt`, `score_positional`: raw NNUE per-head outputs from
///   `Networks::evaluate()`, mover-POV, before any post-processing. Together
///   they're the right targets for hot-swap NNUE-replacement distillation —
///   Stockfish itself applies the post-processing on top, so the student must
///   not have it baked in. Same Some/None rule as `score_eval_v` — present
///   on evallegal-sourced rows, absent on multipv-sourced rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegalMoveEval {
    pub move_idx: i16,
    pub score_cp: i16,
    pub score_eval_v: Option<i16>,
    pub score_psqt: Option<i16>,
    pub score_positional: Option<i16>,
}

/// Fields of the `legal_move_evals` Struct element. Hoisted so the
/// schema and the StructBuilder agree on order/nullability.
fn legal_move_eval_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("move_idx", DataType::Int16, false),
        Field::new("score_cp", DataType::Int16, false),
        // Three nullable raw-eval columns: only the `evallegal` protocol
        // (sf_18-v0.3.0+ patched binary) provides any of them. Multipv-derived
        // candidates leave all three null.
        Field::new("score_eval_v", DataType::Int16, true),
        Field::new("score_psqt", DataType::Int16, true),
        Field::new("score_positional", DataType::Int16, true),
    ])
}

fn legal_move_eval_struct_builders() -> Vec<Box<dyn ArrayBuilder>> {
    // Drive the builder count from the field count so adding / removing a
    // field can never desync the two — `StructBuilder` panics at append
    // time on a length mismatch, which is far harder to diagnose than the
    // schema panicking at construction.
    (0..legal_move_eval_struct_fields().len())
        .map(|_| Box::new(Int16Builder::new()) as Box<dyn ArrayBuilder>)
        .collect()
}

/// Append one row's per-ply per-move payload to a `List<List<Struct>>`
/// column builder. Used by both `legal_move_evals` (non-nullable column,
/// `none_is_empty_list = true`) and `static_legal_move_evals` (nullable
/// column, `none_is_empty_list = false`).
///
/// When `plies` is `None`:
/// - `none_is_empty_list = true`: append a row-level empty list `[]`.
///   Schema is non-null at the row level; downstream readers see an empty
///   outer list rather than a row-level null.
/// - `none_is_empty_list = false`: append a row-level null. Distinguishes
///   "tier opted out of this column" (null) from "tier opted in but the
///   game had zero plies" (empty list).
fn append_eval_column(
    col: &mut ListBuilder<ListBuilder<StructBuilder>>,
    plies: Option<&[Vec<LegalMoveEval>]>,
    none_is_empty_list: bool,
) {
    if let Some(plies) = plies {
        let inner_list = col.values();
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
                struct_b
                    .field_builder::<Int16Builder>(2)
                    .expect("score_eval_v field 2")
                    .append_option(ev.score_eval_v);
                struct_b
                    .field_builder::<Int16Builder>(3)
                    .expect("score_psqt field 3")
                    .append_option(ev.score_psqt);
                struct_b
                    .field_builder::<Int16Builder>(4)
                    .expect("score_positional field 4")
                    .append_option(ev.score_positional);
                struct_b.append(true);
            }
            inner_list.append(true);
        }
        col.append(true);
    } else if none_is_empty_list {
        col.append(true); // empty outer list at row level
    } else {
        col.append(false); // row-level null
    }
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
    /// Searchless-only: `"cp"` or `"v"`, the score field the sampler
    /// softmaxed over. `None` for non-searchless rows (the multipv parsing
    /// path only surfaces cp). Persisted per-row so a shard moved out of
    /// its tier directory remains attributable.
    pub sample_score: Option<String>,
    /// Either tier mode: the UCI `NetSelection` value (`"auto"`, `"small"`,
    /// `"large"`) the engine was configured to use. `None` when the tier
    /// left the engine on its default (`auto` for the patched binary,
    /// vanilla SF dynamic for unpatched). Both search-mode and searchless
    /// tiers may set this; `sample_score IS NULL` does NOT imply
    /// `net_selection IS NULL`.
    pub net_selection: Option<String>,
    pub temperature: f32,
    /// Canonical per-tier global game index, independent of which worker
    /// thread generated it. game_seed is `mix(tier_seed, global_game_index)`,
    /// so this is the reproduction key together with the tier's config.
    pub global_game_index: u64,
    pub game_seed: u64,
    pub stockfish_version: String,
    /// Per-ply per-legal-move payload from the tier's *selection* engine
    /// call. `None` when the tier did not request distillation data
    /// (non-storing tiers leave the parquet column as an empty list,
    /// paying only a few bytes of offsets per row). `Some(Vec)` outer
    /// length equals `game_length`; inner lengths are the number of
    /// candidates the selection engine produced at each ply (capped by
    /// the tier's MultiPV for search-mode; full legal-move set for
    /// searchless / evallegal). The semantics of each `LegalMoveEval`'s
    /// score fields therefore depend on tier mode — search-mode rows
    /// only populate `score_cp`; searchless rows populate all five.
    /// See `TierConfig::store_legal_move_evals`.
    pub legal_move_evals: Option<Vec<Vec<LegalMoveEval>>>,
    /// Per-ply per-legal-move *canonical* NNUE static eval payload (full
    /// `evallegal` output), captured by a separate engine call regardless
    /// of how the move was actually selected. Same Struct shape as
    /// `legal_move_evals` but with all five score fields populated and
    /// the full legal-move set per ply (no MultiPV cap, move-gen ordering).
    ///
    /// `None` for: tiers that didn't set `store_legal_move_evals: true`,
    /// AND for searchless tiers (where the same data already lives in
    /// `legal_move_evals` — capturing it again would double the storage
    /// cost on tier-0 distillation data).
    ///
    /// Convention for downstream consumers: read the canonical static
    /// eval as `static_legal_move_evals if not None else legal_move_evals`.
    /// This null-on-searchless rule keeps the schema uniform without
    /// duplicating ~16 KB/game on the largest tier.
    pub static_legal_move_evals: Option<Vec<Vec<LegalMoveEval>>>,
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
        // sample_score: searchless-only, "cp"|"v"; null on non-searchless rows.
        // net_selection: any tier may set it ("auto"|"small"|"large"); null
        // when the tier left the engine on its default. Persisted per-row so
        // a shard moved out of its directory context remains attributable.
        Field::new("sample_score", DataType::Utf8, true),
        Field::new("net_selection", DataType::Utf8, true),
        Field::new("temperature", DataType::Float32, false),
        // Canonical per-tier global game index. Stored as `UInt64` so 100M+
        // game datasets fit without overflow. Together with the tier
        // fingerprint, this uniquely identifies a game in the dataset.
        Field::new("global_game_index", DataType::UInt64, false),
        // game_seed is the per-game RNG seed, derived via splitmix64 — a
        // full u64. Stored as `UInt64` (Arrow + Parquet support it
        // natively) so any reader sees the actual value without
        // bit-pattern reinterpretation. Polars / pyarrow / pandas all
        // handle this correctly.
        Field::new("game_seed", DataType::UInt64, false),
        Field::new("stockfish_version", DataType::Utf8, false),
        Field::new("legal_move_evals", eval_outer_list_dt.clone(), false),
        // Canonical NNUE static eval per legal move per ply, captured via
        // a separate `evallegal` call regardless of selection mode.
        // Nullable at the row level so searchless tiers can leave it null
        // (their `legal_move_evals` already contains the same data) and
        // for forward-compat with shards generated before this column
        // existed (readers see null, not a missing-column error).
        Field::new("static_legal_move_evals", eval_outer_list_dt, true),
    ]))
});

/// Buffer rows in column builders, write once at close. Memory cap is the
/// shard size (~10k games × ~150 plies × few bytes/move = ~15MB), so we
/// don't need streaming writes within a shard.
pub struct ShardWriter {
    tier_dir: PathBuf,
    shard_id: u64,
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
    sample_score: StringBuilder,
    net_selection: StringBuilder,
    temperature: Float32Builder,
    global_game_index: UInt64Builder,
    game_seed: UInt64Builder,
    stockfish_version: StringBuilder,
    legal_move_evals: ListBuilder<ListBuilder<StructBuilder>>,
    static_legal_move_evals: ListBuilder<ListBuilder<StructBuilder>>,
    n_rows: usize,
}

impl ShardWriter {
    /// Open a writer for the given `shard_id` under `tier_dir`. The final
    /// filename — which includes the row count — is determined at
    /// `close()` time, so writes go to a row-count-free `.parquet.tmp`
    /// path and the rename happens at the end.
    pub fn create(tier_dir: PathBuf, shard_id: u64) -> anyhow::Result<Self> {
        fs::create_dir_all(&tier_dir)
            .with_context(|| format!("creating shard dir {}", tier_dir.display()))?;
        let tmp_path = shard_tmp_path(&tier_dir, shard_id);
        // Best effort: clean up any leftover .tmp from a prior crash before
        // we start writing this shard.
        let _ = fs::remove_file(&tmp_path);

        Ok(Self {
            tier_dir,
            shard_id,
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
            sample_score: StringBuilder::new(),
            net_selection: StringBuilder::new(),
            temperature: Float32Builder::new(),
            global_game_index: UInt64Builder::new(),
            game_seed: UInt64Builder::new(),
            stockfish_version: StringBuilder::new(),
            legal_move_evals: ListBuilder::new(ListBuilder::new(StructBuilder::new(
                legal_move_eval_struct_fields(),
                legal_move_eval_struct_builders(),
            ))),
            static_legal_move_evals: ListBuilder::new(ListBuilder::new(StructBuilder::new(
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
        self.sample_score.append_option(row.sample_score.as_deref());
        self.net_selection.append_option(row.net_selection.as_deref());
        self.temperature.append_value(row.temperature);
        self.global_game_index.append_value(row.global_game_index);
        self.game_seed.append_value(row.game_seed);
        self.stockfish_version.append_value(&row.stockfish_version);

        // legal_move_evals: outer list = per-ply, inner list = per-move.
        // Tiers that don't store distillation data leave the outer list
        // empty (still non-null at row level — schema declares the column
        // non-null and downstream readers find an empty `[]` rather than
        // having to handle row-level nulls).
        append_eval_column(&mut self.legal_move_evals, row.legal_move_evals.as_deref(), true);

        // static_legal_move_evals: nullable column (schema-level), so when
        // None we append a row-level null rather than an empty list. This
        // distinguishes "tier opted out of static labels" (null) from
        // "tier opted in but the game had zero plies" (empty list).
        append_eval_column(
            &mut self.static_legal_move_evals,
            row.static_legal_move_evals.as_deref(),
            false,
        );

        self.n_rows += 1;
    }

    /// Materialize the buffered rows, write the parquet, and atomically
    /// rename `.tmp` → final path (which includes the row count). Consumes self.
    pub fn close(mut self) -> anyhow::Result<PathBuf> {
        if self.n_rows == 0 {
            // Nothing to write — clean up the (never-created) tmp and return.
            let _ = fs::remove_file(&self.tmp_path);
            anyhow::bail!(
                "ShardWriter::close called with zero rows for s{:06} in {}",
                self.shard_id, self.tier_dir.display(),
            );
        }
        let final_path = shard_final_path(
            &self.tier_dir, self.shard_id, self.n_rows as u64,
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
            Arc::new(self.sample_score.finish()),
            Arc::new(self.net_selection.finish()),
            Arc::new(self.temperature.finish()),
            Arc::new(self.global_game_index.finish()),
            Arc::new(self.game_seed.finish()),
            Arc::new(self.stockfish_version.finish()),
            Arc::new(self.legal_move_evals.finish()),
            Arc::new(self.static_legal_move_evals.finish()),
        ];
        let batch = RecordBatch::try_new(SCHEMA.clone(), columns)
            .context("building record batch")?;

        // zstd-19 + 16 MB pages chosen empirically: ~22% smaller files vs the
        // writer default (zstd-3, 1 MB pages). Sweep on a 1000-game shard
        // (pyarrow rewrite, comparable but not identical to the rust crate's
        // numbers):
        //   zstd-3  +  1 MB:   baseline
        //   zstd-9  +  8 MB:  -14.8%, 0.97 s/shard
        //   zstd-15 +  8 MB:  -16.8%, 2.51 s/shard
        //   zstd-19 +  8 MB:  -21.0%, 4.40 s/shard
        //   zstd-19 + 16 MB:  -22.6%, 4.85 s/shard   ← chosen
        //   zstd-22 +  8 MB:  -21.0%, 4.94 s/shard   (plateau, no gain past 19)
        //
        // Decompression speed is independent of zstd level, so training
        // pipelines and downstream consumers see no read-side change.
        //
        // Per-shard write is BLOCKING on the worker thread (compress + fsync +
        // rename happen inline), so a higher zstd level eats into wall-clock.
        // At ~5 s/shard close cost and 1000-game shards, this is roughly
        // 1 s blocked per ~70 s worker step ≈ 7 % wall-clock — acceptable for
        // multi-day generation runs that pay it back many times over via
        // smaller HF dataset uploads, training cache, etc.
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(19)?))
            .set_data_page_size_limit(16 * 1024 * 1024)
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

/// In-progress path: `<tier_dir>/shard-s<NNNNNN>.parquet.tmp`.
/// The row count is unknown until close, so the tmp filename omits it.
pub fn shard_tmp_path(tier_dir: &Path, shard_id: u64) -> PathBuf {
    tier_dir.join(format!("shard-s{shard_id:06}.parquet.tmp"))
}

/// Final, post-rename path: `<tier_dir>/shard-s<NNNNNN>-r<NNNNNN>.parquet`.
/// Encoding the row count in the filename lets resume — and remote-sync
/// placeholder files — recover `(shard_id, n_rows)` from a directory
/// listing alone. Both fields are zero-padded to a *minimum* of 6 digits;
/// values that exceed 999_999 simply produce a wider field, which
/// `parse_shard_filename` and the Python regex `\d{6,}` both accept.
pub fn shard_final_path(tier_dir: &Path, shard_id: u64, n_rows: u64) -> PathBuf {
    tier_dir.join(format!("shard-s{shard_id:06}-r{n_rows:06}.parquet"))
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
            sample_score: None,
            net_selection: None,
            temperature: 1.0,
            global_game_index: seed, // tests use seed as a stand-in index
            game_seed: seed,
            stockfish_version: "Stockfish 18 by ...".into(),
            legal_move_evals: None,
            static_legal_move_evals: None,
        }
    }

    #[test]
    fn shard_paths_are_zero_padded() {
        let tmp = shard_tmp_path(Path::new("/tmp/x"), 17);
        assert_eq!(tmp, PathBuf::from("/tmp/x/shard-s000017.parquet.tmp"));
        let final_p = shard_final_path(Path::new("/tmp/x"), 17, 834);
        assert_eq!(final_p, PathBuf::from("/tmp/x/shard-s000017-r000834.parquet"));
    }

    #[test]
    fn write_then_read_round_trips() {
        let dir = tempdir().unwrap();
        let mut w = ShardWriter::create(dir.path().to_path_buf(), 0).unwrap();
        for i in 0..5 {
            w.append(&fake_row(1000 + i));
        }
        let written = w.close().unwrap();
        assert_eq!(written, shard_final_path(dir.path(), 0, 5));

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
        let mut w = ShardWriter::create(dir.path().to_path_buf(), 0).unwrap();

        // Game 0: distillation tier — populated. Two plies, varying legal-move counts.
        // First ply mixes Some / None across all three nullable raw-eval fields
        // to exercise the nullable column path independently for each.
        let mut row0 = fake_row(1);
        row0.legal_move_evals = Some(vec![
            vec![
                LegalMoveEval {
                    move_idx: 100,
                    score_cp: 50,
                    score_eval_v: Some(125),
                    score_psqt: Some(140),
                    score_positional: Some(-15),
                },
                LegalMoveEval {
                    move_idx: 200,
                    score_cp: -25,
                    score_eval_v: None,
                    score_psqt: None,
                    score_positional: None,
                },
            ],
            vec![LegalMoveEval {
                move_idx: 300,
                score_cp: 10,
                score_eval_v: Some(28),
                score_psqt: Some(35),
                score_positional: Some(-7),
            }],
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
        let score_eval_v = ply0_0.column(2).as_any().downcast_ref::<Int16Array>().unwrap();
        let score_psqt = ply0_0.column(3).as_any().downcast_ref::<Int16Array>().unwrap();
        let score_positional = ply0_0.column(4).as_any().downcast_ref::<Int16Array>().unwrap();
        assert_eq!(move_idx.value(0), 100);
        assert_eq!(score_cp.value(0), 50);
        assert_eq!(score_eval_v.value(0), 125);
        assert!(score_eval_v.is_valid(0));
        assert_eq!(score_psqt.value(0), 140);
        assert!(score_psqt.is_valid(0));
        assert_eq!(score_positional.value(0), -15);
        assert!(score_positional.is_valid(0));
        assert_eq!(move_idx.value(1), 200);
        assert_eq!(score_cp.value(1), -25);
        // All three nullable raw-eval fields independently null out per row.
        assert!(!score_eval_v.is_valid(1));
        assert!(!score_psqt.is_valid(1));
        assert!(!score_positional.is_valid(1));

        // Row 1: empty outer list (no per-ply data captured).
        let plies1 = outer.value(1);
        let plies1 = plies1.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(plies1.len(), 0);
    }

    /// `static_legal_move_evals` is nullable at the row level (the schema
    /// declares it nullable, and `append_eval_column(..., none_is_empty_list:
    /// false)` appends a row-level null when the field is `None`). This test
    /// pins both halves of that contract:
    /// - `Some(...)` round-trips with full per-ply per-move data
    /// - `None` produces a row-level null (NOT an empty outer list, which
    ///   is what `legal_move_evals` does for its non-null column)
    /// Without this, a future swap of the `col.append(true/false)` flag in
    /// `append_eval_column` (or a schema-nullability change) would silently
    /// drop static labels, since the live integration tests only assert on
    /// in-memory `PlayedGame` shape, never re-read the parquet column.
    #[test]
    fn static_legal_move_evals_round_trip() {
        use arrow::array::{Array, Int16Array, ListArray, StructArray};
        let dir = tempdir().unwrap();
        let mut w = ShardWriter::create(dir.path().to_path_buf(), 0).unwrap();

        // Row 0: search-mode tier with the static column populated. Mirrors
        // the live test's expected shape: every entry has all five struct
        // fields populated (move_idx + 4 score fields), distinguishing the
        // evallegal-sourced data from the multipv-sourced legal_move_evals.
        let mut row0 = fake_row(101);
        row0.legal_move_evals = Some(vec![vec![LegalMoveEval {
            move_idx: 50,
            score_cp: 10,
            score_eval_v: None, // multipv-sourced — raw v not surfaced
            score_psqt: None,
            score_positional: None,
        }]]);
        row0.static_legal_move_evals = Some(vec![vec![
            LegalMoveEval {
                move_idx: 50,
                score_cp: 10,
                score_eval_v: Some(35),
                score_psqt: Some(40),
                score_positional: Some(-5),
            },
            LegalMoveEval {
                move_idx: 75,
                score_cp: 8,
                score_eval_v: Some(28),
                score_psqt: Some(30),
                score_positional: Some(-2),
            },
        ]]);
        w.append(&row0);

        // Row 1: searchless tier — `static_legal_move_evals` left None, so
        // the row-level null path of append_eval_column fires.
        let mut row1 = fake_row(102);
        row1.legal_move_evals = Some(vec![vec![LegalMoveEval {
            move_idx: 100,
            score_cp: 5,
            score_eval_v: Some(17),
            score_psqt: Some(20),
            score_positional: Some(-3),
        }]]);
        row1.static_legal_move_evals = None;
        w.append(&row1);

        let path = w.close().unwrap();
        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        let batch = &batches[0];
        let static_col = batch
            .column_by_name("static_legal_move_evals")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        // Row 0 should be NON-null, with one ply containing two moves.
        assert!(static_col.is_valid(0), "row 0 static_legal_move_evals should be non-null");
        let plies0 = static_col.value(0);
        let plies0 = plies0.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(plies0.len(), 1, "row 0 should have 1 ply");
        let ply0 = plies0.value(0);
        let ply0 = ply0.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(ply0.len(), 2, "row 0 ply 0 should have 2 moves");
        let move_idx = ply0.column(0).as_any().downcast_ref::<Int16Array>().unwrap();
        let score_eval_v = ply0.column(2).as_any().downcast_ref::<Int16Array>().unwrap();
        assert_eq!(move_idx.value(0), 50);
        assert_eq!(score_eval_v.value(0), 35);
        assert_eq!(move_idx.value(1), 75);
        assert_eq!(score_eval_v.value(1), 28);

        // Row 1 should be ROW-LEVEL NULL — pin this against any future
        // refactor that confuses "no data" with "empty outer list".
        assert!(!static_col.is_valid(1), "row 1 static_legal_move_evals should be row-level null");
    }

    /// Regression test for the round-7 `game_seed: i64 -> UInt64` change.
    /// A high-bit seed (>= 2^63) MUST round-trip as itself, not flip to a
    /// negative integer. With the prior Int64 schema this would store a
    /// negative value and require explicit bit-reinterpretation in every
    /// reader.
    #[test]
    fn high_bit_game_seed_round_trips() {
        let dir = tempdir().unwrap();
        let mut w = ShardWriter::create(dir.path().to_path_buf(), 0).unwrap();
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
        let w = ShardWriter::create(dir.path().to_path_buf(), 0).unwrap();
        assert!(w.close().is_err());
    }

    #[test]
    fn tmp_orphan_is_cleaned_up_on_create() {
        let dir = tempdir().unwrap();
        let tmp = shard_tmp_path(dir.path(), 0);
        // Plant a fake .tmp orphan.
        std::fs::write(&tmp, b"garbage").unwrap();
        assert!(tmp.exists());
        let mut w = ShardWriter::create(dir.path().to_path_buf(), 0).unwrap();
        assert!(!tmp.exists(), "create() should have cleaned up the orphan tmp");
        w.append(&fake_row(0));
        let final_path = w.close().unwrap();
        assert!(final_path.exists());
        assert_eq!(final_path, shard_final_path(dir.path(), 0, 1));
    }
}
