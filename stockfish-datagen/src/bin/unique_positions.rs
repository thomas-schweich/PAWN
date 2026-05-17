//! Counts, with HyperLogLog++ (~0.2% standard error at the default
//! precision), two quantities in the `thomas-schweich/pawn-stockfish-100m`
//! HuggingFace dataset:
//!   * unique positions — distinct board states across all five tiers.
//!     Also the count of unique *raw evaluations*: a raw NNUE eval is a
//!     search-free, pure function of the position.
//!   * unique positions with a MultiPV evaluation — distinct board states
//!     across the four search tiers (tier 0 is searchless, no MultiPV).
//!
//! Both come from one pass over the same replayed games — the MultiPV
//! sketch simply ignores tier-0 shards.
//!
//! Minimal data transfer: only the `uci` column is read, via parquet
//! column projection over HTTP range requests. `get_read` streams its
//! response rather than buffering to EOF, and the parquet reader stops
//! after the `uci` column — so the large eval columns are never
//! transferred and a full run moves ~25 GB, not the 987 GB dataset. The
//! exact bytes transferred are reported at the end.
//!
//! Build:   cargo build --release --bin unique_positions
//! Run:     HF_TOKEN=hf_xxx ./target/release/unique_positions [opts]
//!   --precision N        HLL++ precision 4..=18 (default 18, ~0.20% error)
//!   --threads N          concurrent shards (default 16; lower if HF
//!                        rate-limits, raise on a fat pod)
//!   --shards-per-tier N  cap shards/tier — use a small N to smoke-test
//!                        (default 10000 = the full dataset). Shard ids
//!                        count from 0, so a small N covers only `train`.
//!   --repo OWNER/NAME    dataset repo (default the 100M repo)
//!
//! HF_TOKEN is read from the env or ~/.cache/huggingface/token; omit it if
//! the dataset is public. Exits non-zero if any shard fails to download.

use std::hash::{BuildHasher, Hasher};
use std::io::Read;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{Array, ListArray, StringArray};
use bytes::Bytes;
use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use parquet::errors::ParquetError;
use parquet::file::reader::{ChunkReader, Length};
use rayon::prelude::*;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{Chess, EnPassantMode, Position};

const TIERS: [&str; 5] = [
    "tier0_evallegal",
    "nodes_0001",
    "nodes_0128",
    "nodes_0256",
    "nodes_1024",
];
const SHARDS_PER_TIER: usize = 10_000;
const SHARD_ROWS: usize = 2_000;

/// Total HTTP payload bytes pulled from the dataset, tallied across all
/// threads and reported at the end so a run's transfer volume is visible.
static DOWNLOADED: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Hashing — feed HLL++ a fully-mixed, deterministic 64-bit key.
// ---------------------------------------------------------------------------

/// splitmix64 finalizer — full avalanche, so a Zobrist key becomes a
/// uniformly-distributed 64-bit hash suitable for HLL register extraction.
#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Identity `Hasher`: we hand HLL++ an already-mixed `u64` and want it to
/// use exactly those bits. `u64::hash` calls `write_u64`, so the value
/// passes straight through with no re-hashing.
#[derive(Default)]
struct IdHasher(u64);
impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _bytes: &[u8]) {
        // Only `u64` keys are ever inserted (via `write_u64`); a byte-slice
        // write would mean a different key type and a silently wrong hash.
        unreachable!("IdHasher only accepts u64 keys via write_u64");
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}
#[derive(Default, Clone)]
struct IdBuild;
impl BuildHasher for IdBuild {
    type Hasher = IdHasher;
    fn build_hasher(&self) -> IdHasher {
        IdHasher(0)
    }
}

type Hll = HyperLogLogPlus<u64, IdBuild>;

fn new_hll(precision: u8) -> Hll {
    HyperLogLogPlus::new(precision, IdBuild).expect("HLL precision out of range (use 4..=18)")
}

/// The two sketches accumulated together in one pass over the `uci` column.
struct Acc {
    /// Distinct positions across all five tiers.
    all: Hll,
    /// Distinct positions across the four search tiers — i.e. distinct
    /// positions that carry a MultiPV evaluation (tier 0 is searchless).
    mpv: Hll,
}
impl Acc {
    fn new(precision: u8) -> Self {
        Acc {
            all: new_hll(precision),
            mpv: new_hll(precision),
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP range reader — a parquet `ChunkReader` backed by HTTP range requests.
// `get_read` streams its response rather than buffering to EOF, so a
// projected read transfers only the bytes the parquet reader consumes.
// ---------------------------------------------------------------------------

fn agent() -> &'static ureq::Agent {
    static A: OnceLock<ureq::Agent> = OnceLock::new();
    A.get_or_init(|| {
        ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(30))
            // Per-read inactivity timeout — NOT a whole-request deadline. A
            // healthy stream keeps resetting it; a genuinely stalled socket
            // trips it. A whole-request `.timeout()` would spuriously fail a
            // slow-but-progressing streamed `get_read` body.
            .timeout_read(Duration::from_secs(120))
            .build()
    })
}

fn hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.is_empty() {
            return Some(t);
        }
    }
    let home = std::env::var("HOME").ok()?;
    std::fs::read_to_string(format!("{home}/.cache/huggingface/token"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// A `Read` that streams an HTTP response body and tallies consumed bytes
/// into `DOWNLOADED`. Used by `get_read`: parquet stops reading after the
/// projected column chunk and drops this, closing the connection, so only
/// the consumed prefix — never the rest of the shard — is transferred.
struct CountingReader<R> {
    inner: R,
}
impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        DOWNLOADED.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

struct HttpRangeReader {
    url: String,
    token: Option<String>,
    len: u64,
}

impl HttpRangeReader {
    fn new(url: String, token: Option<String>) -> Result<Self, ParquetError> {
        let mut r = Self { url, token, len: 0 };
        // A 4-byte suffix range yields `Content-Range: bytes A-B/TOTAL`,
        // which gives us the file length in one cheap request.
        let (_, total) = r.fetch_buffered("bytes=-4", 4)?;
        r.len = total.ok_or_else(|| ParquetError::General("no Content-Range header".into()))?;
        Ok(r)
    }

    /// Issue a range GET with retry/backoff, returning the live response.
    fn request(&self, range: &str) -> Result<ureq::Response, ParquetError> {
        let mut last = String::new();
        for attempt in 0..8u32 {
            let mut req = agent().get(&self.url).set("Range", range);
            if let Some(t) = &self.token {
                req = req.set("Authorization", &format!("Bearer {t}"));
            }
            match req.call() {
                Ok(resp) if resp.status() == 206 => return Ok(resp),
                Ok(resp) => {
                    // A 2xx that is not 206 means the server ignored the
                    // Range header (a proxy/CDN stripping it) and sent the
                    // whole file from offset 0 — parquet would then read at
                    // the wrong offset. Retrying won't fix it; fail loud.
                    return Err(ParquetError::General(format!(
                        "range {range} on {}: got HTTP {} (expected 206 \
                         Partial Content) — a proxy/CDN may be stripping \
                         Range headers",
                        self.url,
                        resp.status()
                    )));
                }
                Err(ureq::Error::Status(code, _))
                    if matches!(code, 429 | 500 | 502 | 503 | 504) =>
                {
                    last = format!("HTTP {code}");
                }
                Err(ureq::Error::Transport(t)) => last = t.to_string(),
                Err(e) => return Err(ParquetError::General(format!("{e}"))),
            }
            // Exponential backoff with jitter — without the jitter, parallel
            // workers that hit a 429 together retry in lockstep and
            // immediately re-trigger the rate limit. The jitter is salted
            // with the per-shard URL so concurrent workers stay decorrelated
            // even when their clocks read the same nanosecond.
            let base = 300u64 << attempt.min(7);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64)
                .unwrap_or(0);
            let salt = self
                .url
                .bytes()
                .fold(0u64, |a, b| a.wrapping_mul(131).wrapping_add(b as u64));
            let jitter = mix64(nanos ^ salt ^ attempt as u64) % (base / 2 + 1);
            std::thread::sleep(Duration::from_millis(base + jitter));
        }
        Err(ParquetError::General(format!(
            "range {range} on {}: exhausted retries ({last})",
            self.url
        )))
    }

    /// Buffered range fetch (footer probe + `get_bytes`); `cap` pre-sizes
    /// the receive buffer to the expected length.
    fn fetch_buffered(
        &self,
        range: &str,
        cap: usize,
    ) -> Result<(Bytes, Option<u64>), ParquetError> {
        let resp = self.request(range)?;
        let total = resp
            .header("Content-Range")
            .and_then(|cr| cr.rsplit('/').next())
            .and_then(|t| t.trim().parse::<u64>().ok());
        let mut buf = Vec::with_capacity(cap);
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| ParquetError::General(format!("body read: {e}")))?;
        DOWNLOADED.fetch_add(buf.len() as u64, Ordering::Relaxed);
        Ok((Bytes::from(buf), total))
    }
}

impl Length for HttpRangeReader {
    fn len(&self) -> u64 {
        self.len
    }
}

impl ChunkReader for HttpRangeReader {
    type T = CountingReader<Box<dyn Read + Send + Sync>>;

    /// Streams `start..EOF`. parquet stops after the projected column
    /// chunk and drops the reader, closing the connection — so the eval
    /// columns past `uci` are never transferred, even on the page-by-page
    /// (no-offset-index) read path.
    fn get_read(&self, start: u64) -> Result<Self::T, ParquetError> {
        if start >= self.len {
            // Empty read at/past EOF — avoid forming a reversed
            // `bytes=N-(N-1)` range (answered 416), mirroring get_bytes's
            // length==0 guard.
            return Ok(CountingReader {
                inner: Box::new(std::io::empty()),
            });
        }
        let resp = self.request(&format!("bytes={}-{}", start, self.len - 1))?;
        Ok(CountingReader {
            inner: resp.into_reader(),
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes, ParquetError> {
        if length == 0 {
            return Ok(Bytes::new());
        }
        let (b, _) = self.fetch_buffered(
            &format!("bytes={}-{}", start, start + length as u64 - 1),
            length,
        )?;
        Ok(b)
    }
}

// ---------------------------------------------------------------------------
// Per-shard processing
// ---------------------------------------------------------------------------

/// Replay every game in a shard, feeding the Zobrist hash of each evaluated
/// position into `acc.all` (and, when `is_search_tier`, also `acc.mpv`).
/// Returns the number of positions seen.
fn scan_shard(
    url: String,
    token: Option<String>,
    acc: &mut Acc,
    is_search_tier: bool,
) -> Result<u64, String> {
    let reader = HttpRangeReader::new(url, token).map_err(|e| e.to_string())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(reader).map_err(|e| e.to_string())?;
    let idx = builder
        .schema()
        .fields()
        .iter()
        .position(|f| f.name().as_str() == "uci")
        .ok_or("no `uci` column")?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), [idx]);
    let rb = builder
        .with_projection(mask)
        .with_batch_size(SHARD_ROWS)
        .build()
        .map_err(|e| e.to_string())?;

    let mut n = 0u64;
    for batch in rb {
        let batch = batch.map_err(|e| e.to_string())?;
        let games = batch
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("`uci` is not a list")?;
        let moves = games
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("`uci` values are not strings")?;
        let off = games.value_offsets();
        for row in 0..games.len() {
            if games.is_null(row) {
                continue;
            }
            let (s, e) = (off[row] as usize, off[row + 1] as usize);
            let mut pos = Chess::default();
            for k in s..e {
                if moves.is_null(k) {
                    break;
                }
                // The position *before* the move is the one Stockfish
                // evaluated; the terminal position is never evaluated.
                let zob: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
                let h = mix64(zob.0);
                acc.all.insert(&h);
                if is_search_tier {
                    acc.mpv.insert(&h);
                }
                n += 1;
                let mv = match moves
                    .value(k)
                    .parse::<UciMove>()
                    .ok()
                    .and_then(|u| u.to_move(&pos).ok())
                {
                    Some(m) => m,
                    None => break, // unparseable move — stop this game
                };
                pos.play_unchecked(mv);
            }
        }
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn split_of(shard_id: usize) -> &'static str {
    if shard_id < 9950 {
        "train"
    } else if shard_id < 9975 {
        "val"
    } else {
        "test"
    }
}

fn commas(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, &c) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c as char);
    }
    out
}

fn main() {
    let mut precision: u8 = 18;
    let mut threads: usize = 16;
    let mut spt: usize = SHARDS_PER_TIER;
    let mut repo = String::from("thomas-schweich/pawn-stockfish-100m");

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let val = |args: &[String], i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| {
            eprintln!("missing value for {}", args[*i - 1]);
            std::process::exit(2);
        })
    };
    let bad_num = |flag: &str| -> ! {
        eprintln!("{flag} expects an integer");
        std::process::exit(2);
    };
    while i < args.len() {
        match args[i].as_str() {
            "--precision" => {
                precision = val(&args, &mut i)
                    .parse()
                    .unwrap_or_else(|_| bad_num("--precision"));
            }
            "--threads" => {
                threads = val(&args, &mut i)
                    .parse()
                    .unwrap_or_else(|_| bad_num("--threads"));
            }
            "--shards-per-tier" => {
                spt = val(&args, &mut i)
                    .parse()
                    .unwrap_or_else(|_| bad_num("--shards-per-tier"));
            }
            "--repo" => repo = val(&args, &mut i),
            "-h" | "--help" => {
                eprintln!("unique_positions — HLL++ unique-position counter for pawn-stockfish-100m");
                eprintln!("opts: --precision N  --threads N  --shards-per-tier N  --repo OWNER/NAME");
                eprintln!("(see the file header for details)");
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if spt > SHARDS_PER_TIER {
        eprintln!("note: capping --shards-per-tier at {SHARDS_PER_TIER} (the dataset has that many shards per tier)");
    }
    let spt = spt.min(SHARDS_PER_TIER);

    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("rayon pool");
    }
    let token = hf_token();
    let rel_err = 1.04 / (2f64.powi(precision as i32)).sqrt() * 100.0;
    eprintln!(
        "repo={repo}  HLL++ p={precision} (std error ~{rel_err:.3}%)  \
         threads={threads}  shards/tier={spt}  auth={}",
        if token.is_some() { "yes" } else { "no" }
    );

    // Preflight: a systemic failure (bad token, no repo access, no network)
    // should abort immediately, not surface as 50,000 per-shard failures.
    let probe = format!(
        "https://huggingface.co/datasets/{repo}/resolve/main/train/{}/shard-s000000-r002000.parquet",
        TIERS[0]
    );
    if let Err(e) = HttpRangeReader::new(probe, token.clone()) {
        eprintln!("preflight failed — cannot read the dataset: {e}");
        eprintln!("check HF_TOKEN and that `{repo}` is accessible.");
        std::process::exit(1);
    }

    // All five tiers carry positions.
    let shards: Vec<(usize, usize)> = (0..TIERS.len())
        .flat_map(|ti| (0..spt).map(move |s| (ti, s)))
        .collect();
    let n_shards = shards.len();
    eprintln!("scanning {n_shards} shards (uci column only)...");

    let started = Instant::now();
    let done = AtomicUsize::new(0);
    let fails = AtomicUsize::new(0);
    let scanned = AtomicU64::new(0);

    let mut acc = shards
        .par_iter()
        .fold(
            || Acc::new(precision),
            |mut acc, &(ti, sid)| {
                let path = format!(
                    "{}/{}/shard-s{:06}-r002000.parquet",
                    split_of(sid),
                    TIERS[ti],
                    sid
                );
                let url = format!("https://huggingface.co/datasets/{repo}/resolve/main/{path}");
                // Tier 0 is searchless — its positions count toward `all`
                // but never toward the MultiPV sketch.
                match scan_shard(url, token.clone(), &mut acc, ti >= 1) {
                    Ok(c) => {
                        scanned.fetch_add(c, Ordering::Relaxed);
                    }
                    Err(e) => {
                        fails.fetch_add(1, Ordering::Relaxed);
                        eprintln!("  FAIL {path}: {e}");
                    }
                }
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 1000 == 0 || d == n_shards {
                    eprintln!("  {d}/{n_shards} shards");
                }
                acc
            },
        )
        .reduce(
            || Acc::new(precision),
            |mut a, b| {
                a.all.merge(&b.all).expect("HLL merge (same precision)");
                a.mpv.merge(&b.mpv).expect("HLL merge (same precision)");
                a
            },
        );

    let unique = acc.all.count().round() as u64;
    let unique_mpv = acc.mpv.count().round() as u64;
    let fails = fails.load(Ordering::Relaxed);
    let scanned = scanned.load(Ordering::Relaxed);
    let downloaded = DOWNLOADED.load(Ordering::Relaxed);

    println!("\n=========== result (HLL++ p={precision}, ~{rel_err:.2}% std err) ===========");
    println!("data transferred                 : {:>15} MB", commas(downloaded / 1_000_000));
    println!("positions scanned                : {:>18}", commas(scanned));
    println!("unique positions                 : {:>18}", commas(unique));
    println!("  (= unique raw evaluations — a raw NNUE eval is a pure function of the position)");
    println!("unique positions w/ MultiPV eval : {:>18}", commas(unique_mpv));
    println!("  (distinct positions across the four search tiers; tier 0 is searchless)");
    if fails > 0 {
        println!("shards FAILED                    : {fails}  (result is INCOMPLETE — exit code 1)");
    }
    println!("elapsed: {:.1?}", started.elapsed());

    if fails > 0 {
        std::process::exit(1);
    }
}
