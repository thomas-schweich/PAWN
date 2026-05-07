//! Run config — declarative description of a multi-tier generation run.
//!
//! Loaded from a single JSON file via `--config`; everything that isn't a
//! seed is configured here. The config is hashed (sha256) and the digest is
//! stored in each tier's `_manifest.json` so resume can detect a config drift.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Tilde-expand a path. Only handles `~/...` (the common case);
/// `~user/...` is returned as-is.
fn expand_tilde(p: &Path) -> PathBuf {
    if let Ok(s) = p.strip_prefix("~") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(s);
        }
    }
    p.to_path_buf()
}

/// Top-level run config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Path to the Stockfish binary. `~` is expanded.
    pub stockfish_path: PathBuf,

    /// Exact Stockfish version string we will accept (matched against
    /// the `id name <X>` line of the UCI handshake). Pinning this makes
    /// games reproducible from `(game_seed, config)`.
    pub stockfish_version: String,

    /// Output directory. One subdirectory per tier.
    pub output_dir: PathBuf,

    /// Master RNG seed. All per-tier and per-worker / per-game seeds are
    /// derived from this deterministically.
    pub master_seed: u64,

    /// Number of worker threads (each owns one Stockfish subprocess).
    pub n_workers: u32,

    /// Per-game ply cap. Beyond this the game terminates with PLY_LIMIT.
    #[serde(default = "default_max_ply")]
    pub max_ply: u32,

    /// Stockfish `Hash` option (MB).
    #[serde(default = "default_stockfish_hash_mb")]
    pub stockfish_hash_mb: u32,

    /// Games per parquet shard. Smaller = finer resume granularity at the
    /// cost of more shard files.
    #[serde(default = "default_shard_size_games")]
    pub shard_size_games: u32,

    /// Tiers to generate, in declaration order.
    pub tiers: Vec<TierConfig>,
}

/// One tier of self-play games. Two protocol modes:
///
/// - **Search mode** (`searchless: false`, the default): drives Stockfish
///   via `go nodes N` with MultiPV. Requires `nodes`, `multi_pv`,
///   `opening_multi_pv`, `opening_plies`, `sample_plies`. `sample_score`
///   must be omitted (only normalized cp is available from multipv parsing).
///
/// - **Searchless mode** (`searchless: true`): drives Stockfish via the
///   patched binary's `evallegal` command — pure NNUE static eval per
///   legal move, no search. Requires `sample_score` (deliberate cp/v
///   choice). Forbids `nodes`, `multi_pv`, `opening_multi_pv`,
///   `opening_plies`, `sample_plies` — those all describe how the search
///   budget is partitioned and have no meaning here.
///
/// `name`, `n_games`, `temperature`, `store_legal_move_evals` apply to
/// both modes. `validate()` enforces the field/mode invariants at config
/// load time, so all downstream `unwrap()`s on the protocol-specific
/// `Option` fields are safe within their respective code paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierConfig {
    /// Human-readable name; used as a subdirectory.
    pub name: String,

    /// Total games to generate for this tier (across all workers).
    pub n_games: u64,

    /// Softmax temperature over the chosen score field. Scaling factor is
    /// `100 * T` regardless of mode — so for cp this means "100 cp shifts
    /// probability by an e-fold at T=1.0", and for raw v this is sharper
    /// (raw v is ~3–5× larger than cp). `T <= 0` falls back to argmax.
    pub temperature: f32,

    /// When true, use the patched binary's `evallegal` protocol instead
    /// of `go nodes N`.
    #[serde(default)]
    pub searchless: bool,

    /// When true, persist the full Stockfish candidates list for every ply
    /// alongside the played move. The parquet column is
    /// `legal_move_evals: List<List<Struct{move_idx: i16, score_cp: i16, score_v: i16?}>>`,
    /// indexed per game and per ply. Designed for distillation-style
    /// training (KL loss vs the per-move softmax). Independent of
    /// `searchless` — though the typical use case is `searchless=true`
    /// + `store_legal_move_evals=true` for "tier 0" datasets.
    #[serde(default)]
    pub store_legal_move_evals: bool,

    // === Searchless-mode only ===
    /// Which per-move score the sampler softmaxes over. Required for
    /// searchless tiers (no default — choosing cp vs v is a meaningful
    /// decision worth ~10 Elo per the cp-vs-v tournament). Forbidden for
    /// non-searchless tiers (multipv parsing only surfaces normalized cp,
    /// raw v has nothing to refer to).
    #[serde(default)]
    pub sample_score: Option<SampleScore>,

    /// Force a specific NNUE network for all evaluation (`auto` / `small`
    /// / `large`). `None` (the default) leaves the engine's default in
    /// place (`auto`, vanilla SF18 dynamic selection). Applies to both
    /// search-mode and searchless tiers — vanilla SF18 picks small for
    /// material-imbalanced positions and may re-evaluate with big, which
    /// introduces position-dependent eval-source heterogeneity that's
    /// awkward for ML use cases. `Large` is the natural pick for
    /// distillation labelling (uniform high-quality teacher network).
    ///
    /// Requires the patched binary (vanilla SF18 doesn't recognize the
    /// `NetSelection` UCI option and would silently ignore the setoption);
    /// the preflight check in `main.rs` triggers whenever any tier sets
    /// this field.
    #[serde(default)]
    pub net_selection: Option<NetSelection>,

    // === Search-mode only ===
    /// `go nodes N` per-move budget. Forbidden in searchless mode
    /// (`evallegal` doesn't take a node budget).
    #[serde(default)]
    pub nodes: Option<u32>,

    /// MultiPV used for plies in `[opening_plies, sample_plies)`. Forbidden
    /// in searchless mode (`evallegal` returns every legal move; there's
    /// no top-K cap). Could mislead future readers as "limit softmax to
    /// top N" if accepted there.
    #[serde(default)]
    pub multi_pv: Option<u32>,

    /// MultiPV used for the very first plies; widens initial branching
    /// without slowing the rest of the game. Forbidden in searchless mode
    /// for the same reason as `multi_pv`.
    #[serde(default)]
    pub opening_multi_pv: Option<u32>,

    /// Number of plies (from move 0) that use `opening_multi_pv`. Forbidden
    /// in searchless mode (no opening/main split when every ply enumerates
    /// all legals).
    #[serde(default)]
    pub opening_plies: Option<u32>,

    /// First N plies use MultiPV+softmax; beyond N, take top-1 only.
    /// Forbidden in searchless mode (would silently misbehave: under
    /// `evallegal` the `target_pv == 1` shortcut takes `candidates[0]`,
    /// which is move-generation order, NOT the best move).
    #[serde(default)]
    pub sample_plies: Option<u32>,
}

/// Which per-move score the sampler should softmax over. Only meaningful
/// in searchless mode — the multipv path doesn't surface raw `v`, and
/// validation rejects this field on non-searchless tiers.
///
/// JSON-renamed to lowercase (`"cp"` / `"v"`) so configs stay terse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleScore {
    /// Normalized centipawns (`UCIEngine::to_cp`). Stable units across
    /// game phases.
    Cp,
    /// Raw NNUE `Value`. Equivalent to using the network's policy logits
    /// as the sampling distribution.
    V,
}

/// Forces uniform use of one NNUE network across every evaluation, mapped
/// to the patched binary's `NetSelection` UCI option (`auto` / `small` /
/// `large`). See `TierConfig::net_selection` for usage.
///
/// JSON-renamed to lowercase (`"auto"` / `"small"` / `"large"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetSelection {
    /// Vanilla SF18 dynamic selection — small net for large material
    /// imbalances, big net otherwise, with optional re-evaluation.
    Auto,
    /// Always use the small NNUE network. Fast but lower quality.
    Small,
    /// Always use the big NNUE network. Slower but uniform high quality.
    /// Recommended for distillation labelling.
    Large,
}

impl NetSelection {
    /// String form expected by the patched binary's UCI handler.
    pub fn as_uci_str(self) -> &'static str {
        match self {
            NetSelection::Auto => "auto",
            NetSelection::Small => "small",
            NetSelection::Large => "large",
        }
    }
}

/// Enforce the search-mode / searchless-mode field invariants. The two
/// modes use disjoint subsets of `TierConfig`'s protocol-specific fields;
/// the rules below make the user's choice explicit at config-load time
/// rather than letting it silently corrupt a tier mid-run.
fn validate_protocol_fields(tier: &TierConfig) -> anyhow::Result<()> {
    let name = &tier.name;
    if tier.searchless {
        // Searchless requires sample_score (deliberate cp/v pick), forbids
        // every search-budget knob.
        if tier.sample_score.is_none() {
            anyhow::bail!(
                "tier {name}: searchless=true requires sample_score (\"cp\" or \"v\"); \
                 evallegal surfaces both per-move scores and the choice meaningfully \
                 affects play strength (~10 Elo per the cp-vs-v tournament)"
            );
        }
        for (field, present) in [
            ("nodes", tier.nodes.is_some()),
            ("multi_pv", tier.multi_pv.is_some()),
            ("opening_multi_pv", tier.opening_multi_pv.is_some()),
            ("opening_plies", tier.opening_plies.is_some()),
            ("sample_plies", tier.sample_plies.is_some()),
        ] {
            if present {
                anyhow::bail!(
                    "tier {name}: searchless=true forbids {field}; evallegal evaluates \
                     every legal move with no search and no MultiPV cap, so the field \
                     would be silently ignored or misleading"
                );
            }
        }
    } else {
        // Search mode requires every search-budget knob, forbids sample_score.
        if tier.sample_score.is_some() {
            anyhow::bail!(
                "tier {name}: sample_score is only valid for searchless=true tiers; \
                 the multipv parsing path only surfaces normalized cp (raw v requires \
                 the patched binary's evallegal command)"
            );
        }
        let nodes = tier.nodes.ok_or_else(|| anyhow::anyhow!(
            "tier {name}: searchless=false requires nodes (the `go nodes N` budget)"
        ))?;
        if nodes == 0 {
            anyhow::bail!("tier {name}: nodes must be >= 1 (or set searchless=true)");
        }
        let multi_pv = tier.multi_pv.ok_or_else(|| anyhow::anyhow!(
            "tier {name}: searchless=false requires multi_pv"
        ))?;
        let opening_multi_pv = tier.opening_multi_pv.ok_or_else(|| anyhow::anyhow!(
            "tier {name}: searchless=false requires opening_multi_pv"
        ))?;
        if multi_pv == 0 || opening_multi_pv == 0 {
            anyhow::bail!("tier {name}: multi_pv values must be >= 1");
        }
        let opening_plies = tier.opening_plies.ok_or_else(|| anyhow::anyhow!(
            "tier {name}: searchless=false requires opening_plies"
        ))?;
        let sample_plies = tier.sample_plies.ok_or_else(|| anyhow::anyhow!(
            "tier {name}: searchless=false requires sample_plies"
        ))?;
        if opening_plies > sample_plies {
            anyhow::bail!(
                "tier {name}: opening_plies ({opening_plies}) must be <= sample_plies ({sample_plies})"
            );
        }
    }
    Ok(())
}

fn default_max_ply() -> u32 {
    512
}
fn default_stockfish_hash_mb() -> u32 {
    16
}
fn default_shard_size_games() -> u32 {
    10_000
}

impl RunConfig {
    /// Load + validate a config from a JSON file.
    ///
    /// Path fields (`stockfish_path`, `output_dir`) are tilde-expanded
    /// at load time — `~/sf-data` becomes `$HOME/sf-data` before any
    /// downstream consumer sees the value. Doing this once at the
    /// boundary prevents downstream callers (the CLI's preflight
    /// `create_dir_all`, `print_plan`, the python sync orchestrator)
    /// from each having to remember to expand it themselves.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        let mut cfg: RunConfig = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))?;
        cfg.stockfish_path = expand_tilde(&cfg.stockfish_path);
        cfg.output_dir = expand_tilde(&cfg.output_dir);
        cfg.validate()?;
        Ok(cfg)
    }

    /// Sanity check: tier names unique, counts positive, scheduling consistent.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.n_workers == 0 {
            anyhow::bail!("n_workers must be > 0");
        }
        // worker_id is stored as i16 in the parquet schema.
        if self.n_workers > i16::MAX as u32 {
            anyhow::bail!(
                "n_workers={} exceeds i16::MAX ({}); the parquet `worker_id` column is Int16",
                self.n_workers,
                i16::MAX,
            );
        }
        // game_length is stored as u16. With max_ply=512 (default) we're
        // nowhere near this, but reject pathological configs explicitly.
        if self.max_ply == 0 {
            // play_game's `for ply in 0..max_ply` loop would never run,
            // returning an empty move list that the worker would mis-
            // attribute to a "terminal-check bug" hard error.
            anyhow::bail!("max_ply must be >= 1");
        }
        if self.max_ply > u16::MAX as u32 {
            anyhow::bail!(
                "max_ply={} exceeds u16::MAX ({}); the parquet `game_length` column is UInt16",
                self.max_ply,
                u16::MAX,
            );
        }
        if self.shard_size_games == 0 {
            anyhow::bail!("shard_size_games must be > 0");
        }
        if self.stockfish_hash_mb == 0 {
            // Stockfish silently ignores `setoption Hash 0` and falls
            // back to its built-in default (~16MB), so the configured
            // value would no longer reflect what's actually in use,
            // breaking the fingerprint's "this matches the running run"
            // contract. Reject up-front.
            anyhow::bail!("stockfish_hash_mb must be > 0");
        }
        if self.tiers.is_empty() {
            anyhow::bail!("at least one tier required");
        }
        let mut seen = std::collections::HashSet::new();
        for tier in &self.tiers {
            if !seen.insert(&tier.name) {
                anyhow::bail!("duplicate tier name: {}", tier.name);
            }
            if tier.n_games == 0 {
                anyhow::bail!("tier {}: n_games must be > 0", tier.name);
            }
            validate_protocol_fields(tier)?;
        }
        Ok(())
    }

    /// Per-tier games-per-worker breakdown. Surplus games go to the
    /// lowest worker indices, matching the Python script's behavior.
    pub fn games_per_worker(&self, tier: &TierConfig) -> Vec<u64> {
        let n = self.n_workers as u64;
        let base = tier.n_games / n;
        let rem = tier.n_games % n;
        (0..n).map(|i| base + if i < rem { 1 } else { 0 }).collect()
    }

    /// Hex-encoded sha256 of the config bytes (after re-serialization, so
    /// formatting differences are normalized away). Stored alongside each
    /// tier's manifest so resume can refuse to continue under a changed
    /// config.
    ///
    /// Prefer [`Self::tier_fingerprint`] for resume-time tier validation —
    /// this whole-config hash is too coarse (adding a new tier to the
    /// config invalidates every existing tier's manifest).
    pub fn fingerprint(&self) -> String {
        let canonical = serde_json::to_vec(self).expect("config is round-trippable");
        let mut h = Sha256::new();
        h.update(&canonical);
        hex(&h.finalize())
    }

    /// Per-tier fingerprint covering only the inputs that would change
    /// the *bytes generated for that tier*. Adding/modifying *other*
    /// tiers leaves this fingerprint untouched, so users can grow their
    /// config over time without invalidating prior runs.
    ///
    /// Why every field matters for resume safety:
    /// - `tier` / `tier_index` / `master_seed`: directly seed every game.
    /// - `stockfish_version`: different NNUE → different game outcomes.
    /// - `stockfish_hash_mb`: Stockfish's transposition-table size can
    ///   affect fixed-node search choices (and therefore moves) at the
    ///   higher tiers (nodes >= 32 or so).
    /// - `shard_size_games`: changes which games end up in which shard
    ///   file, which would silently invalidate the resume row-count math.
    /// - `max_ply`: longer games may now resolve where they previously
    ///   hit `PLY_LIMIT`; outcome tokens differ.
    /// - `n_workers`: changes the (worker_id, game_index) partition AND
    ///   each game's seed (since `game_seed = mix(worker_seed, idx)`),
    ///   so an old shard's games are not a prefix of the new run.
    ///
    /// Implementation note: `serde_json::Value::Object` is `BTreeMap`
    /// (we don't enable the `preserve_order` feature), so the serialized
    /// JSON has keys in alphabetical order regardless of how they're
    /// listed below. This makes the fingerprint stable across builds.
    /// **Do not enable `preserve_order` on `serde_json` without a
    /// fingerprint version bump.** A pinned golden-value test
    /// (`tier_fingerprint_golden`) catches accidental drift.
    pub fn tier_fingerprint(&self, tier_index: usize) -> String {
        let payload = serde_json::json!({
            "tier_index": tier_index,
            "tier": &self.tiers[tier_index],
            "master_seed": self.master_seed,
            "stockfish_version": &self.stockfish_version,
            "stockfish_hash_mb": self.stockfish_hash_mb,
            "shard_size_games": self.shard_size_games,
            "max_ply": self.max_ply,
            "n_workers": self.n_workers,
        });
        let mut h = Sha256::new();
        h.update(payload.to_string().as_bytes());
        hex(&h.finalize())
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> RunConfig {
        RunConfig {
            stockfish_path: "/usr/bin/stockfish".into(),
            stockfish_version: "Stockfish 18".into(),
            output_dir: "out".into(),
            master_seed: 42,
            n_workers: 4,
            max_ply: 512,
            stockfish_hash_mb: 16,
            shard_size_games: 1000,
            tiers: vec![TierConfig {
                name: "nodes_0001".into(),
                n_games: 100,
                temperature: 1.0,
                searchless: false,
                store_legal_move_evals: false,
                sample_score: None,
                net_selection: None,
                nodes: Some(1),
                multi_pv: Some(5),
                opening_multi_pv: Some(20),
                opening_plies: Some(1),
                sample_plies: Some(999),
            }],
        }
    }

    /// Searchless-mode tier fixture, parallel to `minimal_config`. Used to
    /// exercise the validation rules that apply to evallegal tiers.
    fn searchless_tier(name: &str) -> TierConfig {
        TierConfig {
            name: name.into(),
            n_games: 100,
            temperature: 0.5,
            searchless: true,
            store_legal_move_evals: true,
            sample_score: Some(SampleScore::V),
            net_selection: None,
            nodes: None,
            multi_pv: None,
            opening_multi_pv: None,
            opening_plies: None,
            sample_plies: None,
        }
    }

    #[test]
    fn validate_accepts_minimal_config() {
        minimal_config().validate().unwrap();
    }

    #[test]
    fn load_expands_tilde_in_path_fields() {
        // Pin the contract that `RunConfig::load` is the one place
        // tilde expansion happens — every downstream consumer (CLI
        // preflight, run_tier, print_plan, the Python orchestrator)
        // assumes the path it sees is already absolute.
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("c.json");
        let body = r#"{
            "stockfish_path": "~/bin/stockfish",
            "stockfish_version": "Stockfish",
            "output_dir": "~/sf-data",
            "master_seed": 1,
            "n_workers": 1,
            "max_ply": 16,
            "stockfish_hash_mb": 1,
            "shard_size_games": 4,
            "tiers": [{
                "name": "t",
                "nodes": 1,
                "n_games": 4,
                "multi_pv": 1,
                "opening_multi_pv": 1,
                "opening_plies": 0,
                "sample_plies": 1,
                "temperature": 1.0
            }]
        }"#;
        std::fs::File::create(&cfg_path).unwrap().write_all(body.as_bytes()).unwrap();
        let cfg = RunConfig::load(&cfg_path).unwrap();
        let home = std::env::var("HOME").unwrap();
        assert_eq!(cfg.output_dir, PathBuf::from(&home).join("sf-data"));
        assert_eq!(cfg.stockfish_path, PathBuf::from(&home).join("bin/stockfish"));
    }

    #[test]
    fn validate_rejects_zero_workers() {
        let mut c = minimal_config();
        c.n_workers = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_duplicate_tier_names() {
        let mut c = minimal_config();
        c.tiers.push(c.tiers[0].clone());
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_opening_after_sample() {
        let mut c = minimal_config();
        c.tiers[0].opening_plies = Some(50);
        c.tiers[0].sample_plies = Some(10);
        assert!(c.validate().is_err());
    }

    #[test]
    fn games_per_worker_distributes_remainder() {
        let mut c = minimal_config();
        c.n_workers = 3;
        c.tiers[0].n_games = 10;
        let split = c.games_per_worker(&c.tiers[0]);
        assert_eq!(split, vec![4, 3, 3]);
        assert_eq!(split.iter().sum::<u64>(), 10);
    }

    #[test]
    fn fingerprint_is_stable() {
        let c = minimal_config();
        assert_eq!(c.fingerprint(), c.fingerprint());
    }

    #[test]
    fn fingerprint_changes_with_seed() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.master_seed = 1;
        b.master_seed = 2;
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn validate_rejects_zero_shard_size() {
        let mut c = minimal_config();
        c.shard_size_games = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_oversized_n_workers() {
        let mut c = minimal_config();
        c.n_workers = i16::MAX as u32 + 1;
        let err = c.validate().unwrap_err();
        assert!(format!("{err:#}").contains("Int16"), "got {err:#}");
    }

    #[test]
    fn validate_rejects_oversized_max_ply() {
        let mut c = minimal_config();
        c.max_ply = u16::MAX as u32 + 1;
        let err = c.validate().unwrap_err();
        assert!(format!("{err:#}").contains("UInt16"), "got {err:#}");
    }

    #[test]
    fn tier_fingerprint_isolates_tiers() {
        let mut a = minimal_config();
        a.tiers.push(TierConfig {
            name: "second".into(),
            n_games: 50,
            temperature: 1.0,
            searchless: false,
            store_legal_move_evals: false,
            sample_score: None,
            net_selection: None,
            nodes: Some(32),
            multi_pv: Some(5),
            opening_multi_pv: Some(5),
            opening_plies: Some(0),
            sample_plies: Some(999),
        });
        let mut b = a.clone();
        // Change only tier 1.
        b.tiers[1].n_games = 75;
        // Tier 0's fingerprint must NOT change just because tier 1 did.
        assert_eq!(a.tier_fingerprint(0), b.tier_fingerprint(0));
        // Tier 1's fingerprint MUST change.
        assert_ne!(a.tier_fingerprint(1), b.tier_fingerprint(1));
    }

    #[test]
    fn tier_fingerprint_changes_with_master_seed() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.master_seed = 1;
        b.master_seed = 2;
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }

    #[test]
    fn tier_fingerprint_changes_with_stockfish_version() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.stockfish_version = "Stockfish 17".into();
        b.stockfish_version = "Stockfish 18".into();
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }

    #[test]
    fn tier_fingerprint_changes_with_max_ply() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.max_ply = 256;
        b.max_ply = 512;
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }

    #[test]
    fn tier_fingerprint_changes_with_n_workers() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.n_workers = 4;
        b.n_workers = 8;
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }

    /// Pin the exact serialized JSON byte-for-byte. If serde or our
    /// payload structure ever changes the emission order or formatting,
    /// this test fires loudly — better than silently invalidating every
    /// existing manifest in production.
    ///
    /// To regenerate after an INTENTIONAL change: run the test, copy
    /// the printed `actual` value below.
    #[test]
    fn tier_fingerprint_golden() {
        let cfg = minimal_config();
        let actual = cfg.tier_fingerprint(0);
        // Updated when search-budget fields became Option<u32> and
        // sample_score became Option<SampleScore> (intentional — the JSON
        // emission for null fields differs from omitted fields, and the
        // protocol-mode invariant means existing production tiers don't
        // round-trip identically).
        let expected = "a36dac69d63481353a7aa5bc1764ab910b022541bf97dd6c2f844870ff8e1491";
        assert_eq!(
            actual, expected,
            "tier_fingerprint changed for minimal_config — verify the change is intentional, then update the expected value"
        );
    }

    #[test]
    fn tier_fingerprint_changes_with_sample_score() {
        let mut a = searchless_tier("t");
        let mut b = a.clone();
        a.sample_score = Some(SampleScore::Cp);
        b.sample_score = Some(SampleScore::V);
        let mut cfg_a = minimal_config();
        let mut cfg_b = minimal_config();
        cfg_a.tiers = vec![a];
        cfg_b.tiers = vec![b];
        assert_ne!(cfg_a.tier_fingerprint(0), cfg_b.tier_fingerprint(0));
    }

    #[test]
    fn validate_rejects_sample_score_on_search_tier() {
        // sample_score on a non-searchless tier suggests a misunderstanding —
        // multipv parsing only ever surfaces cp; raw v has no source.
        let mut c = minimal_config();
        c.tiers[0].sample_score = Some(SampleScore::Cp);
        let err = c.validate().unwrap_err();
        assert!(
            err.to_string().contains("sample_score"),
            "expected error to mention sample_score, got: {err}"
        );
    }

    #[test]
    fn validate_rejects_searchless_without_sample_score() {
        let mut c = minimal_config();
        c.tiers[0] = searchless_tier("t");
        c.tiers[0].sample_score = None;
        let err = c.validate().unwrap_err();
        assert!(
            err.to_string().contains("sample_score"),
            "expected error to mention sample_score, got: {err}"
        );
    }

    #[test]
    fn validate_accepts_searchless_with_explicit_score() {
        let mut c = minimal_config();
        c.tiers[0] = searchless_tier("t");
        c.tiers[0].sample_score = Some(SampleScore::V);
        c.validate().unwrap();
        c.tiers[0].sample_score = Some(SampleScore::Cp);
        c.validate().unwrap();
    }

    #[test]
    fn validate_rejects_searchless_with_search_budget_fields() {
        for (label, mutator) in [
            ("nodes", (|t: &mut TierConfig| t.nodes = Some(1)) as fn(&mut TierConfig)),
            ("multi_pv", |t| t.multi_pv = Some(5)),
            ("opening_multi_pv", |t| t.opening_multi_pv = Some(20)),
            ("opening_plies", |t| t.opening_plies = Some(2)),
            ("sample_plies", |t| t.sample_plies = Some(999)),
        ] {
            let mut c = minimal_config();
            c.tiers[0] = searchless_tier("t");
            mutator(&mut c.tiers[0]);
            let err = c.validate().unwrap_err();
            assert!(
                err.to_string().contains(label),
                "expected error to mention {label}, got: {err}"
            );
        }
    }

    #[test]
    fn validate_rejects_search_tier_missing_required_fields() {
        for (label, mutator) in [
            ("nodes", (|t: &mut TierConfig| t.nodes = None) as fn(&mut TierConfig)),
            ("multi_pv", |t| t.multi_pv = None),
            ("opening_multi_pv", |t| t.opening_multi_pv = None),
            ("opening_plies", |t| t.opening_plies = None),
            ("sample_plies", |t| t.sample_plies = None),
        ] {
            let mut c = minimal_config();
            mutator(&mut c.tiers[0]);
            let err = c.validate().unwrap_err();
            assert!(
                err.to_string().contains(label),
                "expected error to mention {label}, got: {err}"
            );
        }
    }

    #[test]
    fn tier_fingerprint_changes_with_net_selection() {
        // Different network choice ⇒ different evals ⇒ different games.
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.tiers[0].net_selection = None;
        b.tiers[0].net_selection = Some(NetSelection::Large);
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
        // And small vs large differ too.
        a.tiers[0].net_selection = Some(NetSelection::Small);
        b.tiers[0].net_selection = Some(NetSelection::Large);
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }

    #[test]
    fn net_selection_serde_round_trip() {
        let json = r#"{
            "stockfish_path": "/usr/bin/stockfish",
            "stockfish_version": "Stockfish 18",
            "output_dir": "out",
            "master_seed": 42,
            "n_workers": 4,
            "max_ply": 512,
            "stockfish_hash_mb": 16,
            "shard_size_games": 1000,
            "tiers": [{
                "name": "t",
                "n_games": 100,
                "temperature": 1.0,
                "searchless": true,
                "store_legal_move_evals": true,
                "sample_score": "v",
                "net_selection": "large"
            }]
        }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.tiers[0].net_selection, Some(NetSelection::Large));
        assert_eq!(cfg.tiers[0].net_selection.unwrap().as_uci_str(), "large");
        cfg.validate().unwrap();
    }

    #[test]
    fn search_tier_omitting_sample_score_round_trips() {
        // Existing search-tier configs (which never set sample_score)
        // continue to load and validate.
        let json = r#"{
            "stockfish_path": "/usr/bin/stockfish",
            "stockfish_version": "Stockfish 18",
            "output_dir": "out",
            "master_seed": 42,
            "n_workers": 4,
            "max_ply": 512,
            "stockfish_hash_mb": 16,
            "shard_size_games": 1000,
            "tiers": [{
                "name": "t",
                "nodes": 1,
                "n_games": 100,
                "multi_pv": 5,
                "opening_multi_pv": 20,
                "opening_plies": 1,
                "sample_plies": 999,
                "temperature": 1.0
            }]
        }"#;
        let cfg: RunConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.tiers[0].sample_score, None);
        cfg.validate().unwrap();
    }

    #[test]
    fn validate_rejects_zero_max_ply() {
        let mut c = minimal_config();
        c.max_ply = 0;
        let err = c.validate().unwrap_err();
        assert!(format!("{err:#}").contains("max_ply"), "got {err:#}");
    }

    #[test]
    fn validate_rejects_zero_stockfish_hash_mb() {
        let mut c = minimal_config();
        c.stockfish_hash_mb = 0;
        let err = c.validate().unwrap_err();
        assert!(format!("{err:#}").contains("stockfish_hash_mb"), "got {err:#}");
    }

    #[test]
    fn tier_fingerprint_changes_with_stockfish_hash_mb() {
        let mut a = minimal_config();
        let mut b = minimal_config();
        a.stockfish_hash_mb = 16;
        b.stockfish_hash_mb = 64;
        assert_ne!(a.tier_fingerprint(0), b.tier_fingerprint(0));
    }
}
