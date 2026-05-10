//! `stockfish-datagen` — generate Stockfish self-play games as zstd-Parquet
//! shards. All run state (paths, seeds, tier breakdown) lives in a single
//! JSON config; the CLI is intentionally minimal.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Parser, Subcommand};

use stockfish_datagen::config::RunConfig;
use stockfish_datagen::numa::set_interleave_all_nodes;
use stockfish_datagen::resume::ShardRange;
use stockfish_datagen::runner::{RunScope, run_tier};
use stockfish_datagen::stockfish::{GoBudget, StockfishProcess};
use stockfish_datagen::tournament::{TournamentConfig, run_tournament};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Validate the config + print the per-tier plan, do nothing else.
    DryRun {
        #[arg(long)]
        config: PathBuf,
    },
    /// Generate the dataset described by the config. Resumes any
    /// partial tiers automatically (per-shard granularity).
    Run {
        #[arg(long)]
        config: PathBuf,

        /// Comma-separated tier indices to run (e.g. `0,1` or `3`).
        /// Default: all tiers in the config. Skipped tiers are not
        /// preflighted, not started, and don't have state files written.
        /// Useful for multi-pod runs where each pod tackles a different
        /// subset of the tier list.
        #[arg(long, value_delimiter = ',')]
        tiers: Option<Vec<usize>>,

        /// Restrict this pod's work to shard ids in the half-open range
        /// `A:B` (e.g. `0:5000`). Default: full tier range. Used to fan
        /// a single tier across multiple pods — each pod claims a
        /// disjoint slice of the shard ids while the per-tier
        /// `n_games` / `shard_size_games` stay constant across pods.
        ///
        /// Per-pod state and manifest sentinels are suffixed with the
        /// range (`_tier_state-s<A>-s<B>.json`,
        /// `_manifest-s<A>-s<B>.json`) so disjoint pods coexist cleanly
        /// in the same HF dataset folder.
        #[arg(long, value_parser = parse_shard_range)]
        shard_id_range: Option<ShardRange>,
    },
    /// Play two SampleScore × temperature configs against each other and
    /// report W/D/L + Elo difference (Wilson 95% CI). Used for things like
    /// "is cp-policy or v-policy stronger at T=0?". Always uses the
    /// patched binary's evallegal protocol.
    Tournament {
        #[arg(long)]
        config: PathBuf,
    },
}

/// Parse `A:B` (half-open) into a `ShardRange`. Reject `A >= B`.
fn parse_shard_range(s: &str) -> Result<ShardRange, String> {
    let (a, b) = s.split_once(':').ok_or_else(|| {
        format!("expected `<start>:<end>` (half-open), got `{s}`")
    })?;
    let start: u64 = a.parse().map_err(|e| format!("invalid start: {e}"))?;
    let end: u64 = b.parse().map_err(|e| format!("invalid end: {e}"))?;
    if start >= end {
        return Err(format!("empty range: start {start} >= end {end}"));
    }
    Ok(ShardRange { start, end })
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::DryRun { config } => {
            let cfg = RunConfig::load(&config)
                .with_context(|| format!("loading config {}", config.display()))?;
            print_plan(&cfg, &RunScope::default());
            preflight_check_patched_binary(&cfg)?;
            Ok(())
        }
        Command::Run { config, tiers, shard_id_range } => {
            // Set MPOL_INTERLEAVE before any allocation / spawn so the
            // policy is inherited by every stockfish child. On
            // single-socket pods this is effectively a no-op; on
            // multi-socket pods it spreads NNUE page-cache first-touches
            // evenly across NUMA nodes. See `numa.rs` for details.
            set_interleave_all_nodes();

            let cfg = RunConfig::load(&config)
                .with_context(|| format!("loading config {}", config.display()))?;
            // Validate tier subset against the config NOW so a typo
            // surfaces before any expensive setup (preflight spawn, etc).
            if let Some(t) = &tiers {
                for idx in t {
                    if *idx >= cfg.tiers.len() {
                        anyhow::bail!(
                            "--tiers {idx} out of range; config has {} tier(s) (0..{})",
                            cfg.tiers.len(),
                            cfg.tiers.len().saturating_sub(1),
                        );
                    }
                }
            }
            let scope = RunScope {
                tiers: tiers.clone(),
                shard_range: shard_id_range,
            };
            print_plan(&cfg, &scope);
            preflight_check_patched_binary(&cfg)?;
            std::fs::create_dir_all(&cfg.output_dir).with_context(|| {
                format!("creating output dir {}", cfg.output_dir.display())
            })?;

            let t0 = std::time::Instant::now();
            let mut totals = Totals::default();
            for tier_index in 0..cfg.tiers.len() {
                if !scope.includes_tier(tier_index) {
                    continue;
                }
                let tier_t0 = std::time::Instant::now();
                let result = run_tier(&cfg, tier_index, &scope)
                    .with_context(|| format!("tier {} failed", cfg.tiers[tier_index].name))?;
                let elapsed = tier_t0.elapsed();
                let rate = result.n_games_written as f64 / elapsed.as_secs_f64().max(1e-9);
                eprintln!(
                    "[{}] {}/{} games in {:.1}s ({:.1} games/s); {} shards",
                    cfg.tiers[tier_index].name,
                    result.n_games_written,
                    cfg.tiers[tier_index].n_games,
                    elapsed.as_secs_f64(),
                    rate,
                    result.shards.len(),
                );
                totals.games += result.n_games_written;
                totals.shards += result.shards.len() as u64;
            }

            let total_elapsed = t0.elapsed();
            eprintln!();
            eprintln!("=== run complete ===");
            eprintln!(
                "wrote {} games across {} shards in {:.1}m",
                totals.games,
                totals.shards,
                total_elapsed.as_secs_f64() / 60.0,
            );
            eprintln!("output: {}", cfg.output_dir.display());
            Ok(())
        }
        Command::Tournament { config } => {
            let cfg = TournamentConfig::load(&config)
                .with_context(|| format!("loading tournament config {}", config.display()))?;
            print_tournament_plan(&cfg);
            preflight_check_tournament_binary(&cfg)?;

            let t0 = std::time::Instant::now();
            let result = run_tournament(&cfg).context("running tournament")?;
            let elapsed = t0.elapsed();

            print_tournament_summary(&cfg, &result, elapsed);
            Ok(())
        }
    }
}

fn print_tournament_plan(cfg: &TournamentConfig) {
    println!("=== tournament plan ===");
    println!("stockfish:        {}", cfg.stockfish_path.display());
    println!("master_seed:      {}", cfg.master_seed);
    println!("workers:          {}", cfg.n_workers);
    println!("pairs × 2:        {} × 2 = {} games", cfg.n_pairs, 2 * cfg.n_pairs);
    println!("opening_plies:    {}", cfg.opening_plies);
    println!("max_ply:          {}", cfg.max_ply);
    println!(
        "side_a:           {} (sample_score={:?}, T={})",
        cfg.side_a.name, cfg.side_a.sample_score, cfg.side_a.temperature,
    );
    println!(
        "side_b:           {} (sample_score={:?}, T={})",
        cfg.side_b.name, cfg.side_b.sample_score, cfg.side_b.temperature,
    );
    println!("config fingerprint: {}", cfg.fingerprint());
    println!();
}

fn print_tournament_summary(
    cfg: &TournamentConfig,
    result: &stockfish_datagen::tournament::TournamentResult,
    elapsed: std::time::Duration,
) {
    let (lo_wr, hi_wr) = result.a_win_rate_ci95();
    let (lo_elo, hi_elo) = result.a_elo_ci95();
    eprintln!();
    eprintln!("=== tournament complete ===");
    eprintln!("elapsed:    {:.1}s", elapsed.as_secs_f64());
    eprintln!(
        "side_a ({}): {} wins",
        cfg.side_a.name, result.a_wins,
    );
    eprintln!(
        "side_b ({}): {} wins",
        cfg.side_b.name, result.b_wins,
    );
    eprintln!("draws:      {}", result.draws);
    eprintln!("total:      {}", result.total);
    eprintln!();
    eprintln!(
        "{} win rate: {:.4} (Wilson 95% CI: {:.4} – {:.4})",
        cfg.side_a.name, result.a_win_rate(), lo_wr, hi_wr,
    );
    eprintln!(
        "{} − {} Elo: {:+.1} (95% CI: {:+.1} – {:+.1})",
        cfg.side_a.name, cfg.side_b.name, result.a_elo(), lo_elo, hi_elo,
    );
    if let Some(out) = &cfg.output_path {
        eprintln!("per-game records: {}", out.display());
    }
}

#[derive(Default)]
struct Totals {
    games: u64,
    shards: u64,
}

/// Tournament-side counterpart to `preflight_check_patched_binary`.
/// Tournaments always drive the patched binary's `evallegal` command
/// (`GoBudget::EvalLegal`), so the only requirement here is that the binary
/// recognizes that command **with the v0.3.0+ output shape**
/// (`<uci> <cp> <eval_v> <psqt> <positional>` per legal move). NetSelection
/// is irrelevant — tournament workers don't apply per-side `net_selection`
/// overrides (`TournamentSide` has no such field) — but the v0.3.0 shape
/// check is non-negotiable: the parser silently degrades to zero candidates
/// against pre-v0.3.0 patched binaries, so without the probe-time check a
/// stale binary would crash workers mid-match with `NoCandidates`. The
/// `is_patched` flag set by `StockfishProcess::spawn` is true only when
/// **both** conditions hold (patched + v0.3.0 shape), so this is the same
/// check as the runner's. Spawn one throwaway probe; fail fast before
/// spawning N tournament workers against a vanilla or stale SF.
fn preflight_check_tournament_binary(cfg: &TournamentConfig) -> anyhow::Result<()> {
    let probe = StockfishProcess::spawn(
        &cfg.stockfish_path,
        &cfg.stockfish_version,
        cfg.stockfish_hash_mb,
        GoBudget::Nodes(1),
    )
    .with_context(|| {
        format!("preflight: spawning {} for tournament probe", cfg.stockfish_path.display())
    })?;
    eprintln!(
        "stockfish patched (evallegal v0.3.0 shape): {}",
        if probe.is_patched { "yes" } else { "NO" },
    );
    if !probe.is_patched {
        anyhow::bail!(
            "tournament requires the v0.3.0+ patched binary (always runs evallegal), \
             but {} does not recognize the `evallegal` UCI command or emits a stale \
             pre-v0.3.0 output shape. Build / rebuild via \
             `bash stockfish-datagen/scripts/build_patched_stockfish.sh` (currently \
             pinned to fork tag `sf_18-v0.3.0`).",
            cfg.stockfish_path.display(),
        );
    }
    Ok(())
}

/// If any tier sets `searchless: true` or `net_selection: ...`, spawn one
/// throwaway Stockfish process and verify it recognizes the patched
/// binary's surface area (`evallegal` command **with the v0.3.0+ output
/// shape** + `NetSelection` UCI option).
///
/// Vanilla Stockfish responds with `Unknown command: 'evallegal'` to the
/// probe and silently ignores unknown setoption names — both are silent
/// failures that would corrupt a tier mid-run. Pre-v0.3.0 patched binaries
/// recognize `evallegal` but emit a 3-tuple shape (`<uci> <cp> <v>`) the
/// current parser silently degrades to zero candidates on, which is the
/// same class of silent corruption with a worse failure mode (worker dies
/// 200 plies into a game with `NoCandidates`). Fail loudly at startup
/// instead — the v0.3.0 shape check is built into the spawn-time probe
/// and surfaces as `is_patched: false` for stale binaries.
fn preflight_check_patched_binary(cfg: &RunConfig) -> anyhow::Result<()> {
    // A tier needs the patched binary when:
    //   - searchless: the whole selection loop runs `evallegal`
    //   - net_selection: vanilla SF18 silently ignores the unknown setoption,
    //     and the shard fingerprint would lie about the network in use
    //   - non-searchless + store_legal_move_evals: the per-ply teacher signal
    //     is captured by a separate `evallegal` call after each search-mode
    //     selection (the static_legal_move_evals column). Without the patch
    //     the call would emit "Unknown command" and the worker would die on
    //     the first ply of the first game.
    let needs_patched: Vec<String> = cfg
        .tiers
        .iter()
        .filter(|t| {
            t.searchless
                || t.net_selection.is_some()
                || (t.store_legal_move_evals && !t.searchless)
        })
        .map(|t| {
            let mut why: Vec<&str> = Vec::new();
            if t.searchless { why.push("searchless"); }
            if t.net_selection.is_some() { why.push("net_selection"); }
            if t.store_legal_move_evals && !t.searchless {
                why.push("store_legal_move_evals");
            }
            format!("{} ({})", t.name, why.join("+"))
        })
        .collect();
    if needs_patched.is_empty() {
        return Ok(());
    }
    // One throwaway probe — `spawn()` always sends `evallegal` against startpos
    // post-handshake and tags `is_patched` based on the response shape, so
    // the budget choice here doesn't matter.
    let probe = StockfishProcess::spawn(
        &cfg.stockfish_path,
        &cfg.stockfish_version,
        cfg.stockfish_hash_mb,
        GoBudget::Nodes(1),
    )
    .with_context(|| {
        format!("preflight: spawning {} to check for evallegal patch", cfg.stockfish_path.display())
    })?;
    eprintln!(
        "stockfish patched (evallegal v0.3.0 shape): {}",
        if probe.is_patched { "yes" } else { "NO" },
    );
    eprintln!(
        "stockfish patched (NetSelection option): {}",
        if probe.has_net_selection { "yes" } else { "NO" },
    );
    if !probe.is_patched {
        anyhow::bail!(
            "tier(s) {:?} require the v0.3.0+ patched binary, but {} either does not \
             recognize the `evallegal` UCI command or emits a stale pre-v0.3.0 output \
             shape. Build / rebuild via \
             `bash stockfish-datagen/scripts/build_patched_stockfish.sh` (currently \
             pinned to fork tag `sf_18-v0.3.0`) and point `stockfish_path` at the \
             resulting `stockfish-datagen/stockfish-patched`.",
            needs_patched, cfg.stockfish_path.display(),
        );
    }
    // Separate check: any tier that sets `net_selection` needs the binary
    // to ALSO advertise the NetSelection UCI option. The v0.3.0 patched
    // binary advertises both, but theoretically a future fork build could
    // diverge. Older fork builds (pre-`sf_18-v0.2.0`) had `evallegal` but
    // not NetSelection — UCI silently ignores unknown setoption names, so
    // without this gate the engine would fall back to its default network
    // while the shard fingerprint claims `large` / `small`. Reject loudly.
    let needs_net_selection: Vec<&str> = cfg
        .tiers
        .iter()
        .filter(|t| t.net_selection.is_some())
        .map(|t| t.name.as_str())
        .collect();
    if !needs_net_selection.is_empty() && !probe.has_net_selection {
        anyhow::bail!(
            "tier(s) {:?} set `net_selection`, but {} does not advertise the \
             `NetSelection` UCI option. The bundled v0.3.0 patched binary advertises \
             it; a binary that has `evallegal` but not `NetSelection` is either a \
             pre-v0.2.0 fork build or a divergent fork. UCI silently ignores unknown \
             setoption names, so a setoption send here would leave the engine on its \
             default network while the shard fingerprint claims the requested choice. \
             Rebuild via `bash stockfish-datagen/scripts/build_patched_stockfish.sh`.",
            needs_net_selection, cfg.stockfish_path.display(),
        );
    }
    Ok(())
}

fn print_plan(cfg: &RunConfig, scope: &RunScope) {
    println!("=== stockfish-datagen plan ===");
    println!("stockfish:          {}", cfg.stockfish_path.display());
    println!("stockfish_version:  {}", cfg.stockfish_version);
    println!("output_dir:         {}", cfg.output_dir.display());
    println!("master_seed:        {}", cfg.master_seed);
    println!("workers:            {}", cfg.n_workers);
    println!("max_ply:            {}", cfg.max_ply);
    println!("shard_size_games:   {}", cfg.shard_size_games);
    println!("stockfish_hash_mb:  {} (top-level default)", cfg.stockfish_hash_mb);
    if let Some(t) = &scope.tiers {
        println!("--tiers:            {t:?}");
    }
    if let Some(r) = scope.shard_range {
        println!("--shard-id-range:   {}..{}", r.start, r.end);
    }
    println!("config fingerprint: {}", cfg.fingerprint());
    println!();
    println!("tiers:");
    for (i, tier) in cfg.tiers.iter().enumerate() {
        let active = scope.includes_tier(i);
        let active_marker = if active { " " } else { "*" }; // * = skipped this run
        let total_shards = cfg.total_shards(tier);
        let pod_range = scope.effective_shard_range(cfg, tier);
        let pod_shards = pod_range.end.saturating_sub(pod_range.start);
        let net = tier.net_selection
            .map(|n| format!(" net={:?}", n))
            .unwrap_or_default();
        let hash_mb_str = match tier.stockfish_hash_mb {
            Some(v) => format!(" hash_mb={v}"),
            None => String::new(),
        };
        if tier.searchless {
            let score = tier.sample_score.expect("validated: searchless has sample_score");
            println!(
                "  [{i}]{active_marker}{name:<14} EVALLEGAL  games={n_games:>10} \
                 sample_score={score:?} temp={temp:.2}{store}{net}{hash}",
                name = tier.name,
                n_games = tier.n_games,
                temp = tier.temperature,
                store = if tier.store_legal_move_evals { " store_legal_move_evals=true" } else { "" },
                hash = hash_mb_str,
            );
        } else {
            println!(
                "  [{i}]{active_marker}{name:<14} nodes={nodes:>4} games={n_games:>10} \
                 multi_pv={mpv:>2} opening={ompv:>2}/{op_plies} \
                 sample_plies={spl:<3} temp={temp:.2}{net}{store}{hash}",
                name = tier.name,
                nodes = tier.nodes.expect("validated"),
                n_games = tier.n_games,
                mpv = tier.multi_pv.expect("validated"),
                ompv = tier.opening_multi_pv.expect("validated"),
                op_plies = tier.opening_plies.expect("validated"),
                spl = tier.sample_plies.expect("validated"),
                temp = tier.temperature,
                store = if tier.store_legal_move_evals { " store_legal_move_evals=true" } else { "" },
                hash = hash_mb_str,
            );
        }
        let scope_note = if active && pod_shards < total_shards {
            format!(", this pod owns {pod_shards} ({:?})", pod_range)
        } else if !active {
            " — SKIPPED via --tiers".into()
        } else {
            String::new()
        };
        println!(
            "       total shards: {total_shards}{scope_note}",
        );
    }
}
