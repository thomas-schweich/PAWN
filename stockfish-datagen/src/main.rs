//! `stockfish-datagen` — generate Stockfish self-play games as zstd-Parquet
//! shards. All run state (paths, seeds, tier breakdown) lives in a single
//! JSON config; the CLI is intentionally minimal.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Parser, Subcommand};

use stockfish_datagen::config::RunConfig;
use stockfish_datagen::runner::run_tier;
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
            print_plan(&cfg);
            preflight_check_patched_binary(&cfg)?;
            Ok(())
        }
        Command::Run { config } => {
            let cfg = RunConfig::load(&config)
                .with_context(|| format!("loading config {}", config.display()))?;
            print_plan(&cfg);
            preflight_check_patched_binary(&cfg)?;
            std::fs::create_dir_all(&cfg.output_dir).with_context(|| {
                format!("creating output dir {}", cfg.output_dir.display())
            })?;

            let t0 = std::time::Instant::now();
            let mut totals = Totals::default();
            for tier_index in 0..cfg.tiers.len() {
                let tier_t0 = std::time::Instant::now();
                let result = run_tier(&cfg, tier_index)
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

/// If any tier sets `searchless: true` or `net_selection: ...`, spawn one
/// throwaway Stockfish process and verify it recognizes the patched
/// binary's surface area (`evallegal` command + `NetSelection` UCI option).
///
/// Vanilla Stockfish responds with `Unknown command: 'evallegal'` to the
/// probe and silently ignores unknown setoption names — both are silent
/// failures that would corrupt a tier mid-run. Fail loudly at startup
/// instead.
fn preflight_check_patched_binary(cfg: &RunConfig) -> anyhow::Result<()> {
    let needs_patched: Vec<String> = cfg
        .tiers
        .iter()
        .filter(|t| t.searchless || t.net_selection.is_some())
        .map(|t| {
            let mut why: Vec<&str> = Vec::new();
            if t.searchless { why.push("searchless"); }
            if t.net_selection.is_some() { why.push("net_selection"); }
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
        "stockfish patched (evallegal command): {}",
        if probe.is_patched { "yes" } else { "NO" },
    );
    if !probe.is_patched {
        anyhow::bail!(
            "tier(s) {:?} require the patched binary, but {} does not recognize the \
             `evallegal` UCI command that marks it. Build it via \
             `bash stockfish-datagen/scripts/build_patched_stockfish.sh` and point \
             `stockfish_path` at the resulting `stockfish-datagen/stockfish-patched`.",
            needs_patched, cfg.stockfish_path.display(),
        );
    }
    Ok(())
}

fn print_plan(cfg: &RunConfig) {
    println!("=== stockfish-datagen plan ===");
    println!("stockfish:          {}", cfg.stockfish_path.display());
    println!("stockfish_version:  {}", cfg.stockfish_version);
    println!("output_dir:         {}", cfg.output_dir.display());
    println!("master_seed:        {}", cfg.master_seed);
    println!("workers:            {}", cfg.n_workers);
    println!("max_ply:            {}", cfg.max_ply);
    println!("shard_size_games:   {}", cfg.shard_size_games);
    println!("config fingerprint: {}", cfg.fingerprint());
    println!();
    println!("tiers:");
    for (i, tier) in cfg.tiers.iter().enumerate() {
        let split = cfg.games_per_worker(tier);
        let max_per_worker = split.iter().copied().max().unwrap_or(0);
        let total_shards =
            split.iter().map(|n| n.div_ceil(cfg.shard_size_games as u64)).sum::<u64>();
        let net = tier.net_selection
            .map(|n| format!(" net={:?}", n))
            .unwrap_or_default();
        if tier.searchless {
            // Searchless tier: no nodes / multi_pv / opening_* / sample_plies
            // — those are the search-mode knobs. Show what actually applies.
            let score = tier.sample_score.expect("validated: searchless has sample_score");
            println!(
                "  [{i}] {name:<14} EVALLEGAL  games={n_games:>10} \
                 sample_score={score:?} temp={temp:.2}{store}{net}",
                name = tier.name,
                n_games = tier.n_games,
                temp = tier.temperature,
                store = if tier.store_legal_move_evals { " store_legal_move_evals=true" } else { "" },
            );
        } else {
            println!(
                "  [{i}] {name:<14} nodes={nodes:>4} games={n_games:>10} \
                 multi_pv={mpv:>2} opening={ompv:>2}/{op_plies} \
                 sample_plies={spl:<3} temp={temp:.2}{net}",
                name = tier.name,
                nodes = tier.nodes.expect("validated"),
                n_games = tier.n_games,
                mpv = tier.multi_pv.expect("validated"),
                ompv = tier.opening_multi_pv.expect("validated"),
                op_plies = tier.opening_plies.expect("validated"),
                spl = tier.sample_plies.expect("validated"),
                temp = tier.temperature,
            );
        }
        println!(
            "       per-worker max: {max_per_worker}, total shards: {total_shards}"
        );
    }
}
