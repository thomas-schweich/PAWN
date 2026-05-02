//! `stockfish-datagen` — generate Stockfish self-play games as zstd-Parquet
//! shards. All run state (paths, seeds, tier breakdown) lives in a single
//! JSON config; the CLI is intentionally minimal.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Parser, Subcommand};

use stockfish_datagen::config::RunConfig;

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
            Ok(())
        }
        Command::Run { config: _ } => {
            anyhow::bail!("run is not implemented yet — coming in the next commit");
        }
    }
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
        println!(
            "  [{i}] {name:<14} nodes={nodes:>4} games={n_games:>10} \
             multi_pv={mpv:>2} opening={ompv:>2}/{op_plies} \
             sample_plies={spl:<3} temp={temp:.2}",
            name = tier.name,
            nodes = tier.nodes,
            n_games = tier.n_games,
            mpv = tier.multi_pv,
            ompv = tier.opening_multi_pv,
            op_plies = tier.opening_plies,
            spl = tier.sample_plies,
            temp = tier.temperature,
        );
        println!(
            "       per-worker max: {max_per_worker}, total shards: {total_shards}"
        );
    }
}
