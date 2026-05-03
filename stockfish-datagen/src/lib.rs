//! Standalone Stockfish self-play data generator. Entry point in `main.rs`;
//! everything testable lives in modules so the binary stays a thin wrapper.

pub mod affinity;
pub mod config;
pub mod game;
pub mod outcome;
pub mod resume;
pub mod runner;
pub mod sampler;
pub mod seed;
pub mod shard;
pub mod stockfish;
