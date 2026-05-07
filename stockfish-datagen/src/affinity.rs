//! CPU pinning for the (worker thread, Stockfish process) pair.
//!
//! Each worker drives one Stockfish subprocess synchronously over a pipe
//! (write `position+go`, wait, read `bestmove`). The two are intrinsically
//! serialized — at any instant only one is on-CPU — so they want to share
//! the same core's L1/L2. Without pinning, the OS scheduler load-balances
//! them across cores and the cache evicts on every round-trip; we
//! measured this as ~23% kernel time + 537K context switches/sec on a
//! 16-core box at 16 workers.
//!
//! With more workers than physical cores, pinning round-robins
//! `worker_id % n_cores` so SMT siblings get used. With fewer workers
//! than cores, each pair gets a dedicated core. NUMA effects matter
//! more at high core counts (e.g. 128-thread vast.ai pods); pinning is
//! a precondition for any further NUMA-aware tuning.
//!
//! ## n_workers tuning under pinning
//!
//! Pinning shifts the optimal worker count DOWN versus the unpinned
//! baseline. On a local 16-thread / 8-physical-core box at nodes=1,
//! 5k games:
//!
//! ```text
//! workers   no-pin    pinned    delta
//! ------------------------------------
//!   8        179.3    244.1    +36%
//!  12        219.2    301.5    +38%
//!  13          —      312.0    (peak − 3%)
//!  14        261.4    321.4    +23%   ← pinned peak
//!  15        284.6    309.9     +9%
//!  16        300.6    301.3     ≈0%   ← unpinned peak
//!  20        298.3    231.7    -22%
//!  24        303.9    260.0    -14%
//! ```
//!
//! Three observations:
//! 1. Pinning peaks ~7% higher than the unpinned best, AND uses fewer
//!    cores to do it (14 vs 16 logical in the local sweep).
//! 2. Pinning gives a 36-38% boost in the under-saturated regime.
//!    Each (worker, SF) pair gets a dedicated core with cache locality.
//! 3. Pinning *hurts* once workers ≥ cores: with multiple pairs
//!    pinned to the same logical core via `worker_id % n_logical`,
//!    they fight over the time slice instead of being able to migrate.
//!
//! ## Rule of thumb: `n_workers = total_threads - threads_per_core`
//!
//! Equivalent reading: "fully occupy every physical core *except one*,
//! and leave that one core entirely free for the kernel + the
//! orchestrator's watcher thread + HF upload network handling +
//! parquet write-back I/O".
//!
//! On the local 16-thread / 8-physical / 2-SMT box: 16 − 2 = 14.
//! Verified by the table above.
//!
//! Why this works (packed Linux topology — the modern default):
//! `core_affinity::get_core_ids()` returns logical cpus in topology
//! order: cpus 0,1 are SMT siblings on physical core 0; cpus 2,3 on
//! physical 1; …; cpus N-2, N-1 on the last physical core. With
//! `worker_id % n_logical` pinning and `n_workers = n_logical -
//! threads_per_core`, the LAST physical core's siblings are entirely
//! out of the rotation — that core is uncontested for everything
//! else the pod is doing.
//!
//! Predictions for other topologies:
//!
//! ```text
//! box                                          rule → workers
//! ----------------------------------------------------------
//! local: 16 thread / 8 phys / 2 SMT            16 − 2 = 14   (verified)
//! vast.ai 128 thread / 64 phys / 2 SMT         128 − 2 = 126
//! same chip with SMT off (64 phys / 1 SMT)     64 − 1 = 63
//! POWER-style 32 phys / 4 SMT                  128 − 4 = 124
//! ```
//!
//! Caveats:
//! - Empirically verified at SMT=2 / 16-thread only. The 128-thread
//!   case is theoretical extrapolation. A 30-second 5k-game sweep at
//!   {n−4, n−2, n} workers on the actual pod is cheap insurance.
//! - Assumes packed topology. On scattered topology (older systems),
//!   `worker_id % n_logical` would NOT leave a fully-free physical core.
//! - NUMA: the rule doesn't care about socket count — and that's fine.
//!   What the free core absorbs (kernel networking for HF uploads,
//!   the watcher thread, parquet I/O write-back, the orchestrator's
//!   periodic work) doesn't scale with worker count or socket count.
//!   A 1-socket box and a 2-socket box have the same kernel/watcher
//!   load, so the same single free physical core suffices for both.
//!   The realistic NUMA cost — NIC softirqs from a remote socket
//!   preempting worker pairs because the free core is on the other
//!   socket — is ~1% throughput on the affected socket at production
//!   interrupt rates, not worth burning a second physical core to
//!   avoid.
//!
//! Linux-only at runtime. `pick_core` returns `None`, and `pin_thread_to`
//! and `pin_child_to` no-op on other platforms — macOS in particular has
//! no `sched_setaffinity` equivalent, and we don't need to pin during
//! local dev tests.
//!
//! ## API ordering
//!
//! Callers should pin the worker thread BEFORE spawning Stockfish, so the
//! child inherits affinity via `fork()` and does its NNUE init / hash-table
//! allocation on the target core's L1/L2. After spawn, `pin_child` is a
//! defensive re-pin — typically a no-op since the child already inherited,
//! but it confirms the pinning if the kernel scheduled the child elsewhere
//! during the brief unpinned window between fork and the parent's first
//! sched_setaffinity (theoretical, but cheap to defend against).
//!
//! The single-call rule: resolve the core ONCE per worker via
//! `pick_core(worker_id)`, then pass the resulting `CoreId` to both pin
//! functions. Two independent `get_core_ids()` calls would risk seeing
//! different cpuset lists under cgroup mutation (vast.ai-style containers
//! occasionally retighten cpusets mid-run) and land the worker thread and
//! its Stockfish child on different cores.

/// Resolve the core this worker should pin to, using `worker_id % n_cores`.
/// Call ONCE per worker, then pass the `CoreId` to both `pin_thread_to`
/// (before spawn) and `pin_child_to` (after spawn). Returns `None` and
/// logs a warning if the OS reports no cores.
#[cfg(target_os = "linux")]
pub fn pick_core(worker_id: u32) -> Option<core_affinity::CoreId> {
    match core_affinity::get_core_ids() {
        Some(c) if !c.is_empty() => Some(c[(worker_id as usize) % c.len()]),
        _ => {
            eprintln!(
                "[affinity] worker {worker_id}: core_affinity::get_core_ids \
                 returned no cores; pinning will be skipped",
            );
            None
        }
    }
}

/// Stable `Option`-returning shape on non-Linux too, so callers don't need
/// `#[cfg]` blocks. Always returns `None` off Linux.
#[cfg(not(target_os = "linux"))]
pub fn pick_core(_worker_id: u32) -> Option<()> {
    None
}

/// Pin the calling thread to `core`. Best-effort: failures are logged.
/// Call before spawning Stockfish so the child inherits the affinity via
/// `fork()` and runs its NNUE init on the target core's cache.
#[cfg(target_os = "linux")]
pub fn pin_thread_to(core: core_affinity::CoreId, worker_id: u32) {
    if !core_affinity::set_for_current(core) {
        eprintln!(
            "[affinity] worker {worker_id}: set_for_current({core:?}) failed; \
             thread will run on whatever core the scheduler picks",
        );
    }
}

/// No-op on non-Linux. The signature uses `()` because `pick_core` returns
/// `Option<()>` off Linux, so callers can write `if let Some(c) = pick_core(id)
/// { pin_thread_to(c, id); }` portably without `#[cfg]` blocks.
#[cfg(not(target_os = "linux"))]
pub fn pin_thread_to(_core: (), _worker_id: u32) {}

/// Pin a child process to `core`. Defensive re-pin — typically a no-op
/// because the child already inherited the parent's pin via `fork()`.
#[cfg(target_os = "linux")]
pub fn pin_child_to(pid: u32, core: core_affinity::CoreId, worker_id: u32) {
    let core_id = core.id;
    // Build a single-CPU mask. cpu_set_t is opaque; zero it out then
    // set the target bit via CPU_SET. SAFETY: cpu_set_t is POD; the
    // libc helpers don't escape pointers; sched_setaffinity is the
    // canonical Linux syscall for this.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(core_id, &mut set);
        let rc = libc::sched_setaffinity(
            pid as libc::pid_t,
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        );
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            eprintln!(
                "[affinity] worker {worker_id}: sched_setaffinity(pid={pid}, core={core_id}) \
                 failed: {err}; child will run on whatever core the scheduler picks",
            );
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_child_to(_pid: u32, _core: (), _worker_id: u32) {}
