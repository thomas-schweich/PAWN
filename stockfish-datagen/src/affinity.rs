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
//! Linux-only at runtime. Both `pin_current_thread` and `pin_pid` are
//! no-ops on other platforms — macOS in particular has no
//! `sched_setaffinity` equivalent, and we don't need to pin during
//! local dev tests.

/// Pin the calling thread to core `worker_id % n_cores`. Best-effort:
/// failures are logged via `eprintln!` and otherwise ignored.
pub fn pin_current_thread(worker_id: u32) {
    #[cfg(target_os = "linux")]
    {
        let cores = match core_affinity::get_core_ids() {
            Some(c) if !c.is_empty() => c,
            _ => {
                eprintln!(
                    "[affinity] worker {worker_id}: core_affinity::get_core_ids \
                     returned no cores; skipping thread pin",
                );
                return;
            }
        };
        let pick = cores[(worker_id as usize) % cores.len()];
        if !core_affinity::set_for_current(pick) {
            eprintln!(
                "[affinity] worker {worker_id}: set_for_current({pick:?}) failed; \
                 thread will run on whatever core the scheduler picks",
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = worker_id;
    }
}

/// Pin process `pid` to core `worker_id % n_cores`. Best-effort: a
/// failure here just leaves the child where the OS scheduler put it,
/// which is the pre-pinning baseline.
pub fn pin_pid(pid: u32, worker_id: u32) {
    #[cfg(target_os = "linux")]
    {
        let cores = match core_affinity::get_core_ids() {
            Some(c) if !c.is_empty() => c,
            _ => return, // already logged in pin_current_thread
        };
        let core = cores[(worker_id as usize) % cores.len()].id;

        // Build a single-CPU mask. cpu_set_t is opaque; zero it out then
        // set the target bit via CPU_SET. SAFETY: cpu_set_t is POD; the
        // libc helpers don't escape pointers; sched_setaffinity is the
        // canonical Linux syscall for this.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(core, &mut set);
            let rc = libc::sched_setaffinity(
                pid as libc::pid_t,
                std::mem::size_of::<libc::cpu_set_t>(),
                &set,
            );
            if rc != 0 {
                let err = std::io::Error::last_os_error();
                eprintln!(
                    "[affinity] worker {worker_id}: sched_setaffinity(pid={pid}, core={core}) \
                     failed: {err}; child will run on whatever core the scheduler picks",
                );
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (pid, worker_id);
    }
}
