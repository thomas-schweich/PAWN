//! NUMA memory-policy helpers. The only one we use today is
//! `set_interleave_all_nodes()`, called at startup so DRAM allocations
//! (and, transitively, page-cache fills) round-robin across NUMA nodes.
//!
//! ## Why we need this
//!
//! On a dual-socket EPYC pod the patched stockfish binary's NNUE weights
//! (~50 MB rodata) live in the kernel's page cache. The first stockfish
//! worker to fault those pages in determines their physical placement;
//! every other worker then either shares those local pages (same socket,
//! ~85 ns DRAM latency) or pays a 3.2× penalty crossing the inter-socket
//! interconnect (~270 ns). Roughly half the workers eat that penalty by
//! default — first-touch is socket-dependent and the cache is a single
//! shared inode.
//!
//! `MPOL_INTERLEAVE` round-robins fresh page allocations across nodes
//! per the policy mask. `set_mempolicy` is inherited across `execve` and
//! `fork`, so a `set_mempolicy(MPOL_INTERLEAVE, all_nodes)` call in the
//! rust binary's `main` propagates to every stockfish child it spawns.
//! Each worker's first-touch on a fresh NNUE page therefore lands on the
//! next node in rotation, evenly spreading the page set across both
//! sockets. Workers' subsequent reads of pages first-touched by other
//! workers still cross sockets, but on average every worker pays half
//! the cross-socket penalty instead of half the workers paying all of it.
//!
//! Single-socket pods see this as a no-op (the policy mask has one node,
//! interleave degenerates to local). NUMA-unaware hosts (or
//! `set_mempolicy` returning ENOSYS) are non-fatal: log and continue.
//!
//! See `ANALYSIS.md` "Phase 5 — NUMA wrapping" for the full rationale.

// x86_64-specific syscall numbers — Linux's syscall numbering is
// per-architecture. aarch64 uses 237, riscv uses different numbers, etc.
// Restrict to x86_64 + Linux; the no-op stub catches every other host.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod imp {
    // `mempolicy.h` constants. Hardcoded rather than depending on the
    // `nix` crate or a custom build script — the values are part of the
    // kernel ABI and have been stable since the syscall was introduced.
    const MPOL_INTERLEAVE: libc::c_int = 3;
    const SYS_SET_MEMPOLICY: libc::c_long = 238;

    /// Configure this process (and inherited children) to interleave new
    /// page allocations across every NUMA node available to the process.
    /// Best-effort: prints a single-line note on failure and returns —
    /// callers shouldn't abort because NUMA was missing.
    pub fn set_interleave_all_nodes() {
        // Discover the configured node count from `/sys/devices/system/node`.
        // Cheaper and more reliable than `numa_max_node()` (which requires
        // `libnuma`); the kernel always exposes one `node<N>` directory per
        // configured node.
        let mut max_node: usize = 0;
        let entries = match std::fs::read_dir("/sys/devices/system/node") {
            Ok(e) => e,
            Err(_) => {
                // Not on Linux's standard sysfs layout — skip silently.
                return;
            }
        };
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(rest) = name.strip_prefix("node") {
                    if let Ok(n) = rest.parse::<usize>() {
                        max_node = max_node.max(n);
                    }
                }
            }
        }

        // Build the nodemask bitmap covering nodes 0..=max_node. Each u64
        // covers 64 nodes; a dual-socket EPYC has 2 (or 16 if NPS=4), so
        // one u64 is plenty for foreseeable pods. Defensive: round up the
        // word count.
        let n_nodes = max_node + 1;
        let n_words = n_nodes.div_ceil(64).max(1);
        let mut mask: Vec<u64> = vec![0; n_words];
        for n in 0..n_nodes {
            mask[n / 64] |= 1u64 << (n % 64);
        }
        // The kernel's nodemask uses an EXCLUSIVE upper bound: `maxnode`
        // is the highest valid node bit *plus one* (in bits, not bytes).
        // `n_nodes` already represents that "one past the last" count.
        let maxnode_arg = n_nodes as libc::c_ulong;

        let rc = unsafe {
            libc::syscall(
                SYS_SET_MEMPOLICY,
                MPOL_INTERLEAVE as libc::c_long,
                mask.as_ptr() as libc::c_long,
                maxnode_arg as libc::c_long,
            )
        };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            eprintln!(
                "[numa] set_mempolicy(MPOL_INTERLEAVE, nodes={n_nodes}) failed: {err}; \
                 NUMA placement will fall back to first-touch (no-op on single-socket pods)"
            );
        } else if n_nodes >= 2 {
            eprintln!(
                "[numa] mempolicy=MPOL_INTERLEAVE over {n_nodes} NUMA node(s); \
                 stockfish children inherit this and spread NNUE first-touches evenly"
            );
        }
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod imp {
    pub fn set_interleave_all_nodes() {
        // No-op everywhere except Linux x86_64: the syscall number we
        // hardcode is x86_64-specific (aarch64 Linux uses 237, etc.) and
        // macOS / Windows don't have `set_mempolicy` at all. Production
        // pods are linux/amd64, so this is the only path that ever
        // executes the real syscall; the stub keeps tests building on
        // macOS dev machines and on a hypothetical aarch64 host.
    }
}

pub use imp::set_interleave_all_nodes;
