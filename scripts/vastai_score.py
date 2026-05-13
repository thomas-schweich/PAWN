#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "beautifulsoup4"]
# ///
"""
Score vast.ai CPU offers by PassMark multi-thread CPU Mark.

We're shopping for CPU-bound workloads (stockfish-datagen — many independent
SF workers, each pinned to its own thread). The right figure of merit is
aggregate multi-thread throughput per dollar-hour, accounting for two things
the raw `vastai search offers` output doesn't surface:

  1. **Partial container residency.** vast.ai allocates CPU proportional to
     the GPU fraction rented (`cpu_cores_effective` ≈ `cpu_cores × gpu_frac`).
     A 384-thread host with 8 GPUs gives you only ~48 threads if you rent
     one GPU. PassMark scores the whole chip, not the slice.
  2. **Multi-socket motherboards.** Many EPYC offers are dual-socket
     (96-core × 2 sockets × 2 SMT = 384 threads). PassMark's CPU Mark is
     per-chip; two chips on one board scale ~linearly at high thread counts.

Effective score = chip_passmark × n_sockets × (cpu_cores_effective / cpu_cores)
n_sockets is inferred from total threads / (cores_per_chip × SMT_2) where
cores_per_chip is parsed from the vast.ai `cpu_name` ("AMD EPYC 9654 96-Core
Processor" → 96). Falls back gracefully if the name has no core count.

Cached PassMark scrape under ~/.cache/pawn/passmark.json (30-day TTL).
Use --refresh to force a re-fetch.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# PassMark's per-tier ranked charts. Each `<li id="rk...">` carries a
# `<span class="prdname">` (CPU name) and `<span class="count">` (CPU Mark).
# Single-CPU pages first, then `multi_cpu.html` for measured multi-socket
# (`[Dual CPU] …` / `[Quad CPU] …` entries with rk ids like `rk####_2`).
# We merge the lot into one dict keyed by `normalize_name(prdname)`.
PASSMARK_URLS = [
    "https://www.cpubenchmark.net/high_end_cpus.html",
    "https://www.cpubenchmark.net/mid_range_cpus.html",
    "https://www.cpubenchmark.net/low_end_cpus.html",
    "https://www.cpubenchmark.net/multi_cpu.html",
]
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
CACHE_DIR = Path.home() / ".cache" / "pawn"
CACHE_FILE = CACHE_DIR / "passmark.json"
CACHE_TTL_SEC = 30 * 86400

DEFAULT_QUERY = (
    "cpu_cores>=128 cpu_cores_effective>=64 reliability>=0.97 "
    "rentable=true verified=any"
)


_ONCLICK_RE = re.compile(
    r"x\(event,\s*\d+,\s*\d+,\s*(?P<cores>\d+),\s*(?P<smt>\d+)"
)


def _parse_passmark_page(html: str) -> dict[str, dict]:
    """
    Parse one PassMark tier page. Each CPU is rendered as

        <li id="rk####"><span class="more_details"
                           onclick="x(event, 24, 45, 96, 2, '...', null, null);"></span>
          <a href="/cpu.php?cpu=...">
            <span class="prdname">CPU Name</span>
            <div>...</div>
            <span class="count">119,259</span>
            ...
          </a>
        </li>

    `count` is the multi-thread CPU Mark. The `onclick` args carry
    `(event, rank_today, rank_3mo_avg, cores_per_chip, threads_per_core,
    price, ...)` — we keep `cores` and `smt` so the scorer can recover
    socket count when the vast.ai `cpu_name` is missing the `X-Core`
    suffix.

    `multi_cpu.html` uses `id="rk####_2"` for dual-CPU rows (with a
    `[Dual CPU] ` name prefix); the regex below accepts both forms.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, dict] = {}
    for li in soup.find_all("li", id=re.compile(r"^rk\d+(_\d+)?$")):
        name_el = li.find("span", class_="prdname")
        mark_el = li.find("span", class_="count")
        details_el = li.find("span", class_="more_details")
        if name_el is None or mark_el is None:
            continue
        name = name_el.get_text(strip=True)
        mark_text = mark_el.get_text(strip=True).replace(",", "")
        try:
            mark = int(mark_text)
        except ValueError:
            continue
        if not name:
            continue
        cores: int | None = None
        smt: int | None = None
        if details_el is not None:
            onclick = details_el.get("onclick", "")
            m = _ONCLICK_RE.search(onclick)
            if m:
                cores = int(m.group("cores"))
                smt = int(m.group("smt"))
        out[normalize_name(name)] = {
            "mark": mark, "cores": cores, "smt": smt,
        }
    return out


def fetch_passmark(refresh: bool = False) -> dict[str, dict]:
    """
    Return `{normalized_cpu_name: {mark, cores, smt}}` merged across
    PassMark's tier pages and `multi_cpu.html`.
    """
    if CACHE_FILE.exists() and not refresh:
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL_SEC:
            data = json.loads(CACHE_FILE.read_text())
            # Old single-int cache format from earlier iterations: re-fetch.
            sample = next(iter(data.values()), None) if data else None
            if isinstance(sample, dict):
                print(
                    f"[passmark] cache hit: {len(data)} CPUs "
                    f"(age {age / 86400:.1f} days)",
                    file=sys.stderr,
                )
                return data
            print(
                "[passmark] cache schema is stale, re-fetching",
                file=sys.stderr,
            )

    cpus: dict[str, dict] = {}
    for url in PASSMARK_URLS:
        print(f"[passmark] fetching {url}...", file=sys.stderr)
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        page = _parse_passmark_page(r.text)
        print(f"[passmark]   {len(page)} CPUs from this tier", file=sys.stderr)
        # First page that lists a CPU wins (high_end > mid > low > multi).
        for k, v in page.items():
            cpus.setdefault(k, v)

    if not cpus:
        raise RuntimeError("PassMark: parsed 0 CPUs; selectors may be stale.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cpus))
    print(f"[passmark] cached {len(cpus)} CPUs.", file=sys.stderr)
    return cpus


SOCKET_PREFIX_TO_N = {"dual": 2, "tri": 3, "quad": 4, "octa": 8, "octal": 8}


def normalize_name(name: str) -> str:
    """
    Canonicalize a CPU name for fuzzy matching across vast.ai and PassMark.

    Single-chip:
        vast.ai  "AMD EPYC 9654 96-Core Processor" → "amd epyc 9654"
        PassMark "AMD EPYC 9654"                   → "amd epyc 9654"

    Multi-socket (only PassMark's `multi_cpu.html` page uses this form):
        PassMark "[Dual CPU] AMD EPYC 9654"        → "2x amd epyc 9654"
        PassMark "[Quad CPU] Intel Xeon Platinum 8570" → "4x intel xeon platinum 8570"

    Folding the `[Dual CPU] …` prefix down to `Nx …` *before* the noise
    stripper matters: the stripper kills the literal token `CPU`, which
    would otherwise mangle `[Dual CPU]` into `[Dual ]`.
    """
    n = name
    m = re.match(r"^\[\s*(\w+)\s+CPU\s*\]\s*", n, flags=re.IGNORECASE)
    if m:
        prefix_word = m.group(1).lower()
        n_sockets = SOCKET_PREFIX_TO_N.get(prefix_word)
        if n_sockets is not None:
            n = f"{n_sockets}x " + n[m.end():]
    # vast.ai sometimes inserts ® or ™ inside cpu_name (e.g. "Xeon® Gold 6430").
    # PassMark uses no such glyphs. Strip them before the rest of the pipeline.
    n = re.sub(r"[®™©]", "", n)
    n = re.sub(r"\b\d+[- ]?Core\b", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\b(Processor|CPU)\b", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s+@\s*\d+(\.\d+)?\s*GHz", "", n, flags=re.IGNORECASE)
    n = re.sub(r"[(),\[\]]", " ", n)
    n = re.sub(r"\s+", " ", n).strip().lower()
    # vast.ai's Intel listings drop "Intel" sometimes ("Xeon® Gold 6430"),
    # PassMark always carries it. Prepend "intel" if the bare prefix matches
    # known Intel families — the lookup ladder will accept either.
    if re.match(r"^(xeon|core|pentium|celeron|atom|itanium)\b", n):
        n = "intel " + n
    return n


def extract_physical_cores(name: str) -> int | None:
    m = re.search(r"(\d+)[- ]?Core", name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def lookup_cpu_entry(
    name: str, passmark: dict[str, dict]
) -> dict | None:
    """
    Look up a PassMark entry (mark + cores + smt), with a small ladder of
    fuzzy fallbacks:

      1. exact normalized-name match
      2. progressively shorter prefixes (drops trailing tokens until ≥2 left)
      3. for hyperscaler SKUs that PassMark only lists in dual-/quad-socket
         form (e.g. AMD EPYC 7Y83 → "[Dual CPU] AMD EPYC 7Y83"), synthesize
         a single-chip score from `measured_multi / n_sockets`. Mark the
         result so the caller can annotate the row.
    """
    def _try(key: str) -> dict | None:
        return passmark.get(key)

    norm = normalize_name(name)
    if (e := _try(norm)) is not None:
        return e
    parts = norm.split()
    while len(parts) > 2:
        parts.pop()
        if (e := _try(" ".join(parts))) is not None:
            return e

    for n_attempt in (2, 4, 8):
        multi = _try(f"{n_attempt}x {norm}")
        if multi is not None:
            return {
                "mark": multi["mark"] // n_attempt,
                "cores": multi.get("cores"),
                "smt": multi.get("smt"),
                "synthesized_from_multi": n_attempt,
            }
    return None


def score_offer(
    offer: dict, passmark: dict[str, dict], scaling: str
) -> tuple[dict | None, str | None]:
    """
    Score one vast.ai offer.

    `scaling` ∈ {"measured", "linear"}:
      - `measured` (default): when n_sockets ≥ 2, prefer PassMark's actual
        `[Dual CPU]` / `[Quad CPU]` benchmark from `multi_cpu.html`; fall
        back to single × n_sockets if no measured entry exists.
      - `linear`: always use single_chip × n_sockets (the right model for
        embarrassingly-parallel workloads like stockfish-datagen, where
        cross-socket bandwidth is barely touched).
    """
    cpu_name = offer.get("cpu_name") or ""
    if not cpu_name:
        return None, "no cpu_name"

    entry = lookup_cpu_entry(cpu_name, passmark)
    if entry is None:
        return None, f"no PassMark match: {cpu_name}"
    cpu_mark_single = entry["mark"]
    single_synthesized_from = entry.get("synthesized_from_multi")

    total_threads = offer.get("cpu_cores") or 0
    effective_threads = offer.get("cpu_cores_effective") or 0
    if not total_threads or not effective_threads:
        return None, "missing thread count"

    # Determine cores-per-chip. vast.ai usually carries it in the cpu_name
    # ("AMD EPYC 9654 96-Core Processor"); when missing, fall back to the
    # PassMark metadata (parsed from the `more_details` onclick payload).
    cores_per_chip = extract_physical_cores(cpu_name) or entry.get("cores")
    smt = entry.get("smt") or 2

    # Detect socket count from total threads vs. per-chip threads.
    if cores_per_chip:
        threads_per_chip = cores_per_chip * smt
        n_sockets = max(1, round(total_threads / threads_per_chip))
        # Sanity: if host has SMT disabled, total threads == cores_per_chip × N
        if n_sockets * threads_per_chip < total_threads * 0.5:
            n_sockets = max(1, round(total_threads / cores_per_chip))
    else:
        n_sockets = 1  # conservative: assume single chip

    # Host (full machine) score: measured if available and applicable.
    cpu_mark_measured: int | None = None
    score_source = "single chip"
    if n_sockets >= 2:
        base_norm = normalize_name(cpu_name)
        measured_entry = passmark.get(f"{n_sockets}x {base_norm}")
        if measured_entry is not None:
            cpu_mark_measured = measured_entry["mark"]
        if cpu_mark_measured is not None and scaling == "measured":
            host_score = cpu_mark_measured
            score_source = f"{n_sockets}-socket measured"
        else:
            host_score = cpu_mark_single * n_sockets
            score_source = f"single×{n_sockets} (linear)"

    else:
        host_score = cpu_mark_single
        if single_synthesized_from:
            score_source = f"single ≈ ÷{single_synthesized_from}-socket"

    effective_share = effective_threads / total_threads
    effective_score = host_score * effective_share

    price = offer.get("dph_total") or 0
    if not price or effective_score <= 0:
        return None, "zero price or score"

    up_cost_per_tb = offer.get("internet_up_cost_per_tb") or 0.0
    down_cost_per_tb = offer.get("internet_down_cost_per_tb") or 0.0
    inet_up_mbps = offer.get("inet_up") or 0.0
    inet_down_mbps = offer.get("inet_down") or 0.0

    return {
        "id": offer["id"],
        "cpu": cpu_name,
        "passmark_chip": cpu_mark_single,
        "passmark_measured_multi": cpu_mark_measured,
        "cores_per_chip": cores_per_chip,
        "score_source": score_source,
        "sockets": n_sockets,
        "threads_eff": effective_threads,
        "threads_total": total_threads,
        "host_score": host_score,
        "effective_score": effective_score,
        "dph": price,
        "dollars_per_kscore_hr": price / (effective_score / 1000),
        "up_cost_per_tb": up_cost_per_tb,
        "down_cost_per_tb": down_cost_per_tb,
        "inet_up_mbps": inet_up_mbps,
        "inet_down_mbps": inet_down_mbps,
        "ram_mb": offer.get("cpu_ram", 0),
        "disk_gb": offer.get("disk_space", 0),
        "reliability": offer.get("reliability", 0),
        "verification": offer.get("verification", ""),
        "geo": offer.get("geolocation", ""),
        "num_gpus": offer.get("num_gpus", 0),
        "gpu_name": offer.get("gpu_name", ""),
        "rentable": offer.get("rentable", False),
    }, None


_DUAL_WORDS = {"dual", "2-socket", "2socket"}
_QUAD_WORDS = {"quad", "4-socket", "4socket"}


def lookup_anchor_cpu(
    user_string: str, passmark: dict[str, dict]
) -> dict | None:
    """
    Permissive PassMark lookup for the --anchor-cpu string. The user might
    type "EPYC 9654 dual", "dual EPYC 9654", "2x EPYC 9654", or "AMD EPYC
    9654" — all should resolve to PassMark's "[Dual CPU] AMD EPYC 9654".

    Strategy: detect a multi-socket hint (word or `Nx` prefix), strip it,
    then run the base name through the standard `lookup_cpu_entry` ladder.
    Re-prepend `Nx ` when looking in the multi-socket pool. Also try with
    an "AMD " prefix since the user may omit it for AMD parts.
    """
    n = normalize_name(user_string)
    n_sockets = 1

    m = re.match(r"^(\d+)x\s+(.*)", n)
    if m:
        n_sockets = int(m.group(1))
        bare = m.group(2)
    else:
        tokens = n.split()
        bare_tokens: list[str] = []
        for t in tokens:
            if t in _DUAL_WORDS:
                n_sockets = max(n_sockets, 2)
            elif t in _QUAD_WORDS:
                n_sockets = max(n_sockets, 4)
            else:
                bare_tokens.append(t)
        bare = " ".join(bare_tokens)

    candidates: list[str] = []
    if n_sockets > 1:
        candidates += [f"{n_sockets}x {bare}", f"{n_sockets}x amd {bare}"]
    candidates += [bare, f"amd {bare}"]

    for key in candidates:
        entry = passmark.get(key)
        if entry is not None:
            return entry
        # Prefix-shortened fallback through the standard ladder.
        entry = lookup_cpu_entry(key, passmark)
        if entry is not None:
            return entry
    return None


def resolve_workload_size(
    *,
    target_tb: float,
    total_kscore_hours: float | None,
    anchor_cpu: str | None,
    anchor_hours: float | None,
    data_gb_per_kscore_h: float,
    passmark: dict[str, dict],
) -> tuple[float, str]:
    """
    Collapse the three workload-size input forms into a single number.

    Precedence: explicit `--total-kscore-hours` > anchor (cpu + hours) > rate.

    Returns `(total_kscore_hours, human-readable_note)` for the run header.
    """
    if total_kscore_hours is not None:
        return total_kscore_hours, (
            f"workload anchor: --total-kscore-hours={total_kscore_hours:.1f} "
            "(direct)"
        )

    if anchor_cpu is not None or anchor_hours is not None:
        if anchor_cpu is None or anchor_hours is None:
            raise SystemExit(
                "--anchor-cpu and --anchor-hours must be set together"
            )
        entry = lookup_anchor_cpu(anchor_cpu, passmark)
        if entry is None:
            raise SystemExit(
                f"--anchor-cpu {anchor_cpu!r} did not match any PassMark "
                "entry. Try forms like 'AMD EPYC 9654', 'EPYC 9654 dual', "
                "'2x EPYC 9654', 'Threadripper PRO 7995WX'."
            )
        anchor_mark = entry["mark"]
        anchor_kscore = anchor_mark / 1000.0
        kscore_hours = anchor_hours * anchor_kscore
        return kscore_hours, (
            f"workload anchor: {anchor_hours:.1f}h on {anchor_cpu!r} "
            f"(PassMark {anchor_mark}, {anchor_kscore:.2f} kscore) "
            f"→ {kscore_hours:.1f} kscore-h total"
        )

    # Fallback: GB-per-kscore-hour rate.
    rate = data_gb_per_kscore_h
    kscore_hours = (target_tb * 1024.0) / rate
    return kscore_hours, (
        f"workload anchor: --data-gb-per-kscore-h={rate} × "
        f"{target_tb} TB → {kscore_hours:.1f} kscore-h total "
        "(loose default — prefer an empirical anchor)"
    )


def add_target_tb_estimates(
    scored: list[dict],
    target_tb: float,
    total_kscore_hours: float,
    download_gb: float,
) -> None:
    """
    Project the total dollar cost of producing `target_tb` of output on each
    offer, given the total CPU work in kscore-hours required to produce that
    much data (`total_kscore_hours`).

    `total_kscore_hours` is the one knob that encodes "how heavy the workload
    is." It can be derived from any of:
      - direct: `--total-kscore-hours K`
      - empirical anchor: `--anchor-cpu CPU --anchor-hours H`
            → W = H × PassMark(CPU) / 1000
      - calibration rate: `--data-gb-per-kscore-h C`
            → W = target_tb × 1024 / C

    Per-machine math (S = effective_score / 1000):
      hours_X      = W / S
      compute_$    = dph_X × hours_X
      upload_$     = target_tb × up_cost_per_tb_X      (exact, no calibration)
      download_$   = (download_gb / 1024) × down_cost_per_tb_X
      total_$      = compute + upload + download

    Bandwidth check: req Mbps = (target_GB × 1024 × 8) / (hours × 3600).
    Flag offers where req > 0.7 × inet_up_mbps.
    """
    target_download_tb = download_gb / 1024.0
    target_gb = target_tb * 1024.0
    for s in scored:
        effective_kscore = s["effective_score"] / 1000.0
        if effective_kscore <= 0 or total_kscore_hours <= 0:
            continue
        hours = total_kscore_hours / effective_kscore
        if hours <= 0:
            continue
        compute = s["dph"] * hours
        upload = target_tb * s["up_cost_per_tb"]
        download = target_download_tb * s["down_cost_per_tb"]
        total = compute + upload + download
        # Required sustained egress Mbps to clear `target_tb` in `hours`.
        required_up_mbps = (target_gb * 1024 * 8) / (hours * 3600)
        bw_limited = (
            s["inet_up_mbps"] > 0
            and required_up_mbps > 0.7 * s["inet_up_mbps"]
        )
        s.update({
            "est_hours": hours,
            "est_compute_cost": compute,
            "est_upload_cost": upload,
            "est_download_cost": download,
            "est_total_cost": total,
            "est_total_dollars_per_tb": total / target_tb,
            "required_up_mbps": required_up_mbps,
            "bw_limited": bw_limited,
        })


def query_vastai(query: str) -> list[dict]:
    print(f"[vastai] search offers: {query}", file=sys.stderr)
    res = subprocess.run(
        ["vastai", "search", "offers", query, "-o", "dph_total", "--raw"],
        capture_output=True,
        text=True,
        check=True,
    )
    offers = json.loads(res.stdout)
    print(f"[vastai] got {len(offers)} offers", file=sys.stderr)
    return offers


def print_table(scored: list[dict], top: int) -> None:
    scored = scored[:top]
    if not scored:
        print("no matches", file=sys.stderr)
        return
    hdr = (
        f"{'#':>3} {'id':>10} {'$/kscore':>10} {'$/h':>7} "
        f"{'score':>9} {'threads':>10} {'sock':>4} "
        f"{'CPU':<26} "
        f"{'up$/TB':>7} {'upMbps':>7} "
        f"{'RAM/GB':>7} {'reliab':>7} {'verif':<11} {'geo':<22}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, s in enumerate(scored, 1):
        threads = f"{s['threads_eff']}/{s['threads_total']}"
        cpu_short = re.sub(r"\s+\d+-Core Processor$", "", s["cpu"])[:26]
        print(
            f"{i:>3} {s['id']:>10} {s['dollars_per_kscore_hr']:>10.4f} "
            f"{s['dph']:>7.3f} {s['effective_score']:>9.0f} "
            f"{threads:>10} {s['sockets']:>4} "
            f"{cpu_short:<26} "
            f"{s['up_cost_per_tb']:>7.2f} {s['inet_up_mbps']:>7.0f} "
            f"{s['ram_mb'] / 1024:>7.0f} {s['reliability']:>7.3f} "
            f"{s['verification']:<11} {s['geo']:<22}"
        )


def print_target_tb_table(
    scored: list[dict], top: int, target_tb: float, download_gb: float
) -> None:
    scored = scored[:top]
    if not scored:
        print("no matches", file=sys.stderr)
        return
    hdr = (
        f"{'#':>3} {'id':>10} {'$/TB':>7} {'total$':>7} {'hours':>6} "
        f"{'comp$':>6} {'up$':>5} {'dn$':>5} "
        f"{'CPU':<26} {'reqMbps':>8} {'upMbps':>7} {'bw?':<4} "
        f"{'reliab':>7} {'verif':<11} {'geo':<22}"
    )
    print(
        f"# target: produce {target_tb:.2f} TB of output "
        f"+ {download_gb:.1f} GB ingress (image pulls etc.)"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, s in enumerate(scored, 1):
        cpu_short = re.sub(r"\s+\d+-Core Processor$", "", s["cpu"])[:26]
        bw_flag = "!!" if s["bw_limited"] else ""
        print(
            f"{i:>3} {s['id']:>10} {s['est_total_dollars_per_tb']:>7.2f} "
            f"{s['est_total_cost']:>7.2f} {s['est_hours']:>6.1f} "
            f"{s['est_compute_cost']:>6.2f} {s['est_upload_cost']:>5.2f} "
            f"{s['est_download_cost']:>5.2f} "
            f"{cpu_short:<26} {s['required_up_mbps']:>8.1f} "
            f"{s['inet_up_mbps']:>7.0f} {bw_flag:<4} "
            f"{s['reliability']:>7.3f} {s['verification']:<11} "
            f"{s['geo']:<22}"
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--query", default=DEFAULT_QUERY,
        help=f"vast.ai search filter (default: {DEFAULT_QUERY!r})",
    )
    p.add_argument(
        "--top", type=int, default=20, help="show top N offers (default 20)",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="force re-fetch of PassMark cache",
    )
    p.add_argument(
        "--json", action="store_true",
        help="emit scored offers as JSON instead of a table",
    )
    p.add_argument(
        "--show-unmatched", action="store_true",
        help="print all CPU names that failed PassMark lookup",
    )
    p.add_argument(
        "--scaling", choices=["measured", "linear"], default="measured",
        help=(
            "multi-socket scaling. 'measured' uses PassMark's [Dual CPU]/[Quad "
            "CPU] benchmarks (a mix of bandwidth-bound and compute-bound work, "
            "scales sub-linearly). 'linear' uses single_chip × n_sockets, "
            "appropriate for embarrassingly-parallel workloads like "
            "stockfish-datagen."
        ),
    )
    p.add_argument(
        "--target-tb", type=float, default=None,
        help=(
            "if set, project total $ to produce this many TB of output "
            "(compute + upload + download), sort by total $/TB, and add a "
            "bandwidth-bottleneck check. Requires one workload-size knob: "
            "--total-kscore-hours OR (--anchor-cpu + --anchor-hours) OR "
            "--data-gb-per-kscore-h. Anchor mode is preferred — it pins the "
            "workload to a single empirical observation instead of a "
            "two-unknown rate guess."
        ),
    )
    p.add_argument(
        "--total-kscore-hours", type=float, default=None,
        help=(
            "direct workload size: total CPU work (in kscore-hours) needed "
            "to produce --target-tb of output. Once you've run the workload "
            "once and measured (compute_hours × kscore), this is the most "
            "honest number to pass forward."
        ),
    )
    p.add_argument(
        "--anchor-cpu", type=str, default=None,
        help=(
            "empirical-anchor mode: a CPU description (e.g. 'EPYC 9654 dual', "
            "'AMD EPYC 7B13', '2x EPYC 9654') used to look up PassMark. "
            "Paired with --anchor-hours, this says 'the --target-tb workload "
            "takes H hours on this CPU'. The script then back-derives "
            "total_kscore_hours = H × PassMark(CPU) / 1000 and projects every "
            "other offer from there."
        ),
    )
    p.add_argument(
        "--anchor-hours", type=float, default=None,
        help="hours the --target-tb workload takes on --anchor-cpu",
    )
    p.add_argument(
        "--data-gb-per-kscore-h", type=float, default=0.1,
        help=(
            "fallback workload calibration when no anchor or direct kscore-h "
            "is provided: GB of output produced per kscore-hour. Default 0.1, "
            "a rough order-of-magnitude estimate for stockfish-datagen "
            "(evallegal mode, ~5 KB/game compressed, ~800 g/s on a dual "
            "EPYC 9654 at PassMark 141k ≈ 14.4 GB/h ÷ 141.8 kscore). Each "
            "factor in that chain is loose — prefer the anchor form once "
            "you have one real run to anchor against."
        ),
    )
    p.add_argument(
        "--download-gb", type=float, default=5.0,
        help=(
            "GB of ingress per run (Docker image pull, HF dataset prime). "
            "Default 5. Multiplied by the host's down-cost-per-TB."
        ),
    )
    args = p.parse_args()

    passmark = fetch_passmark(refresh=args.refresh)
    offers = query_vastai(args.query)

    scored: list[dict] = []
    skip_reasons: dict[str, int] = {}
    unmatched_cpus: list[str] = []
    for offer in offers:
        s, reason = score_offer(offer, passmark, args.scaling)
        if s is not None:
            scored.append(s)
        else:
            assert reason is not None
            skip_reasons[reason.split(":", 1)[0]] = (
                skip_reasons.get(reason.split(":", 1)[0], 0) + 1
            )
            if reason.startswith("no PassMark match"):
                unmatched_cpus.append(offer.get("cpu_name", "?"))

    if args.target_tb is not None:
        total_kscore_hours, anchor_note = resolve_workload_size(
            target_tb=args.target_tb,
            total_kscore_hours=args.total_kscore_hours,
            anchor_cpu=args.anchor_cpu,
            anchor_hours=args.anchor_hours,
            data_gb_per_kscore_h=args.data_gb_per_kscore_h,
            passmark=passmark,
        )
        print(f"[workload] {anchor_note}", file=sys.stderr)
        add_target_tb_estimates(
            scored,
            target_tb=args.target_tb,
            total_kscore_hours=total_kscore_hours,
            download_gb=args.download_gb,
        )
        scored.sort(key=lambda x: x.get("est_total_dollars_per_tb", float("inf")))
    else:
        scored.sort(key=lambda x: x["dollars_per_kscore_hr"])

    if args.json:
        print(json.dumps(scored[: args.top], indent=2))
    else:
        print(
            f"\n[result] scored {len(scored)}/{len(offers)} offers "
            f"(skip reasons: {skip_reasons})\n",
            file=sys.stderr,
        )
        if args.target_tb is not None:
            print_target_tb_table(
                scored, args.top, args.target_tb, args.download_gb
            )
        else:
            print_table(scored, args.top)

    if args.show_unmatched and unmatched_cpus:
        print(
            f"\n[unmatched] {len(set(unmatched_cpus))} unique CPU names "
            "had no PassMark entry:",
            file=sys.stderr,
        )
        for n in sorted(set(unmatched_cpus)):
            print(f"  {n}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
