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
        "ram_mb": offer.get("cpu_ram", 0),
        "disk_gb": offer.get("disk_space", 0),
        "reliability": offer.get("reliability", 0),
        "verification": offer.get("verification", ""),
        "geo": offer.get("geolocation", ""),
        "num_gpus": offer.get("num_gpus", 0),
        "gpu_name": offer.get("gpu_name", ""),
        "rentable": offer.get("rentable", False),
    }, None


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
        f"{'CPU':<28} {'src':<20} "
        f"{'RAM/GB':>7} {'reliab':>7} {'verif':<11} {'geo':<22}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, s in enumerate(scored, 1):
        threads = f"{s['threads_eff']}/{s['threads_total']}"
        cpu_short = re.sub(r"\s+\d+-Core Processor$", "", s["cpu"])[:28]
        print(
            f"{i:>3} {s['id']:>10} {s['dollars_per_kscore_hr']:>10.4f} "
            f"{s['dph']:>7.3f} {s['effective_score']:>9.0f} "
            f"{threads:>10} {s['sockets']:>4} "
            f"{cpu_short:<28} {s['score_source']:<20} "
            f"{s['ram_mb'] / 1024:>7.0f} {s['reliability']:>7.3f} "
            f"{s['verification']:<11} {s['geo']:<22}"
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

    scored.sort(key=lambda x: x["dollars_per_kscore_hr"])

    if args.json:
        print(json.dumps(scored[: args.top], indent=2))
    else:
        print(
            f"\n[result] scored {len(scored)}/{len(offers)} offers "
            f"(skip reasons: {skip_reasons})\n",
            file=sys.stderr,
        )
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
