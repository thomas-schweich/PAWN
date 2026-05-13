#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "beautifulsoup4"]
# ///
"""
Score vast.ai CPU offers by PassMark multi-thread CPU Mark.

Shopping criterion is aggregate multi-thread throughput per dollar-hour for
CPU-bound, embarrassingly-parallel workloads (stockfish-datagen — many
independent SF workers, each pinned to its own thread). The raw
`vastai search offers` output hides three things the script makes explicit:

  1. **Partial container residency.** vast.ai allocates CPU proportional to
     the GPU fraction rented (`cpu_cores_effective` ≈ `cpu_cores × gpu_frac`).
     A 384-thread host with 8 GPUs gives you only ~48 threads if you rent
     one GPU. PassMark scores the whole chip, not the slice.

  2. **Multi-socket motherboards.** vast.ai's `cpu_cores` is host-total.
     We infer socket count from `total_threads / (cores_per_chip × smt)`
     and use single_chip × n_sockets (linear). The linear assumption is
     correct for embarrassingly-parallel workloads with worker pinning;
     PassMark's own dual-CPU measurements are bandwidth-bound and
     under-credit parallel-friendly workloads.

  3. **Network egress.** At TB-scale, $/TB upload to HuggingFace can be the
     second-biggest line item after compute. `up$/TB` and `upMbps` columns
     are always shown.

For TB-scale runs, pass `--target-tb N` plus an empirical anchor
`--anchor-cpu CPU --anchor-hours H` ("this 1 TB workload takes ~H hours on
CPU"). The script back-derives total_kscore_hours = H × PassMark(CPU) ×
n_sockets / 1000 and projects every other listing's wall-clock and cost.
Anchor mode is mandatory for --target-tb — no silent calibration default.

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

# PassMark per-tier ranked charts. Each <li id="rk####"> renders:
#   <span class="more_details" onclick="x(event, NN, NN, CORES, SMT, …)"></span>
#   <a><span class="prdname">CPU Name</span> … <span class="count">123,456</span></a>
# `count` is the multi-thread CPU Mark; the onclick args carry cores-per-chip
# and threads-per-core, which we need when vast.ai's cpu_name omits the
# "X-Core Processor" suffix (e.g. some EPYC 9654 listings).
PASSMARK_URLS = [
    "https://www.cpubenchmark.net/high_end_cpus.html",
    "https://www.cpubenchmark.net/mid_range_cpus.html",
    "https://www.cpubenchmark.net/low_end_cpus.html",
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
_DUAL_WORDS = {"dual", "2-socket", "2socket"}
_QUAD_WORDS = {"quad", "4-socket", "4socket"}


def normalize_name(name: str) -> str:
    """
    Canonicalize a CPU name for fuzzy matching across vast.ai, PassMark, and
    --anchor-cpu user strings.

      vast.ai  "AMD EPYC 9654 96-Core Processor"  → "amd epyc 9654"
      vast.ai  "Xeon® Gold 6430"                  → "intel xeon gold 6430"
      user     "Threadripper PRO 7995WX 96-Core"  → "threadripper pro 7995wx"
    """
    n = re.sub(r"[®™©]", "", name)
    n = re.sub(r"\b\d+[- ]?Core\b", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\b(Processor|CPU)\b", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s+@\s*\d+(\.\d+)?\s*GHz", "", n, flags=re.IGNORECASE)
    n = re.sub(r"[(),\[\]]", " ", n)
    n = re.sub(r"\s+", " ", n).strip().lower()
    if re.match(r"^(xeon|core|pentium|celeron|atom|itanium)\b", n):
        n = "intel " + n
    return n


def extract_physical_cores(name: str) -> int | None:
    m = re.search(r"(\d+)[- ]?Core", name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_passmark_page(html: str) -> dict[str, dict]:
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, dict] = {}
    for li in soup.find_all("li", id=re.compile(r"^rk\d+$")):
        name_el = li.find("span", class_="prdname")
        mark_el = li.find("span", class_="count")
        details_el = li.find("span", class_="more_details")
        if name_el is None or mark_el is None:
            continue
        name = name_el.get_text(strip=True)
        if not name:
            continue
        try:
            mark = int(mark_el.get_text(strip=True).replace(",", ""))
        except ValueError:
            continue
        cores: int | None = None
        smt: int | None = None
        if details_el is not None:
            m = _ONCLICK_RE.search(details_el.get("onclick", ""))
            if m:
                cores = int(m.group("cores"))
                smt = int(m.group("smt"))
        out[normalize_name(name)] = {"mark": mark, "cores": cores, "smt": smt}
    return out


def fetch_passmark(refresh: bool = False) -> dict[str, dict]:
    """Return {normalized_cpu_name: {mark, cores, smt}} merged across tiers."""
    if CACHE_FILE.exists() and not refresh:
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL_SEC:
            data = json.loads(CACHE_FILE.read_text())
            sample = next(iter(data.values()), None) if data else None
            if isinstance(sample, dict):
                print(
                    f"[passmark] cache hit: {len(data)} CPUs "
                    f"(age {age / 86400:.1f} days)",
                    file=sys.stderr,
                )
                return data
            print(
                "[passmark] cache schema is stale, re-fetching", file=sys.stderr
            )

    cpus: dict[str, dict] = {}
    for url in PASSMARK_URLS:
        print(f"[passmark] fetching {url}...", file=sys.stderr)
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        page = _parse_passmark_page(r.text)
        print(f"[passmark]   {len(page)} CPUs from this tier", file=sys.stderr)
        for k, v in page.items():
            cpus.setdefault(k, v)

    if not cpus:
        raise RuntimeError("PassMark: parsed 0 CPUs; selectors may be stale.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cpus))
    print(f"[passmark] cached {len(cpus)} CPUs.", file=sys.stderr)
    return cpus


def lookup_cpu_entry(name: str, passmark: dict[str, dict]) -> dict | None:
    """
    Match a CPU name to a PassMark entry. Tries the exact normalized form,
    then progressively shorter prefixes (drops trailing tokens until ≥2 left),
    then with "amd " prepended.
    """
    norm = normalize_name(name)
    if (e := passmark.get(norm)) is not None:
        return e
    parts = norm.split()
    while len(parts) > 2:
        parts.pop()
        if (e := passmark.get(" ".join(parts))) is not None:
            return e
    if not norm.startswith("amd "):
        return passmark.get("amd " + norm)
    return None


def parse_anchor_cpu(
    user_string: str, passmark: dict[str, dict]
) -> tuple[dict, int] | None:
    """
    Resolve --anchor-cpu to (single_chip_entry, n_sockets). Accepts
    "EPYC 9654 dual", "dual EPYC 9654", "2x EPYC 9654", "AMD EPYC 9654".
    """
    n = normalize_name(user_string)
    n_sockets = 1
    m = re.match(r"^(\d+)x\s+(.*)", n)
    if m:
        n_sockets = int(m.group(1))
        bare = m.group(2)
    else:
        bare_tokens: list[str] = []
        for t in n.split():
            if t in _DUAL_WORDS:
                n_sockets = max(n_sockets, 2)
            elif t in _QUAD_WORDS:
                n_sockets = max(n_sockets, 4)
            else:
                bare_tokens.append(t)
        bare = " ".join(bare_tokens)
    entry = lookup_cpu_entry(bare, passmark)
    if entry is None:
        return None
    return entry, n_sockets


def score_offer(
    offer: dict, passmark: dict[str, dict]
) -> tuple[dict | None, str | None]:
    """
    Score one vast.ai offer under linear multi-socket scaling.

        host_score      = single_chip_mark × n_sockets
        effective_score = host_score × (cpu_cores_effective / cpu_cores)
    """
    cpu_name = offer.get("cpu_name") or ""
    if not cpu_name:
        return None, "no cpu_name"
    entry = lookup_cpu_entry(cpu_name, passmark)
    if entry is None:
        return None, f"no PassMark match: {cpu_name}"

    total_threads = offer.get("cpu_cores") or 0
    effective_threads = offer.get("cpu_cores_effective") or 0
    if not total_threads or not effective_threads:
        return None, "missing thread count"

    cores_per_chip = extract_physical_cores(cpu_name) or entry.get("cores")
    smt = entry.get("smt") or 2

    if cores_per_chip:
        threads_per_chip = cores_per_chip * smt
        n_sockets = max(1, round(total_threads / threads_per_chip))
        # SMT-disabled fallback: total threads == cores_per_chip × N
        if n_sockets * threads_per_chip < total_threads * 0.5:
            n_sockets = max(1, round(total_threads / cores_per_chip))
    else:
        n_sockets = 1

    host_score = entry["mark"] * n_sockets
    effective_score = host_score * (effective_threads / total_threads)
    price = offer.get("dph_total") or 0
    if not price or effective_score <= 0:
        return None, "zero price or score"

    return {
        "id": offer["id"],
        "cpu": cpu_name,
        "passmark_chip": entry["mark"],
        "sockets": n_sockets,
        "threads_eff": effective_threads,
        "threads_total": total_threads,
        "host_score": host_score,
        "effective_score": effective_score,
        "dph": price,
        "dollars_per_kscore_hr": price / (effective_score / 1000),
        "up_cost_per_tb": offer.get("internet_up_cost_per_tb") or 0.0,
        "down_cost_per_tb": offer.get("internet_down_cost_per_tb") or 0.0,
        "inet_up_mbps": offer.get("inet_up") or 0.0,
        "ram_mb": offer.get("cpu_ram", 0),
        "reliability": offer.get("reliability", 0),
        "verification": offer.get("verification", ""),
        "geo": offer.get("geolocation", ""),
    }, None


def add_target_tb_estimates(
    scored: list[dict],
    target_tb: float,
    total_kscore_hours: float,
    download_gb: float,
) -> None:
    """
    Project cost to produce `target_tb` of output given total kscore-hours of
    work required.

        hours_X    = total_kscore_hours / (effective_score_X / 1000)
        compute_$  = dph_X × hours_X
        upload_$   = target_tb × up_cost_per_tb_X        (exact)
        download_$ = (download_gb / 1024) × down_cost_per_tb_X
        total_$    = compute + upload + download

    Bandwidth flag fires when required egress (target bytes ÷ wall-clock)
    exceeds 70 % of the host's published upload Mbps.
    """
    target_gb = target_tb * 1024.0
    target_download_tb = download_gb / 1024.0
    for s in scored:
        eff_kscore = s["effective_score"] / 1000.0
        if eff_kscore <= 0:
            continue
        hours = total_kscore_hours / eff_kscore
        compute = s["dph"] * hours
        upload = target_tb * s["up_cost_per_tb"]
        download = target_download_tb * s["down_cost_per_tb"]
        total = compute + upload + download
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


def print_kscore_table(scored: list[dict], top: int) -> None:
    scored = scored[:top]
    if not scored:
        print("no matches", file=sys.stderr)
        return
    hdr = (
        f"{'#':>3} {'id':>10} {'$/kscore':>10} {'$/h':>7} "
        f"{'score':>9} {'threads':>10} {'sock':>4} "
        f"{'CPU':<26} {'up$/TB':>7} {'upMbps':>7} "
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
    print(
        f"# target: produce {target_tb:.2f} TB of output "
        f"+ {download_gb:.1f} GB ingress (image pulls etc.)"
    )
    hdr = (
        f"{'#':>3} {'id':>10} {'$/TB':>7} {'total$':>7} {'hours':>6} "
        f"{'comp$':>6} {'up$':>5} {'dn$':>5} "
        f"{'CPU':<26} {'reqMbps':>8} {'upMbps':>7} {'bw?':<4} "
        f"{'reliab':>7} {'verif':<11} {'geo':<22}"
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
        "--top", type=int, default=20, help="show top N (default 20)",
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
        "--target-tb", type=float, default=None,
        help=(
            "project total $ to produce this many TB of output (compute + "
            "upload + download), sort by total $/TB, and add a "
            "bandwidth-bottleneck check. Requires --anchor-cpu and "
            "--anchor-hours."
        ),
    )
    p.add_argument(
        "--anchor-cpu", type=str, default=None,
        help=(
            "CPU description for the empirical anchor (e.g. 'EPYC 9654 dual', "
            "'AMD EPYC 7B13', '2x EPYC 9654'). Used with --anchor-hours to "
            "derive total_kscore_hours = H × PassMark(CPU) × n_sockets / 1000."
        ),
    )
    p.add_argument(
        "--anchor-hours", type=float, default=None,
        help="hours the --target-tb workload takes on --anchor-cpu",
    )
    p.add_argument(
        "--download-gb", type=float, default=5.0,
        help="GB of ingress per run (image pull, etc.). Default 5.",
    )
    args = p.parse_args()

    if args.target_tb is not None:
        if args.anchor_cpu is None or args.anchor_hours is None:
            raise SystemExit(
                "--target-tb requires --anchor-cpu and --anchor-hours "
                "(no silent calibration default — anchor on one real run)."
            )
    elif args.anchor_cpu is not None or args.anchor_hours is not None:
        raise SystemExit("--anchor-* flags are only meaningful with --target-tb")

    passmark = fetch_passmark(refresh=args.refresh)
    offers = query_vastai(args.query)

    scored: list[dict] = []
    skip_reasons: dict[str, int] = {}
    for offer in offers:
        s, reason = score_offer(offer, passmark)
        if s is not None:
            scored.append(s)
        else:
            assert reason is not None
            skip_reasons[reason.split(":", 1)[0]] = (
                skip_reasons.get(reason.split(":", 1)[0], 0) + 1
            )

    if args.target_tb is not None:
        anchor = parse_anchor_cpu(args.anchor_cpu, passmark)
        if anchor is None:
            raise SystemExit(
                f"--anchor-cpu {args.anchor_cpu!r} did not match any PassMark "
                "entry. Try 'AMD EPYC 9654', 'EPYC 9654 dual', "
                "'2x EPYC 9654', 'Threadripper PRO 7995WX'."
            )
        anchor_entry, anchor_sockets = anchor
        anchor_mark_total = anchor_entry["mark"] * anchor_sockets
        total_kscore_hours = args.anchor_hours * (anchor_mark_total / 1000.0)
        print(
            f"[workload] anchor: {args.anchor_hours:.1f}h on "
            f"{args.anchor_cpu!r} (PassMark {anchor_entry['mark']} × "
            f"{anchor_sockets} sockets = {anchor_mark_total / 1000:.2f} "
            f"kscore) → {total_kscore_hours:.1f} kscore-h",
            file=sys.stderr,
        )
        add_target_tb_estimates(
            scored,
            target_tb=args.target_tb,
            total_kscore_hours=total_kscore_hours,
            download_gb=args.download_gb,
        )
        scored.sort(
            key=lambda x: x.get("est_total_dollars_per_tb", float("inf"))
        )
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
            print_kscore_table(scored, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
