#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 -- DATA COLLECTION
==========================

Harvests vacancies from the Trudvsem (Работа России) open data API into
immutable raw JSON. Acquisition only: no cleaning, no feature engineering.

TWO SETS
--------
  * IT set       -- sliced by IT keywords. The subject of the model.
  * control set  -- sliced by non-IT keywords (accountant, driver, teacher...).
                    Needed to measure the IT salary premium with region and
                    qualification held constant (research sub-question 2).

HOW COVERAGE IS OBTAINED -- measured constraints, not assumptions
-----------------------------------------------------------------
All of the following was established by direct measurement (see
api_investigation/). Three of them forced the design:

1. **Paging is dead.** `offset > 0` now returns HTTP 200 with ZERO records.
   Earlier in the same session offsets up to 999 returned data, so the
   behaviour changed under us. Only `offset=0` is reliable.

2. **Page size 100 works.** `limit=100` at `offset=0` returns 100 records in a
   single request; `limit` above 100 is silently clamped to 100.

3. **The server takes 5-6 s per request**, and connection pooling does not
   help -- it is server-side latency. Concurrency does help: 4 parallel workers
   measured a 3.9x speedup with zero failed or truncated responses.

=> Therefore: one request per (keyword, region) cell at limit=100, fanned out
   across 4 workers. Coverage comes from slicing, never from paging. A cell with
   more than 100 vacancies yields only its first 100 -- an accepted, documented
   loss.

SAMPLING -- read before drawing conclusions
-------------------------------------------
Within one query the API returns its own ordering, which is recency-weighted but
is NOT random and NOT explicitly sorted. The result is a *convenience sample*,
not a probability sample. This is stated in the data dictionary and must be
repeated in any write-up.

USAGE
-----
    python collect.py                      # full run
    python collect.py --budget 40          # smoke test
    python collect.py --phase national     # skip the regional fan-out
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys
import threading
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    import requests
except ImportError:
    sys.exit("Missing dependency. Run:  pip install requests")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

API_BASE = "https://opendata.trudvsem.ru/api/v1/vacancies"
USER_AGENT = "SalaryResearchProject/0.1 (1870037962@qq.com)"
HEADERS = {"User-Agent": USER_AGENT}

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw_samples"
RAW.mkdir(exist_ok=True)
LOG_PATH = HERE / "collection_log.txt"
REGION_CACHE = RAW / "regions.json"

PAGE_SIZE = 100         # measured ceiling; larger values are clamped
WORKERS = 8             # measured: 4 -> 0.56 req/s, 8 -> 1.20 req/s, 0 failures
REQUEST_DELAY = 0.15
MAX_RETRIES = 3
TIMEOUT = 60

DEFAULT_BUDGET = 1400   # HTTP requests
DEFAULT_SECONDS = 2100  # wall-clock safety stop

PROBE_RANGE = range(1, 93)   # region codes to probe empirically

# Control set uses the regions that actually carry volume, to keep the request
# count down -- the IT set is the one that needs full geographic coverage.
CONTROL_REGIONS = [
    "77", "78", "59", "66", "54", "16", "23", "52", "63", "74", "61", "2",
]

IT_KEYWORDS = [
    "программист",
    "разработчик",
    "python",
    "1С",
    "аналитик данных",
    "тестировщик",
    "системный администратор",
    "инженер-программист",
    "веб-разработчик",
    "java-разработчик",
    "devops",
    "frontend-разработчик",
    "backend-разработчик",
    "мобильный разработчик",
    "qa-инженер",
    "системный аналитик",
    "информационная безопасность",
    "data scientist",
]

CONTROL_KEYWORDS = [
    "бухгалтер",
    "водитель",
    "продавец",
    "учитель",
    "медицинская сестра",
    "повар",
    "слесарь",
    "экономист",
    "юрист",
    "инженер-механик",
]

# --------------------------------------------------------------------------
# Logging and shared state
# --------------------------------------------------------------------------

class Log:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.lock = threading.Lock()
        self.t0 = time.time()

    def __call__(self, msg: str = "") -> None:
        with self.lock:
            print(msg, flush=True)
            self.lines.append(str(msg))

    def save(self) -> None:
        LOG_PATH.write_text("\n".join(self.lines), encoding="utf-8")


log = Log()


class Budget:
    def __init__(self, limit: int, seconds: float) -> None:
        self.limit = limit
        self.deadline = time.time() + seconds
        self.lock = threading.Lock()
        self.used = 0
        self.failed = 0

    @property
    def exhausted(self) -> bool:
        if time.time() > self.deadline:
            return True
        with self.lock:
            return self.used >= self.limit

    def spend(self) -> None:
        with self.lock:
            self.used += 1

    def fail(self) -> None:
        with self.lock:
            self.failed += 1


BUDGET = Budget(DEFAULT_BUDGET, DEFAULT_SECONDS)

# One requests.Session per worker thread, with a matching connection pool.
_local = threading.local()


def sess() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=WORKERS, pool_maxsize=WORKERS
        )
        s.mount("https://", adapter)
        _local.s = s
    return _local.s


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def fetch(params: dict) -> tuple[int | None, list[dict]]:
    """Return (total, records). Retries transient failures."""
    if BUDGET.exhausted:
        return None, []
    url = API_BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(MAX_RETRIES):
        BUDGET.spend()
        try:
            r = sess().get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                j = r.json()
                meta = j.get("meta") or {}
                if "total" not in meta:
                    time.sleep(REQUEST_DELAY * (2 ** attempt))
                    continue
                rows = [
                    (v or {}).get("vacancy") or {}
                    for v in (j.get("results") or {}).get("vacancies") or []
                ]
                return meta["total"], rows
        except Exception:                               # noqa: BLE001
            pass
        BUDGET.fail()
        time.sleep(REQUEST_DELAY * (2 ** attempt))
    return None, []


# --------------------------------------------------------------------------
# Region directory
# --------------------------------------------------------------------------

def discover_regions(force: bool = False) -> dict[str, str]:
    """Learn region code -> name empirically, and remember what was probed.

    The API silently ignores unknown parameter names, so the mapping cannot be
    assumed -- it must be read back out of the records. Single-digit probes are
    ambiguous (the parameter behaves as a prefix match), which is why the
    canonical code stored here is the one that comes back INSIDE the record,
    not the one that was sent.
    """
    state = {"probed": [], "regions": {}}
    if REGION_CACHE.exists() and not force:
        raw = json.loads(REGION_CACHE.read_text(encoding="utf-8"))
        if "regions" in raw and "probed" in raw:
            state = raw
        else:
            # legacy flat {code: name} layout -- keep the names, re-probe codes
            state = {"probed": [], "regions": raw}
        state.setdefault("probed", [])
        state.setdefault("regions", {})

    todo = [n for n in PROBE_RANGE if n not in state["probed"]]
    if not todo:
        log(f"  region directory: {len(state['regions'])} regions (cached)")
        return state["regions"]

    log(f"  probing {len(todo)} region codes ...")
    lock = threading.Lock()

    def probe(n: int) -> None:
        _total, rows = fetch({"region_code": n, "limit": 1})
        if _total is None:
            return          # never answered -- leave it unprobed so a later run retries
        with lock:
            state["probed"].append(n)
            if rows:
                reg = rows[0].get("region") or {}
                canonical = str(reg.get("region_code") or "")[:2]
                if canonical and reg.get("name"):
                    state["regions"].setdefault(canonical, reg["name"])

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(probe, todo))

    state["probed"] = sorted(set(state["probed"]))
    REGION_CACHE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log(f"  region directory: {len(state['regions'])} regions")
    return state["regions"]


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

class Store:
    def __init__(self, name: str) -> None:
        self.name = name
        self.data: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.seen_pages = 0
        self.added = 0

    def add_many(self, rows: list[dict]) -> int:
        n = 0
        with self.lock:
            for v in rows:
                vid = v.get("id")
                if vid and vid not in self.data:
                    self.data[vid] = v
                    n += 1
            self.added += n
        return n

    def save(self) -> pathlib.Path:
        """Write the harvest gzipped.

        The raw records are dominated by Russian free text, so gzip takes
        105 MB of pretty-printed JSON down to 13 MB -- an 87% cut. The
        uncompressed form is gitignored; nothing in the pipeline needs it.
        """
        path = RAW / self.name
        with self.lock:
            payload = list(self.data.values())
        # Never clobber a real harvest with the empty result of a starved run.
        if not payload and path.exists() and path.stat().st_size > 10:
            return path
        blob = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        path.write_bytes(gzip.compress(blob, compresslevel=9))
        return path


def harvest_cell(store: Store, label: str, **query) -> None:
    """One (keyword, region) cell: a single request at limit=100, offset=0.

    A second request at offset=100 is attempted only when the first reports more
    than 100 matches -- it usually returns nothing, so it is not worth 990
    wasted requests across the whole grid.
    """
    if BUDGET.exhausted:
        return
    total, rows = fetch({**query, "limit": PAGE_SIZE, "offset": 0})
    if total is None:
        return
    store.add_many(rows)
    with store.lock:
        store.seen_pages += 1

    if total > PAGE_SIZE and not BUDGET.exhausted:
        time.sleep(REQUEST_DELAY)
        _t, more = fetch({**query, "limit": PAGE_SIZE, "offset": PAGE_SIZE})
        if more:
            store.add_many(more)

    time.sleep(REQUEST_DELAY)


def run_grid(store: Store, keywords: list[str], regions: list[str] | None,
             tag: str) -> None:
    log("")
    log("-" * 78)
    log(f"{tag} -- {len(keywords)} keywords"
        + (f" x {len(regions)} regions" if regions else " (national only)"))
    log("-" * 78)

    tasks: list[tuple[str, str | None]] = []
    for kw in keywords:
        if regions:
            for rc in regions:
                tasks.append((kw, rc))
        else:
            tasks.append((kw, None))

    done = 0
    total_tasks = len(tasks)
    started = time.time()

    def work(task: tuple[str, str | None]) -> None:
        kw, rc = task
        q = {"text": kw}
        if rc:
            q["region_code"] = rc
        harvest_cell(store, kw, **q)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        for _ in as_completed(futures):
            done += 1
            if done % 50 == 0 or done == total_tasks:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0
                eta = (total_tasks - done) / rate if rate else 0
                log(f"    {done:>5}/{total_tasks} cells  store={len(store.data):>7} "
                    f"req={BUDGET.used:>5}  {rate:.1f} cell/s  ETA {eta/60:.1f} min")
            if BUDGET.exhausted:
                for f in futures:
                    f.cancel()
                break

    path = store.save()
    log(f"  -> {path.name}: {len(store.data):,} unique vacancies "
        f"({store.added:,} added this phase)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--seconds", type=int, default=DEFAULT_SECONDS)
    ap.add_argument("--phase", choices=["both", "national", "grid"],
                    default="both")
    ap.add_argument("--refresh-regions", action="store_true")
    args = ap.parse_args()

    BUDGET.limit = args.budget
    BUDGET.deadline = time.time() + args.seconds

    log("Trudvsem collection -- Stage 1")
    log(f"run at {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"budget: {args.budget} requests / {args.seconds} s   "
        f"workers: {WORKERS}   phase: {args.phase}")
    log("")

    log("=" * 78)
    log("REGION DIRECTORY")
    log("=" * 78)
    regions = discover_regions(force=args.refresh_regions)
    region_codes = sorted(regions)
    log("")

    it = Store("trudvsem_it_harvest.json.gz")
    control = Store("trudvsem_control_harvest.json.gz")

    if args.phase in ("both", "national"):
        run_grid(it, IT_KEYWORDS, None, "PHASE 1a -- IT national sweep")
        run_grid(control, CONTROL_KEYWORDS, None,
                 "PHASE 1b -- control national sweep")

    if args.phase in ("both", "grid"):
        run_grid(it, IT_KEYWORDS, region_codes,
                 "PHASE 2a -- IT keyword x region grid")
        run_grid(control, CONTROL_KEYWORDS,
                 [r for r in CONTROL_REGIONS if r in regions],
                 "PHASE 2b -- control keyword x region grid")

    # ---------------------------------------------------------------- summary
    log("")
    log("=" * 78)
    log("SUMMARY")
    log("=" * 78)
    log(f"  HTTP requests used        : {BUDGET.used} / {BUDGET.limit}")
    log(f"  failed requests           : {BUDGET.failed}")
    log(f"  elapsed                   : {time.time() - log.t0:.0f} s")
    log(f"  IT vacancies (unique)     : {len(it.data):,}")
    log(f"  control vacancies (unique): {len(control.data):,}")
    log("")

    for tag, store in (("IT", it), ("control", control)):
        if not store.data:
            continue
        vals = list(store.data.values())
        with_salary = sum(1 for v in vals if v.get("salary_min"))
        seen = Counter((v.get("region") or {}).get("name") for v in vals)
        dates = [str(v.get("creation-date")) for v in vals if v.get("creation-date")]
        log(f"  [{tag}]")
        log(f"    with salary_min         : {with_salary:,} "
            f"({100 * with_salary / len(vals):.0f}%)")
        log(f"    distinct regions        : {len(seen)}")
        log(f"    top regions             : "
            f"{', '.join(f'{n} {c}' for n, c in seen.most_common(6))}")
        if dates:
            log(f"    creation-date range     : {min(dates)} .. {max(dates)}")
        log("")

    log("  files:")
    for p in sorted(RAW.iterdir()):
        if p.is_file():
            log(f"    {p.name:<40} {p.stat().st_size:>10,} bytes")
    log("")
    log("=" * 78)
    log.save()


if __name__ == "__main__":
    main()
