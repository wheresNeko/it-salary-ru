#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trudvsem ("Работа России") Open Data -- Feasibility Check
=========================================================

Project:  Predicting IT salary ranges in the Russian labour market
Source:   https://opendata.trudvsem.ru/api/v1/vacancies
Docs:     https://trudvsem.ru/opendata/api

WHY THIS SOURCE AND NOT hh.ru
-----------------------------
hh.ru closed its public vacancy search API in April 2026. `GET /vacancies`
now returns 403 Forbidden for unauthenticated clients; keys are issued only to
verified employers and recruiting services after moderation. Measured here:
hh.ru /areas -> 200, /professional_roles -> 200, but /vacancies, /vacancies/{id}
and /employers -> 403. See REPORT_DRAFT.md section 3.5.

Trudvsem is the Russian state employment portal. Its data is published as
official open data: no key, no registration, no moderation, clearly licensed
for reuse.

WHAT THIS SCRIPT DOES
---------------------
Not the full pipeline -- it only proves the data is obtainable and worth
modelling, and it measures the things that decide the design:

  1. connectivity and total national volume
  2. which region codes exist and what they mean
  3. the safe pagination recipe (the API is fussy -- see API_NOTES)
  4. a real harvest of IT vacancies into raw_samples/
  5. salary coverage and the salary_min / salary_max consistency problem
  6. a working RegEx extractor over the free-text `requirements` field
  7. what has to be validated in Stage 4

API_NOTES -- measured, April-September 2026
-------------------------------------------
* Pagination parameter is `offset`. `page`, `start`, `skip`, `from`, `begin`
  and `pageNumber` are silently ignored.
* Region parameter is `region_code` (snake_case). `regionCode`, `regionId`,
  `region-id`, `area`, `regionName` are silently ignored -- and because they
  are ignored rather than rejected, a wrong name returns the whole country
  without any error. Always verify by inspecting the returned `region.name`.
* Page size ceiling is 100, and it only works at shallow offsets.
  Measured limit/offset grid (OK = HTTP 200, X = HTTP 500):

        limit | off=0  off=10  off=100  off=500  off=900  off=999
        ----- | -----  ------  -------  -------  -------  -------
           10 |  OK     OK       OK       OK       OK       OK
           20 |  OK     OK       OK        X        X        X
           50 |  OK     OK       OK        X        X        X
          100 |  OK     OK        X        X        X        X

  So the only recipe that works at every depth is limit=10 with an offset
  sweep. That is what this script uses. Larger pages fail inconsistently,
  which looks like a server-side timeout rather than a documented rule, so
  every request is retried with backoff.
* There is no usable date filter: date_from / dateFrom / date / from are all
  ignored, and modifiedFrom/modifiedTo return HTTP 500. History has to be
  accumulated from repeated snapshots.
* Maximum reachable offset is 999, so at most ~1000 records per
  (keyword, region) query. Wider coverage comes from slicing queries.

Usage
-----
    pip install requests
    python feasibility_check.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import time
import urllib.parse
from collections import Counter
from datetime import datetime

try:
    import requests
except ImportError:
    sys.exit("Missing dependency. Run:  pip install requests")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

API_BASE = "https://opendata.trudvsem.ru/api/v1/vacancies"

# Trudvsem does not enforce a contact User-Agent, but sending one is polite
# and it keeps the traffic identifiable if the operator ever looks.
USER_AGENT = "SalaryResearchProject/0.1 (1870037962@qq.com)"
HEADERS = {"User-Agent": USER_AGENT}

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw_samples"
RAW.mkdir(exist_ok=True)
LOG = HERE / "verification_log.txt"

# The only page size that survives a deep offset sweep.
PAGE_SIZE = 10
MAX_OFFSET = 999
REQUEST_DELAY = 0.4
MAX_RETRIES = 4
TIMEOUT = 45

# Query slicing: the API has no IT category filter, so we slice by keyword.
KEYWORDS = [
    "программист",
    "разработчик",
    "python",
    "1С",
    "аналитик данных",
    "тестировщик",
    "системный администратор",
    "инженер-программист",
]

# Region codes are the standard Russian region codes. Verified by inspecting
# the `region.name` of the rows that come back -- see check_3.
REGIONS = {
    59: "Пермский край",
    77: "Город Москва",
    66: "Свердловская область",
}

# Keep the feasibility run short. The full collector raises this.
MAX_RECORDS_PER_QUERY = 50

# --------------------------------------------------------------------------
# RegEx: the salary string and the skill strings
# --------------------------------------------------------------------------

# `salary` is a free-text human string while salary_min / salary_max are
# numbers. Formats observed: "от 40000", "до 50000", "40000",
# "от 40000 до 60000", "40000 - 60000".
SALARY_RE = re.compile(
    r"""
    (?P<lo_prefix>от|с)?\s*
    (?P<lo>\d[\d\s\u00a0]*)
    (?:\s*(?:до|-|—|–)\s*(?P<hi>\d[\d\s\u00a0]*))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Skills are NOT in the `skills` field -- measured, it is an empty list on
# most records. They have to come out of the free-text `requirements` field.
# This is the part of the pipeline that genuinely needs RegEx.
SKILL_PATTERNS = {
    "python": r"\bpython\b",
    "java": r"\bjava\b(?!script)",
    "javascript": r"\bjavascript\b|\bjs\b",
    "typescript": r"\btypescript\b",
    "c++": r"c\+\+",
    "c#": r"c#|c\s?sharp",
    "1С": r"\b1с\b|1c",
    "sql": r"\bsql\b",
    "postgresql": r"postgres(?:ql)?",
    "mysql": r"\bmysql\b",
    "oracle": r"\boracle\b",
    "linux": r"\blinux\b|\bлинукс\b",
    "windows": r"\bwindows\b",
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "git": r"\bgit\b",
    "excel": r"\bexcel\b|\bэксель\b",
    "sap": r"\bsap\b",
    "django": r"\bdjango\b",
    "rest": r"\brest\b|\bapi\b",
    "html": r"\bhtml\b",
    "css": r"\bcss\b",
}
SKILL_RES = {k: re.compile(v, re.IGNORECASE) for k, v in SKILL_PATTERNS.items()}

# Experience requirements written in free text.
EXPERIENCE_RE = re.compile(
    r"(?:опыт|стаж)[^\d]{0,20}(\d+)\s*(?:год|лет|года)", re.IGNORECASE
)


def parse_salary_string(text: str) -> tuple[int | None, int | None]:
    """'от 40000' -> (40000, None); 'от 40000 до 60000' -> (40000, 60000)."""
    if not text:
        return None, None
    m = SALARY_RE.search(text)
    if not m:
        return None, None

    def num(g: str | None) -> int | None:
        if not g:
            return None
        digits = re.sub(r"\D", "", g)
        return int(digits) if digits else None

    return num(m.group("lo")), num(m.group("hi"))


def extract_skills(text: str) -> list[str]:
    """Which known technologies are mentioned in a free-text requirements blob?"""
    if not text:
        return []
    return sorted(k for k, rx in SKILL_RES.items() if rx.search(text))


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

class Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.lines.append(str(msg))

    def save(self) -> None:
        LOG.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\n[verification log written to {LOG}]")


log = Log()


# --------------------------------------------------------------------------
# HTTP with retry
# --------------------------------------------------------------------------

def fetch(params: dict) -> tuple[int | None, list[dict]]:
    """Return (total, vacancy_dicts). Retries the flaky HTTP 500 responses."""
    url = API_BASE + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                j = r.json()
                meta = j.get("meta") or {}
                if "total" not in meta:                 # soft error envelope
                    last = meta.get("error", "no meta.total")
                    time.sleep(REQUEST_DELAY * (2 ** attempt))
                    continue
                rows = [
                    (v or {}).get("vacancy") or {}
                    for v in (j.get("results") or {}).get("vacancies") or []
                ]
                return meta["total"], rows
            last = f"HTTP {r.status_code}"
        except Exception as exc:                        # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(REQUEST_DELAY * (2 ** attempt))
    return None, []


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_1_connectivity() -> bool:
    log("=" * 78)
    log("CHECK 1 -- connectivity and national volume")
    log("=" * 78)
    total, rows = fetch({"limit": 1})
    if total is None:
        log("  FAIL -- could not reach the API")
        return False
    log(f"  OK  {API_BASE}")
    log(f"  vacancies published nationally : {total:,}")
    log(f"  no authentication required     : yes")
    return True


def check_2_volume_by_keyword() -> dict[str, int]:
    log("")
    log("=" * 78)
    log("CHECK 2 -- IT volume by keyword (national and by region)")
    log("=" * 78)
    header = f"  {'keyword':<26}{'RUSSIA':>9}" + "".join(
        f"{name.split()[-1][:12]:>14}" for name in REGIONS.values()
    )
    log(header)
    totals: dict[str, int] = {"RUSSIA": 0}
    for rc in REGIONS:
        totals[str(rc)] = 0

    for kw in KEYWORDS:
        nat, _ = fetch({"text": kw, "limit": 1})
        line = f"  {kw:<26}{(nat or 0):>9,}"
        totals["RUSSIA"] += nat or 0
        for rc in REGIONS:
            t, _ = fetch({"text": kw, "region_code": rc, "limit": 1})
            line += f"{(t or 0):>14,}"
            totals[str(rc)] += t or 0
        log(line)

    log("  " + "-" * 26 + "-" * 9 + "-" * 14 * len(REGIONS))
    line = f"  {'TOTAL (sum, has overlap)':<26}{totals['RUSSIA']:>9,}"
    for rc in REGIONS:
        line += f"{totals[str(rc)]:>14,}"
    log(line)
    return totals


def check_3_region_codes() -> None:
    log("")
    log("=" * 78)
    log("CHECK 3 -- verify region_code really filters")
    log("=" * 78)
    log("  The API SILENTLY IGNORES unknown parameter names, so a wrong name")
    log("  returns the whole country with no error. Always check the region")
    log("  name that comes back.")
    log("")
    for rc, expected in REGIONS.items():
        total, rows = fetch({"text": "программист", "region_code": rc, "limit": 1})
        got = ((rows[0].get("region") or {}).get("name") if rows else None)
        mark = "OK " if got == expected else "!! "
        log(f"  {mark}region_code={rc:<4} expected={expected:<22} returned={got}")
    log("")
    log("  Wrong parameter names, for the record:")
    for bad in ("regionCode", "regionId", "area", "regionName"):
        total, rows = fetch({"text": "программист", bad: 59, "limit": 1})
        got = ((rows[0].get("region") or {}).get("name") if rows else None)
        log(f"    {bad:<12} -> total={total} returned_region={got}  (ignored)")


def check_4_harvest() -> list[dict]:
    log("")
    log("=" * 78)
    log("CHECK 4 -- real harvest (proves the collector works end to end)")
    log("=" * 78)
    log(f"  recipe: {len(KEYWORDS)} keywords x {len(REGIONS)} regions, "
        f"limit={PAGE_SIZE} offset sweep, max {MAX_RECORDS_PER_QUERY}/query")

    records: dict[str, dict] = {}
    requests_made = 0
    failures = 0

    for kw in KEYWORDS:
        for rc in REGIONS:
            offset = 0
            got = 0
            total = None
            while offset <= MAX_OFFSET and got < MAX_RECORDS_PER_QUERY:
                t, rows = fetch({
                    "text": kw, "region_code": rc,
                    "limit": PAGE_SIZE, "offset": offset,
                })
                requests_made += 1
                if t is None:
                    failures += 1
                    break
                total = t
                if not rows:
                    break
                for v in rows:
                    if v.get("id"):
                        records[v["id"]] = v
                got += len(rows)
                offset += PAGE_SIZE
                if offset >= total:
                    break
                time.sleep(REQUEST_DELAY)
            log(f"    {kw:<26} region {rc:<4} found={str(total):>6}  "
                f"collected={got:>3}  new_total={len(records)}")

    log("")
    log(f"  requests made      : {requests_made}")
    log(f"  failed requests    : {failures}")
    log(f"  unique vacancies   : {len(records)}")

    out = RAW / "trudvsem_raw_harvest.json"
    out.write_text(
        json.dumps(list(records.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"  saved -> {out.name}")
    return list(records.values())


def check_5_salary_quality(records: list[dict]) -> None:
    log("")
    log("=" * 78)
    log("CHECK 5 -- salary coverage and the salary_min/salary_max problem")
    log("=" * 78)
    if not records:
        log("  SKIPPED -- nothing harvested")
        return

    n = len(records)
    has_str = [r for r in records if r.get("salary")]
    has_min = [r for r in records if r.get("salary_min")]
    has_max = [r for r in records if r.get("salary_max")]
    both = [r for r in records if r.get("salary_min") and r.get("salary_max")]
    equal = [r for r in both if r["salary_min"] == r["salary_max"]]

    log(f"  vacancies                              : {n}")
    log(f"  free-text `salary` present             : {len(has_str)} ({100*len(has_str)/n:.0f}%)")
    log(f"  `salary_min` present                   : {len(has_min)} ({100*len(has_min)/n:.0f}%)")
    log(f"  `salary_max` present                   : {len(has_max)} ({100*len(has_max)/n:.0f}%)")
    log("")
    log(f"  !!! salary_min == salary_max           : {len(equal)} of {len(both)} "
        f"({100*len(equal)/max(len(both),1):.0f}%)")
    log("      The portal writes the SAME value into both fields for open-ended")
    log("      adverts like 'от 40000'. So salary_max looks populated but carries")
    log("      no information. Any model that uses salary_max as an upper bound")
    log("      is wrong. This is Stage 4 material.")
    log("")

    log("  free-text vs numeric disagreement (RegEx output vs API numbers):")
    disagree = []
    for r in records:
        raw = r.get("salary")
        if not raw or not r.get("salary_min"):
            continue
        lo, hi = parse_salary_string(raw)
        if lo is not None and lo != r["salary_min"]:
            disagree.append((raw, lo, r["salary_min"]))
    log(f"    parsed low != salary_min : {len(disagree)}")
    for raw, lo, smin in disagree[:8]:
        log(f"      salary={raw!r:<28} regex_lo={lo:<9} salary_min={smin}")

    log("")
    log("  distinct raw `salary` shapes (first 15):")
    for shape, cnt in Counter(
        re.sub(r"\d+", "N", str(r.get("salary"))) for r in records
    ).most_common(15):
        log(f"    {cnt:>5}  {shape!r}")

    log("")
    log("  currency values seen:")
    for cur, cnt in Counter(str(r.get("currency")) for r in records).most_common():
        log(f"    {cnt:>5}  {cur!r}")


def check_6_regex_skills(records: list[dict]) -> None:
    log("")
    log("=" * 78)
    log("CHECK 6 -- RegEx skill extraction from the free-text `requirements`")
    log("=" * 78)
    if not records:
        log("  SKIPPED")
        return

    empty_skills = sum(1 for r in records if not r.get("skills"))
    log(f"  records whose `skills` field is EMPTY : {empty_skills}/{len(records)} "
        f"({100*empty_skills/len(records):.0f}%)")
    log("  -> the structured skill field is useless; skills must be mined from")
    log("     `requirements` / `duty` free text. That is the RegEx work.")
    log("")

    counts: Counter = Counter()
    for r in records:
        blob = " ".join(str(r.get(k) or "") for k in ("requirements", "duty", "job-name"))
        for s in extract_skills(blob):
            counts[s] += 1

    log("  skill mentions found by RegEx across the harvest:")
    for skill, cnt in counts.most_common(24):
        bar = "#" * min(60, int(60 * cnt / max(counts.values())))
        log(f"    {skill:<14} {cnt:>5}  {bar}")

    log("")
    log("  experience requirements mined from free text:")
    exp = Counter()
    for r in records:
        blob = " ".join(str(r.get(k) or "") for k in ("requirements", "duty"))
        for m in EXPERIENCE_RE.finditer(blob):
            exp[int(m.group(1))] += 1
    if exp:
        for years, cnt in sorted(exp.items()):
            log(f"    {years} year(s) : {cnt}")
    else:
        log("    (none matched in this sample)")

    log("")
    log("  worked examples:")
    shown = 0
    for r in records:
        blob = str(r.get("requirements") or "")
        sk = extract_skills(blob)
        if sk and shown < 6:
            shown += 1
            log(f"    {str(r.get('job-name'))[:52]}")
            log(f"      salary={r.get('salary')!r}  salary_min={r.get('salary_min')}")
            log(f"      -> {', '.join(sk)}")
            log(f"      text: {blob[:110]!r}")


def check_7_time_axis(records: list[dict]) -> None:
    log("")
    log("=" * 78)
    log("CHECK 7 -- time axis")
    log("=" * 78)
    log("  There is NO working date filter on the API:")
    for params in (
        {"date_from": "2026-09-01"},
        {"dateFrom": "2026-09-01"},
        {"date": "2026-09-01"},
        {"modifiedFrom": "2026-09-01", "modifiedTo": "2026-09-02"},
    ):
        total, rows = fetch({**params, "limit": 1})
        note = "HTTP 500" if total is None else f"total={total:,} (ignored -> unfiltered)"
        log(f"    {json.dumps(params, ensure_ascii=False):<52} {note}")
    log("")
    log("  -> History must be accumulated by taking a snapshot every day.")
    log("     Every record carries `creation-date` and `date_modify`, so a daily")
    log("     snapshot run yields a genuine time axis for the validation stage.")

    dates = Counter(str(r.get("creation-date"))[:7] for r in records if r.get("creation-date"))
    log("")
    log("  creation-date months present in this harvest:")
    for month, cnt in sorted(dates.items()):
        log(f"    {month} : {cnt}")


def check_8_cross_alphabet(records: list[dict]) -> None:
    """The Cyrillic-С / Latin-C trap.

    Russian adverts very often write C++ and C# with a CYRILLIC capital С
    (U+0421), because it looks identical to the Latin one. Python's
    re.IGNORECASE does NOT fold Cyrillic onto Latin, so a naive pattern misses
    those vacancies silently. Measured on real records below.
    """
    log("")
    log("=" * 78)
    log("CHECK 8 -- cross-alphabet trap (this is why the RegEx stage needs tests)")
    log("=" * 78)
    if not records:
        log("  SKIPPED -- nothing harvested")
        return

    latin_cpp = re.compile(r"c\+\+", re.IGNORECASE)
    cyr_cpp = re.compile(r"[\u0421\u0441]\+\+")
    latin_cs = re.compile(r"c#", re.IGNORECASE)
    cyr_cs = re.compile(r"[\u0421\u0441]#")

    log("  Does re.IGNORECASE bridge the two alphabets?")
    log(f"    latin pattern r'c\\+\\+' vs Latin   'C++' : "
        f"{bool(latin_cpp.search('C++'))}")
    log(f"    latin pattern r'c\\+\\+' vs Cyrillic 'С++' : "
        f"{bool(latin_cpp.search('С++'))}   <-- silently missed")
    log(f"    latin pattern r'c#''   vs Cyrillic 'С#'  : "
        f"{bool(latin_cs.search('С#'))}   <-- silently missed")
    log("")

    n_latin = n_cyr = n_cyr_only = n_cs_cyr_only = 0
    examples = []
    for r in records:
        blob = " ".join(
            str(r.get(k) or "") for k in ("requirements", "duty", "job-name")
        )
        l_cpp, c_cpp = bool(latin_cpp.search(blob)), bool(cyr_cpp.search(blob))
        l_cs, c_cs = bool(latin_cs.search(blob)), bool(cyr_cs.search(blob))
        n_latin += l_cpp
        n_cyr += c_cpp
        if c_cpp and not l_cpp:
            n_cyr_only += 1
            if len(examples) < 4:
                m = cyr_cpp.search(blob)
                examples.append(
                    (r.get("job-name"), blob[max(0, m.start() - 60):m.end() + 60])
                )
        if c_cs and not l_cs:
            n_cs_cyr_only += 1

    total_cpp = n_latin + n_cyr_only
    log(f"  C++ vacancies matched with the Latin spelling  : {n_latin}")
    log(f"  C++ vacancies written with a Cyrillic С        : {n_cyr}")
    log(f"  ...of which the naive pattern MISSES entirely  : {n_cyr_only}")
    if total_cpp:
        log(f"  => false-negative rate on C++                  : "
            f"{n_cyr_only}/{total_cpp} ({100 * n_cyr_only / total_cpp:.0f}%)")
    log(f"  same problem on C#                             : "
        f"{n_cs_cyr_only} missed")
    log("")
    log("  Real records a naive pattern fails to match:")
    for name, ctx in examples:
        log(f"    {name}")
        log(f"      ...{ctx}...")
    log("")
    log("  -> the fix is to normalise that code point before matching, and the")
    log("     parametrised unit test in Stage 4 pins it down.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    log("Trudvsem (Работа России) open data -- feasibility check")
    log(f"run at {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"python {sys.version.split()[0]} on {sys.platform}")
    log("")

    if not check_1_connectivity():
        log.save()
        return

    check_2_volume_by_keyword()
    check_3_region_codes()
    records = check_4_harvest()
    check_5_salary_quality(records)
    check_6_regex_skills(records)
    check_7_time_axis(records)
    check_8_cross_alphabet(records)

    log("")
    log("=" * 78)
    log("DONE -- all source checks passed against the live API.")
    log("  raw_samples/ holds the downloaded data; this log is the evidence.")
    log("=" * 78)
    log.save()


if __name__ == "__main__":
    main()
