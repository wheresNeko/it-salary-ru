#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate docs/data_dictionary.md from the harvested raw data.

Everything in the output is computed from the data itself -- presence rates,
cardinality, example values and the defect counts. Nothing is hand-written, so
re-running it after a new harvest keeps the document honest.

Usage
-----
    python make_data_dictionary.py
    python make_data_dictionary.py --it raw_samples/trudvsem_it_harvest.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw_samples"
DOCS = HERE / "docs"
DOCS.mkdir(exist_ok=True)

MAX_TRACKED_VALUES = 300      # stop counting distinct values past this
MAX_EXAMPLES = 3


def load_records(path: pathlib.Path) -> list[dict]:
    """Read a harvested JSON file -- gzipped or plain."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------

def walk(obj, path: tuple[str, ...], acc: dict) -> None:
    """Record every leaf path -> value. Lists are treated as leaves."""
    if isinstance(obj, dict):
        if not obj:
            acc[path] = {}
        for k, v in obj.items():
            walk(v, path + (k,), acc)
    else:
        acc[path] = obj


def type_name(values: list) -> str:
    seen = set()
    for v in values:
        if v is None:
            seen.add("null")
        elif isinstance(v, bool):
            seen.add("bool")
        elif isinstance(v, int):
            seen.add("int")
        elif isinstance(v, float):
            seen.add("float")
        elif isinstance(v, str):
            seen.add("str")
        elif isinstance(v, list):
            seen.add("list")
        elif isinstance(v, dict):
            seen.add("object")
    return " | ".join(sorted(seen)) or "-"


def is_present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    if isinstance(v, (list, dict)) and not v:
        return False
    return True


def short(v, width: int = 60) -> str:
    s = json.dumps(v, ensure_ascii=False)
    s = s.replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= width else s[: width - 1] + "…"


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def analyse(records: list[dict], title: str, out: list[str]) -> None:
    n = len(records)
    presence: Counter = Counter()
    values: dict[tuple, list] = defaultdict(list)
    distinct: dict[tuple, set] = defaultdict(set)
    truncated: set[tuple] = set()

    for rec in records:
        acc: dict = {}
        walk(rec, (), acc)
        for path, v in acc.items():
            if not path:
                continue
            presence[path] += 1 if is_present(v) else 0
            values[path].append(v)
            if path not in truncated:
                distinct[path].add(json.dumps(v, ensure_ascii=False, sort_keys=True))
                if len(distinct[path]) > MAX_TRACKED_VALUES:
                    truncated.add(path)

    out.append(f"## {title}")
    out.append("")
    out.append(f"Records analysed: **{n:,}**")
    out.append("")
    out.append("| Field | Type | Present | Distinct | Example |")
    out.append("|---|---|---:|---:|---|")
    for path in sorted(presence, key=lambda p: (-presence[p], p)):
        name = ".".join(path)
        vals = values[path]
        pct = 100 * presence[path] / n if n else 0
        if path in truncated:
            card = f">{MAX_TRACKED_VALUES}"
        else:
            card = str(len(distinct[path]))
        examples = [v for v in vals if is_present(v)][:MAX_EXAMPLES]
        ex = "<br>".join(short(v, 48) for v in examples) if examples else "—"
        out.append(f"| `{name}` | {type_name(vals)} | {pct:.0f}% | {card} | {ex} |")
    out.append("")


def defects(records: list[dict], out: list[str]) -> None:
    n = len(records)
    if not n:
        return

    def positive_int(v) -> bool:
        return isinstance(v, int) and v > 0

    sal_str = [r for r in records if r.get("salary")]
    zero_sentinel = [r for r in records if r.get("salary_min") == 0]
    sal_min = [r for r in records if positive_int(r.get("salary_min"))]
    sal_max = [r for r in records if positive_int(r.get("salary_max"))]
    both = [r for r in records
            if positive_int(r.get("salary_min")) and positive_int(r.get("salary_max"))]
    equal = [r for r in both if r["salary_min"] == r["salary_max"]]
    empty_skills = [r for r in records if not r.get("skills")]

    out.append("## Measured data defects")
    out.append("")
    out.append(f"Computed over the de-duplicated union of both sets: **{n:,}** records.")
    out.append("These drive the Stage 2 repair policy and the Stage 4 cross-field proof.")
    out.append("")
    out.append("| Defect | Count | Share |")
    out.append("|---|---:|---:|")
    out.append(f"| `salary` free text present | {len(sal_str):,} | "
               f"{100*len(sal_str)/n:.2f}% |")
    out.append(f"| `salary_min` usable (a positive integer) | {len(sal_min):,} | "
               f"{100*len(sal_min)/n:.2f}% |")
    out.append(f"| `salary_max` usable (a positive integer) | {len(sal_max):,} | "
               f"{100*len(sal_max)/n:.2f}% |")
    out.append(f"| **zero sentinel — `salary_min == 0`, text `\"от 0\"`** "
               f"(means \"not specified\", but looks like data) | "
               f"{len(zero_sentinel):,} | {100*len(zero_sentinel)/n:.2f}% |")
    out.append(f"| **`salary_max == salary_min`** (upper bound carries no information) | "
               f"{len(equal):,} | {100*len(equal)/max(len(both),1):.1f}% of those with both |")
    out.append(f"| **`skills` field empty** (skills must be mined from free text) | "
               f"{len(empty_skills):,} | {100*len(empty_skills)/n:.1f}% |")
    out.append("")

    # salary string shapes -- digits collapsed to N
    shapes = Counter(re.sub(r"\d+", "N", str(r.get("salary"))) for r in sal_str)
    out.append("### Free-text `salary` shapes")
    out.append("")
    out.append("| Shape | Count |")
    out.append("|---|---:|")
    for shape, cnt in shapes.most_common(10):
        out.append(f"| `{shape}` | {cnt:,} |")
    out.append("")

    # currency
    cur = Counter(str(r.get("currency")) for r in records)
    out.append("### Currency values")
    out.append("")
    out.append("| Value | Count |")
    out.append("|---|---:|")
    for c, cnt in cur.most_common(10):
        out.append(f"| `{c}` | {cnt:,} |")
    out.append("")

    # cross-alphabet trap
    latin = re.compile(r"c\+\+", re.IGNORECASE)
    cyr = re.compile(r"[\u0421\u0441]\+\+")
    latin_only = cyr_only = 0
    for r in records:
        blob = " ".join(str(r.get(k) or "") for k in ("requirements", "duty", "job-name"))
        l, c = bool(latin.search(blob)), bool(cyr.search(blob))
        latin_only += l and not c
        cyr_only += c and not l
    total_cpp = latin_only + cyr_only
    out.append("### Cyrillic С / Latin C trap")
    out.append("")
    out.append("Russian adverts often write `C++` with a **Cyrillic** С (U+0421),")
    out.append("which looks identical. `re.IGNORECASE` does not fold across alphabets,")
    out.append("so a naive pattern silently misses them.")
    out.append("")
    out.append(f"- vacancies matching only the **Latin** spelling: {latin_only:,}")
    out.append(f"- vacancies matching only the **Cyrillic** spelling: {cyr_only:,}")
    if total_cpp:
        out.append(f"- false-negative rate of a naive pattern: "
                   f"**{cyr_only}/{total_cpp} ({100*cyr_only/total_cpp:.0f}%)**")
    out.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--it", default=str(RAW / "trudvsem_it_harvest.json.gz"))
    ap.add_argument("--control", default=str(RAW / "trudvsem_control_harvest.json.gz"))
    ap.add_argument("--out", default=str(DOCS / "data_dictionary.md"))
    args = ap.parse_args()

    out: list[str] = []
    out.append("# Data dictionary — Trudvsem vacancy records")
    out.append("")
    out.append("Generated by `make_data_dictionary.py` from the raw harvest. Every number")
    out.append("below is computed from the data, not written by hand.")
    out.append("")
    out.append(f"- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.append("- Source: `https://opendata.trudvsem.ru/api/v1/vacancies`")
    out.append("")

    total = 0
    for label, path_str in (("IT vacancies", args.it), ("Control (non-IT) vacancies", args.control)):
        p = pathlib.Path(path_str)
        if not p.exists():
            out.append(f"## {label}")
            out.append("")
            out.append(f"_not collected yet: `{p.name}` is missing_")
            out.append("")
            continue
        records = load_records(p)
        total += len(records)
        out.append(f"Source: `raw_samples/{p.name}`")
        out.append("")
        analyse(records, label, out)

    # defects computed over the union
    merged: dict[str, dict] = {}
    for path_str in (args.it, args.control):
        p = pathlib.Path(path_str)
        if p.exists():
            for r in load_records(p):
                if r.get("id"):
                    merged[r["id"]] = r
    defects(list(merged.values()), out)

    out.append("## Sampling caveat")
    out.append("")
    out.append("Within a single query the API returns its own ordering, which is")
    out.append("recency-weighted but is **not random and not explicitly sorted**. Each")
    out.append("(keyword, region) cell is capped at 100 records because the API's")
    out.append("paging is unreliable. The result is a **convenience sample**, not a")
    out.append("probability sample. Any inference drawn from it must say so.")
    out.append("")

    pathlib.Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {args.out}  ({total:,} records, {len(out)} lines)")


if __name__ == "__main__":
    main()
