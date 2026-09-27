# -*- coding: utf-8 -*-
"""Final probe: pin down the limit/offset rule, verify region_code, size the dataset."""
import json

import requests

BASE = "https://opendata.trudvsem.ru/api/v1/vacancies"
H = {"User-Agent": "SalaryResearchProject/0.1 (1870037962@qq.com)"}
out = []


def call(**params):
    r = requests.get(BASE, params=params, headers=H, timeout=45)
    if r.status_code != 200:
        return None, None
    j = r.json()
    vacs = (j.get("results") or {}).get("vacancies") or []
    return (j.get("meta") or {}).get("total"), vacs


def t(label, **params):
    total, vacs = call(**params)
    if total is None:
        out.append(f"  {label:<44} HTTP FAIL")
        return
    out.append(f"  {label:<44} total={total:>8,}  n={len(vacs):>4}")
    return vacs


out.append("=" * 96)
out.append("A. LIMIT / OFFSET LOOKUP TABLE  (find the exact rule)")
out.append("=" * 96)
out.append(f"  {'limit':>6} | " + " | ".join(f"off={o:<5}" for o in (0, 10, 100, 500, 900, 999)))
for lim in (10, 20, 50, 100):
    cells = []
    for off in (0, 10, 100, 500, 900, 999):
        total, vacs = call(limit=lim, offset=off)
        cells.append("OK     " if total is not None else "HTTP500")
    out.append(f"  {lim:>6} | " + " | ".join(cells))

out.append("")
out.append("=" * 96)
out.append("B. VERIFY region_code ACTUALLY FILTERS (check the region of returned rows)")
out.append("=" * 96)
REGIONS = {
    1: "Москва", 2: "Санкт-Петербург", 59: "Пермский край", 66: "Свердловская",
    77: "Москва(город)", 23: "Краснодарский край",
}
for rc, guess in REGIONS.items():
    total, vacs = call(text="программист", region_code=rc, limit=1)
    if vacs:
        v = vacs[0]["vacancy"]
        got = (v.get("region") or {}).get("name")
        out.append(f"  region_code={rc:<4} ({guess:<18}) total={total:>6,}  "
                   f"returned_region={got}")
    else:
        out.append(f"  region_code={rc:<4} ({guess:<18}) total={total}")

out.append("")
out.append("=" * 96)
out.append("C. DOES region_code WORK ALONE (no text)?")
out.append("=" * 96)
for rc in (59, 1, 77):
    t(f"region_code={rc}, limit=1", region_code=rc, limit=1)

out.append("")
out.append("=" * 96)
out.append("D. DATASET SIZE: IT keywords x regions")
out.append("=" * 96)
KEYWORDS = ["программист", "разработчик", "python", "1С", "аналитик данных",
            "тестировщик", "системный администратор", "веб-разработчик",
            "инженер-программист", "data scientist"]
out.append(f"  {'keyword':<26} {'RUSSIA':>9} {'Perm(59)':>10} {'Moscow(77)':>11} {'SPb(2)':>9}")
grand = {"RU": 0, "59": 0, "77": 0, "2": 0}
for kw in KEYWORDS:
    row = {}
    for key, rc in (("RU", None), ("59", 59), ("77", 77), ("2", 2)):
        p = {"text": kw, "limit": 1}
        if rc is not None:
            p["region_code"] = rc
        total, _ = call(**p)
        row[key] = total or 0
        grand[key] += row[key]
    out.append(f"  {kw:<26} {row['RU']:>9,} {row['59']:>10,} {row['77']:>11,} {row['2']:>9,}")
out.append(f"  {'-' * 26} {'-' * 9} {'-' * 10} {'-' * 11} {'-' * 9}")
out.append(f"  {'TOTAL (with overlap)':<26} {grand['RU']:>9,} {grand['59']:>10,} "
           f"{grand['77']:>11,} {grand['2']:>9,}")

out.append("")
out.append("=" * 96)
out.append("E. ACTUAL HARVEST FEASIBILITY: pull 300 records in 3 pages of 100")
out.append("=" * 96)
seen = {}
for off in (0,):
    for lim in (100,):
        total, vacs = call(text="программист", region_code=59, limit=lim, offset=off)
        out.append(f"  offset={off} limit={lim} -> got {len(vacs) if vacs else 0} rows")
        if vacs:
            keys = sorted(vacs[0]["vacancy"].keys())
            out.append(f"  field list ({len(keys)}): {keys}")

text = "\n".join(out)
open("trudvsem_final_report.txt", "w", encoding="utf-8").write(text)
print(text)
