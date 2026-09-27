# -*- coding: utf-8 -*-
"""Find the real pagination and region parameters of the Trudvsem API."""
import json

import requests

BASE = "https://opendata.trudvsem.ru/api/v1/vacancies"
H = {"User-Agent": "SalaryResearchProject/0.1 (1870037962@qq.com)"}
out = []


def t(label, **params):
    try:
        r = requests.get(BASE, params=params, headers=H, timeout=45)
    except Exception as e:  # noqa: BLE001
        out.append(f"  {label:<50} EXC {type(e).__name__}: {e}")
        return None
    if r.status_code != 200:
        msg = ""
        try:
            msg = r.json().get("meta", {}).get("error", "")[:70]
        except Exception:  # noqa: BLE001
            msg = r.text[:70]
        out.append(f"  {label:<50} HTTP {r.status_code}  {msg}")
        return None
    j = r.json()
    meta = j.get("meta") or {}
    vacs = (j.get("results") or {}).get("vacancies") or []
    total = meta.get("total")
    ids = [((v or {}).get("vacancy") or {}).get("id", "")[:8] for v in vacs]
    out.append(f"  {label:<50} total={total}  n={len(vacs):>4}  first_id={ids[0] if ids else '-'}")
    return ids


out.append("=" * 104)
out.append("A. PAGINATION PARAMETER NAME -- which one actually moves the window?")
out.append("=" * 104)
base_ids = t("limit=10 (baseline)", limit=10)
for name in ("offset", "page", "start", "skip", "from", "begin", "pageNumber", "page_num"):
    ids = t(f"{name}=10, limit=10", limit=10, **{name: 10})
    if ids and base_ids:
        out.append(f"        -> window moved: {ids[0] != base_ids[0]}")

out.append("")
out.append("=" * 104)
out.append("B. OFFSET VALUES -- where is the ceiling?")
out.append("=" * 104)
for off in (0, 10, 50, 100, 200, 400, 500, 600, 900, 999, 1000):
    t(f"offset={off}, limit=10", offset=off, limit=10)

out.append("")
out.append("=" * 104)
out.append("C. OFFSET WITH MAX PAGE SIZE")
out.append("=" * 104)
for off in (0, 100, 200, 300, 900, 1000, 1900):
    t(f"offset={off}, limit=100", offset=off, limit=100)

out.append("")
out.append("=" * 104)
out.append("D. REGION FILTER -- detect via total change when combined with text")
out.append("=" * 104)
t("text=python (no region)", text="python", limit=1)
for rc in (59, "59", 5900000000000, "5900000000000", 1, 77, "7700000000000"):
    t(f"text=python & regionCode={rc}", text="python", regionCode=rc, limit=1)
for pname in ("region_code", "regionId", "region-id", "area", "regionName"):
    t(f"text=python & {pname}=59", text="python", limit=1, **{pname: 59})

out.append("")
out.append("=" * 104)
out.append("E. OTHER FILTERS")
out.append("=" * 104)
for params in (
    {"text": "python", "salaryMin": 100000},
    {"text": "python", "salary_min": 100000},
    {"text": "python", "minSalary": 100000},
    {"text": "python", "experience": 1},
    {"text": "python", "date_from": "2026-09-01"},
    {"text": "python", "date_to": "2026-09-10"},
    {"text": "python", "sort": "date"},
    {"text": "python", "order": "desc"},
):
    t(json.dumps(params, ensure_ascii=False), limit=1, **params)

text = "\n".join(out)
open("trudvsem_paging_report.txt", "w", encoding="utf-8").write(text)
print(text)
