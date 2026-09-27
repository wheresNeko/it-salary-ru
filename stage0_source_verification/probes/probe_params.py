# -*- coding: utf-8 -*-
"""Work out the correct query parameters for the Trudvsem API."""
import json

import requests

BASE = "https://opendata.trudvsem.ru/api/v1/vacancies"
H = {"User-Agent": "SalaryResearchProject/0.1 (1870037962@qq.com)"}
out = []


def t(label, **params):
    try:
        r = requests.get(BASE, params=params, headers=H, timeout=45)
    except Exception as e:  # noqa: BLE001
        out.append(f"  {label:<46} EXC {type(e).__name__}: {e}")
        return
    if r.status_code != 200:
        out.append(f"  {label:<46} HTTP {r.status_code}  {r.text[:110]}")
        return
    j = r.json()
    vacs = (j.get("results") or {}).get("vacancies") or []
    first = (vacs[0] or {}).get("vacancy") or {} if vacs else {}
    reg = (first.get("region") or {}).get("name")
    total = (j.get("meta") or {}).get("total")
    if total is None:
        out.append(f"  {label:<46} HTTP 200 but no meta.total -> "
                   f"{json.dumps(j, ensure_ascii=False)[:170]}")
        return
    out.append(
        f"  {label:<46} total={total:>9,}  n={len(vacs):>4}  "
        f"first_region={reg}"
    )


out.append("=" * 100)
out.append("A. REGION FILTER -- which code format actually works?")
out.append("=" * 100)
t("no region filter", limit=1)
t("regionCode=59", regionCode=59, limit=1)
t("regionCode=5900000000000", regionCode=5900000000000, limit=1)
t("region=59", region=59, limit=1)
t("regionId=59", regionId=59, limit=1)
t("regionCode=1 (Moscow?)", regionCode=1, limit=1)
t("regionCode=7700000000000", regionCode=7700000000000, limit=1)

out.append("")
out.append("=" * 100)
out.append("B. PAGE SIZE LIMITS")
out.append("=" * 100)
for lim in (100, 500, 1000, 2000, 5000):
    t(f"limit={lim}", limit=lim)

out.append("")
out.append("=" * 100)
out.append("C. DEEP PAGINATION")
out.append("=" * 100)
for off in (0, 1000, 10000, 400000):
    t(f"offset={off}, limit=10", offset=off, limit=10)

out.append("")
out.append("=" * 100)
out.append("D. TEXT SEARCH + IT KEYWORDS")
out.append("=" * 100)
for kw in ("программист", "python", "1С", "разработчик", "аналитик данных", "тестировщик"):
    t(f"text={kw!r}", text=kw, limit=1)

out.append("")
out.append("=" * 100)
out.append("E. DATE PARAMETER NAMES")
out.append("=" * 100)
for params in (
    {"date_from": "2026-01-01", "date_to": "2026-01-02"},
    {"dateFrom": "2026-01-01"},
    {"from": "2026-01-01"},
    {"date_from": "2026-01-01"},
    {"modifiedFrom": "2026-01-01"},
    {"date": "2026-01-01"},
):
    t(json.dumps(params, ensure_ascii=False), limit=1, **params)

out.append("")
out.append("=" * 100)
out.append("F. REGION DIRECTORY ENDPOINT")
out.append("=" * 100)
for ep in (
    "https://opendata.trudvsem.ru/api/v1/vacancies/region",
    "https://opendata.trudvsem.ru/api/v1/regions",
    "https://opendata.trudvsem.ru/api/v1/vacancies/regions",
):
    try:
        r = requests.get(ep, headers=H, timeout=30)
        out.append(f"  {ep:<58} HTTP {r.status_code}  {r.text[:120]}")
    except Exception as e:  # noqa: BLE001
        out.append(f"  {ep:<58} EXC {e}")

text = "\n".join(out)
open("trudvsem_params_report.txt", "w", encoding="utf-8").write(text)
print(text)
