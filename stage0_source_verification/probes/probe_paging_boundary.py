# -*- coding: utf-8 -*-
"""
Test the hypothesis: the API fails when offset * limit exceeds ~10000.

Earlier grid (limit x offset): failures were at (100,100), (50,500), (20,500)
and successes at (10,999), (50,100), (20,100), (100,10).
offset*limit:  10000 X | 25000 X | 10000 X | 9990 OK | 5000 OK | 2000 OK | 1000 OK

If the rule is offset*limit <= 10000, we can fetch 100 records in ONE request
for most queries instead of ten -- a 10x cut in wall-clock time.
"""
import time

import requests

URL = "https://opendata.trudvsem.ru/api/v1/vacancies"
H = {"User-Agent": "SalaryResearchProject/0.1 (1870037962@qq.com)"}
S = requests.Session()

print(f"{'limit':>6} {'offset':>7} {'o*l':>8}  result")
print("-" * 52)
for limit, offset in (
    (100, 0), (100, 100), (100, 101), (100, 200),
    (50, 200), (50, 201),
    (99, 101), (99, 102),
    (200, 0), (200, 50),
    (10, 999), (10, 1000),
):
    t = time.time()
    try:
        r = S.get(URL, params={"limit": limit, "offset": offset, "text": "программист"},
                  headers=H, timeout=60)
        if r.status_code == 200:
            j = r.json()
            meta = j.get("meta") or {}
            n = len((j.get("results") or {}).get("vacancies") or [])
            total = meta.get("total")
            result = f"OK   n={n:<4} total={total}"
        else:
            result = f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        result = f"EXC {type(e).__name__}"
    print(f"{limit:>6} {offset:>7} {limit*offset:>8}  {result}   ({time.time()-t:.1f}s)")
