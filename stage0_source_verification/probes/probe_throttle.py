# -*- coding: utf-8 -*-
"""
Two questions:
 1. Was the earlier n=0 at a deep offset a transient throttle, or the real behaviour?
 2. Does modest concurrency help, or does the server throttle harder?
"""
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "https://opendata.trudvsem.ru/api/v1/vacancies"
H = {"User-Agent": "SalaryResearchProject/0.1 (1870037962@qq.com)"}
S = requests.Session()


def one(**params):
    t = time.time()
    try:
        r = S.get(URL, params={**params, "limit": params.pop("limit", 10)}, headers=H, timeout=60)
        if r.status_code != 200:
            return f"HTTP {r.status_code}", time.time() - t, 0
        j = r.json()
        n = len((j.get("results") or {}).get("vacancies") or [])
        return "OK", time.time() - t, n
    except Exception as e:  # noqa: BLE001
        return f"EXC {type(e).__name__}", time.time() - t, 0


print("=== 1. deep offset, twice in a row (throttle or real?) ===")
for label, params in (
    ("limit=10  offset=999", {"limit": 10, "offset": 999, "text": "программист"}),
    ("limit=100 offset=0  ", {"limit": 100, "offset": 0, "text": "программист"}),
    ("limit=10  offset=990", {"limit": 10, "offset": 990, "text": "программист"}),
    ("limit=10  offset=500", {"limit": 10, "offset": 500, "text": "программист"}),
    ("limit=10  offset=999", {"limit": 10, "offset": 999, "text": "программист"}),
):
    st, dt, n = one(**params)
    print(f"  {label}  {st:<8} n={n:<4} {dt:5.1f}s")
    time.sleep(0.3)

print()
print("=== 2. concurrency test: 8 distinct queries, 1 vs 4 workers ===")
queries = [{"text": k, "limit": 100, "offset": 0} for k in (
    "программист", "разработчик", "python", "1С",
    "аналитик данных", "тестировщик", "системный администратор", "devops",
)]

t = time.time()
seq = [one(**q) for q in queries]
seq_t = time.time() - t
seq_ok = sum(1 for s, _, _ in seq if s == "OK")
seq_rows = sum(n for _, _, n in seq)
print(f"  sequential : {seq_t:6.1f}s  ok={seq_ok}/8  rows={seq_rows}")

time.sleep(2)
t = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    par = list(ex.map(lambda q: one(**q), queries))
par_t = time.time() - t
par_ok = sum(1 for s, _, _ in par if s == "OK")
par_rows = sum(n for _, _, n in par)
print(f"  4 workers  : {par_t:6.1f}s  ok={par_ok}/8  rows={par_rows}")
print()
print(f"  per-request: sequential {seq_t/8:.1f}s    concurrent {par_t/8:.1f}s")
print(f"  speedup    : {seq_t/par_t:.1f}x")
print(f"  rows lost  : {seq_rows - par_rows}")
