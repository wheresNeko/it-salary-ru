# -*- coding: utf-8 -*-
"""Measure request latency: is the API slow, or is it the connection setup?"""
import time

import requests

URL = "https://opendata.trudvsem.ru/api/v1/vacancies"
H = {"User-Agent": "SalaryResearchProject/0.1 (1870037962@qq.com)"}
P = {"limit": 10, "offset": 0, "text": "программист"}


def bench(label, fn, n=5):
    fn()                                   # warm-up, not timed
    t = time.time()
    for _ in range(n):
        fn()
    dt = (time.time() - t) / n
    print(f"  {label:<34} {dt:6.2f} s/request")
    return dt


print("=== 5 requests each, warm-up excluded ===")
d1 = bench("requests.get (new conn each time)",
           lambda: requests.get(URL, params=P, headers=H, timeout=40))

s = requests.Session()
d2 = bench("requests.Session (pooled conn)",
           lambda: s.get(URL, params=P, headers=H, timeout=40))

print()
print(f"  speedup: {d1 / d2:.1f}x")
print()
print("=== server-side time only (TTFB minus connect) ===")
r = s.get(URL, params=P, headers=H, timeout=40)
print(f"  elapsed={r.elapsed.total_seconds():.2f}s  status={r.status_code}")
print()
print("=== does response time grow with offset? ===")
for off in (0, 100, 500, 900):
    t = time.time()
    r = s.get(URL, params={**P, "offset": off}, headers=H, timeout=40)
    print(f"  offset={off:<5} {time.time() - t:6.2f}s  status={r.status_code}")
