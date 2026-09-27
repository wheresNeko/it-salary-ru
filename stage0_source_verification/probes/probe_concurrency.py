# -*- coding: utf-8 -*-
"""How far does concurrency scale before the server pushes back?"""
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "https://opendata.trudvsem.ru/api/v1/vacancies"
H = {"User-Agent": "SalaryResearchProject/0.1 (1870037962@qq.com)"}
KW = ["программист", "разработчик", "python", "1С", "аналитик данных",
      "тестировщик", "системный администратор", "devops", "экономист",
      "юрист", "бухгалтер", "водитель", "продавец", "учитель", "повар",
      "слесарь"]

_local = {}


def sess():
    import threading
    tid = threading.get_ident()
    if tid not in _local:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=16, pool_maxsize=16))
        _local[tid] = s
    return _local[tid]


def one(kw):
    t = time.time()
    try:
        r = sess().get(URL, params={"text": kw, "limit": 100, "offset": 0},
                       headers=H, timeout=60)
        if r.status_code != 200:
            return "HTTP%d" % r.status_code, time.time() - t, 0
        j = r.json()
        n = len((j.get("results") or {}).get("vacancies") or [])
        return "OK", time.time() - t, n
    except Exception as e:  # noqa: BLE001
        return type(e).__name__, time.time() - t, 0


for workers in (4, 8):
    time.sleep(1)
    t = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, KW))
    dt = time.time() - t
    ok = sum(1 for s, _, _ in res if s == "OK")
    rows = sum(n for _, _, n in res)
    errs = {s for s, _, _ in res if s != "OK"}
    print(f"  {workers:>2} workers : {dt:6.1f}s for {len(KW)} req  "
          f"ok={ok}/{len(KW)}  rows={rows}  "
          f"per-req={dt/len(KW):.2f}s  eff={len(KW)/dt:.2f} req/s  errs={errs or '-'}")
