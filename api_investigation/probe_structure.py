# -*- coding: utf-8 -*-
"""Probe the Trudvsem open-data API and dump a readable structure report."""
import json
import pathlib
import sys
import urllib.parse

import requests

BASE = "https://opendata.trudvsem.ru/api/v1/vacancies"
HEADERS = {"User-Agent": "SalaryResearchProject/0.1 (1870037962@qq.com)"}
RAW = pathlib.Path(__file__).resolve().parent / "raw_samples"
RAW.mkdir(exist_ok=True)

out = []


def p(line=""):
    out.append(str(line))


def get(params):
    url = BASE + "?" + urllib.parse.urlencode(params)
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- 1. paging
p("=" * 78)
p("1. PAGINATION AND TOTAL VOLUME")
p("=" * 78)
for label, params in [
    ("national, limit=1", {"limit": 1}),
    ("national, limit=100", {"limit": 100}),
    ("Perm krai (regionCode=59)", {"regionCode": 59, "limit": 1}),
    ("Perm, text=python", {"text": "python", "limit": 1}),
    ("Perm, text=программист", {"text": "программист", "limit": 1}),
]:
    try:
        j = get(params)
        meta = j.get("meta", {})
        n = len(j.get("results", {}).get("vacancies", []))
        p(f"  {label:<28} total={meta.get('total'):>9,}  returned={n:>4}  limit={meta.get('limit')}")
    except Exception as e:  # noqa: BLE001
        p(f"  {label:<28} FAIL {type(e).__name__}: {e}")

# ------------------------------------------- 2. full record for one vacancy
p("")
p("=" * 78)
p("2. COMPLETE FIELD STRUCTURE OF ONE VACANCY")
p("=" * 78)
data = get({"regionCode": 59, "limit": 100})
vacancies = data["results"]["vacancies"]
(RAW / "trudvsem_perm_sample.json").write_text(
    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
)
p(f"  saved {len(vacancies)} raw vacancies -> raw_samples/trudvsem_perm_sample.json")
p("")
p(json.dumps(vacancies[0]["vacancy"], ensure_ascii=False, indent=2)[:3500])

# ------------------------------------------------------- 3. salary coverage
p("")
p("=" * 78)
p("3. SALARY COVERAGE AND FORMAT  <-- decides the whole project")
p("=" * 78)


def salary_of(v):
    return (v.get("vacancy") or {}).get("salary") or {}


with_salary = [v for v in vacancies if salary_of(v)]
p(f"  vacancies fetched                     : {len(vacancies)}")
p(f"  vacancy.salary is present             : {len(with_salary)} "
  f"({100 * len(with_salary) / max(len(vacancies), 1):.0f}%)")
p(f"  vacancy.region present                : {sum(1 for v in vacancies if (v.get('vacancy') or {}).get('region'))}")
p("")
p("  Raw salary objects (first 12 with data):")
shown = 0
for v in vacancies:
    s = salary_of(v)
    if s and shown < 12:
        p(f"    {json.dumps(s, ensure_ascii=False)}")
        shown += 1
if shown == 0:
    p("    (none)")

# --------------------------------------------------- 4. free-text for RegEx
p("")
p("=" * 78)
p("4. FREE-TEXT FIELDS  <-- the RegEx target")
p("=" * 78)
v0 = vacancies[0]["vacancy"]
for key in ("vacancyDuty", "requirements", "skills", "company", "region",
            "employment", "schedule", "vacancyName", "address"):
    val = v0.get(key)
    if isinstance(val, str):
        p(f"  {key:<14} (str)  {val[:200]!r}")
    elif isinstance(val, (dict, list)):
        p(f"  {key:<14} ({type(val).__name__})  {json.dumps(val, ensure_ascii=False)[:250]}")
    else:
        p(f"  {key:<14} = {val!r}")

# --------------------------------- 5. how many carry salary ONLY in text
p("")
p("=" * 78)
p("5. HOW MANY VACANCIES MENTION A SALARY IN FREE TEXT BUT HAVE NO FIELD?")
p("=" * 78)
import re  # noqa: E402

SAL = re.compile(
    r"(?:от|до)?\s*(\d[\d\s\u00a0]*)\s*(?:руб|₽|тыс|к\b)",
    re.IGNORECASE,
)
text_only = 0
examples = []
for v in vacancies:
    vac = v.get("vacancy") or {}
    if vac.get("salary"):
        continue
    blob = " ".join(
        str(vac.get(k) or "")
        for k in ("vacancyDuty", "requirements", "vacancyName", "skills")
    )
    hits = SAL.findall(blob)
    if hits:
        text_only += 1
        if len(examples) < 5:
            examples.append((vac.get("vacancyName", "")[:50], SAL.findall(blob)[:3]))
p(f"  no salary field BUT a salary-like string in the text: {text_only}")
for name, hits in examples:
    p(f"    {name!r} -> matches {hits}")

# ------------------------------------------------------------- 6. time axis
p("")
p("=" * 78)
p("6. TIME AXIS -- can we get history without daily snapshots?")
p("=" * 78)
sample = vacancies[0]["vacancy"]
for k in sample:
    if any(t in k.lower() for t in ("date", "creation", "modified", "publish")):
        p(f"  {k} = {sample[k]!r}")
for params in [
    {"modifiedFrom": "2026-01-01", "modifiedTo": "2026-01-02", "limit": 1},
    {"dateFrom": "2026-01-01", "limit": 1},
    {"regionCode": 59, "offset": 100, "limit": 5},
]:
    try:
        j = get(params)
        p(f"  {params} -> total={j.get('meta', {}).get('total')}")
    except Exception as e:  # noqa: BLE001
        p(f"  {params} -> FAIL {type(e).__name__}: {e}")

# ------------------------------------------------------------- 7. IT subset
p("")
p("=" * 78)
p("7. WHY THE FIELD STRUCTURE IS MESSY (justifies the validation stage)")
p("=" * 78)
for v in vacancies[:6]:
    vac = v.get("vacancy") or {}
    comp = vac.get("company") or {}
    p(f"  {str(vac.get('vacancyName'))[:46]:<46} | "
      f"salary={json.dumps(salary_of(v), ensure_ascii=False):<46} | "
      f"company={str(comp.get('name'))[:28]}")

pathlib.Path(__file__).with_name("trudvsem_probe_report.txt").write_text(
    "\n".join(out), encoding="utf-8"
)
print("\n".join(out))
