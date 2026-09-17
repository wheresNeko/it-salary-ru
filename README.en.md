# Predicting IT Salary Levels in the Russian Labour Market

A machine learning project built on open data from the Russian state employment
portal. The goal is to predict the salary offered for an IT vacancy and to
quantify which factors actually drive it.

> **Русская / 中文版: [README.md](README.md)**

---

## Research question

> Which factors determine the salary level of an IT specialist in the Russian
> labour market, and how accurately can the offered salary be predicted from the
> vacancy text and its structured attributes?

Sub-questions:

1. How much of the salary signal is carried by structured attributes (region,
   education, experience, employer, schedule) versus by the free-text
   requirements — and how much has to be mined from text because the portal
   publishes no structured skill field?
2. How large is the IT salary premium relative to the rest of the labour market
   once region and qualification are controlled for?
3. Does the prediction quality justify a neural network, or does a simpler model
   perform equally well?

Sub-question 3 is deliberate: the validation stage should produce a defensible
conclusion, not just a list of test names.

---

## Why this data source

**Primary source:** [Trudvsem / «Работа России»](https://trudvsem.ru/opendata/api)
(the Russian state employment portal) official open data API.

| Property | Value |
|---|---|
| Endpoint | `https://opendata.trudvsem.ru/api/v1/vacancies` |
| Operator | Russian federal labour and employment service (Роструд) |
| Authentication | **None required** |
| Format | JSON |
| National volume | **522,303** vacancies |
| Cost | Free |
| Legal basis | Published as official open data, explicitly licensed for reuse |

hh.ru (the largest Russian job board) was the first candidate. **It closed its
public vacancy search API in April 2026**: `/vacancies` returns `403` to
unauthenticated clients, and keys are issued only to verified employers and
recruiting services after moderation. Measured directly: hh.ru `/areas` and
`/professional_roles` return 200, while `/vacancies`, `/vacancies/{id}` and
`/employers` all return 403.

This is not merely a source swap. A design that depends on scraping a large
commercial job board sits on a legally and technically unstable foundation,
whereas open data carries an explicit reuse licence.

---

## What the data looks like

Each vacancy record carries **31 fields**:

| Group | Fields |
|---|---|
| Salary | `salary` (free text), `salary_min`, `salary_max`, `currency` |
| Vacancy | `job-name`, `qualification`, `schedule`, `employment`, `code_profession`, `typicalPosition`, `work_places` |
| Text | `requirements`, `duty` (Russian free text), `skills` (**empty on 81%**) |
| Requirements | `requirement.education`, `requirement.experience` |
| Employer | `company.name`, `inn`, `ogrn`, `kpp`, `site`, industry |
| Region | `region.name`, `region.region_code`, `addresses` (with lat/lng) |
| Time | `creation-date`, `date_modify` |

### Volume (measured)

The IT subset is obtained by slicing `text=` keywords against `region_code`:

| Keyword | Russia | Perm (59) | Moscow (77) | Sverdlovsk (66) |
|---|---:|---:|---:|---:|
| 1С | 9,042 | 133 | 592 | 327 |
| программист | 2,089 | 23 | 220 | 97 |
| инженер-программист | 996 | 15 | 83 | 46 |
| системный администратор | 894 | 9 | 44 | 16 |
| разработчик | 742 | 11 | 167 | 37 |
| аналитик данных | 511 | 19 | 118 | 11 |
| python | 431 | 5 | 144 | 21 |
| тестировщик | 79 | 1 | 19 | 7 |
| **sum (overlapping)** | **14,784** | **216** | **1,387** | **562** |

---

## Pipeline design

```
STAGE 1  DATA MINING
         keyword x region slicing -> RegEx skill and experience extraction from
         free text -> salary string validation
         output: immutable raw JSON snapshots

STAGE 2  DATA PROCESSING
         salary normalisation -> deduplication -> defect repair policy ->
         feature table -> quality gates
         output: Parquet analytical table + data-quality report

STAGE 3  PREDICTIVE ANALYTICS
         M0 baseline      (median of the region x education cell)
         M1 Ridge         (one-hot + TF-IDF)
         M2 LightGBM
         M3 small MLP
         output: model comparison table + error analysis

STAGE 4  PROOF THAT THE DATA AND RESULTS ARE CORRECT
         parametrised unit tests -> integration test -> schema tests ->
         cross-field consistency proof -> leakage audit -> baseline comparison
         -> robustness analysis
         output: test suite + validation report
```

**The core of Stage 3 is the M0→M3 comparison.** It answers sub-question 3
instead of assuming that "neural" means "better".

---

## Project status

Done: source verification and API reverse-engineering → **Stage 1 full
collection** → data dictionary.

### Stage 1 collection results

`collect.py` used 8 concurrent workers and finished 2,002 requests in 24.7
minutes:

| | IT vacancies | Non-IT control |
|---|---:|---:|
| Unique vacancies | **12,656** | **9,731** |
| Distinct regions | 89 | 87 |
| `salary_min` present | 99% | 99% |
| `creation-date` range | 2016-07 … 2026-09 | 2015-08 … 2026-09 |

**22,387 records** in total, 13.4 MB gzipped in the repository. Field structure
and defect statistics are in [`docs/data_dictionary.md`](docs/data_dictionary.md)
— every number there is computed from the data, none is hand-written.

The region directory `raw_samples/regions.json` records the **78 regions** found
by probing codes 1–92; it is measured, not a hard-coded list.

### Source verification (earlier stage)

`feasibility_check.py` runs 8 checks against the live API; all pass. Output is in
[`verification_log.txt`](verification_log.txt).

| Check | Result |
|---|---|
| Connectivity and national volume | HTTP 200, **522,303** vacancies, no authentication |
| Region filter verified | `region_code` 59 / 77 / 66 returned exactly Пермский край / Город Москва / Свердловская область |
| `salary` field coverage | **100%** |
| RegEx salary parse vs `salary_min` | **100% agreement** |
| RegEx skill extraction | 1С / python / sql / REST / Excel / Linux / C++ — 22 patterns |
| `creation-date` range | **2020-08 … 2026-09** |

---

## Known data-quality problems

All of these were measured, not assumed. They are the real material for Stage 4.

### 1. `salary_min == salary_max` on 36% of records

For open-ended adverts such as `от 40000` ("from 40,000") the portal writes **the
same value into both bounds**. So `salary_max` looks populated but carries no
information. Any model that treats `salary_max` as an upper bound is wrong.

Measured across 21,941 records carrying salary: **7,654 (36%) have identical
bounds**.

→ Approach: model `salary_min` only; quantify the defect, choose and defend a
repair rule, and report the effect of that choice on the results.

### 2. The `skills` field is empty on 77% of records

The technology skills that should be the strongest salary predictors **cannot be
read off** — they have to be mined from the Russian free-text `requirements` and
`duty` fields. This is the core RegEx work, not decoration.

Measured: **16,978 of 22,387 records (77%)** have an empty `skills` field.

### 3. The Cyrillic trap: a naive pattern misses 43% of C++ vacancies

Russian adverts often write `С++` with a **Cyrillic** С (U+0421) because it looks
identical to the Latin one. Python's `re.IGNORECASE` does **not** fold across
alphabets:

```python
re.search(r"c\+\+", "C++", re.I)   # -> match
re.search(r"c\+\+", "С++", re.I)   # -> None   silently missed
```

Measured across all 22,387 records: 139 match only the Latin spelling, **102 match
only the Cyrillic spelling**, and 100 use both — a naive pattern's false-negative
rate is **102/239 ≈ 43%**. (The earlier 317-record sample suggested 31%; the
larger sample made the problem worse, not better.)

Defects of this kind **raise no error**; they silently turn a feature into zero.
They only surface by looking at real records, never by reading documentation.
This is precisely why Stage 4 needs parametrised unit tests.

### 4. The `salary` free text is highly uniform

Across the 21,941 records carrying salary, **100% of values are of the form
`"от N"`** and **100% of currencies are `«руб.»`**. So salary string parsing is a
*validation* tool rather than a feature source — counter-intuitive, but that is
what the measurement shows.

---

## API notes (read before re-running)

The API has several **silent failure** modes, fully documented in the
`feasibility_check.py` docstring:

### Unknown parameter names are silently ignored

`regionCode`, `regionId`, `area` and `regionName` all return the **whole
country** with HTTP 200 and no warning. The correct name is `region_code`
(snake_case).

> ⚠️ A client that does not verify the returned `region.name` will silently train
> on unfiltered national data.
> `feasibility_check.py` asserts the returned region on every request.

### Paging stopped working (the behaviour changed mid-investigation)

Earlier measurements reached `offset` 999. **`offset > 0` now returns HTTP 200
with ZERO records** — the behaviour changed within the same session. Only
`offset=0` is reliable.

| limit | offset=0 | offset>0 |
|---|---|---|
| 100 | OK, returns 100 records | **returns 0 records** |
| 10 | OK | **returns 0 records** |

A `limit` above 100 is silently clamped to 100.

The collection strategy is therefore: **one `limit=100&offset=0` request per
(keyword, region) cell, with coverage coming entirely from slicing.** A cell with
more than 100 vacancies yields only its first 100 — an accepted and documented
loss.

### Server latency is 5–6 s per request, so concurrency is mandatory

A single request takes 5–6 seconds, and **connection pooling does not help** —
the bottleneck is server-side processing, not handshakes. Measured scaling:

| workers | throughput | failures |
|---|---|---|
| sequential | 0.19 req/s | 0 |
| 4 | 0.56 req/s | 0 |
| 8 | **1.20 req/s** | 0 |

`collect.py` therefore uses 8 workers — the only way to bring the harvest down
from eight hours to under half an hour.

### There is no usable date filter

`date_from`, `dateFrom`, `date` and `from` are ignored; `modifiedFrom` /
`modifiedTo` return HTTP 500.

However, every record carries `creation-date`, and the harvested sample spans
2020-08 to 2026-09 — **the time axis exists in the data**, it just cannot be used
as a query filter.

---

## Repository layout

```
it-salary-ru/
├── README.md                    Chinese / Russian-facing notes
├── README.en.md                 This file
├── feasibility_check.py         Source verification script (8 checks)
├── collect.py                   Stage 1 collector (8 concurrent workers)
├── make_data_dictionary.py      Generates the data dictionary from raw data
├── verification_log.txt         Verification script output
├── collection_log.txt           Collection run log
├── docs/
│   └── data_dictionary.md       Data dictionary (computed, not hand-written)
├── raw_samples/                 Real downloaded data
│   ├── regions.json                 78 region directory (measured, not hard-coded)
│   ├── sample_3_records.json        3 complete records (human-readable)
│   ├── trudvsem_it_harvest.json     IT vacancy harvest
│   ├── trudvsem_control_harvest.json Non-IT control set
│   └── verification_sample.json     Sample from the verification script
└── api_investigation/           Probe scripts from the API reverse-engineering
    ├── probe_structure.py           record field structure
    ├── probe_params.py              parameter names (how region_code was found)
    ├── probe_paging.py              pagination parameter
    ├── probe_limits.py              limit/offset combination limits
    ├── probe_paging_boundary.py     falsification of the offset*limit hypothesis
    ├── probe_latency.py             latency attribution (server vs handshake)
    ├── probe_throttle.py            deep-offset failure and concurrency
    └── probe_concurrency.py         scaling (4 vs 8 workers)
```

`api_investigation/` is not scratch work — it is the reproducible evidence behind
claims such as "`regionCode` does not work" and "paging stopped working".

---

## How to run

```powershell
cd C:\Users\Neko\Desktop\Workspace\it-salary-ru
pip install requests

python collect.py                    # Stage 1 collection (~25 minutes)
python make_data_dictionary.py       # regenerate the data dictionary
python feasibility_check.py          # 8 source-verification checks
```

`collect.py` accepts `--budget N` (request cap), `--seconds N` (time cap),
`--phase national|grid|both`, and `--refresh-regions`. An interrupted run keeps
its region directory and resumes from it.

Dependencies: `requests` (required). The DOM-parsing stage will additionally need
`pip install beautifulsoup4 lxml`.

Environment: Anaconda Python 3.14.6 at `C:\ProgramData\anaconda3\python.exe`.

---

## Roadmap

| Stage | Milestone | Deliverable |
|---|---|---|
| ✅ Done | Source verification and API reverse-engineering | Verification script, log, raw samples |
| ✅ Done | **Stage 1 full collection** | 22,387 raw records, 78-region directory, data dictionary |
| Next | Cleaning, RegEx extraction, quality gates | Analytical table + data-quality report |
| | Exploratory analysis, feature engineering | EDA notebook, feature specification |
| | M0 baseline and M1 Ridge | Evaluation harness, first honest numbers |
| | M2 LightGBM and M3 MLP comparison | Model comparison table, error analysis |
| | Unit tests, integration test, leakage audit | Test suite, validation report |
| | Write-up and presentation | Final report, reproducible repository |

---

## Known limitations of the source

Stated plainly: the portal skews towards state-sector and blue-collar roles, and
IT accounts for only about 15k of 522k vacancies. That is an objective limitation
of the source, and conclusions hold only within the population it describes. The
IT analysis will be national in scope, with Perm presented as a sub-analysis.
