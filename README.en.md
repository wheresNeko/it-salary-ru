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

**Done:** source selection and live verification, API behaviour
reverse-engineering, pipeline design.

`feasibility_check.py` runs 8 checks against the live API; all pass. Full output
is in [`verification_log.txt`](verification_log.txt).

| Check | Result |
|---|---|
| Connectivity and national volume | HTTP 200, **522,303** vacancies, no authentication |
| Actual harvest | **317 unique vacancies** from 56 requests, **0 failures** |
| Region filter verified | `region_code` 59 / 77 / 66 returned exactly Пермский край / Город Москва / Свердловская область |
| `salary` field coverage | **100%** |
| `salary_min` / `salary_max` | 97% / 91% |
| RegEx salary parse vs `salary_min` | **100% agreement** |
| RegEx skill extraction | 1С=134, python=52, sql=35, REST=30, Excel=29, Linux=26, git=19, C++=18 |
| Experience mined from free text | 1 yr ×10, 2 yr ×3, 3 yr ×7, 5 yr ×4, 7 yr ×7 |
| `creation-date` range | **2020-08 … 2026-09** |

---

## Known data-quality problems

All of these were measured, not assumed. They are the real material for Stage 4.

### 1. `salary_min == salary_max` on 34% of records

For open-ended adverts such as `от 40000` ("from 40,000") the portal writes **the
same value into both bounds**. So `salary_max` looks populated but carries no
information. Any model that treats `salary_max` as an upper bound is wrong.

→ Approach: model `salary_min` only; quantify the defect, choose and defend a
repair rule, and report the effect of that choice on the results.

### 2. The `skills` field is empty on 81% of records

The technology skills that should be the strongest salary predictors **cannot be
read off** — they have to be mined from the Russian free-text `requirements` and
`duty` fields. This is the core RegEx work, not decoration.

### 3. The Cyrillic trap: a naive pattern misses 31% of C++ vacancies

Russian adverts often write `С++` with a **Cyrillic** С (U+0421) because it looks
identical to the Latin one. Python's `re.IGNORECASE` does **not** fold across
alphabets:

```python
re.search(r"c\+\+", "C++", re.I)   # -> match
re.search(r"c\+\+", "С++", re.I)   # -> None   silently missed
```

Measured: of 26 C++ vacancies, **8 (31%) appear only in the Cyrillic spelling**
and are missed entirely by the naive pattern — including one whose job title is
*"Ведущий программист С++(Qt)"*.

Defects of this kind **raise no error**; they silently turn a feature into zero.
They only surface by looking at real records, never by reading documentation.
This is precisely why Stage 4 needs parametrised unit tests.

### 4. The `salary` free text is highly uniform

In the sample, 100% of values are of the form `"от N"` and 100% of currencies are
`«руб.»`. So salary string parsing is a *validation* tool rather than a feature
source — counter-intuitive, but that is what the measurement shows.

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

### Only one pagination recipe is reliable

Measured grid (OK = HTTP 200, X = HTTP 500):

| limit | off=0 | off=10 | off=100 | off=500 | off=900 | off=999 |
|---|---|---|---|---|---|---|
| 10 | OK | OK | OK | OK | OK | OK |
| 20 | OK | OK | OK | X | X | X |
| 50 | OK | OK | OK | X | X | X |
| 100 | OK | OK | OK | X | X | X |

Only `limit=10` with an offset sweep works at every depth. The maximum offset is
**999**, so a single (keyword × region) query yields at most ~1000 records —
coverage comes from **slicing queries**, not from deep paging.

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
├── feasibility_check.py         Source verification script (8 checks, passing)
├── verification_log.txt         Full output of the script (generated on run)
├── raw_samples/                 Real downloaded data
│   ├── sample_3_records.json        3 complete records (human-readable, shows the field structure)
│   └── trudvsem_raw_harvest.json    317 vacancies (1.4 MB)
└── api_investigation/           Probe scripts from the API reverse-engineering
    ├── probe_structure.py           record field structure
    ├── probe_params.py              parameter names (how region_code was found)
    ├── probe_paging.py              pagination parameter and offset ceiling
    └── probe_limits.py              limit/offset combination limits
```

`api_investigation/` is not scratch work — it is the reproducible evidence behind
claims such as "`regionCode` does not work".

---

## How to run

```powershell
cd C:\Users\Neko\Desktop\Workspace\it-salary-ru
pip install requests
python feasibility_check.py
```

The script contacts the live API, harvests a sample, and rewrites
`verification_log.txt` and `raw_samples/`.

Dependencies: `requests` (required). The DOM-parsing stage will additionally need
`pip install beautifulsoup4 lxml`.

Environment: Anaconda Python 3.14.6 at `C:\ProgramData\anaconda3\python.exe`.

---

## Roadmap

| Stage | Milestone | Deliverable |
|---|---|---|
| ✅ Done | Source verification and API reverse-engineering | Verification script, log, raw samples |
| Next | Full collection (all regions × all keywords + non-IT control sample) | Raw dataset + data dictionary |
| | Cleaning, RegEx extraction, quality gates | Analytical table + data-quality report |
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
