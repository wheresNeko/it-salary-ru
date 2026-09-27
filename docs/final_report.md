# Predicting IT Salary Levels from Russian State Employment Open Data

**Final report.** This document consolidates the project; the four stage reports
it draws on are listed in [§9](#9-references).

*中文版: [final_report.zh.md](final_report.zh.md)*

---

## Abstract

This project asks which factors determine the salary offered for an IT vacancy in
the Russian labour market, and how accurately that salary can be predicted from
the vacancy text and its structured attributes.

The work follows a four-stage pipeline — data mining, data processing,
predictive analytics, validation — applied to 22,387 vacancy records harvested
from the Trudvsem ("Работа России") state employment portal, whose data is
published as official open data.

Three results are worth stating up front.

**The most valuable output is not a model but a set of defects.** Four data
problems were found and quantified, three of which fail silently: a zero sentinel
that encodes "salary not specified" as the literal value 0; a field documented as
an integer that is actually an undocumented code and had to be withdrawn from
the feature set; and a Cyrillic/Latin character confusion that makes a naive
regular expression miss 43% of all C++ vacancies. Each would have produced a
plausible-looking wrong answer.

**On this dataset, a neural network does not justify itself.** Given
byte-identical inputs, gradient boosting reaches R² = 0.615 against 0.595 for a
multi-layer perceptron, and beats it on all seven seeds tested.

**The IT premium largely disappears under controls.** The raw gap between the IT
and non-IT medians is 5,000 RUR (11.1%); with region, education, occupation and
skill features held constant, the estimated premium is **+1.0%**.

---

## 1. Research question

> Which factors determine the salary level of an IT specialist in the Russian
> labour market, and how accurately can the offered salary be predicted from the
> vacancy text and its structured attributes?

Three sub-questions structure the work:

1. How much of the salary signal is carried by structured attributes versus by
   the free text — and how much of the free text must be mined to recover the
   technology skills the portal does not publish in structured form?
2. How large is the IT salary premium once region and qualification are
   controlled for?
3. Does the prediction quality justify a neural network, or does a simpler model
   perform equally well?

Sub-question 3 is deliberately falsifiable. It is answered in §4.2, and the
answer is no.

---

## 2. Data source

### 2.1 The first candidate turned out to be closed

The obvious source, and the first one selected, was hh.ru — the largest Russian
job board. **It is no longer usable for this purpose.** From April 2026 hh.ru
closed public access to its vacancy search endpoint: `GET /vacancies` returns
`403 Forbidden` to unauthenticated clients, and keys are issued only to verified
employers and recruiting services after moderation.

This was confirmed by direct measurement rather than by reading about it:

| Endpoint | Result |
|---|---|
| `hh.ru /areas` | **200** (2,425,807 bytes) |
| `hh.ru /professional_roles` | **200** (53,921 bytes) |
| `hh.ru /vacancies` | **403 Forbidden** |
| `hh.ru /vacancies/{id}` | **403 Forbidden** |
| `hh.ru /employers` | **403 Forbidden** |

The pattern is endpoint-level authorisation, not a network or header problem:
reference endpoints answer normally while every endpoint returning vacancy
records is refused. The distinction matters beyond this project — a design that
depends on scraping a large commercial job board sits on a legally and
technically unstable foundation, whereas open data carries an explicit reuse
licence.

### 2.2 What replaced it

Trudvsem is the Russian state employment portal, operated by Роструд. Its
vacancy API requires no key, no registration and no moderation, and the data is
published as official open data.

| Property | Value |
|---|---|
| Endpoint | `https://opendata.trudvsem.ru/api/v1/vacancies` |
| Authentication | none |
| Fields per record | 31 |
| National volume at collection time | 522,303 vacancies |
| Licence | official open data (see [LICENSE](../LICENSE)) |

The portal publishes no occupation-category filter, so the IT subset is obtained
by slicing `text=` keywords against `region_code`.

### 2.3 Collection, and what the collection constraints imply

The API's behaviour was reverse-engineered before any large harvest, because
three of its properties determine what is possible. All three are documented
with reproducible probes in `stage0_source_verification/probes/`.

**Unknown parameter names are silently ignored.** `regionCode`, `regionId`,
`area` and `regionName` all return the *whole country* with HTTP 200 and no
warning. The correct name is `region_code`. A client that does not verify the
returned `region.name` will silently train on unfiltered national data; this
collector asserts the returned region on every request.

**Paging stopped working during the investigation.** Offsets up to 999 returned
data early on; `offset > 0` now returns HTTP 200 with **zero records**. Coverage
therefore comes from slicing queries, never from paging, and each
(keyword, region) cell is capped at 100 records.

**The server takes 5–6 s per request**, and connection pooling does not help —
it is server-side latency. Concurrency does help, and was measured rather than
assumed: 4 workers give 0.56 req/s, 8 give 1.20 req/s, with zero failures. The
collector uses 8, which is what brings the harvest from eight hours to 24.7
minutes.

The result is 2,002 requests yielding **22,387 records** — 12,656 IT and 9,731
non-IT control — across 89 regions, plus a region directory of 78 regions
discovered by probing codes 1–92 rather than hard-coded.

**These constraints are a sampling method, not merely an implementation
detail.** Within one query the API returns its own ordering, which is
recency-weighted but neither random nor sorted. The harvest is therefore a
convenience sample, not a probability sample, and every estimate in this report
describes that sample.

---

## 3. Method

### 3.1 Processing: two repairs, one withdrawal

Stage 2 turns 22,387 raw records into a 19,578 × 72 analytical table. Four
decisions in that step change what the data means, and all four are recorded
rather than performed silently.

**Duplicates.** 446 records appeared under more than one keyword (a vacancy
matching both an IT and a control term). These are collapsed into the IT set so
the two groups are disjoint — otherwise the IT premium would partly compare a
group with itself. A further 2,363 near-duplicates (same employer, same
normalised title, same salary, reposted) are collapsed to the most recent copy.

**The zero sentinel.** The portal encodes "salary not specified" as the literal
value `0` — the free text reads `"от 0"` — rather than leaving the field null.
179 records are affected. A sentinel that looks like real data is the most
dangerous kind: `log(0)` is `-inf`, and a model will otherwise train on 0-rouble
monthly salaries. These are set to missing and flagged.

**The degenerate range.** The free-text `salary` field is `"от N"` in every one
of the 21,941 salaried records, so the text never carries an upper bound. Where
`salary_max <= salary_min` — 38.5% of the table — the range carries no
information, so `salary_max` is set to missing and the row flagged. The raw
values are retained, so a different policy can be evaluated later without
re-harvesting.

**A field that had to be withdrawn.** `requirement.experience` is documented as
an integer and reads like a year count. It is neither. The observed code space is
`0,1,2,3,4,5,6,7,10,18,20,21,23,31,42` with no documented scale, and
cross-checking against the free text settles it: adverts carrying `code = 0` ask
for up to **35 years** of experience, so `0` cannot mean "no experience
required"; `code = 10` appears on adverts reading "от 10 лет" while `code = 18`
appears on ones reading "от 1 года". The column is kept as an opaque code for
audit and **excluded from the feature set**. Experience is consequently not a
usable variable from this source.

### 3.2 The text mining, and why the alphabet matters

The structured `skills` field is empty on 77.4% of records, so technology
features must be mined from Russian free text: 24.1% of records have a populated
structured field against 36.7% where RegEx recovered at least one skill.

The naive approach to that mining fails quietly. Russian adverts routinely write
`C++` and `C#` with a **Cyrillic** capital С (U+0421), which is visually
identical to the Latin C, and Python's `re.IGNORECASE` does not fold Cyrillic
onto Latin:

```python
re.search(r"c\+\+", "C++", re.I)   # -> match
re.search(r"c\+\+", "С++", re.I)   # -> None    silently missed
```

Measured across the corpus: 137 vacancies match only the Latin spelling, **102
match only the Cyrillic spelling**, and 100 use both. A naive pattern's
false-negative rate is **102/239 = 43%**.

The tempting fix — normalising the whole text — would corrupt ordinary Russian,
where `с` is one of the most common prepositions and `Р`, `О`, `А` are extremely
common letters. Instead the patterns accept either alphabet at the few positions
where the confusion actually occurs, anchored with an explicit left boundary so
they cannot fire from inside a Russian word. A test asserts that
`"Работа с детьми"` yields no technology at all.

### 3.3 Modelling

**The split is temporal, not random.** Training uses the 14,513 postings created
before 2026-08-20; testing uses the 4,886 created after it. A random split would
be invalid twice over: employers repost identical adverts, so near-duplicates
survive Stage 2's deduplication and would be scattered across both folds; and the
sample is recency-weighted, so a random split asks the model to interpolate
within a period rather than extrapolate forward.

**Four model classes, on controlled inputs.** Two feature designs are used:
a sparse one (one-hot + the full TF-IDF vocabulary) for the linear model, and a
dense one (one-hot + numerics + skill flags + a truncated-SVD compression of the
same TF-IDF) shared byte-for-byte by three models. M1b, M2 and M3 receive
identical matrices on purpose, which is what makes the comparison a test of the
model class rather than of the feature engineering.

**The leakage audit is exhaustive by construction.** The target is
`log(salary_min)`. Every column of the table must carry an explicit decision —
feature, text source, target-derived, or excluded with a reason — and a test
fails when one does not. This matters because `salary_text` literally reads
`"от <target>"`; it would have produced a near-perfect and entirely meaningless
model. A further test pins the known-dangerous column names as leakage, so the
audit cannot be passed by *deleting declarations* rather than classifying
columns.

### 3.4 Validation

103 tests, running in ~16 s with no network, enforced on every push by CI on
Python 3.12 and 3.13. Four seams carry the claims about the data — the parsing
functions, the repair policy, the leakage invariant, and the end-to-end path —
and were agreed before any test was written. A fifth file checks that every
relative link in every document still resolves, which is repository hygiene
rather than a claim about the data. The tests found two real bugs and forced one
interface improvement; all three are described in the validation report.

---

## 4. Results

### 4.1 What the data turned out to be

| Measurement | Value |
|---|---:|
| Raw records | 22,387 |
| Analytical table | 19,578 rows × 72 columns |
| Salary usable | 99.09% |
| Zero sentinel (`"от 0"`) | 179 (0.91%) |
| `salary_max == salary_min` | 35.8% of those with both |
| `skills` field empty | 77.4% |
| Cyrillic-С false-negative rate on C++ | **43%** |
| Quality gates passed | **10/10** |

The salary distribution is heavily right-skewed — a median of 47,000 RUR against
a 99th percentile of 200,000 — which is why the target is modelled on the log
scale.

### 4.2 The model comparison

All models are evaluated on the same temporal hold-out. MAE is reported in RUR
after exponentiating back from the log scale; the test-set median salary is
49,999 RUR.

| Model | MAE (RUR) | As % of median | Median AE | R² (log) |
|---|---:|---:|---:|---:|
| M0a global train median | 28,594 | 57.2% | 15,000 | −0.177 |
| M0b (region × education) cell median | 23,674 | 47.3% | 12,668 | 0.187 |
| M1 Ridge — sparse, full TF-IDF | 17,966 | 35.9% | 10,224 | 0.592 |
| M1b Ridge — dense design | 18,264 | 36.5% | 10,402 | 0.588 |
| **M2 gradient boosting — dense** | **17,478** | **35.0%** | **9,692** | **0.615** |
| M3 neural network (MLP) — dense | 17,945 | 35.9% | 10,032 | 0.595 |

The best learned model improves on the strongest baseline by **26.2%**.

**Answer to sub-question 3: no, the neural network does not justify itself.**
M2 and M3 receive identical inputs, so the only difference is the model class.
The MLP is 2.7% worse on MAE and 0.020 lower on R².

A single seed is not evidence for a neural network, so the MLP was retrained
under seven seeds: R² = 0.6002 ± 0.0038 (range 0.595–0.605). **M2 beats all
seven**, on both metrics. The honest framing is that gradient boosting wins by
roughly the width of the network's own seed noise — it does not win by a margin
that would survive any conceivable tuning. What the neural network fails to do is
provide any benefit *beyond* that noise, while costing interpretability and a
tuning burden.

Moving from the dense design to the full TF-IDF vocabulary (M1b → M1) is worth
1.6% of MAE, which is the measurable contribution of the text mining.

### 4.3 The IT premium largely disappears under controls

Stage 2 reports a raw gap: the IT median is 50,000 RUR against 45,000 for the
control set, **+11.1%**.

That is not a controlled comparison — IT vacancies are concentrated in Moscow
and St Petersburg, and in higher-qualification occupations. Placing the `is_it`
indicator inside a Ridge regression alongside region, education, occupation,
schedule and skill features gives:

- coefficient on `is_it` (log scale): **+0.0103**
- implied salary premium, all else equal: **+1.0%**

**The raw gap is therefore almost entirely composition, not a premium.** Read
the residual 1% as a partial association rather than a causal effect: the
controls are whatever this dataset measures, and unmeasured differences
(seniority, contract type, employer sector) remain.

### 4.4 What the model actually uses

Permutation importance on M2, measured on 3,000 test rows:

| Rank | Feature | Increase in MAE when shuffled |
|---:|---|---:|
| 1 | `region_name_Город Москва` | 0.0362 |
| 2 | `lng` | 0.0287 |
| 3 | `lat` | 0.0213 |
| 4 | `specialisation_Образование, наука` | 0.0145 |
| 5 | `svd_17` (text component) | 0.0131 |
| 6 | `schedule_Неполный рабочий день` | 0.0114 |

Geography dominates, and the text contributes through the SVD components rather
than through any single skill flag. That is a substantive finding about the
Russian labour market as much as about the model: where a vacancy is located
matters more than what it asks for.

### 4.5 Error analysis, and a systematic bias

| Subgroup | Rows | MAE (RUR) | Bias (RUR) |
|---|---:|---:|---:|
| IT | 3,015 | 19,206 | −7,792 |
| control | 1,871 | 14,694 | −5,519 |
| `salary_open_ended` = False | 2,886 | 14,665 | −4,026 |
| `salary_open_ended` = True | 2,000 | 21,536 | −11,100 |
| region: Город Москва | 458 | 41,657 | −21,776 |
| region: Город Санкт-Петербург | 323 | 27,256 | −14,857 |

Two things stand out.

**Moscow is by far the hardest subgroup** — an MAE of 41,657 RUR, twice the
overall figure. Salary dispersion in the capital is large, and the model's
features do not resolve it.

**The bias is negative in every subgroup**, averaging roughly **−6,900 RUR**
(13.8% of the median). This is not a subgroup artefact but a known consequence of
the modelling choice: fitting squared error to `log(salary)` estimates the
conditional mean *of the log*, and exponentiating returns the conditional
median. For a right-skewed distribution the median sits below the mean, so
predictions are systematically low. The standard remedies are Duan's smearing
estimator or a loss function specified in the original space; neither is applied
here, and the omission is documented rather than hidden.

---

## 5. Discussion

**The pipeline's fourth stage earned its place.** The defects found in §3.1 and
§3.2 were not discovered by inspecting the model — they were discovered by
treating the data as something to be falsified. Two of the three silent ones were
found only by looking at real records: no amount of reading documentation reveals
that Russian adverts spell C++ with a different alphabet, and no type annotation
reveals that an "integer" field is an undocumented code.

**The negative result about neural networks is a result.** It would have been
straightforward to reach for a larger network and report a slightly better
number. Instead the comparison was designed so that the answer could come out
either way, with identical inputs and a seed sweep sized to the network's own
variance. On tabular data of this size and shape, the added capacity buys
nothing measurable.

**Controlling for composition changes the story.** The finding that the IT
"premium" is mostly geography and occupation is more interesting than the raw gap
it replaces, and it is the kind of claim only a multivariate model can support.

---

## 6. Limitations

1. **The sample is not random.** Within one query the API returns its own
   ordering, and each (keyword, region) cell is capped at 100 records. Estimates
   describe this sample, not the population of Russian vacancies.
2. **The portal skews towards state-sector and blue-collar roles.** IT is a
   minority of the 522k national vacancies. The model is valid within the
   population it was fitted on.
3. **`salary_max` is unusable in 38.5% of records**, and the free-text salary
   never carries an upper bound. Only `salary_min` is modelled.
4. **Experience is unusable.** `requirement.experience` is an undocumented code,
   and the free text states a requirement in a small minority of adverts.
5. **Skill extraction is dictionary-based.** A technology absent from the
   patterns is invisible; no test can detect a missing entry.
6. **The IT premium is a partial association, not a causal effect.**
7. **Predictions are systematically low** by ~13.8%, for the reason given in
   §4.5.
8. **No hyper-parameter search was performed.** The models use reasonable
   defaults, so the comparison is between model classes rather than between
   tuned instances.

---

## 7. Reproducing this

Every number in this report is regenerated from the committed data by three
commands; the harvest itself is committed, so no network access is required.

```powershell
cd it-salary-ru
pip install -r requirements.txt

python stage2_processing/build_features.py     # -> the analytical table + §4.1
python stage3_analytics/train_models.py        # -> §4.2 to §4.5
python -m pytest tests -v                      # -> 103 tests, §3.4
```

The repository is laid out by pipeline stage; see the [README](../README.md) for
the tree and for how to re-harvest from the API.

---

## 8. Sources

| Source | Used for |
|---|---|
| Trudvsem open data API — https://opendata.trudvsem.ru/api/v1/vacancies | All vacancy records |
| Trudvsem API documentation — https://trudvsem.ru/opendata/api | Endpoint and parameter reference |
| hh.ru API — https://github.com/hhru/api | Evaluated and rejected (§2.1) |

Tools: Python 3.14, pandas, scikit-learn, PyTorch (GPU path for M3), pytest,
matplotlib.

---

## 9. References

The four stage reports carry the full detail behind this synthesis.

| Document | Covers |
|---|---|
| [data_dictionary.md](data_dictionary.md) | Field structure, presence rates, cardinality, and every measured defect |
| [data_quality_report.md](data_quality_report.md) | Stage 2: row accounting, the 10 quality gates, and the repair policies |
| [model_report.md](model_report.md) | Stage 3: the leakage audit, the full comparison, feature importance, error analysis and the seed sweep |
| [validation_report.md](validation_report.md) | Stage 4: the seams, the 103 tests, the bugs they found, and what this stage does not prove |
