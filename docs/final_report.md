# Predicting IT Salary Levels from Russian State Employment Open Data

**What this project found.** This document answers the research questions; the
[README](../README.md) covers the data source, the pipeline and how to reproduce
it, and §6 below points to the stage reports that carry the full detail.

*中文版: [final_report.zh.md](final_report.zh.md)*

---

## Abstract

> **Which factors determine the salary level of an IT specialist in the Russian
> labour market, and how accurately can the offered salary be predicted from the
> vacancy text and its structured attributes?**

**Where a vacancy is matters more than what it asks for.** Geography is the
strongest predictor by a wide margin — the Moscow indicator ranks first and the
coordinates second and third — followed by occupation, then schedule. The free
text contributes, but through learned components rather than any single named
skill.

**Being an IT vacancy is worth almost nothing once you control for that.** The
raw gap between the IT and non-IT medians is 11.1%; with region, education,
occupation and skills held constant it falls to **+1.0%**. The apparent premium
is composition: IT vacancies are concentrated where salaries are already high.

**About a third of a typical salary is the typical error.** MAE is 17,478 RUR
against a median of 49,999 — 35.0% — with R² = 0.615 on a forward-looking
temporal hold-out. That is good enough to describe a market and not good enough
to price an individual offer.

**A neural network did not justify itself.** On byte-identical inputs, gradient
boosting scores 0.615 against 0.595 for an MLP, and wins on all seven seeds
tested.

---

## 1. The questions

The main question is stated above. Three sub-questions structure the work, and
each is answered in §2.

| | Sub-question | Answered in |
|---|---|---|
| **Q1** | How much salary signal is carried by structured attributes versus free text — and how much of the text must be mined to recover what the portal does not publish structurally? | §2.3 |
| **Q2** | How large is the IT salary premium once region and qualification are controlled for? | §2.4 |
| **Q3** | Does the prediction quality justify a neural network, or does a simpler model perform equally well? | §2.5 |

Q3 is deliberately falsifiable. The answer is no.

---

## 2. The answers

### 2.1 Which factors determine the salary

**Geography first, by a wide margin.** Permutation importance on the best model,
measured on 3,000 held-out rows:

| Rank | Feature | Increase in MAE when shuffled |
|---:|---|---:|
| 1 | `region_name_Город Москва` | 0.0362 |
| 2 | `lng` | 0.0287 |
| 3 | `lat` | 0.0213 |
| 4 | `specialisation_Образование, наука` | 0.0145 |
| 5 | `svd_17` (a text component) | 0.0131 |
| 6 | `schedule_Неполный рабочий день` | 0.0114 |

Two of the top three are location, and the coordinate features rank nearly as
high as the Moscow indicator itself — the model is learning a salary surface
over geography rather than memorising one city. Occupation follows. The free
text enters at rank 5, and through a **learned component**, not through any
single named technology.

The ordering is the finding: **where the job is, and what kind of work it is,
dominate what the advertisement asks for.**

### 2.2 How accurately the salary can be predicted

**MAE 17,478 RUR, R² = 0.615**, on postings created after the training window —
a forward extrapolation, not an interpolation.

| Model | MAE (RUR) | As % of median | R² (log) |
|---|---:|---:|---:|
| M0a global train median | 28,594 | 57.2% | −0.177 |
| M0b (region × education) cell median | 23,674 | 47.3% | 0.187 |
| M1 Ridge — sparse, full TF-IDF | 17,966 | 35.9% | 0.592 |
| M1b Ridge — dense design | 18,264 | 36.5% | 0.588 |
| **M2 gradient boosting — dense** | **17,478** | **35.0%** | **0.615** |
| M3 neural network (MLP) — dense | 17,945 | 35.9% | 0.595 |

The best learned model beats the strongest baseline by **26.2%**.

**How to read that number.** A third of a typical salary is the typical error.
The model is useful for questions like "is this offer in the normal band for
this region and role?" and not for "what exactly should this person be paid?".
Two further results bound it:

- **Moscow is far harder than anywhere else** — MAE 41,657 RUR, more than twice
  the overall figure. Salary dispersion in the capital is large and the features
  do not resolve it.
- **Predictions are systematically low by about 13.8%**, for a reason given in
  §3.2. Every subgroup shows it.

A secondary classification task — which train-set salary quartile does this fall
in — reaches 57.5% accuracy against a 28.1% majority baseline, with a macro F1
of 0.550. The band is usually right; the exact figure rarely is.

### 2.3 Q1 — structured attributes or free text?

**The text mining is necessary but is not where the salary signal lives.**

Necessary, because the portal's structured `skills` field is empty on **77.4%**
of records. Mining recovers at least one skill for 36.7% of records against the
24.1% that have a structured one — the mining more than doubles the coverage.

But its marginal contribution is modest and diffuse:

- Moving from a 128-component SVD compression of the text to the **full TF-IDF
  vocabulary** is worth **1.6% of MAE** (M1b → M1).
- No individual skill flag reaches the top six features. The text enters through
  components that mix many terms.

So the honest answer is: mine it, because you cannot answer skill questions
without it, but **do not expect it to carry the prediction**. Region and
occupation do that.

The mining is also where the **most dangerous defects** live — a naive pattern
misses 43% of C++ vacancies (§3.1).

### 2.4 Q2 — how large is the IT premium?

**Almost none: +1.0%.**

The raw comparison suggests otherwise. The IT median is 50,000 RUR against
45,000 for the control set — **+11.1%**. But that is not a controlled
comparison: IT vacancies are concentrated in Moscow and St Petersburg, and in
higher-qualification occupations.

Placing the `is_it` indicator inside a Ridge regression alongside region,
education, occupation, schedule and skill features:

- coefficient on `is_it` (log scale): **+0.0103**
- implied salary premium, all else equal: **+1.0%**

**The raw gap is therefore almost entirely composition, not a premium.** IT pays
more in Russia largely because of *where* the jobs are and *what kind* of work
they are — not because the label "IT" commands a premium.

Read the residual 1% as a partial association, not a causal effect. The controls
are whatever this dataset measures, and unmeasured differences — seniority,
contract type, employer sector — remain.

### 2.5 Q3 — does the neural network justify itself?

**No.**

M2 and M3 receive **byte-identical input matrices**, so the only difference is
the model class:

| | MAE (RUR) | R² (log) |
|---|---:|---:|
| M2 gradient boosting | **17,478** | **0.615** |
| M3 MLP | 17,945 | 0.595 |

A single seed is not evidence for a neural network, so the MLP was retrained
under **seven seeds**: R² = 0.6002 ± 0.0038 (range 0.595–0.605). **M2 beats all
seven, on both metrics.**

The honest framing matters here. Gradient boosting wins by roughly the width of
the network's own seed noise — it does not win by a margin that would survive any
conceivable tuning. What the neural network fails to do is deliver **any benefit
beyond that noise**, while costing interpretability and a tuning burden.

This comparison is only meaningful because of a design decision: an earlier
version gave the booster SVD-compressed text while giving the linear model the
full TF-IDF vocabulary. The booster "lost". That was a design error, not a
finding, and §4 records it.

---

## 3. Findings that qualify the answers

These are not incidental. Each one would have produced a plausible, wrong answer
had it gone unnoticed.

### 3.1 Three defects that fail silently

**A zero sentinel that looks like data.** The portal encodes "salary not
specified" as the literal value `0` — the free text reads `"от 0"` — rather than
leaving the field null. **179 records (0.91%)**. A model would otherwise learn
from 0-rouble monthly salaries, and `log(0)` is `-inf`.

**A field that is not what it says.** `requirement.experience` is documented as
an integer and reads like a year count. It is neither. Cross-checking against the
free text settles it: adverts carrying `code = 0` ask for up to **35 years** of
experience, so `0` cannot mean "none required", while `code = 18` appears on
adverts reading "от 1 года". The column is an undocumented code and was
**withdrawn from the feature set** — experience is simply not usable from this
source.

**An alphabet trap that costs 43% of a skill.** Russian adverts routinely write
`C++` with a **Cyrillic** capital С (U+0421), visually identical to the Latin C,
and Python's `re.IGNORECASE` does not fold Cyrillic onto Latin:

```python
re.search(r"c\+\+", "С++", re.I)   # -> None    silently missed
```

Measured: 137 vacancies match only the Latin spelling, **102 match only the
Cyrillic**, 100 use both — a false-negative rate of **102/239 = 43%**. The
obvious fix, normalising the text, would corrupt ordinary Russian, where `с` is
one of the most common prepositions. The patterns accept either alphabet at the
few positions where the confusion occurs instead.

### 3.2 Predictions are systematically low, and it is explainable

The bias is negative in **every** subgroup — IT −7,792 RUR, control −5,519,
Moscow −21,776, St Petersburg −14,857 — averaging about **−6,900 RUR (13.8% of
the median)**.

This is not a subgroup artefact but a consequence of the modelling choice.
Fitting squared error to `log(salary)` estimates the conditional mean *of the
log*, and exponentiating returns the conditional **median**, which sits below the
mean for a right-skewed distribution.

The standard remedies are Duan's smearing estimator or a loss specified in the
original space. **Neither is applied here**, and the omission is documented
rather than hidden: fixing it would change every number in §2, which is a
separate decision from reporting them.

### 3.3 A range that usually is not a range

The free-text salary field is `"от N"` in **100%** of the 21,941 salaried
records, so the text never carries an upper bound, and `salary_max` equals
`salary_min` on **38.5%** of the table. Where the bounds are equal the range
carries no information, so `salary_max` is set to missing and the row flagged.
Only `salary_min` is modelled.

---

## 4. What the answers rest on

Stated compactly; the [README](../README.md) and the stage reports carry the
detail.

| | |
|---|---|
| **Data** | 22,387 vacancy records from the Trudvsem state employment portal's official open data — 12,656 IT and 9,731 non-IT control, harvested with 2,002 requests across 89 regions |
| **After processing** | 19,578 rows × 72 columns, **10/10 quality gates** passing; 446 exact and 2,363 near-duplicates removed |
| **Split** | Temporal, not random: 14,513 postings before 2026-08-20 to train, 4,886 after to test. A random split would be invalid twice over — employers repost identical adverts, and the sample is recency-weighted |
| **Comparison design** | M1b, M2 and M3 share one byte-identical matrix, so the comparison tests model classes rather than feature engineering |
| **Leakage audit** | Every column carries an explicit decision. `salary_text` literally reads `"от <target>"` and would have produced a near-perfect, meaningless model |
| **Validation** | 103 tests, ~16 s, no network, enforced by CI on Python 3.12 and 3.13. They found two real bugs and forced one interface improvement |

Two limitations of the evidence itself belong here rather than in §5, because
they bound every number above:

- **The sample is not random.** Each (keyword, region) query returns the API's
  own ordering and is capped at 100 records. Every figure describes this sample,
  not the population of Russian vacancies.
- **No hyper-parameter search was performed.** The comparison is between model
  classes on reasonable defaults, not between tuned instances.

---

## 5. What would change these answers

1. **A probability sample.** The current harvest is a convenience sample. Nothing
   here estimates the national distribution.
2. **A working date filter.** The API has none that functions, so the temporal
   split is the only control on drift.
3. **Experience data.** The one field that would most plausibly improve accuracy
   is the one that turned out to be an undocumented code.
4. **A corrected retransformation.** Duan's smearing would remove the systematic
   13.8% under-prediction and change the MAE figures.
5. **Salary maxima.** Unusable in 38.5% of records, so the model is trained on
   lower bounds only.
6. **The portal skews towards state-sector and blue-collar roles** — IT is a
   minority of the 522k national vacancies. Conclusions hold within the
   population the sample describes.

---

## 6. References

The four stage reports carry the full detail behind every claim above.

| Document | Covers |
|---|---|
| [data_dictionary.md](data_dictionary.md) | Field structure, presence rates, and every measured defect |
| [data_quality_report.md](data_quality_report.md) | Row accounting, the 10 quality gates, the repair policies |
| [model_report.md](model_report.md) | The leakage audit, the full comparison, feature importance, error analysis, the seed sweep |
| [validation_report.md](validation_report.md) | The test seams, the bugs they found, and what validation does not prove |

Data source: Trudvsem open data — https://opendata.trudvsem.ru/api/v1/vacancies
(terms: https://trudvsem.ru/opendata/api). hh.ru was evaluated and rejected; see
the README. Licence: [MIT](../LICENSE), with a data notice for `data/raw/`.
