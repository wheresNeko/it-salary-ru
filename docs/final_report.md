# Predicting IT Salary Levels from Russian State Employment Open Data

**In one paragraph.** I took 22,387 job adverts published by the Russian state
employment portal and asked two things: what actually decides the salary on
offer, and how close a computer model can get to predicting it. The answer to
the first is *where the job is* — a vacancy in Moscow pays far more than the
same job elsewhere, and geography matters more than anything the advert asks
for. The answer to the second is *about a third of a typical salary*: useful for
judging whether an offer is normal, not for setting one. Along the way I found
that "IT pays more" is almost entirely an illusion created by IT jobs being
concentrated in expensive cities, and that a neural network — the fashionable
choice — was beaten by a simpler model.

*中文版: [final_report.zh.md](final_report.zh.md). Data source, pipeline and reproduction: [README](../README.md).*

---

## The answers at a glance

| The question | The short answer |
|---|---|
| **What decides the salary?** | Where the job is, first and foremost. Then what kind of work it is. What the advert *asks for* matters least. |
| **How accurate is the prediction?** | Off by about 35% of a typical salary — closer than any simple rule of thumb, but far from exact. |
| **Does the advert text need mining?** | Yes, there is no choice — but it is not what drives the prediction. |
| **Is there an "IT premium"?** | Almost none. It looks like 11%, but it is **+1%** once you account for where the jobs are. |
| **Was the neural network worth it?** | No. A simpler model won, and won consistently. |

Every number below is reproduced from the committed data by three commands; see
§7 of the README.

---

## 1. What this project set out to answer

> **Which factors determine the salary level of an IT specialist in the Russian
> labour market, and how accurately can the offered salary be predicted from the
> vacancy text and its structured attributes?**

Three smaller questions break that down, and §2 answers each one:

| | Question | Answered in |
|---|---|---|
| **Q1** | How much comes from the structured fields, and how much has to be dug out of the free text? | [§2.3](#23-q1--is-the-text-worth-mining) |
| **Q2** | How big is the IT salary premium once you account for region and qualification? | [§2.4](#24-q2--is-there-an-it-premium) |
| **Q3** | Is a neural network worth it, or does a simpler model do just as well? | [§2.5](#25-q3--was-the-neural-network-worth-it) |

Q3 was designed to be falsifiable. It would have been easy to build something
complicated and report a good-looking number; the point was to make it possible
for the answer to come out *no*. It did.

---

## 2. The answers

### 2.1 What decides the salary

> **Where the job is. Then what kind of job it is. The advert's requirements come
> last.**

To find out which columns the model actually leans on, I scrambled one column at
a time, left everything else alone, and measured how much worse the predictions
got. A big drop means the model depended on that column.

![Which columns the model relies on](figures/fig_importance.png)

The three longest bars are **location**: Moscow, then longitude, then latitude.
The latitude and longitude bars are nearly as long as the Moscow one, which tells
us the model is not simply memorising one city — it has learned a salary gradient
across the whole map.

Fourth comes the occupation category, then a text component. No individual
technology — not Python, not SQL, not 1C — appears anywhere near the top.

**The ordering is the finding.** Where a job is located, and what kind of work it
is, dominate what the advertisement asks for.

### 2.2 How accurate the prediction is

> **Off by about 17,500 roubles on a typical salary of 50,000 — roughly a third.
> Useful for "is this offer normal?", not for "what should this person be paid?".**

The headline numbers, with the two terms explained:

- **MAE (mean absolute error) = 17,478 RUR.** This is the average size of the
  miss, ignoring direction. If the model predicts a salary, it is wrong by about
  17,500 roubles on average.
- **R² = 0.615.** A score from 0 to 1 for how much of the variation between
  vacancies the model explains. 0 would mean "no better than always guessing the
  average"; 1 would be perfect. 0.615 means it accounts for about 62% of the
  differences. (Scores can go negative — M0a below does, meaning it was worse
  than guessing.)

| Model | MAE (RUR) | As % of median | R² |
|---|---:|---:|---:|
| M0a — always guess the training average | 28,594 | 57.2% | −0.177 |
| M0b — guess the average for this region and education level | 23,674 | 47.3% | 0.187 |
| M1 — Ridge on the full text vocabulary | 17,966 | 35.9% | 0.592 |
| M1b — Ridge on compressed text | 18,264 | 36.5% | 0.588 |
| **M2 — gradient boosting** | **17,478** | **35.0%** | **0.615** |
| M3 — neural network | 17,945 | 35.9% | 0.595 |

The best model beats the strongest simple rule by **26.2%**.

![Predictions against reality, and the spread of errors](figures/fig_model_diagnostics.png)

Two things to see in that figure.

**Left:** each dot is a vacancy, plotted by its real salary (across) against the
predicted one (up). If predictions were perfect every dot would sit on the dashed
line. They cluster around it in a broad band — which is exactly what a 35% error
looks like. Note also how the cloud flattens out at the top right: the model
never predicts the very highest salaries.

**Right:** how far off each model was. Every curve sits slightly to the **left**
of zero, meaning every model tends to *under*-predict. That is systematic, it has
a cause, and §3.2 explains it.

**One more thing worth knowing: Moscow is much harder.** The average miss there
is 41,657 roubles — more than twice the overall figure. Salaries in the capital
vary enormously, and nothing in the advert explains why.

As a side check I also asked the model to sort vacancies into four salary bands
instead of predicting a number. It gets the band right 57.5% of the time, against
28.1% for always guessing the most common band. **The band is usually right; the
exact figure rarely is.**

### 2.3 Q1 — is the text worth mining?

> **You have to mine it — the portal leaves the skills field blank three times
> out of four. But do not expect it to carry the prediction.**

The structured `skills` field is empty on **77.4%** of records. Mining the free
text recovers at least one skill for 36.7% of records, against 24.1% that have a
structured one — so the mining more than doubles the coverage. Without it, any
question about skills simply cannot be answered.

But its contribution to *prediction* is modest:

- Swapping the compressed text for the **full** vocabulary improves the error by
  **1.6%** and nothing more.
- No single skill appears in the top features. The text helps only as a blur of
  many words at once.

So: mine the text, because you need it to describe the market — but the
prediction is carried by region and occupation, not by technology keywords.

**It is also where the most dangerous bugs hide** — see §3.1.

### 2.4 Q2 — is there an IT premium?

> **Almost none: 1%. It looks like 11%, but that is because IT jobs are
> concentrated in the expensive cities.**

The raw comparison suggests a real premium. The middle IT salary is 50,000
roubles, against 45,000 for everything else — **11.1% higher**.

But that is not a fair comparison. IT vacancies cluster in Moscow and St
Petersburg, and in better-paid occupations. So I put the "is this IT?" flag into
a regression alongside region, education, occupation, working hours and skills —
asking, in effect: *between two otherwise identical vacancies, one IT and one
not, how much more does the IT one pay?*

**The answer: +1.0%.**

The apparent 11% is therefore not a premium for being in IT. It is a premium for
being in Moscow, doing a well-paid kind of work. **An IT vacancy in Moscow pays
more because it is in Moscow.**

Treat that remaining 1% as a weak association, not proof of cause. The comparison
only controls for what this dataset happens to record; experience, seniority and
type of employer are not in it.

### 2.5 Q3 — was the neural network worth it?

> **No. The simpler model won, and won on every single run.**

The two models were given **exactly the same input data** — the same numbers, in
the same order. The only difference was the kind of model.

| | Average miss (RUR) | R² |
|---|---:|---:|
| Gradient boosting | **17,478** | **0.615** |
| Neural network | 17,945 | 0.595 |

A neural network starts from random values, so a single run proves nothing. I
retrained it seven times with seven different random starts. Its score wandered
between 0.595 and 0.605 — and **the simpler model beat all seven**, on both
measures.

The honest way to put it: the simpler model won by about as much as the neural
network's own randomness. The point is not that it crushed it. The point is that
**the neural network delivered no benefit at all beyond its own noise**, while
being harder to explain and harder to tune.

---

## 3. How far to trust these answers

### 3.1 Three problems that would have fooled us silently

None of these raise an error. Each would have produced a believable, wrong
answer.

**A salary of zero that means "no salary given".** The portal writes `0` — with
the text `"от 0"`, "from 0" — when an employer has not stated a salary, instead
of leaving the field blank. **179 records.** Taken at face value, the model would
have learned from 0-rouble monthly salaries.

**A field that is not what it claims.** `requirement.experience` is documented as
a whole number of years. It is not. Adverts carrying `0` ask for up to **35
years** of experience, so `0` cannot mean "no experience needed"; another value
appears on adverts that say "from 1 year". It is an undocumented internal code,
so **I removed it from the model entirely** — which means experience is simply
not available from this source.

**The Cyrillic C that hides 43% of a skill.** Russian adverts often write `C++`
using a **Cyrillic** letter С, which looks identical to the Latin C but is a
different character. Standard case-insensitive matching does not treat them as
the same:

```python
re.search(r"c\+\+", "C++", re.I)   # matches
re.search(r"c\+\+", "С++", re.I)   # no match -- silently missed
```

Counting across the whole dataset: 137 vacancies use only the Latin spelling,
**102 use only the Cyrillic**, 100 use both. A naive search misses **43%** of all
C++ vacancies. The tempting fix — rewriting all text to one alphabet — would
break ordinary Russian, where `с` is one of the commonest words. Instead the
search accepts either alphabet, but only where the confusion actually happens.

### 3.2 Why predictions are always a little low

Look again at the right-hand panel of the figure in §2.2: every error curve sits
to the left of zero. In every subgroup — IT, non-IT, Moscow, St Petersburg — the
model under-predicts, by about **6,900 roubles on average, or 13.8%**.

This is not a quirk of one group. It is a direct consequence of a modelling
choice, and it has a name. Salaries are modelled on a **logarithmic** scale,
which compares them as multiples rather than differences. Fitting a model that
way estimates the *middle* of the salary distribution, not its *average* — and
for salaries, which have a long tail of very high earners, the middle sits below
the average.

The standard fix is a correction known as Duan's smearing estimator. **I have not
applied it**, and I am saying so rather than hiding it: applying it would change
every number in §2, which is a separate decision from reporting them honestly.

### 3.3 A "range" that usually is not one

Every salaried vacancy writes its pay as `"от N"` — "from N". Not one of them
gives an upper bound. And on **38.5%** of records the stored maximum is simply a
copy of the minimum, so it carries no information at all. Those maxima were
discarded, and the model predicts the lower bound only.

---

## 4. What the answers rest on

Stated briefly. The [README](../README.md) has the detail.

| | |
|---|---|
| **The data** | 22,387 adverts from the Trudvsem state employment portal's official open data — 12,656 IT and 9,731 others for comparison |
| **After cleaning** | 19,578 rows, 72 columns, **10 out of 10 quality checks** passing; 446 exact and 2,363 near-duplicate adverts removed |
| **How it was tested** | Trained on adverts posted before 20 Aug 2026, tested on those after — so the model is predicting forward, not filling in gaps |
| **Fair comparison** | The three main models received *identical* data, so the comparison is between model types rather than between feature sets |
| **Leakage check** | The raw salary text literally reads `"от <the answer>"`. Left in, it would have produced a near-perfect and completely meaningless model. Every column now carries an explicit in-or-out decision |
| **Automated tests** | 103 tests run in about 16 seconds, with no internet, on every change |

Two limits belong here rather than in §5, because they apply to every number
above:

- **This is not a random sample.** Each search returns the portal's own ordering
  and is capped at 100 adverts. Everything here describes *this sample*.
- **No settings were tuned.** The models were compared at sensible defaults, not
  at their individually best configurations.

---

## 5. What would change these answers

1. **A random sample.** The current harvest is a convenience sample. Nothing here
   estimates the true national picture.
2. **Working date filtering.** The portal's date filters do not work, so the
   before/after split is the only control on change over time.
3. **Experience data.** The single field most likely to improve accuracy turned
   out to be an undocumented code.
4. **The log correction described in §3.2.** It would remove the systematic
   13.8% under-prediction and shift the error figures.
5. **Salary maxima.** Missing or meaningless on 38.5% of records, so only lower
   bounds are modelled.
6. **The portal's own bias.** It skews towards state-sector and manual jobs; IT
   is a small minority of its 522,000 national vacancies. The conclusions hold
   for the population this sample describes.

---

## Glossary

Plain-language definitions of the terms used above.

| Term | What it means here |
|---|---|
| **MAE** (mean absolute error) | The average size of the prediction miss, in roubles, ignoring direction. Lower is better. |
| **R²** | How much of the variation between salaries the model explains, from 0 (no better than guessing the average) to 1 (perfect). Can be negative if the model is worse than guessing. |
| **Baseline** | A deliberately dumb rule to beat. M0a guesses one number for everything; M0b guesses the average for each region-and-education combination. |
| **Ridge regression** | Linear regression with a built-in penalty that stops it clinging to noise. |
| **Gradient boosting** | A model that builds many small decision trees, each one correcting the previous ones' mistakes. |
| **MLP** (multilayer perceptron) | A small neural network. The "M3" model here. |
| **Temporal hold-out** | Training on older records and testing on newer ones, so the model is predicting forward rather than filling in gaps. |
| **Permutation importance** | How much worse the model gets when one column's values are shuffled and everything else is left alone. Larger means the model leans on that column more. |
| **TF-IDF** | A way of turning text into numbers that emphasises words distinctive to a vacancy rather than words that are merely common. |
| **SVD component** | A compressed stand-in for the text. Instead of thousands of individual words, the model gets a smaller set of blended "topics". |
| **One-hot encoding** | Turning a category such as a region name into a set of yes/no columns. |
| **Leakage** | Accidentally letting the answer into the inputs. It produces flattering scores and a useless model. |
| **Log scale** | Comparing salaries as multiples ("twice as much") rather than differences ("20,000 more"). |
| **Retransformation bias** | The systematic under-prediction explained in §3.2, caused by modelling on a log scale and converting back. |

---

## 6. References

| Document | Covers |
|---|---|
| [data_dictionary.md](data_dictionary.md) | Every field, how often it is filled in, and every measured defect |
| [data_quality_report.md](data_quality_report.md) | Row-by-row accounting, the 10 quality checks, the repair rules |
| [model_report.md](model_report.md) | The leakage audit, the full comparison, error analysis, the seven-seed test |
| [validation_report.md](validation_report.md) | The automated tests, the two bugs they caught, and what they do not prove |

**Data source:** Trudvsem open data — https://opendata.trudvsem.ru/api/v1/vacancies
(terms: https://trudvsem.ru/opendata/api). hh.ru was evaluated first and rejected;
see the README. **Licence:** [MIT](../LICENSE), with a separate data notice for
`data/raw/`.
