# Validation report — Stage 4

Stage 4 of the project pipeline is *"proof that the data is correct"*. This
document is that proof, and it is enforced rather than asserted: the suite runs
on every push via [`.github/workflows/tests.yml`](../.github/workflows/tests.yml).

**94 tests, 4 seams, ~12 s, no network.**

```
tests/test_textmining.py             43   the parsing rules
tests/test_repair_policy.py          15   what Stage 2 decides about the data
tests/test_leakage.py                12   what may not reach the model
tests/test_pipeline_integration.py   24   the whole Stage 1 -> 2 path
```

---

## 1. What "correct" means here

"The data is correct" is not one claim, and treating it as one is how projects
end up testing the easy part. Four separable claims are made, and each has its
own seam:

| Claim | Why it is not obvious | Seam |
|---|---|---|
| **Parsed correctly** — the RegEx rules read Russian job adverts the way a human would | The defects are silent. A wrong pattern raises nothing; it turns a feature into zero at scale | `textmining.py` public functions |
| **Repaired correctly** — the two measured defects are handled by a stated policy | The repairs change the regression target, so the target depends on a decision | `build_features.flatten` → `derive` → `dedupe` |
| **Not leaking** — no target-derived column reaches the model | `salary_text` literally reads `"от <target>"`. It would give a near-perfect, meaningless model | `train_models` column classification |
| **Still working end to end** — the stages agree on the data between them | Each stage passed alone while the interface between them drifted | The full Stage 1 → 2 path |

### Seams were agreed before any test was written

Testing effort is finite, so the seams were chosen in advance and confirmed:
the four public boundaries above. Internals — the regex objects, the
`_to_int` helper, the column lists inside `derive` — are deliberately not
tested, because a test bound to them breaks on refactoring while the behaviour
is unchanged.

---

## 2. Two defects the suite found

Both are real. Neither affected any published result, and saying so precisely is
the point of this section.

### 2.1 `parse_salary_text("до 50000")` returned the bound inverted

`"до 50000"` means *up to* 50,000. The parser returned `(50000, None)` — reading
the upper bound as the lower one, which the pipeline uses as the regression
target. The `lo_prefix` capture group never contained `до` at all.

The first fix attempt patched the function and did **nothing**, because the
missing alternative was in the regular expression, not the branch. The test
caught that too: the assertion stayed red after the edit, which is what
prompted reading the regex rather than trusting the edit.

**Impact on published results: none.** All 21,941 salaried records in the corpus
are the shape `"от N"`, so the branch never fires. The cross-field gate
(`parsed low == salary_min`, 100%) would fail loudly if that ever changed — the
gate exists for exactly this.

### 2.2 `"Java Script"` matched neither Java nor JavaScript

The `java` pattern carries `(?!\s*script)` to stop it swallowing JavaScript, and
the `javascript` pattern required the two words joined. A spaced spelling fell
through both: a false negative that silently zeroed a feature.

Fixing it changed the feature matrix, which moved every Stage 3 metric slightly
(see §5).

---

## 3. An interface the tests forced

`build_features.build_table()` did not exist before this stage. The integration
test called `flatten` → `derive` → `dedupe` directly and got a *different* frame
from the one Stage 3 sees, because the dropping of two scratch columns lived
inside `main()`, which also writes files.

There was no way to obtain the real Stage 2 output without running the whole
script. The test exposed that and the function was extracted. This is the
concrete return on testing at a seam rather than reaching into steps.

---

## 4. The leakage audit

The highest-stakes property in the project. Every column of the analytical table
must carry an explicit decision, and the test fails when one does not — so
adding a field in Stage 2 forces the question.

The audit immediately found three problems that had gone unnoticed:

| Finding | Detail |
|---|---|
| **2 columns unclassified** | `region_code` (a deterministic recoding of a feature) and `currency` (constant on every record) had no recorded decision |
| **3 columns declared twice** | `text_blob`, `qualification`, `typical_position` were listed both as TF-IDF sources *and* as exclusions — the documentation contradicted itself |
| **1 undeclared derived feature** | `creation_month` is built in Stage 3 rather than read from the table |

A further test pins the known-dangerous column names (`salary`, `salary_min_raw`,
`salary_max`, `salary_text`, `log_salary`, `salary_mid`, `has_salary`) as
leakage. Without it, the audit could be made to pass by *deleting the
declarations* instead of classifying the columns — the failure mode of every
"check that everything is documented" test.

---

## 5. The corpus claim, and the metric drift it implies

`test_pipeline_integration.py::TestCommittedCorpus` runs the real Stage 2 over
the harvest committed to the repository and asserts:

- all **10 quality gates pass**
- the two groups are **disjoint** (no id appears in both)
- the table is **19,578 rows**, matching the documentation
- salary coverage is 99.09%, with exactly **179 zero sentinels**
- `skill_c++` finds **more than 150** vacancies — the regression net for the
  Cyrillic trap, which is the defect this whole stage exists for

**Documentation drift is a real hazard here.** Fixing the `"Java Script"` gap in
§2.2 added a few `skill_javascript` hits, which moved the Stage 3 metrics:

| | before §2.2 | after |
|---|---:|---:|
| M2 gradient boosting, R² | 0.613 | **0.615** |
| M3 MLP, R² | 0.596 | **0.595** |
| MLP across 7 seeds | 0.5950 ± 0.0134 | **0.6002 ± 0.0038** |

The conclusion does not move — M2 beats every seed before and after — but the
figures in the README were updated by hand. A test that asserted the metric
values would have caught that automatically, and the reason it does not exist is
stated in §6.

---

## 6. What this stage does not prove

Stated plainly, because a validation report that only lists successes is
marketing.

1. **The collector is untested.** `collect.py` owns the network, the retries and
   the region discovery, and it has no tests: they would either hit the live API
   or mock it so heavily that they assert the mock. Its behaviour is documented
   instead, in `api_investigation/` and the README troubleshooting section.
2. **The model metrics are not pinned by a test.** M2's R² is asserted nowhere,
   so §5's drift needed a human to notice. Pinning a float makes a suite that
   fails on every dependency bump; the seed sweep in `model_report.md` is the
   robustness evidence instead, and this trade-off is deliberate.
3. **The fixture is six records, not a sample.** It proves the pipeline's *rules*
   hold; it says nothing statistical. Corpus-level claims rest on the committed
   harvest.
4. **Nothing here validates the sampling.** The harvest is recency-weighted and
   capped at 100 records per (keyword, region) cell. No test can fix that — it is
   a property of the API, recorded in the quality report.
5. **The regexes are dictionary-based.** A technology absent from
   `SKILL_PATTERNS` is invisible, and no test can detect a missing entry. The
   dictionary is a reviewable artefact for that reason.

---

## 7. Running it

```powershell
cd C:\Users\Neko\Desktop\Workspace\it-salary-ru
python -m pytest tests -v
```

No network and no build artifacts are required. The corpus tests read the
gzipped harvests committed to the repository and skip with a clear message if
they are absent.

Continuous integration runs the suite on Python 3.12 and 3.13, then runs Stage 2
and Stage 3 end to end and uploads the generated reports as artifacts. `torch` is
deliberately **not** installed in CI: Stage 3 falls back to scikit-learn's MLP
when CUDA is unavailable, so the GPU path is not exercised in CI and no test
depends on it.
