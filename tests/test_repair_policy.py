"""Behavioural tests for the Stage 2 repair policy.

Stage 2 does not merely reshape the data — it *decides* what two measured
defects mean, and those decisions change the regression target. This module
pins them:

  * "от 0" is a sentinel for "salary not specified", not a 0-rouble salary.
    Measured on 179 records (0.82%).
  * `salary_max == salary_min` means the upper bound carries no information.
    Measured on 36-38% of salaried records.

The tests exercise the pipeline through `flatten` -> `derive` -> `dedupe`,
which is the public boundary Stage 3 consumes. They never reach into the
intermediate helper functions.
"""

import math

import pandas as pd
import pytest

from build_features import Gates, dedupe, derive, flatten, run_gates


def record(**overrides) -> dict:
    """A minimal but realistic API record, with named fields overridable."""
    rec = {
        "id": "00000000-0000-0000-0000-000000000001",
        "company": {"inn": "7701234567", "name": "ООО Тест", "site": "https://x.ru"},
        "region": {"name": "Город Москва", "region_code": "7700000000000"},
        "salary": "от 40000",
        "salary_min": 40000,
        "salary_max": 40000,
        "currency": "«руб.»",
        "job-name": "Программист",
        "creation-date": "2026-01-15",
        "date_modify": "2026-01-15T10:00:00+0300",
        "requirement": {"education": "Высшее образование — бакалавриат",
                        "experience": 1},
        "category": {"specialisation": "Информационные технологии"},
        "requirements": "Опыт работы в 1С, знание SQL",
        "duty": "Разработка и сопровождение",
        "addresses": {"address": [{"lat": "55.75", "lng": "37.61"}]},
        "skills": [],
    }
    rec.update(overrides)
    return rec


def rows(*records: dict, is_it: bool = True) -> pd.DataFrame:
    """Run records through the real flatten -> derive path."""
    flat = pd.DataFrame([flatten(r, is_it) for r in records])
    return derive(flat)


# --------------------------------------------------------------------------
# Seam 2a: the zero sentinel
# --------------------------------------------------------------------------

class TestZeroSentinel:
    """The portal writes the literal value 0 for "not specified", and the free
    text reads "от 0". Left alone, log(0) is -inf and a model trains happily on
    0-rouble monthly salaries."""

    def test_zero_becomes_missing_not_a_salary(self):
        df = rows(record(salary="от 0", salary_min=0, salary_max=0))
        assert math.isnan(df.loc[0, "salary"])
        assert bool(df.loc[0, "has_salary"]) is False
        assert math.isnan(df.loc[0, "log_salary"])

    def test_the_sentinel_is_recorded_not_silently_dropped(self):
        """A repair that leaves no trace cannot be audited or reversed."""
        df = rows(record(salary="от 0", salary_min=0, salary_max=0))
        assert bool(df.loc[0, "salary_zero_sentinel"]) is True

    def test_a_real_salary_is_untouched(self):
        df = rows(record())
        assert df.loc[0, "salary"] == 40000
        assert bool(df.loc[0, "salary_zero_sentinel"]) is False
        assert bool(df.loc[0, "has_salary"]) is True


# --------------------------------------------------------------------------
# Seam 2b: the degenerate range
# --------------------------------------------------------------------------

class TestDegenerateRange:
    """`salary_max == salary_min` leaves the upper bound uninformative. The
    policy records the fact and drops the bound rather than guessing whether the
    employer meant a fixed salary or simply omitted the range."""

    def test_equal_bounds_drop_the_upper_bound(self):
        df = rows(record(salary="от 40000", salary_min=40000, salary_max=40000))
        assert df.loc[0, "salary"] == 40000
        assert math.isnan(df.loc[0, "salary_max"])
        assert bool(df.loc[0, "salary_open_ended"]) is True

    def test_a_genuine_range_is_kept(self):
        df = rows(record(salary="от 40000", salary_min=40000, salary_max=60000))
        assert df.loc[0, "salary"] == 40000
        assert df.loc[0, "salary_max"] == 60000
        assert bool(df.loc[0, "salary_open_ended"]) is False

    def test_an_upper_bound_below_the_lower_one_is_not_a_range(self):
        """Stage 2 measured 472 records with `salary_max == 0` against a
        positive `salary_min`. A max below the min is not a range either."""
        df = rows(record(salary="от 40000", salary_min=40000, salary_max=0))
        assert df.loc[0, "salary"] == 40000
        assert math.isnan(df.loc[0, "salary_max"])
        assert bool(df.loc[0, "salary_open_ended"]) is True


# --------------------------------------------------------------------------
# Seam 2c: extraction reaches the analytical row
# --------------------------------------------------------------------------

class TestTextMiningReachesTheRow:
    """The mined features are only useful if they survive the flatten step."""

    def test_skills_are_mined_from_the_free_text(self):
        df = rows(record(requirements="Опыт работы в 1С, знание SQL"))
        assert set(df.loc[0, "skills_mined"]) == {"1c", "sql"}

    def test_structured_skills_counted_separately_from_mined(self):
        """The two sources are kept apart so their coverage can be compared.

        The free text is emptied here on purpose: the default fixture mentions
        1С and SQL, which would be mined and would mask what this test is about.
        """
        df = rows(record(skills=[{"name": "Python"}, {"name": "Django"}],
                         requirements="Внимательность и ответственность",
                         duty=""))
        assert df.loc[0, "skills_structured_n"] == 2
        assert df.loc[0, "skills_mined_n"] == 0

    def test_the_withdrawn_experience_column_is_not_read_as_years(self):
        """`requirement.experience` is a code, not a year count; Stage 2
        withdrew it. It must survive as an opaque value and must NOT feed
        `experience_years`."""
        df = rows(record(requirement={"education": "Высшее", "experience": 18}))
        assert df.loc[0, "experience_code_structured"] == 18
        assert math.isnan(df.loc[0, "experience_years"])


# --------------------------------------------------------------------------
# Seam 2d: deduplication
# --------------------------------------------------------------------------

class TestDeduplication:
    """Two distinct mechanisms: the same vacancy arriving twice, and an employer
    reposting a near-identical advert."""

    def test_same_id_collapses_to_one_row(self):
        df = rows(record(), record())
        out = dedupe(df, [])
        assert len(out) == 1

    def test_the_it_copy_wins_when_a_vacancy_is_in_both_sets(self):
        """A vacancy matching both an IT and a control keyword must land in the
        IT set, or the IT/control comparison partly compares a group to itself."""
        it = rows(record(), is_it=True)
        ctrl = rows(record(), is_it=False)
        out = dedupe(pd.concat([it, ctrl], ignore_index=True), [])
        assert len(out) == 1
        assert bool(out.iloc[0]["is_it"]) is True

    def test_near_duplicates_collapse_keeping_the_newest(self):
        """Same employer, same normalised title, same salary — one advert
        reposted. The most recently modified copy is the one worth keeping."""
        old = rows(record(id="old", date_modify="2026-01-01T00:00:00+0300"))
        new = rows(record(id="new", date_modify="2026-06-01T00:00:00+0300"))
        out = dedupe(pd.concat([old, new], ignore_index=True), [])
        assert len(out) == 1
        assert out.iloc[0]["id"] == "new"

    def test_different_salaries_are_not_near_duplicates(self):
        """The key includes salary, so two genuinely different adverts from the
        same employer for the same title both survive."""
        a = rows(record(id="a", salary="от 40000",
                        salary_min=40000, salary_max=40000))
        b = rows(record(id="b", salary="от 90000",
                        salary_min=90000, salary_max=90000))
        out = dedupe(pd.concat([a, b], ignore_index=True), [])
        assert len(out) == 2


# --------------------------------------------------------------------------
# Seam 2e: the gates must be able to fail
# --------------------------------------------------------------------------

class TestQualityGates:
    """A gate that cannot fail is decoration. These tests drive failures
    deliberately, so a silently disabled check cannot pass unnoticed.

    The complementary assertion — that the real dataset passes every gate — is
    an integration-level claim and lives in `test_pipeline_integration.py`. It
    cannot be made here: gates such as "no skill pattern is degenerate" are
    properties of a corpus, not of a two-row fixture.
    """

    def test_a_zero_salary_left_in_the_table_fails_the_gate(self):
        """Proves the zero-sentinel gate actually gates."""
        df = rows(record())
        df.loc[0, "salary"] = 0.0          # bypass the repair deliberately
        gates = Gates()
        run_gates(df, gates)
        failed = [n for n, _e, _a, ok in gates.rows if not ok]
        assert any("zero salary" in n for n in failed), failed

    def test_a_dead_skill_pattern_fails_the_gate(self):
        """A pattern that never fires is a silent defect: the feature is zero
        everywhere and nothing complains. The gate must catch it."""
        df = rows(record())
        df["skill_python"] = False         # a column that can never be true
        gates = Gates()
        run_gates(df, gates)
        failed = [n for n, _e, _a, ok in gates.rows if not ok]
        assert any("degenerate" in n for n in failed), failed
