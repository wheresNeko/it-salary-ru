"""Integration tests: the whole Stage 1 -> Stage 2 path, end to end.

Two levels, because they answer different questions.

* `TestFrozenFixture` runs the real pipeline over a hand-written six-record
  fixture and checks the output against expectations computed by hand. It is the
  regression net: it fails if any stage changes what a record means, and it
  needs no network and no downloaded data.

* `TestCommittedCorpus` runs the same pipeline over the harvest actually
  committed to the repository. It is the claim the project makes in its own
  documentation — that this data passes every quality gate.
"""

import gzip
import json
import math
import pathlib

import pandas as pd
import pytest

import build_features as bf

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "raw_sample.json"

# Which fixture record came from which sweep. Index 3 is the control twin of
# index 0: the same vacancy id, and it must lose to the IT copy.
FIXTURE_IS_IT = [True, True, True, False, True, True]

# Hand-computed from the fixture, not recomputed from the implementation.
EXPECTED_ROW_IDS = {"r1", "r2", "r3", "r6"}
EXPECTED_SALARY = {"r1": 40000.0, "r2": 90000.0, "r3": None, "r6": 50000.0}
EXPECTED_MAX = {"r1": None, "r2": 120000.0, "r3": None, "r6": None}
EXPECTED_OPEN_ENDED = {"r1": True, "r2": False, "r3": True, "r6": True}
EXPECTED_SKILLS = {"r1": {"c++"}, "r2": {"1c", "sql"}, "r3": set(), "r6": set()}


def run_pipeline(records: list[dict], is_it: list[bool]) -> pd.DataFrame:
    flat = pd.DataFrame([bf.flatten(r, flag) for r, flag in zip(records, is_it)])
    return bf.dedupe(bf.derive(flat), [])


# --------------------------------------------------------------------------
# Level 1: the frozen fixture
# --------------------------------------------------------------------------

class TestFrozenFixture:
    """Six raw records in, a known table out.

    The fixture covers, deliberately, every repair and both deduplication
    mechanisms: a degenerate salary range, a genuine range, the zero sentinel,
    a vacancy harvested by both sweeps, and a reposted advert.
    """

    @pytest.fixture(scope="class")
    def table(self) -> pd.DataFrame:
        records = json.loads(FIXTURE.read_text(encoding="utf-8"))
        return run_pipeline(records, FIXTURE_IS_IT)

    def test_six_records_become_four_rows(self, table):
        """Six raw records, two removals: the duplicate id and the older repost.
        An off-by-one here would mean a deduplication rule silently changed."""
        assert len(table) == 4
        assert set(table["id"]) == EXPECTED_ROW_IDS

    def test_the_superseded_repost_is_gone(self, table):
        assert "r5" not in set(table["id"])

    def test_the_it_copy_wins_the_shared_id(self, table):
        """The fixture's control twin shares r1's id. If the control copy won,
        the two groups would overlap and every IT comparison would be partly a
        group compared with itself."""
        row = table.set_index("id").loc["r1"]
        assert bool(row["is_it"]) is True
        assert table["is_it"].all(), "no control record should survive this fixture"

    @pytest.mark.parametrize("row_id", sorted(EXPECTED_ROW_IDS))
    def test_salary_is_what_the_policy_says(self, table, row_id):
        row = table.set_index("id").loc[row_id]
        expected = EXPECTED_SALARY[row_id]
        if expected is None:
            assert math.isnan(row["salary"]), f"{row_id} should have no usable salary"
        else:
            assert row["salary"] == expected
        upper = EXPECTED_MAX[row_id]
        assert math.isnan(row["salary_max"]) if upper is None else row["salary_max"] == upper

    @pytest.mark.parametrize("row_id", sorted(EXPECTED_ROW_IDS))
    def test_open_ended_flag_matches_the_bounds(self, table, row_id):
        row = table.set_index("id").loc[row_id]
        assert bool(row["salary_open_ended"]) is EXPECTED_OPEN_ENDED[row_id]

    @pytest.mark.parametrize("row_id", sorted(EXPECTED_ROW_IDS))
    def test_mined_skills_survive_the_whole_path(self, table, row_id):
        row = table.set_index("id").loc[row_id]
        assert set(row["skills_mined"]) == EXPECTED_SKILLS[row_id]

    def test_the_zero_sentinel_is_flagged(self, table):
        row = table.set_index("id").loc["r3"]
        assert bool(row["salary_zero_sentinel"]) is True
        assert bool(row["has_salary"]) is False

    def test_the_cyrillic_spelling_reaches_the_skill_feature(self, table):
        """r1 writes С++ with a Cyrillic С. If this regresses, the column is
        silently zero and nothing else complains."""
        row = table.set_index("id").loc["r1"]
        assert bool(row["skill_c++"]) is True

    def test_no_row_carries_a_zero_salary(self, table):
        """The invariant the whole repair exists for."""
        assert not (table["salary"] == 0).any()


# --------------------------------------------------------------------------
# Level 2: the corpus committed to the repository
# --------------------------------------------------------------------------

class TestCommittedCorpus:
    """The claim the README and the quality report make about the real data."""

    @pytest.fixture(scope="class")
    def table(self) -> pd.DataFrame:
        it_path = bf.RAW / "trudvsem_it_harvest.json.gz"
        ctrl_path = bf.RAW / "trudvsem_control_harvest.json.gz"
        if not (it_path.exists() and ctrl_path.exists()):
            pytest.skip("committed harvests not present -- run collect.py first")
        return bf.build_table(it_path, ctrl_path)

    def test_every_quality_gate_passes(self, table):
        gates = bf.Gates()
        bf.run_gates(table, gates)
        failures = [(n, a) for n, _e, a, ok in gates.rows if not ok]
        assert gates.failed == 0, f"gate failures on the committed corpus: {failures}"
        assert len(gates.rows) == 10, "a gate was added or removed; update this test"

    def test_the_two_groups_are_disjoint(self, table):
        """Overlap would make the IT premium partly a comparison of a group with
        itself. Stage 2 collapses the 446 shared ids into the IT set."""
        assert not table["id"].duplicated().any()
        assert table["is_it"].any() and (~table["is_it"]).any()

    def test_the_corpus_is_the_size_the_documentation_claims(self, table):
        assert len(table) == 19_578

    def test_salary_coverage_is_what_the_report_claims(self, table):
        """99.09% -- the 0.91% gap is the zero sentinel, not missing data."""
        coverage = float(table["has_salary"].mean())
        assert 0.98 < coverage < 1.0, coverage
        sentinels = int(table["salary_zero_sentinel"].sum())
        assert sentinels == 179, (
            f"the zero sentinel count changed to {sentinels}; if the corpus was "
            "re-collected, update the documented figure too"
        )

    def test_the_cyrillic_trap_is_still_caught_at_scale(self, table):
        """The defect Stage 4 exists for. If the alphabet-aware patterns were
        reverted, `skill_c++` would collapse toward zero."""
        cpp = int(table["skill_c++"].sum())
        assert cpp > 150, f"only {cpp} C++ vacancies found; the patterns regressed"

    def test_no_leakage_column_survived_into_the_table(self, table):
        """Belt and braces with test_leakage.py: the raw salary text must not be
        available to the model under any name it could be picked up by."""
        assert "salary_text" in table.columns          # kept for audit...
        assert table["salary_text"].notna().any()
        # ...but it is declared as leakage in train_models.py, which
        # test_leakage.py enforces. Here we only assert it was not renamed to
        # something innocuous.
        suspicious = [c for c in table.columns
                      if "salary" in c and c not in {
                          "salary", "salary_max", "salary_text", "salary_min_raw",
                          "salary_max_raw", "salary_text_lo",
                          "salary_lo_matches_field", "salary_open_ended",
                          "salary_zero_sentinel", "salary_plausible",
                          "salary_mid", "log_salary", "has_salary"}]
        assert not suspicious, f"undocumented salary column(s): {suspicious}"
