"""The leakage invariant.

The most expensive mistake available in this project is letting `salary_text`
(which literally reads "от <target>"), `salary_min_raw`, or `salary_max` reach
the feature matrix. It would produce a near-perfect and entirely meaningless
model, and nothing downstream would complain.

The only defence is that every column carries an explicit decision, so these
tests make the classification exhaustive: adding a column to the table without
deciding what it is fails the suite.
"""

import json
import pathlib

import pandas as pd
import pytest

import build_features as bf
import train_models as tm

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "raw_sample.json"


def table_schema() -> set[str]:
    """The columns Stage 2 produces.

    Derived from the frozen fixture rather than by reading the Parquet: the
    Parquet is a gitignored build artifact, so continuous integration would not
    have it and the module would fail at import. `flatten` emits a fixed set of
    keys for every record, so a single fixture record yields the whole schema.
    """
    records = json.loads(FIXTURE.read_text(encoding="utf-8"))
    flat = pd.DataFrame([bf.flatten(r, True) for r in records])
    df = bf.derive(flat)
    # The scratch columns that carry the repair decision are dropped by
    # build_features.build_table before the table is written.
    return set(df.columns) - {"_salary_open_ended", "_salary_max_repaired"}


TABLE_COLUMNS = table_schema()

# Columns routed into the model as features.
DIRECT_FEATURES = set(tm.CATEGORICAL + tm.NUMERIC + tm.BOOLEAN + tm.SKILL_FEATURES)

# Columns that carry the target, or are computed from it.
LEAKAGE = set(tm.LEAKAGE)


def test_every_table_column_carries_a_decision():
    """No column may reach the table unclassified.

    This is the test that fires when someone adds a field in Stage 2 and forgets
    to say whether it is a feature. Silence is the failure mode being prevented.
    """
    covered = (DIRECT_FEATURES | set(tm.TEXT_SOURCES) | LEAKAGE
               | set(tm.OTHER_EXCLUSIONS))
    unclassified = sorted(TABLE_COLUMNS - covered)
    assert not unclassified, (
        "these columns reach the analytical table with no recorded decision on "
        f"whether they are features or leakage: {unclassified}"
    )


def test_no_column_is_declared_as_both_a_feature_and_an_exclusion():
    """Contradictory declarations mean at least one of them is wrong."""
    both = sorted((DIRECT_FEATURES | set(tm.TEXT_SOURCES)) & set(tm.OTHER_EXCLUSIONS))
    assert not both, f"declared as a feature and as excluded: {both}"


def test_no_leakage_column_is_reachable_as_a_feature():
    """The target and its derivatives must not enter the matrix by any route."""
    reachable = LEAKAGE & (DIRECT_FEATURES | set(tm.TEXT_SOURCES))
    assert not reachable, f"leakage columns routed into the model: {reachable}"


def test_the_target_itself_is_never_a_feature():
    assert tm.TARGET not in DIRECT_FEATURES
    assert tm.TARGET not in set(tm.TEXT_SOURCES)
    assert tm.TARGET in LEAKAGE


def test_every_declared_feature_actually_exists():
    """A feature named but absent from the table is a silent no-op, and the
    model trains on one column fewer than the code claims."""
    known = TABLE_COLUMNS | set(tm.DERIVED_FEATURES)
    missing = sorted(DIRECT_FEATURES - known)
    assert not missing, f"declared as features but absent from the table: {missing}"


@pytest.mark.parametrize(
    "column",
    ["salary", "salary_min_raw", "salary_max", "salary_text", "log_salary",
     "salary_mid", "has_salary"],
)
def test_the_known_dangerous_columns_stay_declared(column):
    """Without this, the audit above could be made to pass by deleting the
    declarations rather than by classifying the columns."""
    assert column in LEAKAGE, (
        f"{column} is derived from the target and must stay declared as leakage"
    )
