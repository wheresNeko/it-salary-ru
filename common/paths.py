"""Every filesystem path the pipeline uses, in one place.

Before this module existed, five scripts each derived their own paths from
`__file__`, so moving a single directory meant editing every one of them and CI,
.gitignore and the documentation too. Nothing outside this module should build a
path from `__file__`; import from here instead.

Layout, and which stage owns what:

    common/                        shared by more than one stage
    stage0_source_verification/    is the API usable at all?
    stage1_collection/             raw harvest -> data/raw/
    stage2_processing/             data/raw/ -> the analytical table
    stage3_analytics/              the table -> models and a report
    tests/                         the Stage 4 validation suite
    data/raw/                      committed, immutable inputs
    data/processed/                derived, gitignored, regenerable
    docs/                          the reports a reader is meant to read
    logs/                          what each run produced
"""

from __future__ import annotations

import pathlib

# `common/paths.py` is two levels below the repository root.
ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---- inputs --------------------------------------------------------------
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
IT_HARVEST = DATA_RAW / "trudvsem_it_harvest.json.gz"
CONTROL_HARVEST = DATA_RAW / "trudvsem_control_harvest.json.gz"
REGIONS = DATA_RAW / "regions.json"
SAMPLE_RECORDS = DATA_RAW / "sample_3_records.json"

# ---- derived -------------------------------------------------------------
DATA_PROCESSED = DATA / "processed"
TABLE = DATA_PROCESSED / "vacancies.parquet"

# ---- outputs a reader consumes -------------------------------------------
DOCS = ROOT / "docs"
DOCS_FIGURES = DOCS / "figures"
DATA_DICTIONARY = DOCS / "data_dictionary.md"
DATA_QUALITY_REPORT = DOCS / "data_quality_report.md"
MODEL_REPORT = DOCS / "model_report.md"
VALIDATION_REPORT = DOCS / "validation_report.md"

# ---- run logs ------------------------------------------------------------
LOGS = ROOT / "logs"
VERIFICATION_LOG = LOGS / "verification_log.txt"
COLLECTION_LOG = LOGS / "collection_log.txt"

# ---- test assets ---------------------------------------------------------
TESTS = ROOT / "tests"
TEST_FIXTURES = TESTS / "fixtures"
RAW_FIXTURE = TEST_FIXTURES / "raw_sample.json"

# ---- scripts -------------------------------------------------------------
VERIFY_SOURCES = ROOT / "stage0_source_verification" / "verify_sources.py"
COLLECT = ROOT / "stage1_collection" / "collect.py"
BUILD_FEATURES = ROOT / "stage2_processing" / "build_features.py"
TRAIN_MODELS = ROOT / "stage3_analytics" / "train_models.py"


def ensure_output_dirs() -> None:
    """Create the directories the pipeline writes into.

    Input directories are never created: a missing `data/raw/` is a fact to
    report, not something to paper over with an empty directory.
    """
    for directory in (DATA_PROCESSED, DOCS, DOCS_FIGURES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


def rel(path: str | pathlib.Path) -> str:
    """Render a path relative to the repository root, with forward slashes.

    Used for every path that ends up in a message, a log or a report, so those
    stay correct and readable wherever they are run from.
    """
    try:
        return pathlib.Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
