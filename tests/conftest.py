"""Shared pytest configuration.

Puts the repository root on `sys.path`, so the tests reach the pipeline through
its package names -- `from stage2_processing import build_features` -- the same
way the scripts reach the shared code through `common`. Nothing here depends on
the working directory.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
