"""Shared pytest configuration.

Puts the repository root on `sys.path` so the tests import the project modules
the same way the pipeline scripts do, without needing an installable package.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
