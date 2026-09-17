#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure text-mining helpers for the Trudvsem salary project.

Everything here is a pure function over strings: no I/O, no network, no mutable
module state. That is deliberate. Stage 4 parametrises tests over exactly these
functions, and the defects they guard against are the silent kind -- they raise
no error and simply turn a feature into zero.

WHY ALPHABET-AWARE PATTERNS, NOT TEXT NORMALISATION
---------------------------------------------------
Russian adverts routinely write C++ and C# with a CYRILLIC capital С (U+0421),
because it is visually identical to the Latin C. Python's `re.IGNORECASE` does
NOT fold Cyrillic onto Latin, so a naive `r"c\\+\\+"` silently misses them --
measured at 43% of all C++ vacancies in this dataset (102 of 239).

The tempting fix is to normalise the whole text (`С`->`C`, `Р`->`P`, ...). That
would CORRUPT ordinary Russian: `с` is one of the most common prepositions in the
language, and `Р`, `О`, `А` are extremely common letters. Normalising text turns
normal prose into garbage and manufactures false matches everywhere.

So instead the patterns accept either alphabet at the few positions where the
confusion actually occurs, and every pattern is anchored with an explicit left
boundary so it cannot fire from inside a Russian word.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

# Left boundary: the match must not continue a Latin or Cyrillic word/number.
# \b is unusable here because it treats Cyrillic letters as word characters,
# which is exactly the ambiguity we are trying to resolve.
_LB = r"(?<![0-9A-Za-z\u0410-\u044f\u0401\u0451])"

# A character class for the letters where Latin/Cyrillic confusion really
# happens in Russian job adverts: C and c versus С (U+0421) and с (U+0441).
_C = r"[Cc\u0421\u0441]"

# Same idea for the trailing boundary.
_RB = r"(?![0-9A-Za-z\u0410-\u044f\u0401\u0451])"


# --------------------------------------------------------------------------
# Skill dictionary
# --------------------------------------------------------------------------
# Keys are stable slugs used as column names downstream; values are regexes.
# Only the tokens where alphabet confusion is actually observed are folded.

SKILL_PATTERNS: dict[str, str] = {
    "1c":         _LB + r"1\s*" + _C + _RB,
    "c++":        _LB + _C + r"\s*\+\+",
    "c#":         _LB + _C + r"\s*#",
    "python":     _LB + r"python",
    "java":       _LB + r"java(?!\s*script)",
    "javascript": _LB + r"javascript|" + _LB + r"js" + _RB,
    "typescript": _LB + r"typescript",
    "sql":        _LB + r"sql",
    "postgresql": _LB + r"postgres",
    "mysql":      _LB + r"mysql",
    "oracle":     _LB + r"oracle",
    "linux":      _LB + r"linux|" + _LB + r"линукс",
    "windows":    _LB + r"windows",
    "docker":     _LB + r"docker",
    "kubernetes": _LB + r"kubernetes|" + _LB + r"k8s" + _RB,
    "git":        _LB + r"git" + _RB,
    "excel":      _LB + r"excel|" + _LB + r"эксель",
    "sap":        _LB + r"sap" + _RB,
    "django":     _LB + r"django",
    "rest":       _LB + r"rest(?:ful)?" + _RB + r"|" + _LB + r"api" + _RB,
    "html":       _LB + r"html",
    "css":        _LB + r"css",
}

SKILL_SLUGS: tuple[str, ...] = tuple(sorted(SKILL_PATTERNS))

_SKILL_RES: dict[str, re.Pattern] = {
    slug: re.compile(src, re.IGNORECASE) for slug, src in SKILL_PATTERNS.items()
}


def extract_skills(*texts: str | None) -> frozenset[str]:
    """Return the set of known technologies mentioned in the given texts."""
    blob = " ".join(t for t in texts if t)
    if not blob:
        return frozenset()
    return frozenset(
        slug for slug, rx in _SKILL_RES.items() if rx.search(blob)
    )


# --------------------------------------------------------------------------
# Salary
# --------------------------------------------------------------------------

# Observed shapes: "от 40000", "до 50000", "40000", "от 40000 до 60000",
# "40000 - 60000". Digit groups may use a space or NBSP as a thousands
# separator. Across 21,941 salaried records the ONLY shape present is "от N",
# but the parser must not assume that -- the dataset can change.
_SALARY_RE = re.compile(
    r"""
    (?P<lo_prefix>от|с)?\s*
    (?P<lo>\d[\d\s\u00a0]*)
    (?:\s*(?:до|-|—|–)\s*(?P<hi>\d[\d\s\u00a0]*))?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _to_int(raw: str | None) -> int | None:
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    return int(digits) if digits else None


def parse_salary_text(text: str | None) -> tuple[int | None, int | None]:
    """Parse a free-text salary string into (low, high).

    >>> parse_salary_text("от 40000")
    (40000, None)
    >>> parse_salary_text("от 40 000 до 60 000")
    (40000, 60000)
    """
    if not text:
        return None, None
    m = _SALARY_RE.search(text)
    if not m:
        return None, None
    return _to_int(m.group("lo")), _to_int(m.group("hi"))


# --------------------------------------------------------------------------
# Experience
# --------------------------------------------------------------------------

# "Опыт работы аналитиком данных от 3 лет", "стаж не менее 5 лет",
# "опыт от 1 года". The filler window has to be generous: real strings put the
# occupation between the keyword and the number.
_EXPERIENCE_RE = re.compile(
    r"(?:опыт|стаж)[^\d]{0,40}?(\d{1,2})\s*(?:год|лет|года)",
    re.IGNORECASE,
)


def extract_experience_years(text: str | None) -> int | None:
    """Smallest explicitly stated experience requirement, in years."""
    if not text:
        return None
    hits = [int(m.group(1)) for m in _EXPERIENCE_RE.finditer(text)]
    hits = [h for h in hits if 0 < h <= 50]
    return min(hits) if hits else None


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalise_whitespace(text: str | None) -> str:
    """Collapse all whitespace runs, including NBSP, to single spaces."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def job_title_key(title: str | None) -> str:
    """Fold a job title to a comparison key for near-duplicate detection."""
    if not title:
        return ""
    t = normalise_whitespace(title).lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return _WS_RE.sub(" ", t).strip()
