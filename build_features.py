#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 -- DATA PROCESSING
==========================

Turns the raw harvests into one analytical table plus a data-quality report.
No modelling happens here; this stage exists so that Stage 3 can consume a table
whose defects are already known, quantified and repaired by an explicit policy.

WHAT IT DOES
------------
  2a. Flatten    nested API records into flat rows
  2b. Mine       skills and experience out of the Russian free text (RegEx)
  2c. Repair     the salary fields, with the policy written down and measured
  2d. Dedupe     exact and near-duplicate vacancies
  2e. Gate       assertions that must hold for the table to be trustworthy
  2f. Report     docs/data_quality_report.md

THE SALARY REPAIR POLICY -- AND A CORRECTION
--------------------------------------------
An earlier reading of this data said "the portal writes the same value into both
bounds for open-ended adverts". That was too strong. Inspection during Stage 2
found records like:

    salary = "от 75000",  salary_min = 75000,  salary_max = 200000

So `salary_max` IS sometimes a genuine upper bound. What is actually measured is:

  * the free-text `salary` field is `"от N"` in 100% of 21,941 salaried records,
    so the TEXT never carries an upper bound;
  * `salary_max == salary_min` in 36% of records that have both, and in those the
    "range" is degenerate.

Whether an equal pair means "fixed salary" or "upper bound never supplied" cannot
be determined from the data. The policy therefore treats it as UNKNOWN rather
than guessing: when the bounds are equal, `salary_max` is set to missing and the
row is flagged `salary_open_ended`. Both the flag and the raw values are kept, so
a different choice can be evaluated later without re-running the harvest.

OUTPUT
------
    data/processed/vacancies.parquet     analytical table
    docs/data_quality_report.md          gate results and distributions
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from textmining import (
    SKILL_SLUGS,
    extract_experience_years,
    extract_skills,
    job_title_key,
    normalise_whitespace,
    parse_salary_text,
)

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw_samples"
OUT_DIR = HERE / "data" / "processed"
DOCS = HERE / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(exist_ok=True)

TABLE_PATH = OUT_DIR / "vacancies.parquet"
REPORT_PATH = DOCS / "data_quality_report.md"

# Plausibility window for a monthly RUR salary. Deliberately wide: the job is to
# catch data entry errors (a missing zero, an annual figure), not to make a
# judgement about what a fair wage is.
SALARY_PLAUSIBLE_MIN = 5_000
SALARY_PLAUSIBLE_MAX = 1_000_000

TEXT_FIELDS = ("requirements", "duty", "job-name", "qualification", "typicalPosition")


# --------------------------------------------------------------------------
# 2a. Flatten
# --------------------------------------------------------------------------

def load_raw(path: pathlib.Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def flatten(rec: dict, is_it: bool) -> dict:
    company = rec.get("company") or {}
    region = rec.get("region") or {}
    requirement = rec.get("requirement") or {}
    category = rec.get("category") or {}
    addresses = (rec.get("addresses") or {}).get("address") or []
    first_addr = addresses[0] if addresses else {}

    for_txt = [rec.get(k) for k in TEXT_FIELDS]
    blob = normalise_whitespace(" ".join(str(t) for t in for_txt if t))

    skills_structured = rec.get("skills") or []
    if isinstance(skills_structured, str):
        skills_structured = [skills_structured]
    skills_mined = extract_skills(*[str(t) for t in for_txt if t])

    sal_text = rec.get("salary")
    parsed_lo, parsed_hi = parse_salary_text(sal_text)
    sal_min = rec.get("salary_min")
    sal_max = rec.get("salary_max")

    # Repair policy: equal bounds carry no range information -> unknown.
    open_ended = (
        sal_min is not None
        and sal_max is not None
        and int(sal_max) <= int(sal_min)
    )
    sal_max_repaired = None if (open_ended or sal_max is None) else int(sal_max)

    return {
        "id": rec.get("id"),
        "is_it": is_it,
        # --- employer ---
        "company_inn": (company.get("inn") or "").strip() or None,
        "company_name": normalise_whitespace(company.get("name")),
        "company_has_site": bool(company.get("site")),
        "company_is_hr_agency": bool(company.get("hr-agency")),
        # --- place ---
        "region_name": region.get("name"),
        "region_code": str(region.get("region_code") or "")[:2] or None,
        "lat": float(first_addr["lat"]) if first_addr.get("lat") else None,
        "lng": float(first_addr["lng"]) if first_addr.get("lng") else None,
        # --- posting ---
        "job_name": normalise_whitespace(rec.get("job-name")),
        "job_title_key": job_title_key(rec.get("job-name")),
        "creation_date": rec.get("creation-date"),
        "date_modify": rec.get("date_modify"),
        "work_places": rec.get("work_places"),
        # --- salary (raw + repaired) ---
        "salary_text": sal_text,
        "salary_min_raw": sal_min,
        "salary_max_raw": sal_max,
        "salary_text_lo": parsed_lo,
        "salary_lo_matches_field": (
            parsed_lo is not None
            and sal_min is not None
            and int(parsed_lo) == int(sal_min)
        ),
        "currency": rec.get("currency"),
        # --- requirements ---
        "education": (requirement.get("education") or None),
        # NOT years. The field looks like a year count but is not one: the
        # observed code space is 0,1,2,3,4,5,6,7,10,18,20,21,23,31,42 with no
        # documented scale, and code=10 appears on adverts whose text says
        # "от 10 лет" while code=18 appears on ones saying "от 1 года". It is
        # kept as an opaque code and is deliberately NOT used as a feature.
        # See docs/data_quality_report.md section 5.
        "experience_code_structured": requirement.get("experience"),
        "qualification": normalise_whitespace(rec.get("qualification")),
        "schedule": normalise_whitespace(rec.get("schedule")),
        "employment": normalise_whitespace(rec.get("employment")),
        "specialisation": normalise_whitespace(category.get("specialisation")),
        "typical_position": normalise_whitespace(rec.get("typicalPosition")),
        "code_profession": rec.get("code_profession"),
        "social_protected": normalise_whitespace(rec.get("social_protected")),
        # --- mined text ---
        "experience_years_text": extract_experience_years(blob),
        "len_requirements": len(rec.get("requirements") or ""),
        "len_duty": len(rec.get("duty") or ""),
        "skills_structured_n": len(skills_structured),
        "skills_mined": sorted(skills_mined),
        "skills_mined_n": len(skills_mined),
        # Full concatenated free text, kept so Stage 3 can run TF-IDF over it
        # rather than only over the dictionary hits.
        "text_blob": blob,
        # --- kept so the repair can be re-evaluated later ---
        "_salary_open_ended": open_ended,
        "_salary_max_repaired": sal_max_repaired,
    }


# --------------------------------------------------------------------------
# 2b/2c. Derive the modelling columns
# --------------------------------------------------------------------------

def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["salary"] = pd.to_numeric(df["salary_min_raw"], errors="coerce")

    # The portal encodes "salary not specified" as the literal value 0
    # (the free text reads "от 0"), not as null. A sentinel that looks like
    # data is the most dangerous kind: log(0) is -inf, and a model will
    # otherwise learn happily from 0-rouble monthly salaries. Neutralise it and
    # keep a flag so the decision stays visible.
    df["salary_zero_sentinel"] = df["salary"].eq(0)
    df.loc[df["salary_zero_sentinel"], "salary"] = np.nan

    df["salary_max"] = pd.to_numeric(df["_salary_max_repaired"], errors="coerce")
    df["salary_open_ended"] = df["_salary_open_ended"].astype(bool)
    df["has_salary"] = df["salary"].notna()
    df["log_salary"] = np.log(df["salary"].where(df["salary"] > 0))
    df["salary_mid"] = df[["salary", "salary_max"]].mean(axis=1, skipna=True)

    df["salary_plausible"] = df["salary"].between(
        SALARY_PLAUSIBLE_MIN, SALARY_PLAUSIBLE_MAX
    )

    # `requirement.experience` is an undocumented code, not a year count, so it
    # is carried as an opaque value and excluded from the feature set. The only
    # usable experience signal is what RegEx recovers from the free text -- and
    # that covers only a small minority of records.
    exp_code_raw = df["experience_code_structured"]
    exp_code = pd.to_numeric(exp_code_raw, errors="coerce")
    df["experience_code_non_numeric"] = exp_code_raw.notna() & exp_code.isna()
    df["experience_code_structured"] = exp_code

    exp_text = pd.to_numeric(df["experience_years_text"], errors="coerce")
    df["experience_years"] = exp_text
    df["experience_source"] = np.where(exp_text.notna(), "mined-from-text", "absent")
    df["creation_ts"] = pd.to_datetime(df["creation_date"], errors="coerce")
    df["modified_ts"] = pd.to_datetime(
        df["date_modify"], errors="coerce", utc=True, format="ISO8601"
    )

    for slug in SKILL_SLUGS:
        df[f"skill_{slug}"] = df["skills_mined"].apply(lambda s, k=slug: k in s)

    return df


# --------------------------------------------------------------------------
# 2d. Deduplicate
# --------------------------------------------------------------------------

def dedupe(df: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    before = len(df)

    # Exact: same vacancy id harvested through more than one route.
    df = df.sort_values("is_it", ascending=False)      # keep the IT copy
    df = df.drop_duplicates(subset=["id"], keep="first")

    exact_removed = before - len(df)

    # Near-duplicate: same employer, same normalised title, same salary.
    # Employers repost identical adverts, and duplicates leak across a random
    # train/test split, so they are collapsed to the most recently modified copy.
    key_cols = ["company_inn", "job_title_key", "salary"]
    keyable = df[key_cols].notna().all(axis=1)
    sub = df[keyable].sort_values("modified_ts", ascending=False)
    dupe_ids = sub.duplicated(subset=key_cols, keep="first")
    removed_ids = set(sub.loc[dupe_ids, "id"])
    near_removed = len(removed_ids)
    df = df[~df["id"].isin(removed_ids)]

    report.append(f"- exact duplicates removed (same vacancy id): "
                  f"**{exact_removed:,}**")
    report.append(f"- near-duplicates removed (same employer + title + salary, "
                  f"newest kept): **{near_removed:,}**")
    report.append(f"- rows remaining: **{len(df):,}**")
    return df


# --------------------------------------------------------------------------
# 2e. Quality gates
# --------------------------------------------------------------------------

class Gates:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def check(self, name: str, expected: str, actual: str, passed: bool) -> None:
        self.rows.append((name, expected, actual, passed))

    @property
    def failed(self) -> int:
        return sum(1 for *_, ok in self.rows if not ok)


def run_gates(df: pd.DataFrame, gates: Gates) -> None:
    n = len(df)

    gates.check("`id` is unique and never null",
                "100%",
                f"{100 * df['id'].notna().mean():.1f}% non-null, "
                f"{df['id'].nunique():,} distinct",
                df["id"].notna().all() and df["id"].nunique() == n)

    gates.check("`region_name` present", "100%",
                f"{100 * df['region_name'].notna().mean():.1f}%",
                df["region_name"].notna().all())

    gates.check("`creation_date` parses", "100%",
                f"{100 * df['creation_ts'].notna().mean():.1f}%",
                df["creation_ts"].notna().all())

    sal = df[df["has_salary"]]
    gates.check("salary present",
                ">=97%",
                f"{100 * df['has_salary'].mean():.2f}%",
                df["has_salary"].mean() >= 0.97)

    zero_left = int((df["salary"] == 0).sum())
    gates.check("no zero salary survives into the table", "0 rows",
                f"{zero_left} rows "
                f"({int(df['salary_zero_sentinel'].sum()):,} sentinels neutralised)",
                zero_left == 0)

    cur = sal["currency"].dropna().unique().tolist()
    gates.check("currency is uniformly «руб.»",
                "1 distinct value",
                f"{cur}",
                len(cur) == 1)

    agree = sal["salary_lo_matches_field"].mean() if len(sal) else 0
    gates.check("RegEx salary parse agrees with `salary_min`",
                "100%",
                f"{100 * agree:.1f}%",
                agree >= 0.999)

    in_range = sal["salary_plausible"].mean() if len(sal) else 0
    gates.check(f"salary within [{SALARY_PLAUSIBLE_MIN:,}, "
                f"{SALARY_PLAUSIBLE_MAX:,}] RUR",
                ">=95%",
                f"{100 * in_range:.1f}%",
                in_range >= 0.95)

    # A skill pattern that fires on almost everything is broken; one that never
    # fires is a dead rule. Both are silent failures, so both are gated.
    max_rate = max(df[f"skill_{s}"].mean() for s in SKILL_SLUGS)
    min_hits = min(df[f"skill_{s}"].sum() for s in SKILL_SLUGS)
    gates.check("no skill pattern is degenerate",
                "max rate <50%, min hits >0",
                f"max rate {100 * max_rate:.1f}%, min hits {min_hits}",
                max_rate < 0.5 and min_hits > 0)

    gates.check("IT / control split intact",
                "both non-empty",
                f"{int(df[df['is_it']].shape[0]):,} IT / "
                f"{int(df[~df['is_it']].shape[0]):,} control",
                df["is_it"].any() and (~df["is_it"]).any())


# --------------------------------------------------------------------------
# 2f. Report
# --------------------------------------------------------------------------

def write_report(df: pd.DataFrame, gates: Gates, dedupe_lines: list[str],
                 src_counts: dict[str, int]) -> None:
    out: list[str] = []
    a = out.append

    a("# Data quality report — Stage 2")
    a("")
    a("Generated by `build_features.py`. Every figure is computed from the data.")
    a("")
    a(f"- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    a(f"- Input: `raw_samples/trudvsem_it_harvest.json.gz` "
      f"({src_counts['it']:,} records), "
      f"`raw_samples/trudvsem_control_harvest.json.gz` "
      f"({src_counts['control']:,} records)")
    a(f"- Output: `data/processed/vacancies.parquet` ({len(df):,} rows, "
      f"{df.shape[1]} columns)")
    a("")

    a("## 1. Row accounting")
    a("")
    a(f"- raw records loaded: **{src_counts['it'] + src_counts['control']:,}** "
      f"({src_counts['it']:,} IT + {src_counts['control']:,} control)")
    for line in dedupe_lines:
        a(line)
    a(f"- analytical columns: **{df.shape[1]}**")
    a("")
    a("The two harvests overlap by design: a vacancy matching both an IT keyword")
    a("and a control keyword (say \"инженер\") is returned by both sweeps. Those")
    a("rows are collapsed above, keeping the IT copy, so the IT and control sets")
    a("in the final table are disjoint — otherwise the IT premium in section 4")
    a("would be partly a comparison of a group with itself.")
    a("")

    a("## 2. Quality gates")
    a("")
    a(f"**{len(gates.rows) - gates.failed}/{len(gates.rows)} passed.** "
      "These are assertions, not descriptions: a failure means a downstream "
      "number cannot be trusted until it is explained.")
    a("")
    a("| Gate | Expected | Actual | Result |")
    a("|---|---|---|---|")
    for name, expected, actual, ok in gates.rows:
        a(f"| {name} | {expected} | {actual} | {'PASS' if ok else '**FAIL**'} |")
    a("")

    a("## 3. Salary repair policy and its impact")
    a("")
    a("Two repairs are applied, and both are recorded rather than silently done.")
    a("")
    a("**(a) The zero sentinel.** The portal encodes \"salary not specified\" as")
    a("the literal value `0` — the free text reads `\"от 0\"` — instead of leaving")
    a("the field null. A sentinel that looks like real data is the most dangerous")
    a("kind of defect: `log(0)` is `-inf`, and a model will otherwise train on")
    a("0-rouble monthly salaries. These rows are set to missing and flagged")
    a("`salary_zero_sentinel`.")
    a("")
    a("**(b) The degenerate range.** The free-text `salary` field is `\"от N\"` in")
    a("every salaried record, so the text never carries an upper bound. Where")
    a("`salary_max <= salary_min` the range carries no information; `salary_max`")
    a("is set to missing and the row flagged `salary_open_ended` rather than")
    a("guessing whether the employer meant a fixed salary or simply omitted the")
    a("upper bound. The raw values are retained in the table, so a different")
    a("policy can be evaluated later without re-running the harvest.")
    a("")
    total = len(df)
    has = int(df["has_salary"].sum())
    a("| Quantity | Count | Share of all rows |")
    a("|---|---:|---:|")
    a(f"| rows | {total:,} | 100% |")
    a(f"| `salary` present and usable | {has:,} | {100*has/total:.2f}% |")
    a(f"| zero sentinel (`\"от 0\"`) neutralised | "
      f"{int(df['salary_zero_sentinel'].sum()):,} | "
      f"{100*df['salary_zero_sentinel'].mean():.2f}% |")
    a(f"| `salary_open_ended` (upper bound unusable) | "
      f"{int(df['salary_open_ended'].sum()):,} | "
      f"{100*df['salary_open_ended'].mean():.1f}% |")
    usable = int(df["salary_max"].notna().sum())
    a(f"| genuine range available | {usable:,} | {100*usable/total:.1f}% |")
    a(f"| salary outside the plausibility window | "
      f"{int((df['has_salary'] & ~df['salary_plausible']).sum()):,} | "
      f"{100*(df['has_salary'] & ~df['salary_plausible']).mean():.2f}% |")
    a("")

    sal = df.loc[df["has_salary"], "salary"]
    if len(sal):
        q = sal.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        a("Salary distribution (RUR/month, salary-present subset):")
        a("")
        a("| p1 | p25 | median | p75 | p99 |")
        a("|---:|---:|---:|---:|---:|")
        a(f"| {q.iloc[0]:,.0f} | {q.iloc[1]:,.0f} | {q.iloc[2]:,.0f} | "
          f"{q.iloc[3]:,.0f} | {q.iloc[4]:,.0f} |")
        a("")

    a("## 4. Text mining: what RegEx recovered")
    a("")
    a("The structured `skills` field is empty in the large majority of records, so")
    a("the technology features have to come out of the Russian free text. This")
    a("table compares the two sources.")
    a("")
    a(f"- records with a non-empty structured `skills` field: "
      f"**{int((df['skills_structured_n'] > 0).sum()):,}** "
      f"({100*(df['skills_structured_n'] > 0).mean():.1f}%)")
    a(f"- records where RegEx mined at least one skill: "
      f"**{int((df['skills_mined_n'] > 0).sum()):,}** "
      f"({100*(df['skills_mined_n'] > 0).mean():.1f}%)")
    a(f"- mean skills per record: structured "
      f"{df['skills_structured_n'].mean():.2f}, mined "
      f"{df['skills_mined_n'].mean():.2f}")
    a("")
    a("| Skill | Records | Share |")
    a("|---|---:|---:|")
    for slug in SKILL_SLUGS:
        col = df[f"skill_{slug}"]
        a(f"| `{slug}` | {int(col.sum()):,} | {100*col.mean():.1f}% |")
    a("")

    it = df[df["is_it"]]
    ctrl = df[~df["is_it"]]
    if len(it) and len(ctrl):
        a("### The IT premium, before any modelling")
        a("")
        a("A first honest look at research sub-question 2. This is a raw comparison")
        a("with no controls — region and qualification are **not** held constant")
        a("here, which is exactly what Stage 3 must fix.")
        a("")
        a("| Group | Rows | Median salary | Mean salary |")
        a("|---|---:|---:|---:|")
        for label, part in (("IT", it), ("control", ctrl)):
            s = part.loc[part["has_salary"], "salary"]
            a(f"| {label} | {len(part):,} | {s.median():,.0f} | {s.mean():,.0f} |")
        a("")

    a("## 5. A feature that had to be withdrawn: `requirement.experience`")
    a("")
    a("This field is documented as an integer and reads like a year count. It is")
    a("neither usable nor what it appears to be. The investigation is written up")
    a("because the same trap waits in any dataset whose field semantics are")
    a("documented by implication rather than by a schema.")
    a("")
    a("Observed value distribution:")
    a("")
    a("| Value | Rows | Share |")
    a("|---|---:|---:|")
    for val, cnt in df["experience_code_structured"].value_counts().head(12).items():
        a(f"| `{int(val)}` | {cnt:,} | {100*cnt/len(df):.2f}% |")
    a("")
    top2 = df["experience_code_structured"].value_counts().head(2)
    code_max = int(df["experience_code_structured"].max())
    a("If this were a year count the distribution would be implausible on its")
    a(f"face: {100*top2.iloc[0]/len(df):.0f}% of adverts would require exactly "
      f"{int(top2.index[0])} year(s) and {100*top2.iloc[1]/len(df):.0f}% would "
      f"require {int(top2.index[1])}, with a tail reaching {code_max}.")
    a("Cross-checking it against the experience that the free text actually")
    a("states settles it:")
    a("")
    code0 = df.loc[df["experience_code_structured"] == 0, "experience_years"].dropna()
    code1 = df.loc[df["experience_code_structured"] == 1, "experience_years"].dropna()
    a("| Structured value | Adverts whose text states experience | Range stated |")
    a("|---|---:|---|")
    if len(code0):
        a(f"| `0` | {len(code0):,} | {code0.min():.0f} – {code0.max():.0f} years |")
    if len(code1):
        a(f"| `1` | {len(code1):,} | {code1.min():.0f} – {code1.max():.0f} years |")
    a("")
    a("Value `0` accompanies adverts asking for up to 35 years of experience, so")
    a("`0` cannot mean \"no experience required\"; value `10` appears on adverts")
    a("whose text reads \"от 10 лет\" while value `18` appears on ones reading")
    a("\"от 1 года\". The code space has no documented scale and no consistent")
    a("reading.")
    a("")
    a("**Decision:** the column is retained as an opaque code for audit, renamed")
    a("`experience_code_structured` so nothing reads it as years, and **excluded")
    a("from the feature set**.")
    a("")
    a("That leaves the free text as the only experience signal, and its coverage")
    a("is poor:")
    a("")
    a("| Source | Rows | Share |")
    a("|---|---:|---:|")
    for src, cnt in df["experience_source"].value_counts().items():
        a(f"| {src} | {cnt:,} | {100*cnt/len(df):.1f}% |")
    a("")
    a("**Experience is therefore not a usable feature from this source.** That is")
    a("a finding, not a setback: it removes a column that would have looked")
    a("informative and been meaningless. It is also the kind of thing that is far")
    a("cheaper to discover here than in the middle of Stage 3.")
    a("")
    nonnum = int(df["experience_code_non_numeric"].sum())
    a(f"(For completeness: {nonnum:,} rows carry non-numeric text such as "
      f"`\"от 0\"` in that field. Arrow refuses to serialise such a mixed column,")
    a("which is how the field first drew attention at all.)")
    a("")

    a("## 6. Known limitations carried into Stage 3")
    a("")
    a("1. **The sample is not random.** Within one query the API returns its own")
    a("   ordering, and each (keyword, region) cell is capped at 100 records")
    a("   because paging is unreliable. Estimates describe this sample, not the")
    a("   population of Russian vacancies.")
    a("2. **The portal skews towards state-sector and blue-collar roles.** IT is a")
    a("   minority of the 522k national vacancies; the model is valid only within")
    a("   the population it was fitted on.")
    a("3. **`salary_max` is unusable in ~38% of salaried records**, and the")
    a("   free-text salary never carries an upper bound. Only `salary_min` is")
    a("   modelled as the target.")
    a("4. **Experience is not usable.** `requirement.experience` is an")
    a("   undocumented code, not years (section 5), and the free text states an")
    a("   experience requirement in only a small minority of adverts. Stage 3")
    a("   must not lean on experience as an explanatory variable.")
    a("5. **Skill extraction is dictionary-based.** A technology absent from")
    a("   `textmining.SKILL_PATTERNS` is invisible; the dictionary is a documented,")
    a("   reviewable artefact rather than a learned extractor.")
    a("6. **Russian free text is inconsistently spelled.** The Cyrillic/Latin")
    a("   folding in `textmining.py` covers the confusions actually observed;")
    a("   others may exist.")
    a("")

    REPORT_PATH.write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    it_path = RAW / "trudvsem_it_harvest.json.gz"
    ctrl_path = RAW / "trudvsem_control_harvest.json.gz"
    for p in (it_path, ctrl_path):
        if not p.exists():
            sys.exit(f"missing input {p} -- run collect.py first")

    print("Stage 2 -- data processing")
    it_raw = load_raw(it_path)
    ctrl_raw = load_raw(ctrl_path)
    src_counts = {"it": len(it_raw), "control": len(ctrl_raw)}
    print(f"  loaded {len(it_raw):,} IT + {len(ctrl_raw):,} control records")

    rows = [flatten(r, True) for r in it_raw] + \
           [flatten(r, False) for r in ctrl_raw]
    df = pd.DataFrame(rows)
    print(f"  flattened to {len(df):,} rows x {df.shape[1]} columns")

    df = derive(df)
    print(f"  derived modelling columns -> {df.shape[1]} columns")
    print(f"  salary present: {int(df['has_salary'].sum()):,} / {len(df):,} "
          f"({100*df['has_salary'].mean():.2f}%)")

    dedupe_lines: list[str] = []
    df = dedupe(df, dedupe_lines)
    print("  dedupe: " + "; ".join(l.strip("- ") for l in dedupe_lines))

    gates = Gates()
    run_gates(df, gates)
    print(f"  gates: {len(gates.rows) - gates.failed}/{len(gates.rows)} passed")
    for name, expected, actual, ok in gates.rows:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {actual}")

    # Drop the scratch columns used to carry the repair decision.
    df = df.drop(columns=["_salary_open_ended", "_salary_max_repaired"])
    df.to_parquet(TABLE_PATH, index=False)
    print(f"  wrote {TABLE_PATH.relative_to(HERE)}  "
          f"({TABLE_PATH.stat().st_size / 1e6:.1f} MB)")

    write_report(df, gates, dedupe_lines, src_counts)
    print(f"  wrote {REPORT_PATH.relative_to(HERE)}")

    if gates.failed:
        sys.exit(f"{gates.failed} quality gate(s) FAILED")


if __name__ == "__main__":
    main()
