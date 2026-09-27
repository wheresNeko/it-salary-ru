# -*- coding: utf-8 -*-
"""Probe textmining edge cases before deciding which seams deserve tests."""
import pathlib
import sys

# This probe sits one level below the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.textmining import (extract_skills, extract_experience_years,
                               job_title_key, normalise_whitespace,
                               parse_salary_text)

print("=== parse_salary_text ===")
for s in ["от 40000", "до 50000", "от 40 000", "40000",
          "от 40000 до 60000", "от 0", "", None]:
    print(f"  {str(s)!r:<24} -> {parse_salary_text(s)}")
print("  NOTE: 'до 50000' means UP TO 50000 -- is (50000, None) the right reading?")

print()
print("=== extract_skills: the traps ===")
cases = [
    ("C++ Cyrillic", "Знание языков С и С++ на базовом уровне"),
    ("C++ Latin", "Knowledge of C++ required"),
    ("C# Cyrillic", "Уверенное знание языка С#"),
    ("1C Cyrillic", "Опыт работы в 1С 8.3"),
    ("1C Latin", "1C:Enterprise developer"),
    ("Java", "Strong Java background"),
    ("JavaScript", "JavaScript and React"),
    ("Java Script", "Java Script developer"),
    ("Russian 'с'", "Работа с детьми, опыт с документами"),
    ("Russian 'С'", "Отдел С по работе с клиентами"),
    ("SQL", "Опыт работы с SQL"),
    ("git in word", "digitalisation project manager"),
    ("rest in word", "Интересная работа, restoran"),
]
for label, text in cases:
    print(f"  {label:<14} {text[:42]!r:<46} -> {sorted(extract_skills(text))}")

print()
print("=== extract_experience_years ===")
for s in ["опыт работы аналитиком данных от 3 лет",
          "стаж не менее 5 лет",
          "опыт от 1 года",
          "требования не предъявляются",
          "опыт работы от 2 лет, опыт от 5 лет"]:
    print(f"  {s[:44]!r:<46} -> {extract_experience_years(s)}")

print()
print("=== job_title_key / normalise_whitespace ===")
for s in ["Программист 1С", "Программист  1С ", "программист-1с",
          "Инженер\xa0по\xa0сетям"]:
    print(f"  {s!r:<28} -> {job_title_key(s)!r}")
print(f"  normalise_whitespace('a\xa0b  c') -> {normalise_whitespace('a\xa0b  c')!r}")
