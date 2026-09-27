"""Behavioural tests for the pure functions in `textmining.py`.

These are the functions Stage 4 exists for. The defects they guard against are
silent: a wrong regex raises nothing, it just turns a feature into zero at scale.
`textmining.py` records the measured consequence — a naive `c\\+\\+` misses 102
of 239 C++ vacancies (43%) because Russian adverts write it with a Cyrillic С.

Every expected value below is a hand-worked literal for the string under test.
None is produced by re-running the implementation's own expression, which would
make the assertion true by construction and unable to disagree with the code.
"""

import pytest

from textmining import (extract_experience_years, extract_skills, job_title_key,
                        normalise_whitespace, parse_salary_text)

# --------------------------------------------------------------------------
# Seam 1a: the free-text salary field
# --------------------------------------------------------------------------

class TestSalaryParsing:
    """`parse_salary_text` returns (low, high).

    Measured on the corpus: 100% of the 21,941 salaried records are the single
    shape "от N". The other shapes are tested anyway, because the parser must
    not silently invert its meaning if the source ever changes.
    """

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("от 40000", (40000, None)),        # the only shape in the corpus
            ("40000", (40000, None)),           # bare number: a stated value
            ("от 40 000", (40000, None)),       # space as thousands separator
            ("от 40\u00a0000", (40000, None)),  # NBSP as thousands separator
            ("от 40000 до 60000", (40000, 60000)),
            ("от 0", (0, None)),                # the zero sentinel
            ("", (None, None)),
            (None, (None, None)),
        ],
    )
    def test_known_shapes(self, text, expected):
        assert parse_salary_text(text) == expected

    def test_up_to_states_an_upper_bound_only(self):
        """'до 50000' means "up to 50000".

        The number is an upper bound, not a lower one. Reporting it as the low
        bound would invert the meaning of the advert — and the pipeline uses the
        low bound as the regression target.
        """
        assert parse_salary_text("до 50000") == (None, 50000)


# --------------------------------------------------------------------------
# Seam 1b: technology extraction from the Russian free text
# --------------------------------------------------------------------------

class TestSkillExtraction:
    """`extract_skills` returns the set of known technologies in a text blob.

    This is the function the headline defect lives in. The structured `skills`
    field is empty on 77% of records, so these patterns are the only source of
    technology features — and a pattern that fails silently turns a feature into
    zero at scale without raising anything.
    """

    @pytest.mark.parametrize(
        "text, expected",
        [
            # Cyrillic С (U+0421) -- measured: a naive pattern misses 43% of
            # C++ vacancies because Russian adverts write it this way.
            ("Знание языков С и С++ на базовом уровне", {"c++"}),
            ("Knowledge of C++ required", {"c++"}),
            ("Уверенное знание языка С#", {"c#"}),
            ("C# and .NET", {"c#"}),
            # 1С is a Russian product; both the Cyrillic and Latin spellings
            # appear, sometimes in the same corpus.
            ("Опыт работы в 1С 8.3", {"1c"}),
            ("1C:Enterprise developer", {"1c"}),
            # Java must not swallow JavaScript, nor the reverse.
            ("Strong Java background", {"java"}),
            ("JavaScript and React", {"javascript"}),
            ("Strong Java, some JavaScript", {"java", "javascript"}),
            ("Java Script developer", {"javascript"}),
            ("Опыт работы с SQL", {"sql"}),
            ("Docker and Kubernetes", {"docker", "kubernetes"}),
            ("PostgreSQL, Linux, Git", {"postgresql", "linux", "git"}),
        ],
    )
    def test_matches_the_technology_present(self, text, expected):
        assert set(extract_skills(text)) == expected

    @pytest.mark.parametrize(
        "text",
        [
            # 'с' is one of the most common prepositions in Russian. Fold the
            # alphabet into Latin here and every advert becomes a C++ advert.
            "Работа с детьми, опыт с документами",
            "Отдел С по работе с клиентами",
            # Substring traps: the patterns must not fire from inside a word.
            "digitalisation project manager",
            "Интересная работа, restoran рядом",
            "Требуется внимание и аккуратность",
            "Продавец-консультант, график 5/2",
            "",
        ],
    )
    def test_reports_nothing_when_no_technology_is_present(self, text):
        assert set(extract_skills(text)) == set()


# --------------------------------------------------------------------------
# Seam 1c: experience stated in the free text
# --------------------------------------------------------------------------

class TestExperienceExtraction:
    """`extract_experience_years` returns the smallest stated requirement.

    This is a fallback only: `requirement.experience` is an undocumented code
    rather than a year count (withdrawn in Stage 2), and the free text states a
    requirement in about 10% of records. The function must still be correct for
    the records where it is the only signal.
    """

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Опыт работы аналитиком данных от 3 лет", 3),
            ("стаж не менее 5 лет", 5),
            ("опыт от 1 года", 1),
            # Several statements: the smallest, so the feature reflects the
            # lowest bar the advert states rather than an arbitrary one.
            ("опыт работы от 2 лет, опыт от 5 лет", 2),
            # No requirement stated at all.
            ("требования не предъявляются", None),
            # Implausible values are dropped, not read as 99 years.
            ("опыт от 99 лет", None),
            ("", None),
            (None, None),
        ],
    )
    def test_years(self, text, expected):
        assert extract_experience_years(text) == expected


# --------------------------------------------------------------------------
# Seam 1d: normalisation, used by the near-duplicate key
# --------------------------------------------------------------------------

class TestNormalisation:
    """`job_title_key` feeds near-duplicate detection and `normalise_whitespace`
    feeds the fee-text blob. Both are relied on to make two spellings of the
    same string compare equal."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Программист 1С", "программист 1с"),
            ("Программист  1С ", "программист 1с"),   # collapsed, then trimmed
            ("программист-1с", "программист 1с"),     # punctuation stripped
            ("", ""),
            (None, ""),
        ],
    )
    def test_title_key(self, text, expected):
        assert job_title_key(text) == expected

    def test_whitespace_collapses_nbsp(self):
        """NBSP appears in real titles and must not split a token."""
        assert normalise_whitespace("Инженер\u00a0по\u00a0сетям") == "Инженер по сетям"
        assert normalise_whitespace("a  b\tc") == "a b c"
        assert normalise_whitespace(None) == ""
