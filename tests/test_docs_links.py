"""Every relative link in every markdown file must resolve.

A restructure that moves 24 files leaves links as the easiest thing to break,
and a dead link in a document a reader opens is worse than a failing test —
nobody sees it until the reader does. Checking it here means CI catches it
instead.

Two kinds of target are verified: file paths, and in-document anchors such as
`#42-the-model-comparison`. External URLs are not fetched; a test suite that
reaches the network fails for reasons unrelated to the code.
"""

import pathlib
import re

import pytest

from common import paths

LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def slug(text: str) -> str:
    """Approximate GitHub's heading-anchor algorithm.

    GitHub replaces each space with a hyphen and does NOT collapse runs. That
    matters for headings containing an em dash: `2.3 Q1 — text` loses the dash
    but keeps both surrounding spaces, giving `23-q1--text`. Collapsing the run
    here would report a valid link as broken.
    """
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return s.replace(" ", "-")


def anchors(path: pathlib.Path) -> set[str]:
    return {
        slug(m.group(1))
        for line in path.read_text(encoding="utf-8").splitlines()
        if (m := HEADING.match(line))
    }


def markdown_files() -> list[pathlib.Path]:
    return sorted(
        p for p in paths.ROOT.rglob("*.md")
        if not any(part in {".git", "__pycache__"} for part in p.parts)
    )


@pytest.mark.parametrize("md", markdown_files(), ids=lambda p: p.name)
def test_relative_links_resolve(md: pathlib.Path) -> None:
    text = md.read_text(encoding="utf-8")
    here = md.resolve()
    broken: list[str] = []

    for raw in LINK.findall(text):
        target = raw.split()[0]                    # drop any optional title
        if target.startswith(("http://", "https://", "mailto:")):
            continue

        if target.startswith("#"):                 # same-document anchor
            frag = target[1:]
            if frag and frag not in anchors(here):
                broken.append(f"{target}  (no such heading in this file)")
            continue

        path_part, _, frag = target.partition("#")
        resolved = (here.parent / path_part).resolve()
        if not resolved.exists():
            broken.append(f"{target}  (no such file)")
        elif frag and resolved.suffix == ".md" and frag not in anchors(resolved):
            broken.append(f"{target}  (no such heading in {resolved.name})")

    assert not broken, (
        f"{md.relative_to(paths.ROOT).as_posix()} has "
        f"{len(broken)} unresolved relative link(s):\n  "
        + "\n  ".join(broken)
    )


def test_the_checker_found_some_links_to_check() -> None:
    """Guards against the suite silently passing because the glob broke.

    The threshold is deliberately loose. Its job is to notice a glob that
    matches nothing, not to be updated every time a document gains a link.
    """
    total = sum(
        len([t for t in LINK.findall(p.read_text(encoding="utf-8"))
             if not t.startswith(("http://", "https://", "mailto:"))])
        for p in markdown_files()
    )
    assert total > 30, f"only {total} relative links found; the glob looks wrong"
