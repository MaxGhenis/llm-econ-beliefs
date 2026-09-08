"""Keep the Table 4 support rationale tied to the retained numerical evidence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_paper_prose as prose  # noqa: E402


def test_eti_support_prose_matches_retained_table():
    prose.FAILURES.clear()
    prose.verify_eti_support_prose()
    assert not prose.FAILURES, prose.FAILURES


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("ETI p05 of `0.086`", "ETI p05 of `0.100`"),
        ("for `Claude Opus 5`.", "for `Claude Sonnet 5`."),
        ("zero lower-tail endpoint", "positive conditioning"),
        ("not conditioning an unrestricted pooled", "conditioning the pooled"),
        (
            "do not measure excluded negative-tail mass",
            "measure excluded negative-tail mass",
        ),
    ],
)
def test_eti_support_prose_rejects_changed_claim(monkeypatch, old, new):
    assert old in prose.PAPER
    monkeypatch.setattr(prose, "PAPER", prose.PAPER.replace(old, new))
    prose.FAILURES.clear()
    prose.verify_eti_support_prose()
    assert prose.FAILURES


def test_eti_support_prose_rejects_restored_false_universal(monkeypatch):
    monkeypatch.setattr(
        prose,
        "PAPER",
        prose.PAPER + " Every model's pooled ETI p05 exceeds `0.10` in Table 4.",
    )
    prose.FAILURES.clear()
    prose.verify_eti_support_prose()
    assert any("universal ETI p05" in failure for failure in prose.FAILURES)


def test_eti_support_prose_detects_changed_referenced_table_value(monkeypatch):
    rows = prose.read_rows("toy-top-rate-labor-tax")
    opus = next(row for row in rows if row["Model"] == "Claude Opus 5")
    opus["ETI median [90%]"] = "0.337 [0.080, 0.961]"
    monkeypatch.setattr(prose, "read_rows", lambda stem: rows)
    prose.FAILURES.clear()
    prose.verify_eti_support_prose()
    assert any("Claude Opus 5 ETI p05" in failure for failure in prose.FAILURES)
