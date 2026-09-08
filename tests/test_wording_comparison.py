"""CI gate for the clarifier-wording ablation (Appendix Tables A18-A19).

Hermetic subset of the wording-ablation checks: the committed A18-A19
tables, the A9 audit table, and the manuscript are all in-tree, so CI can
pin the Design, A9-caveat, and appendix prose numbers to the committed
tables without git history. The archive recomputation — rebuilding both
tables from the superseded April 19 runs at commit ``ddca237`` — uses exact
blobs retained under ``archive/`` and runs in the complete cached reproduction
CI gate, including in shallow clones.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verify_paper_prose  # noqa: E402


def test_wording_comparison_prose_matches_committed_tables():
    verify_paper_prose.FAILURES.clear()
    verify_paper_prose.verify_wording_comparison_prose()
    assert not verify_paper_prose.FAILURES, verify_paper_prose.FAILURES
