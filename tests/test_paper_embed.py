"""The /paper wrapper and its embedded render stay in lockstep.

The wrapper page pins a version param on every link to the raw render so
caches cannot serve a stale manuscript behind a fresh wrapper; these
tests lock that contract and the layout it depends on.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "dashboard" / "src" / "app" / "paper" / "page.tsx"
WEB = REPO_ROOT / "dashboard" / "public" / "paper" / "web"


def test_wrapper_declares_exactly_one_version() -> None:
    source = WRAPPER.read_text()
    versions = re.findall(r'PAPER_VERSION = "([^"]+)"', source)
    assert len(versions) == 1
    # Every ?v= reference routes through the constant — no hardcoded
    # version strings that could drift from it.
    assert len(re.findall(r"\?v=", source)) == len(
        re.findall(r"\?v=\$\{PAPER_VERSION\}", source)
    )


def test_synced_render_exists_and_carries_the_manuscript() -> None:
    index = WEB / "index.html"
    assert index.exists()
    assert (WEB / "index.pdf").exists()
    assert (WEB / "paper_files").is_dir()
    html = index.read_text()
    assert "How Large Language Models Answer Questions" in html


def test_old_pdf_path_keeps_serving() -> None:
    # /ai-beliefs/paper.pdf predates the wrapper and has been shared;
    # next.config redirects it to the synced render.
    config = (REPO_ROOT / "dashboard" / "next.config.ts").read_text()
    assert 'source: "/paper.pdf"' in config
    assert 'destination: "/paper/web/index.pdf"' in config


def test_iframe_contract() -> None:
    source = WRAPPER.read_text()
    assert 'sandbox="allow-same-origin allow-popups allow-popups-to-escape-sandbox"' in source
    assert 'loading="lazy"' in source
    assert "STANDALONE_HREF" in source and "PDF_HREF" in source
