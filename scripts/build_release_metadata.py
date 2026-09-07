"""Refresh the citation abstract and machine-readable panel metadata from evidence."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.archive_manifest import verify_manifest  # noqa: E402


def main() -> int:
    manifest = verify_manifest()
    panel = manifest["main_panel"]
    metadata = {
        "schema_version": 1,
        "archive_manifest": "results/archive-manifest.json",
        "main_panel": {
            k: panel[k]
            for k in (
                "model_count",
                "organization_count",
                "quantity_count",
                "repetitions_per_cell",
                "records",
                "successful_records",
                "failed_records",
                "model_ids",
                "prompt_versions",
            )
        },
        "retained_run_scopes": {
            scope: {
                k: census[k]
                for k in (
                    "records",
                    "successful_records",
                    "failed_records",
                )
            }
            for scope, census in manifest["retained_run_scopes"].items()
        },
        "calibration_provenance": manifest["calibration_provenance"],
        "release_status": "archival preparation; no DOI or new release asserted",
    }
    (ROOT / "release-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    abstract = (
        "abstract: >-\n"
        f"  Elicitation harness, {panel['successful_records']:,}-run retained main-panel dataset, and paper studying\n"
        f"  the prompt-conditioned response distributions that {panel['model_count']} frontier large\n"
        f"  language models from {panel['organization_count']} organizations produce when asked about\n"
        f"  {panel['quantity_count']} economic quantities with {panel['repetitions_per_cell']} successful repeated runs per model-quantity\n"
        "  cell. All retained main runs are tagged v4; three sign-clarified quantities\n"
        "  retain two disclosed prompt wordings. Follow-ups, pilots, and archived\n"
        "  failed attempts are counted separately in results/archive-manifest.json.\n"
    )
    citation = ROOT / "CITATION.cff"
    source = citation.read_text()
    # No release is asserted for this retained snapshot.
    source = re.sub(r"^date-released:.*\n", "", source, flags=re.MULTILINE)
    updated, count = re.subn(r"abstract: >-\n(?:[ \t].*(?:\n|$))*", abstract, source)
    if count != 1:
        raise ValueError("Expected one citation abstract")
    citation.write_text(updated)
    print("Citation and release metadata match the retained panel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
