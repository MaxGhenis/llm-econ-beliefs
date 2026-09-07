"""Inventory retained evidence from file contents, with no inference of provenance.

Run --write only when deliberately reviewing an archive update. The ordinary
reproduction command verifies this manifest and never refreshes evidence hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_econ_beliefs.model_registry import (  # noqa: E402
    PANEL_MODEL_IDS,
    get_panel_model,
)
from llm_econ_beliefs.registry import list_quantities  # noqa: E402
from scripts.cached_calibration import verify_calibration_provenance  # noqa: E402
from scripts.check_panel_grid import load_jsonl_rows, validate_grid_rows  # noqa: E402

MANIFEST = "results/archive-manifest.json"
GENERATED_RESULTS = (
    "model-registry.csv",
    "quantity-registry.csv",
    "elasticity-all-model-comparison.csv",
    "elasticity-model-rollup.csv",
    "correlates-model-summary.csv",
    "correlates-spearman.csv",
    "correlates-sensitivity.csv",
    "correlates-country.csv",
    "correlates-posthoc.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_census(rows: list[dict]) -> dict:
    """Report observed tags and parse outcomes, without pooling prompt families."""
    groups = Counter(
        (r["model_name"], r["prompt_version"], r["parsed_ok"]) for r in rows
    )
    return {
        "records": len(rows),
        "successful_records": sum(r["parsed_ok"] is True for r in rows),
        "failed_records": sum(r["parsed_ok"] is False for r in rows),
        "groups": [
            {"model_id": m, "prompt_version": p, "parsed_ok": ok, "records": n}
            for (m, p, ok), n in sorted(groups.items())
        ],
    }


def build_manifest(root: Path = ROOT, *, model_ids=PANEL_MODEL_IDS) -> dict:
    """Derive the main grid, per-prompt membership and all evidence-file hashes."""
    results = root / "results"
    expected_dirs = {f"{m}-elasticities-batch15" for m in model_ids}
    actual_dirs = {p.name for p in results.glob("*-elasticities-batch15") if p.is_dir()}
    if actual_dirs != expected_dirs:
        raise ValueError(
            "Wrong main-panel inventory: "
            f"missing={sorted(expected_dirs - actual_dirs)}, "
            f"unexpected={sorted(actual_dirs - expected_dirs)}"
        )
    quantities = sorted(q.id for q in list_quantities())
    main_rows = []
    variants = defaultdict(lambda: defaultdict(set))
    for model in model_ids:
        path = results / f"{model}-elasticities-batch15/runs.jsonl"
        rows, errors = load_jsonl_rows([path])
        grid = validate_grid_rows(
            rows,
            model_name=model,
            prompt_version="v4",
            quantity_ids=quantities,
            require_parsed=True,
        )
        if errors or not grid.ok:
            raise ValueError(
                f"Invalid main panel {model}: {errors + list(grid.errors)}"
            )
        main_rows.extend(rows)
        cell_prompts = defaultdict(set)
        for row in rows:
            prompt_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()
            cell_prompts[row["quantity_id"]].add(prompt_hash)
            variants[row["quantity_id"]][prompt_hash].add(row["model_name"])
        if any(len(hashes) != 1 for hashes in cell_prompts.values()):
            raise ValueError(
                f"Mixed prompt texts within a retained main-panel cell: {model}"
            )
    main = run_census(main_rows)
    main.update(
        {
            "model_count": len({r["model_name"] for r in main_rows}),
            "model_ids": sorted({r["model_name"] for r in main_rows}),
            "organization_count": len(
                {get_panel_model(r["model_name"]).organization for r in main_rows}
            ),
            "quantity_count": len(quantities),
            "quantity_ids": quantities,
            "repetitions_per_cell": 15,
            "prompt_versions": dict(
                sorted(Counter(r["prompt_version"] for r in main_rows).items())
            ),
            "prompt_variants": {
                q: [
                    {
                        "prompt_sha256": h,
                        "model_ids": sorted(models),
                        "records": 15 * len(models),
                    }
                    for h, models in sorted(hashes.items())
                ]
                for q, hashes in sorted(variants.items())
            },
        }
    )
    excluded = {results / n for n in GENERATED_RESULTS} | {root / MANIFEST}
    files = []
    scopes = defaultdict(list)
    for path in sorted(results.rglob("*")):
        if (
            not path.is_file()
            or path in excluded
            or path.suffix not in {".csv", ".json", ".jsonl"}
        ):
            continue
        # Runtime staging is never a retained artifact, even if present locally.
        if any(part.startswith("_perquantity_") for part in path.parts):
            continue
        entry = {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
        if path.name in {"runs.jsonl", "failed-runs-archive.jsonl"}:
            rows, errors = load_jsonl_rows([path])
            if errors:
                raise ValueError("; ".join(errors))
            if path.name == "failed-runs-archive.jsonl":
                scope = "archived_failures"
            elif path.parent.name in expected_dirs:
                scope = "retained_main"
            elif path.parent.name in {
                f"{m}-{family}-batch15"
                for m in model_ids
                for family in ("ies-clarify", "armington-clarify")
            }:
                scope = "retained_followups"
            else:
                scope = "retained_auxiliary"
            entry.update({"scope": scope, **run_census(rows)})
            scopes[scope].extend(rows)
        files.append(entry)
    historical_sources = json.loads((root / "archive/sources.json").read_text())
    for entry in historical_sources:
        path = root / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Historical archive hash mismatch: {entry['path']}")
        data = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != entry["git_blob"]:
            raise ValueError(f"Historical Git blob mismatch: {entry['path']}")
    actual_archives = {
        p.relative_to(root).as_posix()
        for p in (root / "archive").rglob("*")
        if p.is_file() and p.name not in {"sources.json", "README.md"}
    }
    if actual_archives != {s["path"] for s in historical_sources}:
        raise ValueError("Historical archive inventory differs from sources.json")
    verify_calibration_provenance(root)
    return {
        "schema_version": 1,
        "scope": "Retained main grid, separate follow-ups/pilots and failure attempts; historical comparison snapshots are not additional main-panel observations.",
        "main_panel": main,
        "retained_run_scopes": {
            scope: run_census(rows) for scope, rows in sorted(scopes.items())
        },
        "evidence_files": files,
        "historical_sources": historical_sources,
        "calibration_provenance": "results/calibration-provenance.json",
    }


def verify_manifest(root: Path = ROOT, *, path: Path | None = None) -> dict:
    """Reject changed evidence, census, inventory or prompt-family metadata."""
    expected = json.loads((path or root / MANIFEST).read_text())
    actual = build_manifest(root)
    if actual != expected:
        old = {f["path"]: f["sha256"] for f in expected["evidence_files"]}
        new = {f["path"]: f["sha256"] for f in actual["evidence_files"]}
        changed = sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))
        raise ValueError(
            f"Archive manifest differs from retained evidence/census: {changed}. "
            "Do not refresh hashes as part of ordinary reproduction."
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="explicitly replace the reviewed manifest"
    )
    args = parser.parse_args()
    if args.write:
        (ROOT / MANIFEST).write_text(
            json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n"
        )
    else:
        verify_manifest()
    print("Retained archive manifest verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
