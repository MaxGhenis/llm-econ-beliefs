"""Failure gates for the retained scientific archive; no providers or microdata."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts.archive_manifest import build_manifest, verify_manifest
from scripts.cached_calibration import load_frozen_calibration
from scripts.reproduce_cached import compare_outputs, require_clean_tree

ROOT = Path(__file__).resolve().parents[1]


def test_missing_calibration_fails_instead_of_substituting(tmp_path):
    with pytest.raises(FileNotFoundError, match="frozen calibration"):
        load_frozen_calibration(tmp_path / "absent.json")


@pytest.mark.parametrize(
    "payload",
    [
        {"a": 1.5, "gbar": 0.6, "threshold": None, "tail_mean": None},
        {"a": 1.6, "gbar": 0.6, "threshold": 100, "tail_mean": 200},
    ],
)
def test_fallback_or_inconsistent_calibration_is_rejected(tmp_path, payload):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_frozen_calibration(path)


def test_frozen_calibration_replays_exact_numbers():
    values = load_frozen_calibration()
    assert values["a"] == 1.6208594549947215
    assert values["gbar"] == 0.6184457743072629


def test_retained_inventory_and_prompt_wordings():
    manifest = build_manifest(ROOT)
    assert manifest["main_panel"]["successful_records"] == 12090
    assert manifest["main_panel"]["model_count"] == 31
    assert manifest["main_panel"]["organization_count"] == 10
    variants = manifest["main_panel"]["prompt_variants"]
    assert sum(len(v) == 1 for v in variants.values()) == 23
    assert all(
        sorted(len(v["model_ids"]) for v in variants[q]) == [7, 24]
        for q in variants
        if len(variants[q]) == 2
    )


def test_wrong_panel_inventory_is_rejected():
    from llm_econ_beliefs.model_registry import PANEL_MODEL_IDS

    with pytest.raises(ValueError, match="main-panel inventory"):
        build_manifest(ROOT, model_ids=PANEL_MODEL_IDS[:-1])


def test_inconsistent_prompt_family_counts_are_rejected(tmp_path):
    manifest = build_manifest(ROOT)
    manifest["main_panel"]["prompt_versions"]["v4"] -= 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest differs"):
        verify_manifest(ROOT, path=path)


def test_clean_tree_gate_includes_untracked_and_staged_changes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    require_clean_tree(tmp_path)
    (tmp_path / "evidence.txt").write_text("changed")
    with pytest.raises(ValueError, match="clean checkout"):
        require_clean_tree(tmp_path)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    with pytest.raises(ValueError, match="clean checkout"):
        require_clean_tree(tmp_path)


def test_output_comparison_detects_missing_and_changed_files(tmp_path):
    expected = {"table.csv": b"verified\n"}
    with pytest.raises(ValueError, match="missing"):
        compare_outputs(tmp_path, expected)
    (tmp_path / "table.csv").write_bytes(b"fallback\n")
    with pytest.raises(ValueError, match="changed"):
        compare_outputs(tmp_path, expected)
    (tmp_path / "table.csv").write_bytes(expected["table.csv"])
    compare_outputs(tmp_path, expected)


def test_every_manuscript_table_is_in_reproduction_inventory():
    import re

    from scripts.reproduce_cached import OUTPUTS

    includes = re.findall(
        r"include (tables/[^ ]+)", (ROOT / "paper/paper.qmd").read_text()
    )
    assert includes
    assert all(f"paper/{path}" in OUTPUTS for path in includes)


def test_absolute_or_escaping_result_paths_are_rejected():
    from paper.build_tables import resolve_result_dir

    for path in ("/another/checkout/results/model", "results/../outside", "elsewhere"):
        with pytest.raises(ValueError):
            resolve_result_dir(path)
    assert (
        resolve_result_dir("results/gpt-5.4-elasticities-batch15")
        == ROOT / "results/gpt-5.4-elasticities-batch15"
    )


@pytest.mark.parametrize(
    "source",
    [
        "import socket; socket.create_connection(('example.com', 443))",
        "import subprocess; subprocess.run(['echo', 'should not execute'])",
    ],
)
def test_offline_steps_cannot_use_network_or_child_processes(tmp_path, source):
    import sys

    from scripts.reproduce_cached import OFFLINE_RUNNER

    script = tmp_path / "step.py"
    script.write_text(source)
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", OFFLINE_RUNNER, str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Cached reproduction forbids" in result.stderr


def test_fresh_calibration_stops_before_writing():
    import sys

    from scripts.archive_manifest import sha256

    path = ROOT / "results/top-rate-calibration.json"
    before = sha256(path)
    result = subprocess.run(
        [sys.executable, str(ROOT / "paper/build_tables.py"), "--fresh-calibration"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Fresh calibration is not implemented" in result.stderr
    assert sha256(path) == before


def test_wrong_raw_prompt_family_is_rejected(monkeypatch):
    from scripts import archive_manifest

    original = archive_manifest.load_jsonl_rows

    def wrong_family(paths):
        rows, errors = original(paths)
        if paths[0].parent.name == "gpt-5.4-elasticities-batch15":
            rows[0] = {**rows[0], "prompt_version": "ies-clarify"}
        return rows, errors

    monkeypatch.setattr(archive_manifest, "load_jsonl_rows", wrong_family)
    with pytest.raises(ValueError, match="Invalid main panel"):
        build_manifest(ROOT)


@pytest.mark.parametrize("target", ["summary.csv", "runs.csv"])
def test_raw_summary_gate_rejects_changed_scientific_values(tmp_path, target):
    import csv

    from scripts.verify_cached_summaries import verify_directory

    source = ROOT / "results/gpt-5.4-elasticities-batch15"
    quantity = "household.annual_discount_factor"
    raw = [
        json.loads(line) for line in (source / "runs.jsonl").read_text().splitlines()
    ]
    raw = [row for row in raw if row["quantity_id"] == quantity]
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    for name in ("summary.csv", "runs.csv"):
        with (source / name).open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [row for row in reader if row["quantity_id"] == quantity]
        if name == target:
            field = (
                "pooled_point_estimate" if target == "summary.csv" else "point_estimate"
            )
            rows[0][field] = "0.12345"
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ValueError, match="differs|cached="):
        verify_directory(tmp_path)
