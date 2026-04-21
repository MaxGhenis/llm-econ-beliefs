"""Tests for the per-quantity rerun merge in scripts/run_v4_per_quantity.py."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from llm_econ_beliefs.models import RequestLog, RunResult


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_merge_module():
    """Import scripts/run_v4_per_quantity.py as a module."""
    script_path = REPO_ROOT / "scripts" / "run_v4_per_quantity.py"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "run_v4_per_quantity", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl_records(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")


def _make_run(
    *,
    model_name: str,
    quantity_id: str,
    run_index: int,
    point_estimate: float,
    p05: float,
    p25: float,
    p50: float,
    p75: float,
    p95: float,
) -> RunResult:
    return RunResult(
        provider="litellm",
        model_name=model_name,
        quantity_id=quantity_id,
        run_index=run_index,
        prompt_version="v4",
        tool_regime="none",
        prompt="prompt",
        raw_response=json.dumps(
            {
                "interpretation": "test",
                "point_estimate": point_estimate,
                "quantiles": {
                    "p05": p05,
                    "p25": p25,
                    "p50": p50,
                    "p75": p75,
                    "p95": p95,
                },
            }
        ),
        parsed_ok=True,
        point_estimate=point_estimate,
        interpretation="test",
        quantiles={"p05": p05, "p25": p25, "p50": p50, "p75": p75, "p95": p95},
        citations=[],
        reasoning_summary="",
    )


def _make_request(
    *,
    model_name: str,
    quantity_id: str,
    request_index: int,
    total_tokens: int,
    cost_usd: float,
) -> RequestLog:
    return RequestLog(
        provider="litellm",
        model_name=model_name,
        quantity_id=quantity_id,
        request_index=request_index,
        prompt_version="v4",
        tool_regime="none",
        batch_size=1,
        prompt_tokens=total_tokens // 2,
        completion_tokens=total_tokens // 2,
        total_tokens=total_tokens,
        estimated_total_cost_usd=cost_usd,
    )


def test_merge_per_quantity_combines_staging_dirs(tmp_path: Path):
    merge_module = _load_merge_module()
    merge_per_quantity = merge_module.merge_per_quantity

    model_name = "test-model"
    prompt_version = "v4"

    staging_root = tmp_path / "staging"
    staging_root.mkdir()

    qa_id = "household.annual_discount_factor"
    qb_id = "labor_supply.frisch_elasticity.prime_age"

    qa_dir = staging_root / qa_id.replace(".", "_")
    qb_dir = staging_root / qb_id.replace(".", "_")

    qa_runs = [
        _make_run(
            model_name=model_name,
            quantity_id=qa_id,
            run_index=i + 1,
            point_estimate=0.96 + 0.01 * i,
            p05=0.94,
            p25=0.95,
            p50=0.96,
            p75=0.97,
            p95=0.99,
        )
        for i in range(3)
    ]
    qb_runs = [
        _make_run(
            model_name=model_name,
            quantity_id=qb_id,
            run_index=i + 1,
            point_estimate=0.5 + 0.1 * i,
            p05=0.2,
            p25=0.35,
            p50=0.5,
            p75=0.8,
            p95=1.5,
        )
        for i in range(2)
    ]

    qa_requests = [
        _make_request(
            model_name=model_name,
            quantity_id=qa_id,
            request_index=i + 1,
            total_tokens=1000,
            cost_usd=0.01,
        )
        for i in range(3)
    ]
    qb_requests = [
        _make_request(
            model_name=model_name,
            quantity_id=qb_id,
            request_index=i + 1,
            total_tokens=2000,
            cost_usd=0.02,
        )
        for i in range(2)
    ]

    _write_jsonl_records(qa_dir / "runs.jsonl", qa_runs)
    _write_jsonl_records(qa_dir / "requests.jsonl", qa_requests)
    _write_jsonl_records(qb_dir / "runs.jsonl", qb_runs)
    _write_jsonl_records(qb_dir / "requests.jsonl", qb_requests)

    target_dir = tmp_path / "merged"

    merge_per_quantity(staging_root, target_dir, model_name, prompt_version)

    merged_runs_path = target_dir / "runs.jsonl"
    merged_requests_path = target_dir / "requests.jsonl"
    summary_path = target_dir / "summary.csv"
    grid_path = target_dir / "prompt_grid.csv"

    assert merged_runs_path.exists()
    assert merged_requests_path.exists()
    assert summary_path.exists()
    assert grid_path.exists()

    merged_run_records = [
        json.loads(line)
        for line in merged_runs_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(merged_run_records) == len(qa_runs) + len(qb_runs)
    seen_pairs = {(r["quantity_id"], r["run_index"]) for r in merged_run_records}
    assert (qa_id, 1) in seen_pairs
    assert (qb_id, 1) in seen_pairs

    merged_request_records = [
        json.loads(line)
        for line in merged_requests_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(merged_request_records) == len(qa_requests) + len(qb_requests)

    merged_indices = [r["request_index"] for r in merged_request_records]
    assert merged_indices == list(range(1, len(merged_indices) + 1))
    assert len(set(merged_indices)) == len(merged_indices)

    with summary_path.open() as handle:
        reader = csv.DictReader(handle)
        summary_rows = list(reader)

    assert len(summary_rows) == 2
    by_quantity = {row["quantity_id"]: row for row in summary_rows}
    assert set(by_quantity) == {qa_id, qb_id}

    for qid, row in by_quantity.items():
        tokens_cell = row["usage_total_tokens_total"]
        cost_cell = row["usage_estimated_total_cost_usd_total"]
        assert tokens_cell not in ("", "None")
        assert cost_cell not in ("", "None")
        assert float(tokens_cell) > 0
        assert float(cost_cell) > 0


def test_merge_per_quantity_is_noop_when_no_records(tmp_path: Path):
    merge_module = _load_merge_module()
    merge_per_quantity = merge_module.merge_per_quantity

    staging_root = tmp_path / "staging_empty"
    staging_root.mkdir()
    target_dir = tmp_path / "merged_empty"

    merge_per_quantity(staging_root, target_dir, "test-model", "v4")

    assert not (target_dir / "runs.jsonl").exists()
