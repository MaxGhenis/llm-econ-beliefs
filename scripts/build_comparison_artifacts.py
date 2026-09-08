"""Rebuild the cross-model comparison artifacts under results/.

Regenerates:

- ``results/elasticity-all-model-comparison.csv`` — one row per
  (model, quantity) stacked from every ``*-elasticities-batch15``
  summary via ``llm_econ_beliefs.compare``.
- ``results/elasticity-model-rollup.csv`` — one row per model with run
  counts, tracked cost, and mean pooled interval geometry.

Usage:
    .venv/bin/python scripts/build_comparison_artifacts.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llm_econ_beliefs import list_quantities  # noqa: E402
from llm_econ_beliefs.compare import (  # noqa: E402
    build_comparison_rows,
    read_summary_rows,
    write_comparison_csv,
)
from llm_econ_beliefs.model_registry import PANEL_MODEL_IDS  # noqa: E402

try:
    from panel_grid_gate import gate_panel_models
except ModuleNotFoundError:  # Imported as ``scripts.build_comparison_artifacts``.
    from scripts.panel_grid_gate import gate_panel_models

RESULTS_DIR = REPO_ROOT / "results"
COMPARISON_PATH = RESULTS_DIR / "elasticity-all-model-comparison.csv"
ROLLUP_PATH = RESULTS_DIR / "elasticity-model-rollup.csv"
RUNS_PER_QUANTITY = 15


def _summary_integrity_errors(
    rows: Sequence[Mapping[str, object]], model_id: str
) -> tuple[str, ...]:
    """Validate the summary consumed after the raw-run grid has passed."""
    expected_quantities = {quantity.id for quantity in list_quantities()}
    errors: list[str] = []
    if not rows:
        return ("summary.csv is empty",)

    raw_model_names = [row.get("model_name") for row in rows]
    model_names = {
        value for value in raw_model_names if isinstance(value, str) and value
    }
    if model_names != {model_id} or len(model_names) != len(set(raw_model_names)):
        errors.append(
            f"model_name values: expected {[model_id]!r}, found {sorted(model_names)!r}"
        )
    quantity_ids = [
        value
        for row in rows
        if isinstance((value := row.get("quantity_id")), str) and value
    ]
    if len(quantity_ids) != len(rows):
        errors.append("every summary row must have a nonblank string quantity_id")
    counts = Counter(quantity_ids)
    duplicates = sorted(quantity for quantity, count in counts.items() if count != 1)
    if duplicates:
        errors.append("duplicate quantity_id values: " + ", ".join(duplicates))
    missing = sorted(expected_quantities - set(quantity_ids))
    unexpected = sorted(set(quantity_ids) - expected_quantities)
    if missing:
        errors.append("missing quantity_id values: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected quantity_id values: " + ", ".join(unexpected))
    for row_number, row in enumerate(rows, 2):
        try:
            successful_runs = int(row["n_successful_runs"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"row {row_number} has invalid n_successful_runs")
            continue
        if successful_runs != RUNS_PER_QUANTITY:
            errors.append(
                f"row {row_number} has n_successful_runs={successful_runs}, "
                f"expected {RUNS_PER_QUANTITY}"
            )
        pooled_values: dict[str, float] = {}
        for field in (
            "pooled_point_estimate",
            "pooled_lower_bound",
            "pooled_upper_bound",
        ):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                errors.append(f"row {row_number} has invalid {field}")
                continue
            pooled_values[field] = float(value)
        if (
            "pooled_lower_bound" in pooled_values
            and "pooled_upper_bound" in pooled_values
            and pooled_values["pooled_lower_bound"]
            > pooled_values["pooled_upper_bound"]
        ):
            errors.append(f"row {row_number} has reversed pooled bounds")
    return tuple(errors)


def elasticities_dirs(
    model_ids: list[str] | tuple[str, ...],
    *,
    allow_partial: bool = False,
) -> list[Path]:
    """Return summary-backed main-panel directories in registry order."""
    result_dirs: list[Path] = []
    errors_by_model: dict[str, list[str]] = {}
    for model_id in model_ids:
        path = RESULTS_DIR / f"{model_id}-elasticities-batch15"
        summary_path = path / "summary.csv"
        if not summary_path.exists():
            errors_by_model[model_id] = [f"missing file: {summary_path}"]
            continue
        try:
            rows = read_summary_rows(path)
        except (KeyError, TypeError, ValueError) as exc:
            errors_by_model[model_id] = [f"invalid summary.csv: {exc}"]
            continue
        errors = _summary_integrity_errors(rows, model_id)
        if errors:
            errors_by_model[model_id] = list(errors)
            if not allow_partial or not rows:
                continue
        result_dirs.append(path)

    included_ids = {
        path.name.removesuffix("-elasticities-batch15") for path in result_dirs
    }
    excluded = [model_id for model_id in model_ids if model_id not in included_ids]
    print(
        "COMPARISON_SUMMARY_EXCLUSIONS="
        + json.dumps(
            {
                "event": "comparison_summary_exclusions",
                "allow_partial": allow_partial,
                "declared_models": list(model_ids),
                "invalid_models": list(errors_by_model),
                "excluded_models": excluded,
                "errors_by_model": errors_by_model,
            },
            sort_keys=True,
        )
    )
    return result_dirs


def build_rollup_rows(result_dirs: list[Path]) -> list[dict[str, object]]:
    expected_per_model = len(list_quantities()) * RUNS_PER_QUANTITY
    rollup_rows: list[dict[str, object]] = []
    for result_dir in result_dirs:
        rows = read_summary_rows(result_dir)
        model_name = rows[0]["model_name"]
        successful_runs = sum(int(row["n_successful_runs"]) for row in rows)
        cost_cells = [row.get("usage_estimated_total_cost_usd_total") for row in rows]
        total_cost = (
            sum(float(value) for value in cost_cells)
            if all(value not in (None, "") for value in cost_cells)
            else None
        )
        widths = [
            float(row["pooled_upper_bound"]) - float(row["pooled_lower_bound"])
            for row in rows
            if row.get("pooled_upper_bound") not in (None, "")
            and row.get("pooled_lower_bound") not in (None, "")
        ]
        points = [
            float(row["pooled_point_estimate"])
            for row in rows
            if row.get("pooled_point_estimate") not in (None, "")
        ]
        rollup_rows.append(
            {
                "model_name": model_name,
                "quantities": len(rows),
                "successful_runs": successful_runs,
                "expected_runs": expected_per_model,
                "success_rate": successful_runs / expected_per_model,
                "total_cost_usd": total_cost,
                "cost_per_successful_run_usd": (
                    total_cost / successful_runs
                    if total_cost is not None and successful_runs
                    else None
                ),
                "mean_pooled_interval_width": (
                    sum(widths) / len(widths) if widths else None
                ),
                "mean_pooled_point_estimate": (
                    sum(points) / len(points) if points else None
                ),
            }
        )
    return rollup_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="include incomplete registered grids for development only",
    )
    args = parser.parse_args()

    gate = gate_panel_models(
        RESULTS_DIR,
        PANEL_MODEL_IDS,
        allow_partial=args.allow_partial,
    )
    result_dirs = elasticities_dirs(
        gate.included_models,
        allow_partial=args.allow_partial,
    )
    if not result_dirs:
        print("No complete registered *-elasticities-batch15 result directories found.")
        return 1

    comparison_rows = build_comparison_rows(result_dirs)
    for row in comparison_rows:
        row["source_dir"] = Path(row["source_dir"]).relative_to(REPO_ROOT).as_posix()
    write_comparison_csv(COMPARISON_PATH, comparison_rows)
    print(f"Wrote {len(comparison_rows)} rows to {COMPARISON_PATH}")

    rollup_rows = build_rollup_rows(result_dirs)
    with ROLLUP_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rollup_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rollup_rows)
    print(f"Wrote {len(rollup_rows)} models to {ROLLUP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
