"""Build paper-friendly comparison tables from experiment result directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_FIELDS = (
    "source_label",
    "model_name",
    "quantity_id",
    "n_successful_runs",
    "pooled_point_estimate",
    "pooled_90_interval",
    "reml_predictive_90_interval",
    "bayes_predictive_90_interval",
    "usage_total_tokens_per_successful_run",
    "usage_estimated_total_cost_usd_per_successful_run",
    "source_dir",
)


def read_summary_rows(result_dir: str | Path) -> list[dict[str, object]]:
    """Read one experiment summary.csv into typed rows."""
    result_dir = Path(result_dir)
    summary_path = result_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv in {result_dir}")

    with summary_path.open(newline="") as handle:
        return [
            {
                **{key: _parse_cell(value) for key, value in row.items()},
                "source_dir": str(result_dir),
                "source_label": result_dir.name,
            }
            for row in csv.DictReader(handle)
        ]


def build_comparison_rows(
    result_dirs: Sequence[str | Path],
    *,
    quantity_id: str | None = None,
) -> list[dict[str, object]]:
    """Stack summaries from multiple result directories into a comparison table."""
    comparison_rows: list[dict[str, object]] = []
    for result_dir in result_dirs:
        rows = read_summary_rows(result_dir)
        if quantity_id is not None:
            rows = [row for row in rows if row.get("quantity_id") == quantity_id]
        if not rows:
            raise ValueError(f"No matching summary rows found in {result_dir}")

        for row in rows:
            comparison_rows.append(
                {
                    "source_label": row["source_label"],
                    "model_name": row["model_name"],
                    "quantity_id": row["quantity_id"],
                    "n_successful_runs": row["n_successful_runs"],
                    "pooled_point_estimate": row["pooled_point_estimate"],
                    "pooled_90_interval": _format_interval(
                        row.get("pooled_lower_bound"),
                        row.get("pooled_upper_bound"),
                    ),
                    "reml_predictive_90_interval": _format_interval(
                        row.get("reml_predictive_lower"),
                        row.get("reml_predictive_upper"),
                    ),
                    "bayes_predictive_90_interval": _format_interval(
                        row.get("bayes_predictive_lower"),
                        row.get("bayes_predictive_upper"),
                    ),
                    "usage_total_tokens_per_successful_run": row.get(
                        "usage_total_tokens_per_successful_run"
                    ),
                    "usage_estimated_total_cost_usd_per_successful_run": row.get(
                        "usage_estimated_total_cost_usd_per_successful_run"
                    ),
                    "source_dir": row["source_dir"],
                }
            )
    return comparison_rows


def write_comparison_csv(
    output_path: str | Path,
    rows: Sequence[dict[str, object]],
    *,
    fieldnames: Sequence[str] = DEFAULT_FIELDS,
) -> None:
    """Write a comparison table to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for comparison-table generation."""
    parser = argparse.ArgumentParser(description="Build a comparison CSV from experiment outputs.")
    parser.add_argument("--result-dir", action="append", required=True)
    parser.add_argument("--quantity")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for comparison-table generation."""
    args = parse_args(argv)
    rows = build_comparison_rows(args.result_dir, quantity_id=args.quantity)
    write_comparison_csv(args.output, rows)
    print(f"Wrote comparison table to {args.output}")
    return 0


def _parse_cell(value: str) -> object:
    if value == "":
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _format_interval(lower: object, upper: object) -> str | None:
    if lower is None or upper is None:
        return None
    return f"[{_format_number(lower)}, {_format_number(upper)}]"


def _format_number(value: object) -> str:
    return f"{float(value):.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
