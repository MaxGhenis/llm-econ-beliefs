"""Check cached scientific summaries and CSV run projections against raw JSONL.

No cost estimates are regenerated: historical request accounting conventions are
retained, hashed inputs. Scientific numeric fields are recomputed from parsed
run fields with the current retained aggregation implementation.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_econ_beliefs.experiment import summarize_run_results  # noqa: E402
from llm_econ_beliefs.model_registry import PANEL_MODEL_IDS  # noqa: E402
from llm_econ_beliefs.models import RunResult  # noqa: E402

SCIENTIFIC_FIELDS = (
    "n_successful_runs",
    "pooled_point_estimate",
    "pooled_lower_bound",
    "pooled_upper_bound",
    "within_run_sd",
    "between_run_sd",
    "total_sd",
    "reml_latent_location",
    "reml_latent_lower",
    "reml_latent_upper",
    "reml_predictive_lower",
    "reml_predictive_upper",
    "reml_tau",
    "reml_typical_within_sd",
    "bayes_latent_location",
    "bayes_latent_lower",
    "bayes_latent_upper",
    "bayes_predictive_lower",
    "bayes_predictive_upper",
    "bayes_tau_mean",
    "bayes_interval_scale_mean",
    "bayes_typical_within_sd",
)


def verify_directory(path: Path) -> int:
    """Compare every scientific summary field, preserving historical usage fields."""
    raw = [
        json.loads(line)
        for line in (path / "runs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    with (path / "runs.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(raw) != len(csv_rows):
        raise ValueError(f"{path.name}: runs.csv row count differs from JSONL")
    # Validate the fields consumed by the manuscript's CSV-based distribution code.
    for index, (record, row) in enumerate(zip(raw, csv_rows), 1):
        for key in (
            "model_name",
            "quantity_id",
            "run_index",
            "prompt_version",
            "parsed_ok",
            "point_estimate",
            "lower_bound",
            "upper_bound",
            "confidence_level",
        ):
            expected = "" if record.get(key) is None else str(record[key])
            if row[key] != expected:
                raise ValueError(f"{path.name}: runs.csv row {index} differs in {key}")
        if json.loads(row["quantiles"] or "{}") != record.get("quantiles", {}):
            raise ValueError(f"{path.name}: runs.csv row {index} differs in quantiles")
    groups = {(r["model_name"], r["quantity_id"], r["prompt_version"]) for r in raw}
    if len(groups) != len({(m, q) for m, q, _ in groups}):
        raise ValueError(f"{path.name}: would pool different prompt versions")
    rebuilt = summarize_run_results([RunResult(**record) for record in raw])
    with (path / "summary.csv").open(newline="") as handle:
        cached = list(csv.DictReader(handle))
    keyed = {(row["model_name"], row["quantity_id"]): row for row in cached}
    if len(keyed) != len(cached) or set(keyed) != {
        (r["model_name"], r["quantity_id"]) for r in rebuilt
    }:
        raise ValueError(f"{path.name}: summary inventory differs from raw runs")
    for row in rebuilt:
        old = keyed[row["model_name"], row["quantity_id"]]
        for key in SCIENTIFIC_FIELDS:
            value = row[key]
            if value is None:
                matches = old[key] == ""
            else:
                matches = old[key] != "" and math.isclose(
                    float(old[key]), value, rel_tol=1e-12, abs_tol=1e-12
                )
            if not matches:
                raise ValueError(
                    f"{path.name}/{row['quantity_id']}: {key} cached={old[key]!r}, rebuilt={value!r}"
                )
    return len(rebuilt)


def main() -> int:
    cells = 0
    paths = [
        ROOT / "results" / f"{model}-{batch}-batch15"
        for model in PANEL_MODEL_IDS
        for batch in ("elasticities", "armington-clarify", "ies-clarify")
    ]
    # The mechanism appendix uses a separate, single-model cached probe.
    paths += sorted((ROOT / "results").glob("*-mechanism-*-batch15"))
    for path in paths:
        cells += verify_directory(path)
        print(f"Verified {path.name} ({cells} cells)", flush=True)
    print(
        f"PASS: {cells} cached scientific summary cells and their run CSV projections match raw JSONL"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
