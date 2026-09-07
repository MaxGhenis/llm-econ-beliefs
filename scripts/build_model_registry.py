"""Write the model and quantity registry CSVs from the canonical registries."""

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llm_econ_beliefs.model_registry import write_model_registry_csv  # noqa: E402
from llm_econ_beliefs.registry import list_quantities  # noqa: E402

QUANTITY_CSV_FIELDS = (
    "quantity_id",
    "name",
    "domain",
    "description",
    "population",
    "unit",
    "preferred_interpretation",
    "lower_support",
    "upper_support",
    "benchmark_summary",
    "benchmark_source",
)


def write_quantity_registry_csv(target: Path) -> Path:
    """Emit the elicited quantity definitions for downstream display."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUANTITY_CSV_FIELDS)
        writer.writeheader()
        for quantity in list_quantities():
            writer.writerow(
                {
                    "quantity_id": quantity.id,
                    "name": quantity.name,
                    "domain": quantity.domain,
                    "description": quantity.description,
                    "population": quantity.population,
                    "unit": quantity.unit,
                    "preferred_interpretation": quantity.preferred_interpretation,
                    "lower_support": quantity.lower_support,
                    "upper_support": quantity.upper_support,
                    "benchmark_summary": quantity.benchmark_summary,
                    "benchmark_source": quantity.benchmark_source,
                }
            )
    os.replace(temporary, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stage-dashboard", action="store_true")
    args = parser.parse_args()
    primary = write_model_registry_csv()
    print(f"Wrote {primary}")
    quantity_primary = write_quantity_registry_csv(
        REPO_ROOT / "results" / "quantity-registry.csv"
    )
    print(f"Wrote {quantity_primary}")

    if args.no_stage_dashboard:
        return 0

    # dashboard/results is the build-root fallback used when Vercel cannot read
    # the repository-level results directory. It is a generated staging copy,
    # not a second metadata source.
    dashboard_copy = REPO_ROOT / "dashboard" / "results" / "model-registry.csv"
    write_model_registry_csv(dashboard_copy)
    print(f"Staged {dashboard_copy}")
    write_quantity_registry_csv(
        REPO_ROOT / "dashboard" / "results" / "quantity-registry.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
