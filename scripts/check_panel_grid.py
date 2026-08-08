"""Validate canonical panel result files against their exact expected grids.

The checker is intentionally read-only.  It treats missing, unexpected, and
duplicate ``(model, prompt_version, quantity, run_index)`` keys as failures.
Use ``--require-parsed`` when completeness also requires every expected slot
to contain a successfully parsed response.

Usage:
    .venv/bin/python scripts/check_panel_grid.py \
        --models deepseek-v4-pro,qwen-3.7-max --require-parsed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llm_econ_beliefs import list_quantities
from llm_econ_beliefs.model_registry import MODEL_REGISTRY, PANEL_MODEL_IDS


RUNS_PER_QUANTITY = 15
# The no-argument invocation verifies exactly the completeness the paper
# claims: the main elasticities batch for every registry model, plus the
# clarify batches for every wave after April 2026. Three April models
# carry documented clarify-batch gaps (claude-opus-4.7's armington runs
# are tagged v4; two Gemini previews hold unparsed clarify slots), so
# April clarify cells are only checked when named explicitly via --models.
POST_APRIL_MODEL_IDS = tuple(
    model.model_id for model in MODEL_REGISTRY if model.wave != "april_2026"
)
BATCH_PROMPT_VERSIONS = {
    "elasticities-batch15": "v4",
    "armington-clarify-batch15": "armington-clarify",
    "ies-clarify-batch15": "ies-clarify",
}

GridKey = tuple[str, str, str, int]


@dataclass(frozen=True)
class GridCheckResult:
    """Result of checking one set of run records."""

    expected_count: int
    observed_count: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def quantity_ids_for_batch(batch: str) -> list[str]:
    """Return the quantities expected in one canonical batch."""
    if batch == "elasticities-batch15":
        return [quantity.id for quantity in list_quantities()]
    if batch == "armington-clarify-batch15":
        return ["trade.armington_elasticity.import_domestic"]
    if batch == "ies-clarify-batch15":
        return ["household.intertemporal_elasticity_of_substitution"]
    raise ValueError(f"Unknown batch: {batch}")


def expected_grid_keys(
    *,
    model_name: str,
    prompt_version: str,
    quantity_ids: Sequence[str],
    n_runs: int = RUNS_PER_QUANTITY,
) -> set[GridKey]:
    """Build the exact expected key set for one model/batch cell."""
    if not model_name:
        raise ValueError("model_name must not be empty")
    if not prompt_version:
        raise ValueError("prompt_version must not be empty")
    if not quantity_ids:
        raise ValueError("quantity_ids must not be empty")
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    return {
        (model_name, prompt_version, quantity_id, run_index)
        for quantity_id in quantity_ids
        for run_index in range(1, n_runs + 1)
    }


def validate_grid_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    model_name: str,
    prompt_version: str,
    quantity_ids: Sequence[str],
    n_runs: int = RUNS_PER_QUANTITY,
    require_parsed: bool = False,
) -> GridCheckResult:
    """Check already-decoded run rows against an exact expected grid."""
    expected = expected_grid_keys(
        model_name=model_name,
        prompt_version=prompt_version,
        quantity_ids=quantity_ids,
        n_runs=n_runs,
    )
    counts: Counter[GridKey] = Counter()
    parsed_by_key: dict[GridKey, list[object]] = {}
    malformed: list[str] = []
    observed_count = 0

    for row_number, row in enumerate(rows, 1):
        observed_count += 1
        try:
            key = (
                row["model_name"],
                row["prompt_version"],
                row["quantity_id"],
                row["run_index"],
            )
        except KeyError as exc:
            malformed.append(f"row {row_number} missing field {exc.args[0]!r}")
            continue
        if (
            not isinstance(key[0], str)
            or not isinstance(key[1], str)
            or not isinstance(key[2], str)
            or not isinstance(key[3], int)
            or isinstance(key[3], bool)
        ):
            malformed.append(f"row {row_number} has invalid grid-key types: {key!r}")
            continue
        typed_key: GridKey = key
        counts[typed_key] += 1
        parsed_by_key.setdefault(typed_key, []).append(row.get("parsed_ok"))

    actual = set(counts)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicates = sorted((key, count) for key, count in counts.items() if count != 1)
    unparsed = sorted(
        key
        for key in expected & actual
        if require_parsed
        and not (len(parsed_by_key.get(key, ())) == 1 and parsed_by_key[key][0] is True)
    )

    errors = list(malformed)
    if missing:
        errors.append(_format_key_problem("missing", missing))
    if unexpected:
        errors.append(_format_key_problem("unexpected", unexpected))
    if duplicates:
        errors.append(_format_key_problem("duplicate", duplicates))
    if unparsed:
        errors.append(_format_key_problem("unparsed", unparsed))
    return GridCheckResult(
        expected_count=len(expected),
        observed_count=observed_count,
        errors=tuple(errors),
    )


def _format_key_problem(label: str, values: Sequence[object]) -> str:
    preview = ", ".join(repr(value) for value in values[:5])
    suffix = "" if len(values) <= 5 else f", ... ({len(values)} total)"
    return f"{label}: {preview}{suffix}"


def load_jsonl_rows(paths: Iterable[Path]) -> tuple[list[dict[str, object]], list[str]]:
    """Read JSONL rows without modifying any input file."""
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            errors.append(f"missing file: {path}")
            continue
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: invalid JSON: {exc.msg}")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"{path}:{line_number}: expected a JSON object")
                    continue
                rows.append(row)
    return rows, errors


def check_run_files(
    paths: Iterable[Path],
    *,
    model_name: str,
    prompt_version: str,
    quantity_ids: Sequence[str],
    n_runs: int = RUNS_PER_QUANTITY,
    require_parsed: bool = False,
) -> GridCheckResult:
    """Read one or more run files and check their combined exact grid."""
    rows, read_errors = load_jsonl_rows(paths)
    result = validate_grid_rows(
        rows,
        model_name=model_name,
        prompt_version=prompt_version,
        quantity_ids=quantity_ids,
        n_runs=n_runs,
        require_parsed=require_parsed,
    )
    return GridCheckResult(
        expected_count=result.expected_count,
        observed_count=result.observed_count,
        errors=tuple(read_errors) + result.errors,
    )


def check_batch_directory(
    results_root: Path,
    *,
    model_name: str,
    batch: str,
    require_parsed: bool = False,
) -> GridCheckResult:
    """Check one canonical ``results/<model>-<batch>`` directory."""
    return check_run_files(
        [results_root / f"{model_name}-{batch}" / "runs.jsonl"],
        model_name=model_name,
        prompt_version=BATCH_PROMPT_VERSIONS[batch],
        quantity_ids=quantity_ids_for_batch(batch),
        require_parsed=require_parsed,
    )


def _parse_csv_list(raw: str, *, option: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{option} must contain at least one value")
    if len(values) != len(set(values)):
        raise ValueError(f"{option} must not contain duplicates")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "comma-separated model names (default: the full registry for the "
            "main batch, July-wave models for the clarify batches)"
        ),
    )
    parser.add_argument(
        "--batches",
        default=None,
        help="comma-separated canonical batch suffixes",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results",
    )
    parser.add_argument(
        "--require-parsed",
        action="store_true",
        help="also require parsed_ok=true for every expected slot",
    )
    args = parser.parse_args()

    try:
        models = (
            _parse_csv_list(args.models, option="--models")
            if args.models is not None
            else None
        )
        batches = (
            _parse_csv_list(args.batches, option="--batches")
            if args.batches is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    if batches is not None:
        unknown_batches = sorted(set(batches) - set(BATCH_PROMPT_VERSIONS))
        if unknown_batches:
            parser.error(f"unknown batches: {', '.join(unknown_batches)}")

    if models is None and batches is None:
        # The paper-claim plan: main batch panel-wide, clarify July-wide.
        plan = [(model, "elasticities-batch15") for model in PANEL_MODEL_IDS]
        plan += [
            (model, batch)
            for model in POST_APRIL_MODEL_IDS
            for batch in ("armington-clarify-batch15", "ies-clarify-batch15")
        ]
    else:
        plan = [
            (model, batch)
            for model in (models if models is not None else PANEL_MODEL_IDS)
            for batch in (batches if batches is not None else BATCH_PROMPT_VERSIONS)
        ]

    all_ok = True
    for model_name, batch in plan:
        result = check_batch_directory(
            args.results_root,
            model_name=model_name,
            batch=batch,
            require_parsed=args.require_parsed,
        )
        label = f"{model_name}/{batch}"
        if result.ok:
            print(f"OK {label}: {result.observed_count}/{result.expected_count}")
            continue
        all_ok = False
        print(
            f"FAIL {label}: {result.observed_count}/{result.expected_count}",
            file=sys.stderr,
        )
        for error in result.errors:
            print(f"  {error}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
