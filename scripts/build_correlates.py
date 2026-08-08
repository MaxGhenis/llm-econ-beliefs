"""Build cross-model correlates from complete raw elicitation grids.

The builder validates the 25-model registry against the pinned PolicyBench
release, gates incomplete panels, reconstructs pooled quantities directly from
``runs.jsonl``, and writes:

- ``results/correlates-model-summary.csv``
- ``results/correlates-spearman.csv``
- ``results/correlates-sensitivity.csv``

Usage:
    .venv/bin/python scripts/build_correlates.py [--allow-partial]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT))

from llm_econ_beliefs import (  # noqa: E402
    BeliefEstimate,
    aggregate_beliefs,
    distribution_from_belief_estimate,
    get_quantity,
    mixture_distribution,
)
from llm_econ_beliefs.model_registry import (  # noqa: E402
    MODEL_REGISTRY_BY_ID,
    PANEL_MODEL_IDS,
)

try:  # noqa: E402
    from panel_grid_gate import gate_panel_models
except ModuleNotFoundError:  # Imported as ``scripts.build_correlates``.
    from scripts.panel_grid_gate import gate_panel_models


LABOR_TAX = {
    "labor_supply.extensive_margin.single_mothers",
    "labor_supply.frisch_elasticity.prime_age",
    "labor_supply.income_elasticity.prime_age",
    "labor_supply.marshallian_wage_elasticity.prime_age",
    "tax.capital_gains_realizations.elasticity",
    "tax.elasticity_of_taxable_income.top_earners",
}
MACRO_TRADE = {
    "household.intertemporal_elasticity_of_substitution",
    "production.capital_labor_substitution",
    "trade.armington_elasticity.import_domestic",
}
CANONICAL = LABOR_TAX | MACRO_TRADE | {
    "household.annual_discount_factor",
    "household.relative_risk_aversion.crra",
    "macro.tfp_persistence.ar1",
    "production.capital_share",
}
ETI_QUANTITY_ID = "tax.elasticity_of_taxable_income.top_earners"

EXPECTED_PANEL_ONLY = {"gpt-5.4", "grok-4.20", "grok-4.1-fast"}
EXPECTED_POLICYBENCH_ONLY = {"grok-build-0.1"}
EXPECTED_POLICYBENCH_CONDITION = "no_tools"
EXPECTED_POLICYBENCH_COUNTRY = "us"

PRIMARY_PREDICTOR_KEY = "policybench_within_dollar"
PRIMARY_PREDICTOR_LABEL = "Overall within-$1 (leaderboard headline)"
PRIMARY_OUTCOME_KEY = "eti_median"
PRIMARY_OUTCOME_LABEL = "ETI pooled median"
DERIVED_TAU_LABEL = (
    "Implied optimal top rate (%) — derived, monotone transform of ETI — "
    "not an additional test"
)

_NUMBER_PATTERN = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
_INTERVAL_PATTERN = re.compile(
    rf"^\s*\[\s*({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*\]\s*$"
)
_ESTIMATE_INTERVAL_PATTERN = re.compile(
    rf"^\s*({_NUMBER_PATTERN})\s*\[\s*({_NUMBER_PATTERN})\s*,\s*"
    rf"({_NUMBER_PATTERN})\s*\]\s*$"
)
_PERCENTAGE_PATTERN = re.compile(rf"^\s*({_NUMBER_PATTERN})\s*%\s*$")
_PERCENTAGE_INTERVAL_PATTERN = re.compile(
    rf"^\s*({_NUMBER_PATTERN})\s*%\s*\[\s*({_NUMBER_PATTERN})\s*%\s*,\s*"
    rf"({_NUMBER_PATTERN})\s*%\s*\]\s*$"
)


def _parse_ordered_bounds(
    lower_text: str, upper_text: str, *, cell: str
) -> tuple[float, float]:
    lower = float(lower_text)
    upper = float(upper_text)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"Interval bounds must be finite: {cell!r}")
    if lower > upper:
        raise ValueError(f"Interval bounds must be ordered: {cell!r}")
    return lower, upper


def parse_interval_cell(cell: str) -> tuple[float, float]:
    """Parse exactly ``[lower, upper]`` with finite, ordered bounds."""
    match = _INTERVAL_PATTERN.fullmatch(cell)
    if match is None:
        raise ValueError(f"Expected '[lower, upper]' interval: {cell!r}")
    return _parse_ordered_bounds(*match.groups(), cell=cell)


def parse_estimate_interval_cell(cell: str) -> tuple[float, float, float]:
    """Parse exactly ``estimate [lower, upper]`` with finite values."""
    match = _ESTIMATE_INTERVAL_PATTERN.fullmatch(cell)
    if match is None:
        raise ValueError(f"Expected 'estimate [lower, upper]' cell: {cell!r}")
    estimate = float(match.group(1))
    if not math.isfinite(estimate):
        raise ValueError(f"Estimate must be finite: {cell!r}")
    lower, upper = _parse_ordered_bounds(
        match.group(2), match.group(3), cell=cell
    )
    return estimate, lower, upper


def parse_percentage_cell(cell: str) -> float:
    """Parse a finite percentage, requiring the explicit percent sign."""
    match = _PERCENTAGE_PATTERN.fullmatch(cell)
    if match is None:
        raise ValueError(f"Expected percentage with explicit '%' sign: {cell!r}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"Percentage must be finite: {cell!r}")
    return value


def parse_percentage_interval_cell(cell: str) -> tuple[float, float, float]:
    """Parse exactly ``x% [lower%, upper%]`` with finite, ordered bounds."""
    match = _PERCENTAGE_INTERVAL_PATTERN.fullmatch(cell)
    if match is None:
        raise ValueError(
            "Expected 'x% [lower%, upper%]' with explicit '%' signs: "
            f"{cell!r}"
        )
    estimate = float(match.group(1))
    if not math.isfinite(estimate):
        raise ValueError(f"Percentage estimate must be finite: {cell!r}")
    lower, upper = _parse_ordered_bounds(
        match.group(2), match.group(3), cell=cell
    )
    return estimate, lower, upper


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw run file: {path}")
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _belief_estimate_from_run(row: dict[str, object]) -> BeliefEstimate:
    quantiles_value = row.get("quantiles") or {}
    if not isinstance(quantiles_value, dict):
        raise ValueError("Run quantiles must be a JSON object")
    point_estimate = row.get("point_estimate")
    if isinstance(point_estimate, bool) or not isinstance(point_estimate, (int, float)):
        raise ValueError("Parsed run is missing a numeric point_estimate")
    return BeliefEstimate(
        point_estimate=float(point_estimate),
        quantity_id=str(row["quantity_id"]),
        interpretation=(
            str(row["interpretation"]) if row.get("interpretation") else None
        ),
        lower_bound=(
            float(row["lower_bound"]) if row.get("lower_bound") is not None else None
        ),
        upper_bound=(
            float(row["upper_bound"]) if row.get("upper_bound") is not None else None
        ),
        confidence_level=(
            float(row["confidence_level"])
            if row.get("confidence_level") is not None
            else None
        ),
        quantiles={str(key): float(value) for key, value in quantiles_value.items()},
        citations=[],
        reasoning_summary=(
            str(row["reasoning_summary"]) if row.get("reasoning_summary") else None
        ),
    )


def pooled_eti_median(run_rows: Iterable[dict[str, object]]) -> float:
    """Reproduce build_tables' equal-weight piecewise-uniform ETI median."""
    estimates = [
        _belief_estimate_from_run(row)
        for row in run_rows
        if row.get("parsed_ok") is True
        and row.get("quantity_id") == ETI_QUANTITY_ID
    ]
    eti_quantity = get_quantity(ETI_QUANTITY_ID)
    components = [
        distribution
        for estimate in estimates
        if (
            distribution := distribution_from_belief_estimate(
                estimate,
                lower_support=0.0,
                upper_support=eti_quantity.upper_support,
            )
        )
        is not None
    ]
    if not components:
        raise ValueError("No quantile-complete ETI runs are available for pooling")
    return mixture_distribution(components).quantile(0.50)


def pool_model_runs(
    model_id: str,
    *,
    results_root: Path = RESULTS,
) -> tuple[dict[str, dict[str, float]], float]:
    """Pool canonical centers/intervals and ETI median from one raw run file."""
    path = results_root / f"{model_id}-elasticities-batch15" / "runs.jsonl"
    run_rows = _read_jsonl(path)
    by_quantity: dict[str, list[BeliefEstimate]] = defaultdict(list)
    eligible_rows = [
        row
        for row in run_rows
        if row.get("model_name") == model_id
        and row.get("prompt_version") == "v4"
        and row.get("parsed_ok") is True
        and row.get("quantity_id") in CANONICAL
    ]
    for row in eligible_rows:
        by_quantity[str(row["quantity_id"])].append(_belief_estimate_from_run(row))

    panel: dict[str, dict[str, float]] = {}
    for quantity_id in sorted(CANONICAL):
        estimates = by_quantity.get(quantity_id, [])
        if not estimates:
            raise ValueError(f"{model_id} has no parsed runs for {quantity_id}")
        quantity = get_quantity(quantity_id)
        pooled = aggregate_beliefs(
            estimates,
            confidence_level=0.9,
            lower_support=quantity.lower_support,
            upper_support=quantity.upper_support,
        )
        if pooled.lower_bound is None or pooled.upper_bound is None:
            raise ValueError(f"{model_id}/{quantity_id} has no pooled 90% interval")
        panel[quantity_id] = {
            "center": pooled.point_estimate,
            "lower": pooled.lower_bound,
            "upper": pooled.upper_bound,
        }
    return panel, pooled_eti_median(eligible_rows)


def load_panel(
    model_ids: Sequence[str],
    *,
    results_root: Path = RESULTS,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, float]]:
    panel: dict[str, dict[str, dict[str, float]]] = {}
    eti_medians: dict[str, float] = {}
    for model_id in model_ids:
        quantities, eti_median = pool_model_runs(model_id, results_root=results_root)
        panel[model_id] = quantities
        eti_medians[model_id] = eti_median
    return panel, eti_medians


def read_policybench_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PolicyBench score file: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_policybench_crosswalk(
    rows: Sequence[dict[str, str]],
    panel_model_ids: Sequence[str] = PANEL_MODEL_IDS,
    *,
    stream: TextIO = sys.stdout,
) -> None:
    """Require the pinned one-to-one PolicyBench/panel nominal crosswalk."""
    if not rows:
        raise ValueError("PolicyBench score file is empty")
    models = [row.get("model", "").strip() for row in rows]
    panel_set = set(panel_model_ids)
    policybench_set = set(models) - {""}
    panel_only = sorted(panel_set - policybench_set)
    policybench_only = sorted(policybench_set - panel_set)
    payload = {
        "event": "policybench_crosswalk",
        "panel_only": panel_only,
        "policybench_only": policybench_only,
    }
    print("POLICYBENCH_CROSSWALK=" + json.dumps(payload, sort_keys=True), file=stream)

    if any(not model for model in models):
        raise ValueError("PolicyBench model IDs must not be blank")
    duplicates = sorted(
        model for model, count in Counter(models).items() if count != 1
    )
    if duplicates:
        raise ValueError(
            "Duplicate PolicyBench model IDs: " + ", ".join(duplicates)
        )
    if len(panel_set) != len(panel_model_ids):
        raise ValueError("Panel registry model IDs must be unique")

    for field in ("source_release", "condition", "country"):
        values = {row.get(field, "").strip() for row in rows}
        if "" in values or len(values) != 1:
            raise ValueError(
                f"PolicyBench {field} must have exactly one nonblank value; "
                f"found {sorted(values)!r}"
            )
    conditions = {row["condition"].strip() for row in rows}
    countries = {row["country"].strip() for row in rows}
    if conditions != {EXPECTED_POLICYBENCH_CONDITION}:
        raise ValueError(
            f"Expected PolicyBench condition={EXPECTED_POLICYBENCH_CONDITION!r}, "
            f"found {sorted(conditions)!r}"
        )
    if countries != {EXPECTED_POLICYBENCH_COUNTRY}:
        raise ValueError(
            f"Expected PolicyBench country={EXPECTED_POLICYBENCH_COUNTRY!r}, "
            f"found {sorted(countries)!r}"
        )
    if set(panel_only) != EXPECTED_PANEL_ONLY:
        raise ValueError(
            "Unexpected panel-only PolicyBench crosswalk IDs: "
            f"expected {sorted(EXPECTED_PANEL_ONLY)!r}, found {panel_only!r}"
        )
    if set(policybench_only) != EXPECTED_POLICYBENCH_ONLY:
        raise ValueError(
            "Unexpected PolicyBench-only crosswalk IDs: "
            f"expected {sorted(EXPECTED_POLICYBENCH_ONLY)!r}, "
            f"found {policybench_only!r}"
        )


def load_policybench(
    path: Path = RESULTS / "policybench-scores.csv",
) -> dict[str, dict[str, float]]:
    rows = read_policybench_rows(path)
    validate_policybench_crosswalk(rows)
    scores: dict[str, dict[str, float]] = {}
    for row in rows:
        values = {
            "overall": float(row["policybench_within_dollar"]),
            "tax": float(row["policybench_tax_within_dollar"]),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"PolicyBench scores must be finite for {row['model']}")
        scores[row["model"]] = values
    return scores


def load_top_rate_calibration(
    path: Path = RESULTS / "top-rate-calibration.json",
) -> dict[str, float]:
    """Load a non-fallback microdata calibration for the top-rate mapping."""
    if not path.exists():
        raise FileNotFoundError(
            "Missing required results/top-rate-calibration.json; run "
            "paper/build_tables.py with the PolicyEngine microdata calibration first."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid top-rate calibration JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Top-rate calibration must be a JSON object")
    required = ("a", "gbar", "threshold", "tail_mean")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(
            "Top-rate calibration is missing fields: " + ", ".join(missing)
        )
    raw_a = payload["a"]
    if isinstance(raw_a, bool) or not isinstance(raw_a, (int, float)):
        raise ValueError("Top-rate calibration a must be numeric")
    a = float(raw_a)
    if not math.isfinite(a):
        raise ValueError("Top-rate calibration a must be finite")
    if math.isclose(a, 1.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "results/top-rate-calibration.json carries fallback a=1.5; rerun "
            "paper/build_tables.py with the PolicyEngine microdata calibration."
        )
    values: dict[str, float] = {"a": a}
    for field in required[1:]:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Top-rate calibration {field} must be numeric")
        values[field] = float(value)
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Top-rate calibration fields must all be finite")
    if values["a"] <= 0:
        raise ValueError("Top-rate calibration a must be positive")
    if not 0 <= values["gbar"] < 1:
        raise ValueError("Top-rate calibration gbar must be in [0, 1)")
    if values["threshold"] <= 0 or values["tail_mean"] <= values["threshold"]:
        raise ValueError(
            "Top-rate calibration requires 0 < threshold < tail_mean"
        )
    return values


def implied_top_rates(eti: float, calibration: dict[str, float]) -> tuple[float, float]:
    """Return calibrated tau* and revenue-max rate in percentage points."""
    a = calibration["a"]
    gbar = calibration["gbar"]
    tau_denominator = 1 - gbar + a * eti
    revenue_denominator = 1 + a * eti
    if tau_denominator <= 0 or revenue_denominator <= 0:
        raise ValueError(f"ETI {eti} yields an invalid top-rate denominator")
    return (
        100 * (1 - gbar) / tau_denominator,
        100 / revenue_denominator,
    )


def rank_map(values: dict[str, float], *, reverse: bool = False) -> dict[str, float]:
    """Tie-averaged ranks, matching build_tables.average_ranks."""
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        tie_end = index
        while (
            tie_end + 1 < len(ordered)
            and ordered[tie_end + 1][1] == ordered[index][1]
        ):
            tie_end += 1
        average = (index + 1 + tie_end + 1) / 2
        for key, _ in ordered[index : tie_end + 1]:
            ranks[key] = average
        index = tie_end + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if not correlation_inputs_are_valid(xs, ys):
        return float("nan")

    def to_ranks(values: list[float]) -> list[float]:
        indexed = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while (
                j + 1 < len(indexed)
                and values[indexed[j + 1]] == values[indexed[i]]
            ):
                j += 1
            average = (i + 1 + j + 1) / 2
            for k in range(i, j + 1):
                ranks[indexed[k]] = average
            i = j + 1
        return ranks

    rx, ry = to_ranks(xs), to_ranks(ys)
    n = len(xs)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    return cov / math.sqrt(var_x * var_y)


def correlation_inputs_are_valid(xs: list[float], ys: list[float]) -> bool:
    """Whether two vectors define a finite, nondegenerate correlation."""
    return (
        len(xs) == len(ys)
        and len(xs) >= 3
        and all(math.isfinite(value) for value in itertools.chain(xs, ys))
        and any(value != xs[0] for value in xs[1:])
        and any(value != ys[0] for value in ys[1:])
    )


def permutation_p(xs: list[float], ys: list[float], *, seed: int = 20260708) -> float:
    """Two-sided permutation p for Spearman rho; exact when feasible."""
    if not correlation_inputs_are_valid(xs, ys):
        return float("nan")

    observed = abs(spearman(xs, ys))
    n = len(xs)
    if n <= 8:
        perms = list(itertools.permutations(ys))
        hits = sum(
            1
            for permutation in perms
            if abs(spearman(xs, list(permutation))) >= observed - 1e-12
        )
        return hits / len(perms)
    rng = random.Random(seed)
    draws = 20000
    hits = 0
    shuffled = list(ys)
    for _ in range(draws):
        rng.shuffle(shuffled)
        if abs(spearman(xs, shuffled)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (draws + 1)


def holm_adjusted_pvalues(pvalues: Sequence[float]) -> list[float]:
    """Holm family-wise-error adjusted p-values in original order."""
    _validate_pvalues(pvalues)
    m = len(pvalues)
    ordered = sorted(range(m), key=lambda index: pvalues[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(ordered):
        candidate = (m - rank) * pvalues[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def bh_adjusted_pvalues(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjusted p-values in original order."""
    _validate_pvalues(pvalues)
    m = len(pvalues)
    ordered = sorted(range(m), key=lambda index: pvalues[index])
    adjusted = [0.0] * m
    running = 1.0
    for position in range(m, 0, -1):
        index = ordered[position - 1]
        candidate = m * pvalues[index] / position
        running = min(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def _validate_pvalues(pvalues: Sequence[float]) -> None:
    if not pvalues:
        raise ValueError("At least one p-value is required")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in pvalues):
        raise ValueError("P-values must be finite and in [0, 1]")


def kruskal_wallis(groups: list[list[float]]) -> tuple[float, float]:
    """Tie-corrected H statistic and chi-square p (df = k-1)."""
    all_values = [value for group in groups for value in group]
    n = len(all_values)
    ranks = rank_map({str(index): value for index, value in enumerate(all_values)})
    flat_ranks = [ranks[str(index)] for index in range(n)]
    position = 0
    h = 0.0
    for group in groups:
        size = len(group)
        group_ranks = flat_ranks[position : position + size]
        position += size
        mean_rank = sum(group_ranks) / size
        h += size * (mean_rank - (n + 1) / 2) ** 2
    h *= 12 / (n * (n + 1))
    tie_correction = 1 - sum(
        count**3 - count for count in Counter(all_values).values()
    ) / (n**3 - n)
    if tie_correction == 0:
        return float("nan"), float("nan")
    h /= tie_correction
    p = chi2_sf(h, len(groups) - 1)
    return h, p


def chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square via the regularized gamma function."""
    return 1 - gamma_inc(k / 2, x / 2)


def gamma_inc(s: float, x: float) -> float:
    """Regularized lower incomplete gamma P(s, x) by series/continued fraction."""
    if x <= 0:
        return 0.0
    if x < s + 1:
        term = 1.0 / s
        total = term
        n = 0
        while abs(term) > 1e-12 * abs(total):
            n += 1
            term *= x / (s + n)
            total += term
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    _, b = 1.0, x + 1 - s
    c, d = 1e300, 1 / b
    h = d
    for index in range(1, 200):
        an = -index * (index - s)
        b += 2
        d = an * d + b
        d = 1 / d if abs(d) > 1e-300 else 1e300
        c = b + an / c if abs(c) > 1e-300 else 1e300
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return 1 - h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def build_summary_rows(
    panel: dict[str, dict[str, dict[str, float]]],
    eti_medians: dict[str, float],
    policybench: dict[str, dict[str, float]],
    calibration: dict[str, float],
) -> list[dict[str, object]]:
    width_values: dict[str, dict[str, float]] = defaultdict(dict)
    for model_id, quantities in panel.items():
        for quantity_id, cell in quantities.items():
            width_values[quantity_id][model_id] = cell["upper"] - cell["lower"]
    width_ranks: dict[str, list[float]] = defaultdict(list)
    for per_model in width_values.values():
        for model_id, rank in rank_map(per_model).items():
            width_ranks[model_id].append(rank)

    rows: list[dict[str, object]] = []
    for model_id in panel:
        metadata = MODEL_REGISTRY_BY_ID[model_id]
        quantities = panel[model_id]
        labor = [
            abs(cell["center"])
            for quantity_id, cell in quantities.items()
            if quantity_id in LABOR_TAX
        ]
        macro = [
            abs(cell["center"])
            for quantity_id, cell in quantities.items()
            if quantity_id in MACRO_TRADE
        ]
        eti_median = eti_medians[model_id]
        tau_star, tau_revmax = implied_top_rates(eti_median, calibration)
        scores = policybench.get(model_id) or {}
        rows.append(
            {
                "model": model_id,
                "organization": metadata.organization,
                "wave": metadata.wave,
                "policybench_within_dollar": scores.get("overall"),
                "policybench_tax_within_dollar": scores.get("tax"),
                "mean_abs_center_labor_tax": sum(labor) / len(labor),
                "mean_abs_center_macro_trade": sum(macro) / len(macro),
                "avg_width_rank": sum(width_ranks[model_id])
                / len(width_ranks[model_id]),
                "eti_median": eti_median,
                "tau_star_pct": tau_star,
                "tau_revmax_pct": tau_revmax,
            }
        )
    return rows


def _correlation_pairs(
    rows: Sequence[dict[str, object]], predictor: str, outcome: str
) -> tuple[list[float], list[float]]:
    pairs = [
        (row[predictor], row[outcome])
        for row in rows
        if row[predictor] is not None and row[outcome] is not None
    ]
    return (
        [float(predictor_value) for predictor_value, _ in pairs],
        [float(outcome_value) for _, outcome_value in pairs],
    )


def build_correlation_rows(
    summary_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    outcomes = [
        ("mean_abs_center_labor_tax", "Mean |center|, labor-and-tax", False),
        ("mean_abs_center_macro_trade", "Mean |center|, macro-and-trade", False),
        ("avg_width_rank", "Avg interval-width rank (1 = tightest)", False),
        ("eti_median", PRIMARY_OUTCOME_LABEL, False),
        ("tau_star_pct", DERIVED_TAU_LABEL, True),
    ]
    predictors = [
        ("policybench_tax_within_dollar", "Tax within-$1 (domain-matched)"),
        (PRIMARY_PREDICTOR_KEY, PRIMARY_PREDICTOR_LABEL),
    ]
    rows: list[dict[str, object]] = []
    for predictor_key, predictor_label in predictors:
        for outcome_key, outcome_label, is_derived in outcomes:
            xs, ys = _correlation_pairs(summary_rows, predictor_key, outcome_key)
            rows.append(
                {
                    "predictor": predictor_label,
                    "outcome": outcome_label,
                    "n_models": len(xs),
                    "spearman_rho": spearman(xs, ys),
                    "raw_p": None if is_derived else permutation_p(xs, ys),
                    "holm_adjusted_p": None,
                    "bh_adjusted_p": None,
                    "family_size": None,
                    "is_derived": is_derived,
                }
            )

    tested_indices = [
        index for index, row in enumerate(rows) if not bool(row["is_derived"])
    ]
    tested_pvalues = [float(rows[index]["raw_p"]) for index in tested_indices]
    holm = holm_adjusted_pvalues(tested_pvalues)
    bh = bh_adjusted_pvalues(tested_pvalues)
    family_size = len(tested_indices)
    for position, index in enumerate(tested_indices):
        rows[index]["holm_adjusted_p"] = holm[position]
        rows[index]["bh_adjusted_p"] = bh[position]
        rows[index]["family_size"] = family_size

    for row in rows:
        if not row["is_derived"]:
            continue
        eti_row = next(
            candidate
            for candidate in rows
            if candidate["predictor"] == row["predictor"]
            and candidate["outcome"] == PRIMARY_OUTCOME_LABEL
        )
        row["raw_p"] = eti_row["raw_p"]
        row["holm_adjusted_p"] = eti_row["holm_adjusted_p"]
        row["bh_adjusted_p"] = eti_row["bh_adjusted_p"]
        row["family_size"] = family_size
    return rows


def build_sensitivity_rows(
    summary_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    primary = [
        row
        for row in summary_rows
        if row[PRIMARY_PREDICTOR_KEY] is not None
        and row[PRIMARY_OUTCOME_KEY] is not None
    ]
    rows: list[dict[str, object]] = []
    organizations = sorted({str(row["organization"]) for row in primary})
    for omitted in organizations:
        retained = [row for row in primary if row["organization"] != omitted]
        xs = [float(row[PRIMARY_PREDICTOR_KEY]) for row in retained]
        ys = [float(row[PRIMARY_OUTCOME_KEY]) for row in retained]
        rows.append(
            {
                "analysis": "leave_one_organization_out",
                "predictor": PRIMARY_PREDICTOR_LABEL,
                "outcome": PRIMARY_OUTCOME_LABEL,
                "omitted_organization": omitted,
                "wave": None,
                "n_models": len(retained),
                "n_organizations": len(
                    {str(row["organization"]) for row in retained}
                ),
                "statistic_level": "model",
                "permutation_unit": None,
                "n_permutations": None,
                "spearman_rho": spearman(xs, ys),
                "permutation_p": None,
            }
        )

    wave_filters = (
        ("april_only", lambda wave: wave == "april_2026"),
        ("july_only", lambda wave: wave.startswith("july_2026_")),
    )
    for wave_label, predicate in wave_filters:
        subset = [row for row in primary if predicate(str(row["wave"]))]
        xs = [float(row[PRIMARY_PREDICTOR_KEY]) for row in subset]
        ys = [float(row[PRIMARY_OUTCOME_KEY]) for row in subset]
        rows.append(
            {
                "analysis": "within_wave",
                "predictor": PRIMARY_PREDICTOR_LABEL,
                "outcome": PRIMARY_OUTCOME_LABEL,
                "omitted_organization": None,
                "wave": wave_label,
                "n_models": len(subset),
                "n_organizations": len(
                    {str(row["organization"]) for row in subset}
                ),
                "statistic_level": "model",
                "permutation_unit": None,
                "n_permutations": None,
                "spearman_rho": spearman(xs, ys),
                "permutation_p": None,
            }
        )

    cluster_rho, cluster_p, n_permutations = organization_block_permutation(primary)
    rows.append(
        {
            "analysis": "organization_cluster_permutation",
            "predictor": PRIMARY_PREDICTOR_LABEL,
            "outcome": PRIMARY_OUTCOME_LABEL,
            "omitted_organization": None,
            "wave": None,
            "n_models": len(primary),
            "n_organizations": len(organizations),
            "statistic_level": "model",
            "permutation_unit": "organization_score_block_within_equal_size_strata",
            "n_permutations": n_permutations,
            "spearman_rho": cluster_rho,
            "permutation_p": cluster_p,
        }
    )
    return rows


def organization_block_permutation(
    rows: Sequence[dict[str, object]],
) -> tuple[float, float, int]:
    """Permute whole PolicyBench score vectors between equal-size organizations.

    Organization sizes differ in the overlap panel, so a whole block can only
    move to another organization with the same number of models. Within-block
    model order follows registry order inherited by ``summary_rows``. The
    resulting exact reference set has 3! x 4! = 144 assignments for the
    current 7/6/5/2/2/2/1/1/1/1 organization-size pattern.
    """
    by_organization: dict[str, list[dict[str, object]]] = defaultdict(list)
    model_ids: list[str] = []
    for row in rows:
        organization = str(row["organization"])
        model_id = str(row["model"])
        if model_id in model_ids:
            raise ValueError(f"Duplicate model in cluster permutation: {model_id}")
        model_ids.append(model_id)
        by_organization[organization].append(row)

    xs = [float(row[PRIMARY_PREDICTOR_KEY]) for row in rows]
    ys = [float(row[PRIMARY_OUTCOME_KEY]) for row in rows]
    if not correlation_inputs_are_valid(xs, ys):
        return float("nan"), float("nan"), 0
    observed_rho = spearman(xs, ys)

    organizations_by_size: dict[int, list[str]] = defaultdict(list)
    for organization, organization_rows in by_organization.items():
        organizations_by_size[len(organization_rows)].append(organization)
    strata: list[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]] = []
    for _, organizations in sorted(organizations_by_size.items()):
        targets = tuple(sorted(organizations))
        strata.append((targets, tuple(itertools.permutations(targets))))

    hits = 0
    total = 0
    for stratum_assignments in itertools.product(
        *(assignments for _, assignments in strata)
    ):
        source_by_target: dict[str, str] = {}
        for (targets, _), sources in zip(strata, stratum_assignments, strict=True):
            source_by_target.update(zip(targets, sources, strict=True))

        permuted_score_by_model: dict[str, float] = {}
        for target, target_rows in by_organization.items():
            source_rows = by_organization[source_by_target[target]]
            source_scores = [
                float(row[PRIMARY_PREDICTOR_KEY]) for row in source_rows
            ]
            for target_row, score in zip(target_rows, source_scores, strict=True):
                permuted_score_by_model[str(target_row["model"])] = score
        permuted_xs = [permuted_score_by_model[model_id] for model_id in model_ids]
        total += 1
        if abs(spearman(permuted_xs, ys)) >= abs(observed_rho) - 1e-12:
            hits += 1
    return observed_rho, hits / total, total


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


COUNTRY_OUTCOMES = [
    # (summary key, label, is_derived) — the top-rate row is a monotone
    # transform of the ETI row, so it tests the same hypothesis and stays
    # outside the adjusted family, mirroring the capability cut.
    ("tau_star_pct", "Implied optimal top rate (%)", True),
    ("eti_median", "ETI pooled median", False),
    ("avg_width_rank", "Avg interval-width rank (1 = tightest)", False),
    ("mean_abs_center_labor_tax", "Mean |center|, labor-and-tax", False),
    ("mean_abs_center_macro_trade", "Mean |center|, macro-and-trade", False),
]

COUNTRY_PRIMARY_OUTCOME_LABEL = "ETI pooled median"

COUNTRY_DISCLOSURE = (
    "Exploratory. Lab country is perfectly confounded with serving path "
    "(every Chinese lab ran via OpenRouter JSON mode) and elicitation "
    "wave, and co-varies with completion budget and reasoning "
    "configuration; prompts are English-language."
)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def country_permutation_p(
    us_values: Sequence[float],
    china_values: Sequence[float],
    *,
    seed: int = 20260710,
    # C(31, 7) = 2,629,575 — keep the panel's country cut on the
    # exact-enumeration path (the paper reports it as exact).
    max_exact: int = 2700000,
) -> float:
    """Two-sided group-label permutation p for the difference in medians.

    Enumerates every assignment of the pooled values into groups of the
    observed sizes when that count is small enough; otherwise samples with
    a fixed seed and applies the +1 correction.
    """
    pooled = [*us_values, *china_values]
    n_china = len(china_values)
    observed = abs(_median(china_values) - _median(us_values))
    total = math.comb(len(pooled), n_china)
    indices = range(len(pooled))
    if total <= max_exact:
        hits = 0
        for combo in itertools.combinations(indices, n_china):
            chosen = set(combo)
            china = [pooled[i] for i in chosen]
            us = [pooled[i] for i in indices if i not in chosen]
            if abs(_median(china) - _median(us)) >= observed - 1e-12:
                hits += 1
        return hits / total
    rng = random.Random(seed)
    draws = 20000
    hits = 0
    for _ in range(draws):
        sample = rng.sample(list(indices), n_china)
        chosen = set(sample)
        china = [pooled[i] for i in chosen]
        us = [pooled[i] for i in indices if i not in chosen]
        if abs(_median(china) - _median(us)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (draws + 1)


def build_country_comparison_rows(
    summary_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """US-lab versus Chinese-lab medians per outcome, with permutation p."""
    from llm_econ_beliefs.model_registry import country_for_organization

    rows: list[dict[str, object]] = []
    for key, label, is_derived in COUNTRY_OUTCOMES:
        groups: dict[str, list[float]] = {"us": [], "china": []}
        for row in summary_rows:
            value = row.get(key)
            if value is None:
                continue
            country = country_for_organization(str(row["organization"]))
            groups[country].append(float(value))
        if not groups["us"] or not groups["china"]:
            continue
        us_median = _median(groups["us"])
        china_median = _median(groups["china"])
        rows.append(
            {
                "outcome": label,
                "us_n": len(groups["us"]),
                "us_median": round(us_median, 4),
                "china_n": len(groups["china"]),
                "china_median": round(china_median, 4),
                "china_minus_us": round(china_median - us_median, 4),
                # Six decimals so downstream 3-decimal display rounding acts
                # on the exact value, not on an already-rounded one.
                "permutation_p": round(
                    country_permutation_p(groups["us"], groups["china"]), 6
                ),
                "holm_adjusted_p": None,
                "bh_adjusted_p": None,
                "family_size": None,
                "is_derived": is_derived,
                "disclosure": COUNTRY_DISCLOSURE,
            }
        )

    # Same multiplicity standard as the capability cut: Holm/BH over the
    # tested outcomes, with the derived top-rate row mirroring the ETI
    # row's adjusted values (they state one hypothesis, not two).
    tested = [i for i, row in enumerate(rows) if not bool(row["is_derived"])]
    pvalues = [float(rows[i]["permutation_p"]) for i in tested]
    holm = holm_adjusted_pvalues(pvalues)
    bh = bh_adjusted_pvalues(pvalues)
    for position, index in enumerate(tested):
        rows[index]["holm_adjusted_p"] = round(holm[position], 4)
        rows[index]["bh_adjusted_p"] = round(bh[position], 4)
        rows[index]["family_size"] = len(tested)
    eti_row = next(
        row for row in rows if row["outcome"] == COUNTRY_PRIMARY_OUTCOME_LABEL
    )
    for row in rows:
        if row["is_derived"]:
            row["holm_adjusted_p"] = eti_row["holm_adjusted_p"]
            row["bh_adjusted_p"] = eti_row["bh_adjusted_p"]
            row["family_size"] = eti_row["family_size"]
    return rows


POSTHOC_NOTE = (
    "Post hoc: computed after the eight-test family was declared, prompted "
    "in review of the width result; raw permutation p only, outside the "
    "Holm/BH family by construction."
)


def consistency_metrics(
    cells: Sequence[tuple[float, float]],
) -> dict[str, float]:
    """Worst-case share and median between-run SD over canonical cells.

    Each cell is (within_run_sd, between_run_sd); the share is the
    between-run component of pooled variance, matching Appendix A15.
    """
    shares = [
        (between * between) / (within * within + between * between)
        for within, between in cells
        if (within * within + between * between) > 0
    ]
    if not shares:
        raise ValueError("No decomposable cells")
    return {
        "max_between_run_share": max(shares),
        "median_between_run_sd": _median([between for _, between in cells]),
    }


def load_consistency_metrics(
    model_ids: Sequence[str],
    results_root: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Per-model run-consistency metrics from the summary artifacts."""
    root = results_root or RESULTS
    metrics: dict[str, dict[str, float]] = {}
    for model_id in model_ids:
        path = root / f"{model_id}-elasticities-batch15" / "summary.csv"
        cells: list[tuple[float, float]] = []
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["quantity_id"] not in CANONICAL:
                    continue
                within = row.get("within_run_sd") or ""
                between = row.get("between_run_sd") or ""
                if within and between:
                    cells.append((float(within), float(between)))
        metrics[model_id] = consistency_metrics(cells)
    return metrics


def build_posthoc_consistency_rows(
    summary_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Capability versus run-to-run consistency, reported outside the family."""
    metrics = load_consistency_metrics(
        [str(row["model"]) for row in summary_rows]
    )
    outcomes = [
        ("max_between_run_share", "Max between-run variance share"),
        ("median_between_run_sd", "Median between-run SD"),
    ]
    predictors = [
        (PRIMARY_PREDICTOR_KEY, PRIMARY_PREDICTOR_LABEL),
        ("policybench_tax_within_dollar", "Tax within-$1 (domain-matched)"),
    ]
    rows: list[dict[str, object]] = []
    for predictor_key, predictor_label in predictors:
        for metric_key, outcome_label in outcomes:
            xs: list[float] = []
            ys: list[float] = []
            for row in summary_rows:
                score = row.get(predictor_key)
                if score is None:
                    continue
                xs.append(float(score))
                ys.append(metrics[str(row["model"])][metric_key])
            rows.append(
                {
                    "predictor": predictor_label,
                    "outcome": outcome_label,
                    "n_models": len(xs),
                    "spearman_rho": spearman(xs, ys),
                    "raw_p": permutation_p(xs, ys),
                    "note": POSTHOC_NOTE,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="include incomplete registered grids for development only",
    )
    args = parser.parse_args()

    try:
        policybench = load_policybench()
        gate = gate_panel_models(
            RESULTS,
            PANEL_MODEL_IDS,
            allow_partial=args.allow_partial,
        )
        if not gate.included_models:
            raise ValueError("No registered panel models passed complete-grid gating")
        calibration = load_top_rate_calibration()
        panel, eti_medians = load_panel(gate.included_models)
        summary_rows = build_summary_rows(
            panel,
            eti_medians,
            policybench,
            calibration,
        )
        correlation_rows = build_correlation_rows(summary_rows)
        sensitivity_rows = build_sensitivity_rows(summary_rows)
        country_rows = build_country_comparison_rows(summary_rows)
        posthoc_rows = build_posthoc_consistency_rows(summary_rows)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _write_csv(RESULTS / "correlates-model-summary.csv", summary_rows)
    _write_csv(RESULTS / "correlates-spearman.csv", correlation_rows)
    _write_csv(RESULTS / "correlates-sensitivity.csv", sensitivity_rows)
    _write_csv(RESULTS / "correlates-posthoc.csv", posthoc_rows)
    if country_rows:
        _write_csv(RESULTS / "correlates-country.csv", country_rows)
        print("\nUS-lab vs Chinese-lab medians (exploratory; see disclosure):")
        for row in country_rows:
            print(
                f"  {str(row['outcome']):42s} "
                f"us={float(row['us_median']):+.3f} (n={row['us_n']}) "
                f"cn={float(row['china_median']):+.3f} (n={row['china_n']}) "
                f"diff={float(row['china_minus_us']):+.3f} "
                f"p={float(row['permutation_p']):.3f}"
            )

    print(f"Wrote {len(summary_rows)} model summaries")
    print("\nSpearman with PolicyBench score:")
    for row in correlation_rows:
        print(
            f"  {str(row['outcome']):72s} n={int(row['n_models']):2d} "
            f"rho={float(row['spearman_rho']):+.3f} "
            f"raw_p={float(row['raw_p']):.3f} "
            f"holm={float(row['holm_adjusted_p']):.3f} "
            f"bh={float(row['bh_adjusted_p']):.3f}"
        )

    tau_groups: dict[str, list[float]] = defaultdict(list)
    for row in summary_rows:
        tau_groups[str(row["organization"])].append(float(row["tau_star_pct"]))
    groups = list(tau_groups.values())
    if len(groups) >= 2:
        h, p = kruskal_wallis(groups)
        print(
            f"\nKruskal-Wallis, tau* by organization "
            f"({len(groups)} groups): H={h:.2f} p={p:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
