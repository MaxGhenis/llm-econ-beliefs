"""Build paper-ready tables from the current elasticity panel."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist, mean, median


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

MAIN_COMPARISON_PATH = RESULTS_DIR / "elasticity-all-model-comparison.csv"
INCOME_DELTA_PATH = RESULTS_DIR / "income-elasticity-signfix-delta.csv"
FULLTOOLS_ARCHIVE_COMMIT = "791b8e392be5c010a74c22b29b731fc2969aaf28"
FULLTOOLS_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]
ARMINGTON_CLARIFY_STEM = "armington-clarify-batch15"
IES_CLARIFY_STEM = "ies-clarify-batch15"
TOP_RATE_PARETO_A = 1.5
TOP_RATE_CRRA_GAMMA = 1.0
TOP_RATE_PARETO_PERCENTILE = 0.99
FLAT_TAX_FRONTIER_DISPLAY_RATES = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
POLICYENGINE_US_REPO = Path("/Users/maxghenis/PolicyEngine/policyengine-us")
POLICYENGINE_US_PYTHON = POLICYENGINE_US_REPO / ".venv" / "bin" / "python"
NORMAL = NormalDist()

sys.path.insert(0, str(REPO_ROOT))
from llm_econ_beliefs.aggregate import _make_transform, aggregate_beliefs  # noqa: E402
from llm_econ_beliefs.distributions import (  # noqa: E402
    distribution_from_belief_estimate,
    mixture_distribution,
)
from llm_econ_beliefs.models import BeliefEstimate  # noqa: E402
from llm_econ_beliefs.registry import get_quantity  # noqa: E402


MODEL_LABELS = {
    "claude-opus-4.7": "Claude Opus 4.7",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "claude-haiku-4.5": "Claude Haiku 4.5",
    "grok-4.20": "Grok 4.20",
    "grok-4.1-fast": "Grok 4.1 Fast",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
}

LEGACY_QUANTITY_LABELS = {
    "labor_supply.policy_response.income_elasticity": (
        "Income elasticity of labor supply in a tax-benefit simulation (legacy)"
    ),
}

SIMULATION_QUANTITY_PREFIX = "labor_supply.policy_response.substitution_elasticity."

LABOR_TAX_QUANTITY_IDS = {
    "labor_supply.extensive_margin.single_mothers",
    "labor_supply.frisch_elasticity.prime_age",
    "labor_supply.income_elasticity.prime_age",
    "labor_supply.marshallian_wage_elasticity.prime_age",
    "tax.capital_gains_realizations.elasticity",
    "tax.elasticity_of_taxable_income.top_earners",
}

MACRO_TRADE_QUANTITY_IDS = {
    "household.intertemporal_elasticity_of_substitution",
    "production.capital_labor_substitution",
    "trade.armington_elasticity.import_domestic",
}

ROUGH_LITERATURE_BENCHMARKS = {
    "labor_supply.income_elasticity.prime_age": {
        "lower": -0.15,
        "upper": -0.05,
        "source": "CBO 2012; Blundell and MaCurdy 1999; Imbens, Rubin, and Sacerdote 2001",
    },
    "labor_supply.frisch_elasticity.prime_age": {
        "lower": 0.25,
        "upper": 0.75,
        "source": "CBO 2012; Peterman 2016; lifecycle and macro-calibration literature",
    },
    "labor_supply.marshallian_wage_elasticity.prime_age": {
        "lower": 0.05,
        "upper": 0.30,
        "source": "CBO 2012; Blundell and MaCurdy 1999",
    },
    "labor_supply.extensive_margin.single_mothers": {
        "lower": 0.30,
        "upper": 1.00,
        "source": "Eissa and Liebman 1996; extensive-margin labor-supply literature",
    },
    "tax.capital_gains_realizations.elasticity": {
        "lower": -1.00,
        "upper": -0.20,
        "source": "Capital-gains realizations literature; medium-run tax-rate elasticity convention",
    },
    "tax.elasticity_of_taxable_income.top_earners": {
        "lower": 0.25,
        "upper": 0.50,
        "source": "Saez, Slemrod, and Giertz 2012; optimal-tax sufficient-statistics literature",
    },
}


@dataclass(frozen=True)
class ComparisonRow:
    model_name: str
    quantity_id: str
    n_successful_runs: int
    pooled_point_estimate: float
    pooled_lower: float
    pooled_upper: float
    pooled_width: float
    cost_per_run_usd: float | None
    source_dir: str


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = read_comparison_rows(MAIN_COMPARISON_PATH)
    income_delta_rows = read_csv(INCOME_DELTA_PATH) if INCOME_DELTA_PATH.exists() else []

    canonical_rows = [
        row for row in comparison_rows if quantity_panel(row.quantity_id) == "canonical"
    ]
    labor_tax_rows = [
        row for row in canonical_rows if row.quantity_id in LABOR_TAX_QUANTITY_IDS
    ]
    macro_trade_rows = [
        row for row in canonical_rows if row.quantity_id in MACRO_TRADE_QUANTITY_IDS
    ]
    simulation_rows = [
        row for row in comparison_rows if quantity_panel(row.quantity_id) == "simulation"
    ]

    model_overview = build_model_overview_table(canonical_rows)
    quantity_disagreement = build_quantity_disagreement_table(canonical_rows)
    labor_tax_overview = build_model_overview_table(labor_tax_rows)
    macro_trade_overview = build_model_overview_table(macro_trade_rows)
    labor_tax_benchmarks = build_benchmark_table(labor_tax_rows)
    top_rate_calibration = estimate_policyengine_top_tail_pareto_parameter()
    top_rate_mapping = build_top_rate_table(
        labor_tax_rows,
        pareto_parameter=top_rate_calibration["a"],
    )
    flat_tax_frontier = build_flat_tax_demogrant_table(
        estimate_policyengine_flat_tax_demogrant_frontier()
    )
    pooling_robustness = build_pooling_robustness_table()
    leave_one_provider_out = build_leave_one_provider_out_table(
        labor_tax_rows=labor_tax_rows,
        macro_trade_rows=macro_trade_rows,
    )
    quantile_rule_robustness = build_quantile_rule_robustness_table(canonical_rows)
    simulation_model_overview = build_model_overview_table(simulation_rows)
    simulation_quantity_disagreement = build_quantity_disagreement_table(simulation_rows)
    income_delta = build_income_delta_table(income_delta_rows)
    armington_delta = build_armington_delta_table()
    ies_delta = build_ies_delta_table()
    stability = build_stability_table(canonical_rows)
    tool_use = build_tool_use_table()
    grok_failures = build_grok_failure_table()

    write_table_bundle(
        stem="model-overview",
        rows=model_overview,
        note=(
            "Canonical elasticity subpanel only: 9 quantities x 11 models x 15 runs. "
            "This table excludes the 12 simulation-facing PolicyEngine substitution-response coefficients "
            "and the legacy deleted income-response row. The canonical income elasticity here is still the "
            "main-panel version; the later sign-clarified rerun is reported separately."
        ),
    )
    write_table_bundle(
        stem="quantity-disagreement",
        rows=quantity_disagreement,
        note=(
            "Canonical elasticity subpanel only, sorted by cross-model spread in pooled point estimates. "
            "Simulation-facing PolicyEngine coefficients are broken out in separate appendix-style tables."
        ),
    )
    write_table_bundle(
        stem="model-overview-labor-tax",
        rows=labor_tax_overview,
        note=(
            "Canonical labor-and-tax subpanel only: 6 quantities x 11 models x 15 runs. "
            "Average ranks are computed within quantity using the absolute value of the pooled point estimate."
        ),
    )
    write_table_bundle(
        stem="model-overview-macro-trade",
        rows=macro_trade_overview,
        note=(
            "Canonical macro-and-trade subpanel only: 3 quantities x 11 models x 15 runs. "
            "Average ranks are computed within quantity using the absolute value of the pooled point estimate."
        ),
    )
    write_table_bundle(
        stem="benchmark-comparison-labor-tax",
        rows=labor_tax_benchmarks,
        note=(
            "Rough review-based benchmark intervals for the canonical labor-and-tax subpanel. "
            "These intervals are hand-coded literature anchors rather than benchmark truths."
        ),
    )
    if math.isfinite(top_rate_calibration["threshold"]) and math.isfinite(
        top_rate_calibration["mean_above"]
    ):
        top_rate_note = (
            "Toy public-finance mapping from each model's pooled ETI distribution to a utilitarian-log optimal top marginal tax rate "
            "under the Saez-style formula tau* = (1 - g_bar) / (1 - g_bar + a e), using log utility (gamma = 1) and a Pareto parameter "
            f"a = {top_rate_calibration['a']:.3f} estimated from the weighted top {100 * (1 - top_rate_calibration['percentile']):.0f}% "
            f"tax-unit AGI tail in PolicyEngine's Enhanced CPS 2024 (threshold ${top_rate_calibration['threshold']:,.0f}, "
            f"tail mean ${top_rate_calibration['mean_above']:,.0f}). Under those assumptions, "
            f"g_bar = a / (a + gamma) = {top_rate_calibration['welfare_weight']:.3f}. "
            "ETI is truncated below at zero for this policy mapping."
        )
    else:
        top_rate_note = (
            "Toy public-finance mapping from each model's pooled ETI distribution to a utilitarian-log optimal top marginal tax rate "
            "under the Saez-style formula tau* = (1 - g_bar) / (1 - g_bar + a e), using log utility (gamma = 1) and fallback Pareto parameter "
            f"a = {top_rate_calibration['a']:.3f}. Under those assumptions, "
            f"g_bar = a / (a + gamma) = {top_rate_calibration['welfare_weight']:.3f}. "
            "ETI is truncated below at zero for this policy mapping."
        )

    write_table_bundle(
        stem="toy-top-rate-labor-tax",
        rows=top_rate_mapping,
        note=top_rate_note,
    )
    if flat_tax_frontier:
        write_table_bundle(
            stem="flat-tax-demogrant-appendix",
            rows=flat_tax_frontier,
            note=(
                "Static PolicyEngine benchmark on Enhanced CPS 2024 microdata. Each row applies a flat tax to the "
                "current positive-AGI base and rebates the revenue as an equal per-person demogrant using tax-unit size. "
                "This is a distributional frontier, not a behavioral or leisure-adjusted optimal-tax exercise. Mean per-person "
                "resources are mechanically constant across rows, so the informative objects are the demogrant, the distributional "
                "quantiles, and the Gini of per-person post-tax positive-AGI resources."
            ),
        )
    write_table_bundle(
        stem="model-overview-simulation",
        rows=simulation_model_overview,
        note=(
            "Simulation-facing subpanel only: 12 PolicyEngine-style substitution-response coefficients "
            "used in a U.S. tax-benefit microsimulation. These rows are reported separately from the "
            "canonical elasticity panel because they are implementation-facing response parameters rather "
            "than standard elasticity objects."
        ),
    )
    write_table_bundle(
        stem="quantity-disagreement-simulation",
        rows=simulation_quantity_disagreement,
        note=(
            "Simulation-facing PolicyEngine substitution-response coefficients only, sorted by cross-model "
            "spread in pooled point estimates."
        ),
    )
    if income_delta:
        write_table_bundle(
            stem="income-signfix-delta",
            rows=income_delta,
            note=(
                "Change in the canonical prime-age income elasticity after adding an explicit sign-convention clarification to the prompt."
            ),
        )
    if armington_delta:
        write_table_bundle(
            stem="armington-clarify-delta",
            rows=armington_delta,
            note=(
                "Change in the Armington elasticity after clarifying that the target is the top-level import-versus-domestic elasticity, not source-country or sector-level substitution."
            ),
        )
    if ies_delta:
        write_table_bundle(
            stem="ies-clarify-delta",
            rows=ies_delta,
            note=(
                "Full-panel follow-up across all 11 models. Change in the intertemporal elasticity of substitution after clarifying that the target is the annual macro-calibration IES for nondurable consumption, not a generic inverse-CRRA or asset-pricing object."
            ),
        )
    if tool_use:
        write_table_bundle(
            stem="tool-use-appendix",
            rows=tool_use,
            note=(
                "Archived GPT-only robustness arm with full web and code-interpreter access on the earlier 8-quantity panel "
                "(8 quantities x 15 runs x 3 GPT models = 360 requests). In realized behavior, tool uptake was rare and "
                "code interpreter was never used."
            ),
        )
    if stability:
        write_table_bundle(
            stem="stability-appendix",
            rows=stability,
            note=(
                "Prefix stability on the 9-quantity canonical subpanel. "
                "Rows compare pooled summaries from the first 5 or 10 runs in a cell to the full 15-run pooled summary."
            ),
        )
    if pooling_robustness:
        write_table_bundle(
            stem="pooling-robustness-appendix",
            rows=pooling_robustness,
            note=(
                "Canonical nine-quantity subpanel only. Average predictive-uncertainty ranks are computed under the headline pooled mixture interval, the REML predictive interval, and the Bayesian predictive interval."
            ),
        )
    if leave_one_provider_out:
        write_table_bundle(
            stem="leave-one-provider-out-appendix",
            rows=leave_one_provider_out,
            note=(
                "Leave-one-provider-out sensitivity of the average absolute-elasticity ranking on the labor-and-tax and macro-and-trade canonical subpanels."
            ),
        )
    if quantile_rule_robustness:
        write_table_bundle(
            stem="quantile-rule-appendix",
            rows=quantile_rule_robustness,
            note=(
                "Canonical nine-quantity subpanel only. The alternative rule approximates each run as a transformed normal calibrated to p05, p50, and p95 instead of the headline piecewise-uniform quantile-bin reconstruction."
            ),
        )
    if grok_failures:
        write_table_bundle(
            stem="grok-failures-appendix",
            rows=grok_failures,
            note=(
                "Quantity-level failure counts for Grok 4.20 in the main 22-quantity no-tools panel. "
                "All failures in this table come from the same structured-output error."
            ),
        )

    cap_gains_convention = build_cap_gains_convention_audit_table(comparison_rows)
    if cap_gains_convention:
        write_table_bundle(
            stem="cap-gains-convention-audit",
            rows=cap_gains_convention,
            note=(
                "Both capital-gains-realizations conventions elicited independently under prompt v4. "
                "Under the identity epsilon_taxrate = -(tau / (1 - tau)) * epsilon_netoftax, the "
                "observed ratio epsilon_taxrate / epsilon_netoftax pins down an implied tau; a model "
                "that answers consistently across conventions implies a positive epsilon_netoftax, a "
                "negative epsilon_taxrate, and an implied tau in the plausible U.S. top-bracket "
                "long-term-capital-gains range. The last column flags each cell as 'in band' (implied "
                "tau between 0.15 and 0.30), 'plausible sign, outside band' (implied tau in (0, 1) but "
                "outside that window), or 'out of band' (implied tau non-positive or > 1)."
            ),
        )

    print(f"Wrote tables to {TABLES_DIR}")
    return 0


def build_model_overview_table(
    comparison_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    if not comparison_rows:
        return []
    rows_by_model: dict[str, list[ComparisonRow]] = {}
    rows_by_quantity: dict[str, list[ComparisonRow]] = {}
    for row in comparison_rows:
        rows_by_model.setdefault(row.model_name, []).append(row)
        rows_by_quantity.setdefault(row.quantity_id, []).append(row)

    responsiveness_ranks: dict[tuple[str, str], float] = {}
    confidence_ranks: dict[tuple[str, str], float] = {}
    for quantity_id, quantity_rows in rows_by_quantity.items():
        abs_points = {row.model_name: abs(row.pooled_point_estimate) for row in quantity_rows}
        widths = {row.model_name: row.pooled_width for row in quantity_rows}
        for model_name, rank in average_ranks(abs_points, reverse=True).items():
            responsiveness_ranks[(model_name, quantity_id)] = rank
        for model_name, rank in average_ranks(widths, reverse=False).items():
            confidence_ranks[(model_name, quantity_id)] = rank

    table_rows: list[dict[str, object]] = []
    for model_name, model_rows in rows_by_model.items():
        total_successful_runs = sum(row.n_successful_runs for row in model_rows)
        total_attempted_runs = len(model_rows) * 15
        total_estimated_cost = sum(
            (row.cost_per_run_usd or 0.0) * row.n_successful_runs for row in model_rows
        )
        table_rows.append(
            {
                "Model": model_label(model_name),
                "Provider": provider_label(model_name),
                "Avg abs-elasticity rank (1=highest)": round(
                    mean(
                        responsiveness_ranks[(model_name, row.quantity_id)]
                        for row in model_rows
                    ),
                    2,
                ),
                "Avg predictive-uncertainty rank (1=narrowest)": round(
                    mean(
                        confidence_ranks[(model_name, row.quantity_id)]
                        for row in model_rows
                    ),
                    2,
                ),
                "Mean absolute pooled center": round(
                    mean(abs(row.pooled_point_estimate) for row in model_rows), 3
                ),
                "Mean pooled 90% width": round(
                    mean(row.pooled_width for row in model_rows), 3
                ),
                "Success rate": round(
                    100 * total_successful_runs / total_attempted_runs, 1
                ),
                "Cost / successful run": round(
                    total_estimated_cost / total_successful_runs, 4
                ),
            }
        )

    table_rows.sort(
        key=lambda row: (
            float(row["Avg abs-elasticity rank (1=highest)"]),
            float(row["Avg predictive-uncertainty rank (1=narrowest)"]),
            str(row["Model"]),
        )
    )
    return table_rows


def build_quantity_disagreement_table(
    comparison_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    if not comparison_rows:
        return []
    rows_by_quantity: dict[str, list[ComparisonRow]] = {}
    for row in comparison_rows:
        rows_by_quantity.setdefault(row.quantity_id, []).append(row)

    table_rows: list[dict[str, object]] = []
    for quantity_id, quantity_rows in rows_by_quantity.items():
        low_row = min(quantity_rows, key=lambda row: row.pooled_point_estimate)
        high_row = max(quantity_rows, key=lambda row: row.pooled_point_estimate)
        spread = high_row.pooled_point_estimate - low_row.pooled_point_estimate
        table_rows.append(
            {
                "Quantity": quantity_label(quantity_id),
                "Lowest model": model_label(low_row.model_name),
                "Lowest center": round(low_row.pooled_point_estimate, 3),
                "Highest model": model_label(high_row.model_name),
                "Highest center": round(high_row.pooled_point_estimate, 3),
                "Spread": round(spread, 3),
                "Mean pooled 90% width": round(
                    mean(row.pooled_width for row in quantity_rows), 3
                ),
                "Spread / mean width": round(
                    spread / mean(row.pooled_width for row in quantity_rows),
                    3,
                ),
            }
        )

    table_rows.sort(
        key=lambda row: (
            -float(row["Spread"]),
            str(row["Quantity"]),
        )
    )
    return table_rows


def build_benchmark_table(
    comparison_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    rows_by_quantity: dict[str, list[ComparisonRow]] = {}
    for row in comparison_rows:
        rows_by_quantity.setdefault(row.quantity_id, []).append(row)

    table_rows: list[dict[str, object]] = []
    for quantity_id in LABOR_TAX_QUANTITY_IDS:
        if quantity_id not in rows_by_quantity or quantity_id not in ROUGH_LITERATURE_BENCHMARKS:
            continue
        benchmark = ROUGH_LITERATURE_BENCHMARKS[quantity_id]
        quantity_rows = rows_by_quantity[quantity_id]
        centers = [row.pooled_point_estimate for row in quantity_rows]
        lower = benchmark["lower"]
        upper = benchmark["upper"]
        in_range = sum(lower <= center <= upper for center in centers)
        table_rows.append(
            {
                "Quantity": quantity_label(quantity_id),
                "Rough review range": f"[{lower:.3g}, {upper:.3g}]",
                "Models in range": f"{in_range} / {len(quantity_rows)}",
                "Model min center": round(min(centers), 3),
                "Model max center": round(max(centers), 3),
                "Benchmark sources": benchmark["source"],
            }
        )

    table_rows.sort(key=lambda row: str(row["Quantity"]))
    return table_rows


def build_top_rate_table(
    comparison_rows: list[ComparisonRow],
    *,
    pareto_parameter: float = TOP_RATE_PARETO_A,
) -> list[dict[str, object]]:
    eti_quantity = get_quantity("tax.elasticity_of_taxable_income.top_earners")
    eti_rows = [
        row
        for row in comparison_rows
        if row.quantity_id == "tax.elasticity_of_taxable_income.top_earners"
    ]
    table_rows = []
    for row in sorted(eti_rows, key=lambda candidate: candidate.pooled_point_estimate):
        run_rows = read_csv(Path(row.source_dir) / "runs.csv")
        estimates = [
            _belief_estimate_from_row(run_row)
            for run_row in run_rows
            if run_row["parsed_ok"] == "True"
            and run_row["quantity_id"] == "tax.elasticity_of_taxable_income.top_earners"
        ]
        run_distributions = [
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
        if not run_distributions:
            continue
        eti_mixture = mixture_distribution(run_distributions)
        eti_q05 = eti_mixture.quantile(0.05)
        eti_q50 = eti_mixture.quantile(0.50)
        eti_q95 = eti_mixture.quantile(0.95)
        top_rate_median = optimal_top_rate_from_eti(
            eti_q50,
            pareto_parameter=pareto_parameter,
        )
        top_rate_lower = optimal_top_rate_from_eti(
            eti_q95,
            pareto_parameter=pareto_parameter,
        )
        top_rate_upper = optimal_top_rate_from_eti(
            eti_q05,
            pareto_parameter=pareto_parameter,
        )
        table_rows.append(
            {
                "Model": model_label(row.model_name),
                "ETI median [90%]": (
                    f"{eti_q50:.3f} "
                    f"[{eti_q05:.3f}, {eti_q95:.3f}]"
                ),
                "Utilitarian top rate median [90%]": (
                    f"{100 * top_rate_median:.1f}% "
                    f"[{100 * top_rate_lower:.1f}%, {100 * top_rate_upper:.1f}%]"
                ),
                "Top-rate 90% width (pp)": round(
                    100 * (top_rate_upper - top_rate_lower),
                    1,
                ),
            }
        )
    return table_rows


def build_flat_tax_demogrant_table(
    frontier: dict[str, object],
) -> list[dict[str, object]]:
    rows = frontier.get("rows")
    if not isinstance(rows, list) or not rows:
        return []

    table_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rate = float(row["rate"])
        table_rows.append(
            {
                "Flat tax rate": f"{100 * rate:.0f}%",
                "Demogrant per person": f"${float(row['demogrant_per_person']):,.0f}",
                "P10 post-tax resources": f"${float(row['p10_resources']):,.0f}",
                "Median post-tax resources": f"${float(row['p50_resources']):,.0f}",
                "P90 post-tax resources": f"${float(row['p90_resources']):,.0f}",
                "Gini": round(float(row["gini"]), 3),
            }
        )
    return table_rows


def build_income_delta_table(
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    if not rows:
        return []

    table_rows = [
        {
            "Model": model_label(row["model_name"]),
            "Old center": round(float(row["old_pooled_point_estimate"]), 3),
            "Sign-fixed center": round(float(row["new_pooled_point_estimate"]), 3),
            "Change": round(float(row["point_estimate_change"]), 3),
            "New pooled 90% interval": row["new_pooled_90_interval"],
            "Cost / successful run": round(float(row["new_cost_per_run_usd"]), 4),
        }
        for row in rows
    ]
    table_rows.sort(key=lambda row: str(row["Model"]))
    return table_rows


def build_tool_use_table() -> list[dict[str, object]]:
    table_rows: list[dict[str, object]] = []
    total_requests = 0
    total_web_requests = 0
    total_web_calls = 0
    total_code_requests = 0
    total_code_calls = 0

    for model_name in FULLTOOLS_MODELS:
        try:
            rows = read_archived_csv(
                FULLTOOLS_ARCHIVE_COMMIT,
                f"results/{model_name}-elasticities-fulltools15-v1/requests.csv",
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

        request_count = len(rows)
        web_requests = sum(as_int(row["web_search_call_count"]) > 0 for row in rows)
        code_requests = sum(
            as_int(row["code_interpreter_call_count"]) > 0 for row in rows
        )
        web_calls = sum(as_int(row["web_search_call_count"]) for row in rows)
        code_calls = sum(as_int(row["code_interpreter_call_count"]) for row in rows)

        total_requests += request_count
        total_web_requests += web_requests
        total_web_calls += web_calls
        total_code_requests += code_requests
        total_code_calls += code_calls

        table_rows.append(
            {
                "Model": model_label(model_name),
                "Requests": request_count,
                "Requests with web use": web_requests,
                "Share with web use": round(100 * web_requests / request_count, 1),
                "Total web calls": web_calls,
                "Requests with code use": code_requests,
                "Share with code use": round(100 * code_requests / request_count, 1),
                "Total code calls": code_calls,
            }
        )

    table_rows.append(
        {
            "Model": "All GPT models",
            "Requests": total_requests,
            "Requests with web use": total_web_requests,
            "Share with web use": round(100 * total_web_requests / total_requests, 1),
            "Total web calls": total_web_calls,
            "Requests with code use": total_code_requests,
            "Share with code use": round(100 * total_code_requests / total_requests, 1),
            "Total code calls": total_code_calls,
        }
    )
    return table_rows


def build_pooling_robustness_table() -> list[dict[str, object]]:
    rows = read_csv(MAIN_COMPARISON_PATH)
    canonical_rows = [
        row for row in rows if quantity_panel(row["quantity_id"]) == "canonical"
    ]
    methods = {
        "pooled": "pooled_90_interval",
        "reml": "reml_predictive_90_interval",
        "bayes": "bayes_predictive_90_interval",
    }
    by_quantity: dict[str, list[dict[str, str]]] = {}
    for row in canonical_rows:
        by_quantity.setdefault(row["quantity_id"], []).append(row)

    ranks_by_method: dict[str, dict[str, list[float]]] = {
        method: {} for method in methods
    }
    for rows_for_quantity in by_quantity.values():
        for method, column in methods.items():
            widths = {}
            for row in rows_for_quantity:
                lower, upper = parse_interval(row[column])
                widths[row["model_name"]] = upper - lower
            for model_name, rank in average_ranks(widths, reverse=False).items():
                ranks_by_method[method].setdefault(model_name, []).append(rank)

    table_rows = []
    for model_name in sorted(ranks_by_method["pooled"], key=lambda name: model_label(name)):
        pooled_rank = mean(ranks_by_method["pooled"][model_name])
        reml_rank = mean(ranks_by_method["reml"][model_name])
        bayes_rank = mean(ranks_by_method["bayes"][model_name])
        table_rows.append(
            {
                "Model": model_label(model_name),
                "Avg pooled rank": round(pooled_rank, 2),
                "Avg REML rank": round(reml_rank, 2),
                "Avg Bayes rank": round(bayes_rank, 2),
                "Max rank spread": round(
                    max(pooled_rank, reml_rank, bayes_rank)
                    - min(pooled_rank, reml_rank, bayes_rank),
                    2,
                ),
            }
        )
    table_rows.sort(key=lambda row: float(row["Avg pooled rank"]))
    return table_rows


def build_leave_one_provider_out_table(
    *,
    labor_tax_rows: list[ComparisonRow],
    macro_trade_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    table_rows: list[dict[str, object]] = []
    for panel_name, rows in (
        ("Labor/tax", labor_tax_rows),
        ("Macro/trade", macro_trade_rows),
    ):
        if not rows:
            continue
        full_ranks = _average_abs_rank_by_model(rows)
        providers = sorted({provider_label(row.model_name) for row in rows})
        for omitted_provider in providers:
            retained_rows = [
                row for row in rows if provider_label(row.model_name) != omitted_provider
            ]
            if not retained_rows:
                continue
            leave_ranks = _average_abs_rank_by_model(retained_rows)
            retained_models = sorted(leave_ranks)
            full_subset = {model_name: full_ranks[model_name] for model_name in retained_models}
            top_full = min(retained_models, key=lambda model_name: full_subset[model_name])
            top_leave = min(retained_models, key=lambda model_name: leave_ranks[model_name])
            max_shift = max(
                abs(full_subset[model_name] - leave_ranks[model_name])
                for model_name in retained_models
            )
            table_rows.append(
                {
                    "Subpanel": panel_name,
                    "Omitted provider": omitted_provider,
                    "Spearman rho": round(
                        spearman_rank_correlation(full_subset, leave_ranks),
                        3,
                    ),
                    "Top retained model, full panel": model_label(top_full),
                    "Top retained model, leave-out": model_label(top_leave),
                    "Max avg-rank shift": round(max_shift, 3),
                }
            )
    return table_rows


def build_quantile_rule_robustness_table(
    comparison_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    if not comparison_rows:
        return []

    current_widths: dict[tuple[str, str], float] = {}
    alt_widths: dict[tuple[str, str], float] = {}
    runs_cache: dict[str, list[dict[str, str]]] = {}

    for row in comparison_rows:
        current_widths[(row.model_name, row.quantity_id)] = row.pooled_width
        if row.source_dir not in runs_cache:
            runs_cache[row.source_dir] = read_csv(Path(row.source_dir) / "runs.csv")
        run_rows = [
            run_row
            for run_row in runs_cache[row.source_dir]
            if run_row["parsed_ok"] == "True" and run_row["quantity_id"] == row.quantity_id
        ]
        quantity = get_quantity(row.quantity_id)
        alt_interval = transformed_normal_mixture_interval(
            run_rows,
            lower_support=quantity.lower_support,
            upper_support=quantity.upper_support,
            confidence_level=0.9,
        )
        alt_widths[(row.model_name, row.quantity_id)] = alt_interval[1] - alt_interval[0]

    current_ranks = _average_width_rank_by_model(comparison_rows, current_widths)
    alt_ranks = _average_width_rank_by_model(comparison_rows, alt_widths)

    table_rows = []
    for model_name in sorted(current_ranks, key=lambda name: current_ranks[name]):
        table_rows.append(
            {
                "Model": model_label(model_name),
                "Avg piecewise-uniform rank": round(current_ranks[model_name], 2),
                "Avg transformed-normal rank": round(alt_ranks[model_name], 2),
                "Rank shift": round(alt_ranks[model_name] - current_ranks[model_name], 2),
            }
        )
    return table_rows


def build_stability_table(
    comparison_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    source_dirs = sorted({row.source_dir for row in comparison_rows})
    point_diffs: dict[int, list[float]] = {5: [], 10: []}
    width_diffs: dict[int, list[float]] = {5: [], 10: []}
    cell_counts: dict[int, int] = {5: 0, 10: 0}

    for source_dir in source_dirs:
        rows = read_csv(Path(source_dir) / "runs.csv")
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            if row["parsed_ok"] != "True" or quantity_panel(row["quantity_id"]) != "canonical":
                continue
            grouped.setdefault((row["model_name"], row["quantity_id"]), []).append(row)

        for (_, quantity_id), grouped_rows in grouped.items():
            grouped_rows.sort(key=lambda row: int(row["run_index"]))
            quantity = get_quantity(quantity_id)
            estimates = [_belief_estimate_from_row(row) for row in grouped_rows]
            full = aggregate_beliefs(
                estimates,
                lower_support=quantity.lower_support,
                upper_support=quantity.upper_support,
            )
            full_width = (full.upper_bound or 0.0) - (full.lower_bound or 0.0)
            for n in (5, 10):
                if len(estimates) < n:
                    continue
                partial = aggregate_beliefs(
                    estimates[:n],
                    lower_support=quantity.lower_support,
                    upper_support=quantity.upper_support,
                )
                partial_width = (partial.upper_bound or 0.0) - (partial.lower_bound or 0.0)
                point_diffs[n].append(abs(partial.point_estimate - full.point_estimate))
                width_diffs[n].append(abs(partial_width - full_width))
                cell_counts[n] += 1

    table_rows: list[dict[str, object]] = []
    for n in (5, 10, 15):
        if n == 15:
            table_rows.append(
                {
                    "Runs used": 15,
                    "Cells compared": sum(
                        1 for row in comparison_rows if row.quantity_id in LABOR_TAX_QUANTITY_IDS | MACRO_TRADE_QUANTITY_IDS
                    ),
                    "Median abs change in pooled center": 0.0,
                    "90th pct abs change in pooled center": 0.0,
                    "Median abs change in pooled width": 0.0,
                    "90th pct abs change in pooled width": 0.0,
                }
            )
            continue
        if not point_diffs[n]:
            continue
        table_rows.append(
            {
                "Runs used": n,
                "Cells compared": cell_counts[n],
                "Median abs change in pooled center": round(median(point_diffs[n]), 4),
                "90th pct abs change in pooled center": round(percentile(point_diffs[n], 0.9), 4),
                "Median abs change in pooled width": round(median(width_diffs[n]), 4),
                "90th pct abs change in pooled width": round(percentile(width_diffs[n], 0.9), 4),
            }
        )
    return table_rows


def build_armington_delta_table() -> list[dict[str, object]]:
    rows = read_csv(MAIN_COMPARISON_PATH)
    old_by_model = {
        row["model_name"]: row
        for row in rows
        if row["quantity_id"] == "trade.armington_elasticity.import_domestic"
    }
    table_rows = []
    for model_name, old_row in old_by_model.items():
        summary_path = RESULTS_DIR / f"{model_name}-{ARMINGTON_CLARIFY_STEM}" / "summary.csv"
        if not summary_path.exists():
            continue
        new_rows = read_csv(summary_path)
        if not new_rows:
            continue
        new_row = new_rows[0]
        table_rows.append(
            {
                "Model": model_label(model_name),
                "Old center": round(float(old_row["pooled_point_estimate"]), 3),
                "Clarified center": round(float(new_row["pooled_point_estimate"]), 3),
                "Change": round(
                    float(new_row["pooled_point_estimate"])
                    - float(old_row["pooled_point_estimate"]),
                    3,
                ),
                "Old pooled 90% interval": old_row["pooled_90_interval"],
                "Clarified pooled 90% interval": format_interval_from_summary_row(new_row),
            }
        )
    table_rows.sort(key=lambda row: str(row["Model"]))
    return table_rows


def build_ies_delta_table() -> list[dict[str, object]]:
    rows = read_csv(MAIN_COMPARISON_PATH)
    old_by_model = {
        row["model_name"]: row
        for row in rows
        if row["quantity_id"] == "household.intertemporal_elasticity_of_substitution"
    }
    table_rows = []
    for model_name, old_row in old_by_model.items():
        summary_path = RESULTS_DIR / f"{model_name}-{IES_CLARIFY_STEM}" / "summary.csv"
        if not summary_path.exists():
            continue
        new_rows = read_csv(summary_path)
        if not new_rows:
            continue
        new_row = new_rows[0]
        table_rows.append(
            {
                "Model": model_label(model_name),
                "Old center": round(float(old_row["pooled_point_estimate"]), 3),
                "Clarified center": round(float(new_row["pooled_point_estimate"]), 3),
                "Change": round(
                    float(new_row["pooled_point_estimate"])
                    - float(old_row["pooled_point_estimate"]),
                    3,
                ),
                "Old pooled 90% interval": old_row["pooled_90_interval"],
                "Clarified pooled 90% interval": format_interval_from_summary_row(new_row),
            }
        )
    table_rows.sort(key=lambda row: str(row["Model"]))
    return table_rows


def format_interval_from_summary_row(row: dict[str, str]) -> str:
    if row.get("pooled_90_interval"):
        return row["pooled_90_interval"]
    lower = float(row["pooled_lower_bound"])
    upper = float(row["pooled_upper_bound"])
    return f"[{lower:.4g}, {upper:.4g}]"


CAP_GAINS_PLAUSIBLE_TAU_RANGE = (0.15, 0.30)


def build_cap_gains_convention_audit_table(
    comparison_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    """One row per model showing both cap-gains parameterizations plus a consistency flag.

    Under the identity epsilon_taxrate = -(tau / (1 - tau)) * epsilon_netoftax,
    the observed ratio epsilon_taxrate / epsilon_netoftax implies a specific tau;
    we flag the implied tau as "in-band" when it falls inside
    CAP_GAINS_PLAUSIBLE_TAU_RANGE (a rough envelope for U.S. top-bracket
    long-term capital-gains marginal rates, federal plus state)."""
    tax_rate_rows = {
        row.model_name: row
        for row in comparison_rows
        if row.quantity_id == CAP_GAINS_TAX_RATE_ID
    }
    net_rows = {
        row.model_name: row
        for row in comparison_rows
        if row.quantity_id == CAP_GAINS_NET_OF_TAX_RATE_ID
    }

    both = sorted(set(tax_rate_rows) & set(net_rows))
    if not both:
        return []

    tau_lo, tau_hi = CAP_GAINS_PLAUSIBLE_TAU_RANGE
    table_rows: list[dict[str, object]] = []
    for model_name in both:
        eps_tax = tax_rate_rows[model_name].pooled_point_estimate
        eps_net = net_rows[model_name].pooled_point_estimate

        if eps_net == 0:
            ratio = float("nan")
            implied_tau = float("nan")
        else:
            ratio = eps_tax / eps_net
            # From epsilon_taxrate = -(tau/(1-tau)) * epsilon_netoftax,
            # tau = -ratio / (1 - ratio).
            denominator = 1.0 - ratio
            implied_tau = (
                -ratio / denominator if denominator != 0 else float("nan")
            )

        if math.isfinite(implied_tau) and tau_lo <= implied_tau <= tau_hi:
            band_flag = "in band"
        elif math.isfinite(implied_tau) and 0.0 < implied_tau < 1.0:
            band_flag = "plausible sign, outside band"
        else:
            band_flag = "out of band"

        if eps_tax <= 0 and eps_net >= 0:
            consistency = "as expected (tax<0, net>0)"
        elif eps_tax >= 0 and eps_net >= 0:
            consistency = "both positive"
        elif eps_tax <= 0 and eps_net <= 0:
            consistency = "both negative"
        else:
            consistency = "reversed (tax>0, net<0)"

        table_rows.append(
            {
                "Model": model_label(model_name),
                "Provider": provider_label(model_name),
                "epsilon w.r.t. tax rate": round(eps_tax, 3),
                "epsilon w.r.t. net-of-tax rate": round(eps_net, 3),
                "Implied tau": (
                    round(implied_tau, 3) if math.isfinite(implied_tau) else "—"
                ),
                "Consistency": consistency,
                f"In plausible tau band [{tau_lo:.2f}, {tau_hi:.2f}]": band_flag,
            }
        )

    table_rows.sort(key=lambda row: (row["Consistency"], str(row["Model"])))
    return table_rows


def build_grok_failure_table() -> list[dict[str, object]]:
    path = RESULTS_DIR / "grok-4.20-elasticities-batch15" / "runs.csv"
    if not path.exists():
        return []

    rows = read_csv(path)
    by_quantity: dict[str, dict[str, object]] = {}
    for row in rows:
        bucket = by_quantity.setdefault(
            row["quantity_id"],
            {"successful": 0, "failed": 0, "error": ""},
        )
        if row["parsed_ok"] == "True":
            bucket["successful"] += 1
        else:
            bucket["failed"] += 1
            if not bucket["error"]:
                bucket["error"] = row["error"]

    table_rows = []
    for quantity_id, payload in sorted(
        by_quantity.items(),
        key=lambda item: (-int(item[1]["failed"]), quantity_label(item[0])),
    ):
        if int(payload["failed"]) == 0:
            continue
        table_rows.append(
            {
                "Quantity": quantity_label(quantity_id),
                "Successful runs": payload["successful"],
                "Failed runs": payload["failed"],
                "Dominant failure mode": str(payload["error"]).split("\n")[0],
            }
        )
    return table_rows


def read_comparison_rows(path: Path) -> list[ComparisonRow]:
    rows = []
    for row in read_csv(path):
        pooled_lower, pooled_upper = parse_interval(row["pooled_90_interval"])
        rows.append(
            ComparisonRow(
                model_name=row["model_name"],
                quantity_id=row["quantity_id"],
                n_successful_runs=int(row["n_successful_runs"]),
                pooled_point_estimate=float(row["pooled_point_estimate"]),
                pooled_lower=pooled_lower,
                pooled_upper=pooled_upper,
                pooled_width=pooled_upper - pooled_lower,
                cost_per_run_usd=float(row["usage_estimated_total_cost_usd_per_successful_run"])
                if row["usage_estimated_total_cost_usd_per_successful_run"]
                else None,
                source_dir=row["source_dir"],
            )
        )
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_archived_csv(commit: str, git_path: str) -> list[dict[str, str]]:
    content = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{git_path}"],
        text=True,
    )
    return list(csv.DictReader(io.StringIO(content)))


def _belief_estimate_from_row(row: dict[str, str]) -> BeliefEstimate:
    quantiles = json.loads(row["quantiles"]) if row["quantiles"] else {}
    return BeliefEstimate(
        point_estimate=float(row["point_estimate"]),
        quantity_id=row["quantity_id"],
        interpretation=row.get("interpretation") or None,
        lower_bound=float(row["lower_bound"]) if row.get("lower_bound") else None,
        upper_bound=float(row["upper_bound"]) if row.get("upper_bound") else None,
        confidence_level=(
            float(row["confidence_level"]) if row.get("confidence_level") else None
        ),
        quantiles={str(key): float(value) for key, value in quantiles.items()},
        citations=[],
        reasoning_summary=row.get("reasoning_summary") or None,
    )


def parse_interval(text: str) -> tuple[float, float]:
    match = re.fullmatch(r"\[\s*([^,]+),\s*([^\]]+)\s*\]", text.strip())
    if not match:
        raise ValueError(f"Could not parse interval: {text}")
    return float(match.group(1)), float(match.group(2))


def optimal_top_rate_from_eti(
    eti: float,
    *,
    pareto_parameter: float = TOP_RATE_PARETO_A,
    welfare_weight: float | None = None,
    crra_gamma: float = TOP_RATE_CRRA_GAMMA,
) -> float:
    if welfare_weight is None:
        welfare_weight = utilitarian_top_welfare_weight(
            pareto_parameter=pareto_parameter,
            crra_gamma=crra_gamma,
        )
    numerator = 1.0 - welfare_weight
    denominator = numerator + pareto_parameter * max(eti, 0.0)
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def estimate_policyengine_top_tail_pareto_parameter() -> dict[str, float]:
    if not POLICYENGINE_US_PYTHON.exists() or not POLICYENGINE_US_REPO.exists():
        welfare_weight = utilitarian_top_welfare_weight(
            pareto_parameter=TOP_RATE_PARETO_A,
            crra_gamma=TOP_RATE_CRRA_GAMMA,
        )
        return {
            "a": TOP_RATE_PARETO_A,
            "percentile": TOP_RATE_PARETO_PERCENTILE,
            "threshold": float("nan"),
            "mean_above": float("nan"),
            "welfare_weight": welfare_weight,
        }

    script = f"""
import json
import numpy as np
from policyengine_us import Microsimulation

sim = Microsimulation()
agi = sim.calc("adjusted_gross_income")
weights = agi.weights.to_numpy()
values = agi.values.astype(float)
mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (values > 0)
values = values[mask]
weights = weights[mask]
order = np.argsort(values)
values = values[order]
weights = weights[order]
cum = np.cumsum(weights)
total = cum[-1]
percentile = {TOP_RATE_PARETO_PERCENTILE}
cutoff = percentile * total
idx = np.searchsorted(cum, cutoff)
threshold = float(values[min(idx, len(values) - 1)])
tail = values >= threshold
mean_above = float(np.average(values[tail], weights=weights[tail]))
a = float(mean_above / (mean_above - threshold))
print(json.dumps({{
    "a": a,
    "percentile": percentile,
    "threshold": threshold,
    "mean_above": mean_above,
}}))
"""
    try:
        completed = subprocess.run(
            [str(POLICYENGINE_US_PYTHON), "-c", script],
            cwd=POLICYENGINE_US_REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(completed.stdout.strip())
        result["welfare_weight"] = utilitarian_top_welfare_weight(
            pareto_parameter=float(result["a"]),
            crra_gamma=TOP_RATE_CRRA_GAMMA,
        )
        return result
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        welfare_weight = utilitarian_top_welfare_weight(
            pareto_parameter=TOP_RATE_PARETO_A,
            crra_gamma=TOP_RATE_CRRA_GAMMA,
        )
        return {
            "a": TOP_RATE_PARETO_A,
            "percentile": TOP_RATE_PARETO_PERCENTILE,
            "threshold": float("nan"),
            "mean_above": float("nan"),
            "welfare_weight": welfare_weight,
        }


def estimate_policyengine_flat_tax_demogrant_frontier() -> dict[str, object]:
    if not POLICYENGINE_US_PYTHON.exists() or not POLICYENGINE_US_REPO.exists():
        return {"rows": []}

    display_rates = ", ".join(repr(rate) for rate in FLAT_TAX_FRONTIER_DISPLAY_RATES)
    script = f"""
import json
import numpy as np
from policyengine_us import Microsimulation

def weighted_quantile(values, weights, probability):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = probability * cumulative[-1]
    index = np.searchsorted(cumulative, cutoff, side="left")
    return float(values[min(index, len(values) - 1)])

def weighted_gini(values, weights):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    weighted_values = values * weights
    cumulative = np.cumsum(weighted_values)
    previous = np.concatenate(([0.0], cumulative[:-1]))
    total_value = float(cumulative[-1])
    total_weight = float(np.sum(weights))
    if total_value <= 0 or total_weight <= 0:
        return float("nan")
    return float(1.0 - np.sum(weights * (previous + cumulative)) / (total_weight * total_value))

sim = Microsimulation()
agi = sim.calc("positive_agi")
size = sim.calc("tax_unit_size")
values = agi.values.astype(float)
weights = agi.weights.to_numpy()
sizes = size.values.astype(float)
mask = (
    np.isfinite(values)
    & np.isfinite(weights)
    & np.isfinite(sizes)
    & (weights > 0)
    & (sizes > 0)
)
values = values[mask]
weights = weights[mask]
sizes = sizes[mask]
person_weights = weights * sizes
total_people = float(np.sum(person_weights))

rows = []
for rate in [{display_rates}]:
    tax = rate * values
    demogrant = float(np.sum(weights * tax) / total_people)
    resources = (values - tax) / sizes + demogrant
    rows.append({{
        "rate": float(rate),
        "demogrant_per_person": demogrant,
        "p10_resources": weighted_quantile(resources, person_weights, 0.10),
        "p50_resources": weighted_quantile(resources, person_weights, 0.50),
        "p90_resources": weighted_quantile(resources, person_weights, 0.90),
        "gini": weighted_gini(resources, person_weights),
    }})

print(json.dumps({{
    "tax_base": "positive_agi",
    "rows": rows,
}}))
"""
    try:
        completed = subprocess.run(
            [str(POLICYENGINE_US_PYTHON), "-c", script],
            cwd=POLICYENGINE_US_REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(completed.stdout.strip())
        if not isinstance(result.get("rows"), list):
            return {"rows": []}
        return result
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        return {"rows": []}


def utilitarian_top_welfare_weight(
    *,
    pareto_parameter: float = TOP_RATE_PARETO_A,
    crra_gamma: float = TOP_RATE_CRRA_GAMMA,
) -> float:
    if pareto_parameter <= 0:
        raise ValueError("Pareto parameter must be positive")
    if crra_gamma < 0:
        raise ValueError("CRRA gamma must be nonnegative")
    return pareto_parameter / (pareto_parameter + crra_gamma)


def as_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take percentile of empty sequence")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(probability * len(ordered)) - 1))
    return ordered[index]


def spearman_rank_correlation(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return 1.0

    left_values = [left[key] for key in keys]
    right_values = [right[key] for key in keys]
    left_mean = mean(left_values)
    right_mean = mean(right_values)

    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    left_denominator = math.sqrt(
        sum((left_value - left_mean) ** 2 for left_value in left_values)
    )
    right_denominator = math.sqrt(
        sum((right_value - right_mean) ** 2 for right_value in right_values)
    )
    if left_denominator == 0 or right_denominator == 0:
        return 1.0
    return numerator / (left_denominator * right_denominator)


def average_ranks(values: dict[str, float], *, reverse: bool) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
    ranks: dict[str, float] = {}
    position = 1
    index = 0
    while index < len(ordered):
        tied = [ordered[index]]
        while (
            index + len(tied) < len(ordered)
            and ordered[index + len(tied)][1] == tied[0][1]
        ):
            tied.append(ordered[index + len(tied)])
        average_rank = (position + position + len(tied) - 1) / 2
        for key, _ in tied:
            ranks[key] = average_rank
        position += len(tied)
        index += len(tied)
    return ranks


def _average_abs_rank_by_model(
    comparison_rows: list[ComparisonRow],
) -> dict[str, float]:
    rows_by_quantity: dict[str, list[ComparisonRow]] = {}
    for row in comparison_rows:
        rows_by_quantity.setdefault(row.quantity_id, []).append(row)

    by_model: dict[str, list[float]] = {}
    for quantity_rows in rows_by_quantity.values():
        abs_points = {row.model_name: abs(row.pooled_point_estimate) for row in quantity_rows}
        for model_name, rank in average_ranks(abs_points, reverse=True).items():
            by_model.setdefault(model_name, []).append(rank)
    return {model_name: mean(ranks) for model_name, ranks in by_model.items()}


def _average_width_rank_by_model(
    comparison_rows: list[ComparisonRow],
    widths: dict[tuple[str, str], float],
) -> dict[str, float]:
    rows_by_quantity: dict[str, list[ComparisonRow]] = {}
    for row in comparison_rows:
        rows_by_quantity.setdefault(row.quantity_id, []).append(row)

    by_model: dict[str, list[float]] = {}
    for quantity_id, quantity_rows in rows_by_quantity.items():
        widths_for_quantity = {
            row.model_name: widths[(row.model_name, quantity_id)] for row in quantity_rows
        }
        for model_name, rank in average_ranks(widths_for_quantity, reverse=False).items():
            by_model.setdefault(model_name, []).append(rank)
    return {model_name: mean(ranks) for model_name, ranks in by_model.items()}


def transformed_normal_mixture_interval(
    run_rows: list[dict[str, str]],
    *,
    lower_support: float | None,
    upper_support: float | None,
    confidence_level: float,
) -> tuple[float, float]:
    transform = _make_transform(lower_support, upper_support)
    z_95 = NORMAL.inv_cdf(0.95)
    components: list[tuple[float, float]] = []
    lower_candidates: list[float] = []
    upper_candidates: list[float] = []

    for run_row in run_rows:
        quantiles = json.loads(run_row["quantiles"]) if run_row.get("quantiles") else {}
        if not {"p05", "p50", "p95"} <= set(quantiles):
            continue
        mu = transform.forward(float(quantiles["p50"]))
        z05 = transform.forward(float(quantiles["p05"]))
        z95 = transform.forward(float(quantiles["p95"]))
        sigma = max(abs(z95 - z05) / (2.0 * z_95), 1e-9)
        components.append((mu, sigma))
        lower_candidates.append(transform.inverse(mu + NORMAL.inv_cdf(1e-4) * sigma))
        upper_candidates.append(transform.inverse(mu + NORMAL.inv_cdf(1 - 1e-4) * sigma))

    if not components:
        raise ValueError("No quantile-complete runs available for transformed-normal mixture")

    search_lower = lower_support if lower_support is not None else min(lower_candidates)
    search_upper = upper_support if upper_support is not None else max(upper_candidates)

    def mixture_cdf(value: float) -> float:
        if lower_support is not None and value <= lower_support:
            return 0.0
        if upper_support is not None and value >= upper_support:
            return 1.0
        z = transform.forward(value)
        return mean(
            NORMAL.cdf((z - mu) / sigma)
            for mu, sigma in components
        )

    lower_probability = (1.0 - confidence_level) / 2.0
    upper_probability = 1.0 - lower_probability
    return (
        bisect_quantile(mixture_cdf, lower_probability, search_lower, search_upper),
        bisect_quantile(mixture_cdf, upper_probability, search_lower, search_upper),
    )


def bisect_quantile(
    cdf,
    probability: float,
    lower: float,
    upper: float,
    *,
    iterations: int = 80,
) -> float:
    lo = lower
    hi = upper
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if cdf(mid) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def provider_label(model_name: str) -> str:
    if model_name.startswith("claude-"):
        return "Anthropic"
    if model_name.startswith("gemini-"):
        return "Google"
    if model_name.startswith("gpt-"):
        return "OpenAI"
    if model_name.startswith("grok-"):
        return "xAI"
    return "Other"


def quantity_label(quantity_id: str) -> str:
    if quantity_id in LEGACY_QUANTITY_LABELS:
        return LEGACY_QUANTITY_LABELS[quantity_id]
    try:
        return get_quantity(quantity_id).name
    except KeyError:
        return quantity_id


CAP_GAINS_TAX_RATE_ID = "tax.capital_gains_realizations.elasticity"
CAP_GAINS_NET_OF_TAX_RATE_ID = "tax.capital_gains_realizations.elasticity.net_of_tax_rate"


def quantity_panel(quantity_id: str) -> str:
    if quantity_id in LEGACY_QUANTITY_LABELS:
        return "legacy"
    if quantity_id.startswith(SIMULATION_QUANTITY_PREFIX):
        return "simulation"
    if quantity_id == CAP_GAINS_NET_OF_TAX_RATE_ID:
        return "convention_sibling"
    return "canonical"


def write_table_bundle(stem: str, rows: list[dict[str, object]], note: str) -> None:
    if not rows:
        return
    csv_path = TABLES_DIR / f"{stem}.csv"
    md_path = TABLES_DIR / f"{stem}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, note=note)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], *, note: str) -> None:
    headers = list(rows[0].keys())
    lines = [f"Note: {note}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [format_markdown_cell(header, row[header]) for header in headers]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def format_markdown_cell(header: str, value: object) -> str:
    if isinstance(value, (int, float)):
        if header == "Success rate":
            return f"{float(value):.1f}%"
        if header.startswith("Share with "):
            return f"{float(value):.1f}%"
        if "top rate" in header.lower():
            return f"{float(value):.1f}%"
        if "Cost /" in header:
            return f"${float(value):.4f}"
        if value == int(value):
            return str(int(value))
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
