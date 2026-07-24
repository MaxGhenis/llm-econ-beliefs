"""Build paper-ready tables from the current elasticity panel."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
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
TOP_RATE_CALIBRATION_PATH = RESULTS_DIR / "top-rate-calibration.json"
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
_POLICYENGINE_US_REPO_DEFAULT = "/Users/maxghenis/PolicyEngine/policyengine-us"
POLICYENGINE_US_REPO = Path(
    os.environ.get("POLICYENGINE_US_REPO", _POLICYENGINE_US_REPO_DEFAULT)
)
_POLICYENGINE_US_PYTHON_DEFAULT = str(POLICYENGINE_US_REPO / ".venv" / "bin" / "python")
POLICYENGINE_US_PYTHON = Path(
    os.environ.get("POLICYENGINE_US_PYTHON", _POLICYENGINE_US_PYTHON_DEFAULT)
)
if not POLICYENGINE_US_PYTHON.exists() or not POLICYENGINE_US_REPO.exists():
    print(
        "warning: PolicyEngine-US repo or venv not found at "
        f"POLICYENGINE_US_REPO={POLICYENGINE_US_REPO} "
        f"POLICYENGINE_US_PYTHON={POLICYENGINE_US_PYTHON}; "
        "PolicyEngine-backed tables will use fallback constants and display NaN "
        "for microdata-derived fields. Set POLICYENGINE_US_REPO and "
        "POLICYENGINE_US_PYTHON to override.",
        file=sys.stderr,
    )
NORMAL = NormalDist()

sys.path.insert(0, str(REPO_ROOT))
from llm_econ_beliefs.aggregate import _make_transform, aggregate_beliefs  # noqa: E402
from llm_econ_beliefs.distributions import (  # noqa: E402
    distribution_from_belief_estimate,
    mixture_distribution,
)
from llm_econ_beliefs.models import BeliefEstimate  # noqa: E402
from llm_econ_beliefs.model_registry import (  # noqa: E402
    WAVE_DISPLAY_LABELS,
    get_panel_model,
    organization_display_label,
    serving_provider_display_label,
)
from llm_econ_beliefs.prompts import create_belief_prompt  # noqa: E402
from llm_econ_beliefs.registry import get_quantity  # noqa: E402

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
        "source": "CBO 2012; Blundell and MaCurdy 1999; Imbens, Rubin, and Sacerdote 2001 (marginal propensity to earn, converted to an elasticity)",
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
        "source": "Chetty, Guren, Manoli, and Weber 2013 (elasticity implied by Eissa and Liebman 1996); Meyer and Rosenbaum 2001",
    },
    "tax.capital_gains_realizations.elasticity": {
        "lower": -1.00,
        "upper": -0.20,
        "source": "Dowd, McClelland, and Muthitacharoen 2015; Burman and Randolph 1994; CBO/JCT medium-run convention",
    },
    "tax.elasticity_of_taxable_income.top_earners": {
        "lower": 0.25,
        "upper": 0.50,
        "source": "Gruber and Saez 2002 (high-income estimates); Saez, Slemrod, and Giertz 2012 (survey range 0.12-0.40, upper half)",
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

    quantity_disagreement = build_quantity_disagreement_table(canonical_rows)
    labor_tax_overview = build_model_overview_table(labor_tax_rows)
    macro_trade_overview = build_model_overview_table(macro_trade_rows)
    labor_tax_benchmarks = build_benchmark_table(labor_tax_rows)
    top_rate_calibration = estimate_policyengine_top_tail_pareto_parameter()
    if math.isfinite(top_rate_calibration["threshold"]):
        write_top_rate_calibration(top_rate_calibration)
    else:
        print(
            "WARNING: fallback calibration in effect — leaving the committed "
            "results/top-rate-calibration.json untouched so build_correlates "
            "keeps the microdata values.",
            file=sys.stderr,
        )
    top_rate_mapping = build_top_rate_table(
        labor_tax_rows,
        pareto_parameter=top_rate_calibration["a"],
    )
    top_rate_robustness = build_top_rate_robustness_table(
        labor_tax_rows,
        baseline_a=top_rate_calibration["a"],
    )
    flat_tax_frontier = build_flat_tax_demogrant_table(
        estimate_policyengine_flat_tax_demogrant_frontier()
    )
    pooling_robustness = build_pooling_robustness_table()
    leave_one_organization_out = build_leave_one_organization_out_table(
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

    canonical_quantity_count = len({row.quantity_id for row in canonical_rows})
    panel_model_count = len({row.model_name for row in canonical_rows})
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
            f"Canonical labor-and-tax subpanel only: {len(LABOR_TAX_QUANTITY_IDS)} quantities x {panel_model_count} models x 15 runs. "
            "Average ranks are computed within quantity using the absolute value of the pooled point estimate."
        ),
    )
    write_table_bundle(
        stem="model-overview-macro-trade",
        rows=macro_trade_overview,
        note=(
            f"Canonical macro-and-trade subpanel only: {len(MACRO_TRADE_QUANTITY_IDS)} quantities x {panel_model_count} models x 15 runs. "
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
            "Toy public-finance mapping from each model's pooled ETI distribution to an optimal top marginal tax rate "
            "under the Saez top-bracket formula tau* = (1 - g_bar) / (1 - g_bar + a e), with a Pareto parameter "
            f"a = {top_rate_calibration['a']:.3f} estimated from the weighted top {100 * (1 - top_rate_calibration['percentile']):.0f}% "
            f"tax-unit AGI tail in PolicyEngine's certified microdata (threshold ${top_rate_calibration['threshold']:,.0f}, "
            f"tail mean ${top_rate_calibration['mean_above']:,.0f}). "
            f"The welfare weight g_bar = a / (a + gamma) = {top_rate_calibration['welfare_weight']:.3f} is the average marginal "
            "utility of top-bracket earners under CRRA (gamma = 1) utility and a Pareto(a) income tail, normalized to the marginal "
            "utility of the earner at the top-bracket threshold. It is a threshold-normalized weight, not the population-normalized "
            "utilitarian weight, which would drive g_bar toward zero; the Revenue-max column reports that g_bar -> 0 "
            "(Diamond-Saez revenue-maximizing) benchmark tau* = 1 / (1 + a e) at the ETI median. "
            "ETI is truncated below at zero for this policy mapping."
        )
    else:
        top_rate_note = (
            "Toy public-finance mapping from each model's pooled ETI distribution to an optimal top marginal tax rate "
            "under the Saez top-bracket formula tau* = (1 - g_bar) / (1 - g_bar + a e), using the FALLBACK Pareto parameter "
            f"a = {top_rate_calibration['a']:.3f} because the PolicyEngine microdata calibration was unavailable at build time "
            "(the build prints a warning when this path is used; the text of the paper assumes the microdata calibration). "
            f"The welfare weight g_bar = a / (a + gamma) = {top_rate_calibration['welfare_weight']:.3f} is the average marginal "
            "utility of top-bracket earners under CRRA (gamma = 1) utility and a Pareto(a) income tail, normalized to the marginal "
            "utility of the earner at the top-bracket threshold; the Revenue-max column reports the g_bar -> 0 "
            "(Diamond-Saez revenue-maximizing) benchmark tau* = 1 / (1 + a e) at the ETI median. "
            "ETI is truncated below at zero for this policy mapping."
        )

    write_table_bundle(
        stem="toy-top-rate-labor-tax",
        rows=top_rate_mapping,
        note=top_rate_note,
    )
    if top_rate_robustness:
        baseline_a_display = top_rate_calibration["a"]
        write_table_bundle(
            stem="top-rate-robustness",
            rows=top_rate_robustness,
            note=(
                "Robustness of the utilitarian optimal top-rate mapping in Table 4 to the Pareto tail "
                f"parameter a and the CRRA coefficient gamma. Each cell is the median implied optimal top rate tau* = "
                "(1 - g_bar) / (1 - g_bar + a e) computed at the model's pooled ETI median under the "
                f"(a, gamma) pair in the column header, where g_bar = a / (a + gamma). The baseline column "
                f"(a = {baseline_a_display:.3f}, gamma = 1) reproduces the median column in Table 4. The "
                "a = 1.3 / 1.5 / 1.7 columns vary only the Pareto tail while keeping log utility (gamma = 1); "
                f"the final column replaces log utility with CRRA at gamma = 2 while holding a at the "
                f"baseline value above. Under log utility the corresponding welfare weights g_bar are "
                f"1.3/2.3 = 0.565, {baseline_a_display:.3f}/(1 + {baseline_a_display:.3f}) = "
                f"{baseline_a_display / (1 + baseline_a_display):.3f}, 1.5/2.5 = 0.600, and 1.7/2.7 = 0.630; "
                f"under gamma = 2 with a = {baseline_a_display:.3f}, g_bar = "
                f"{baseline_a_display:.3f}/({baseline_a_display:.3f} + 2) = "
                f"{baseline_a_display / (baseline_a_display + 2):.3f}."
            ),
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
                f"Prefix stability on the {canonical_quantity_count}-quantity canonical subpanel. "
                "Rows compare pooled summaries from the first 5 or 10 runs in a cell to the full 15-run pooled summary."
            ),
        )
    if pooling_robustness:
        write_table_bundle(
            stem="pooling-robustness-appendix",
            rows=pooling_robustness,
            note=(
                f"Canonical {canonical_quantity_count}-quantity subpanel. Average predictive-uncertainty ranks are computed under the headline pooled mixture interval, the REML predictive interval, and the Bayesian predictive interval."
            ),
        )
    if leave_one_organization_out:
        write_table_bundle(
            stem="leave-one-organization-out-appendix",
            rows=leave_one_organization_out,
            note=(
                "Leave-one-organization-out sensitivity of the average absolute-elasticity ranking on the labor-and-tax and macro-and-trade canonical subpanels."
            ),
        )
        # Keep the existing include path populated until the separate prose pass
        # changes paper.qmd; its table contents and labels are organization-based.
        write_table_bundle(
            stem="leave-one-provider-out-appendix",
            rows=leave_one_organization_out,
            note=(
                "Leave-one-organization-out sensitivity of the average absolute-elasticity ranking on the labor-and-tax and macro-and-trade canonical subpanels."
            ),
        )
    if quantile_rule_robustness:
        write_table_bundle(
            stem="quantile-rule-appendix",
            rows=quantile_rule_robustness,
            note=(
                f"Canonical {canonical_quantity_count}-quantity subpanel. The alternative rule approximates each run as a transformed normal calibrated to p05, p50, and p95 instead of the headline piecewise-uniform quantile-bin reconstruction."
            ),
        )

    write_table_bundle(
        stem="harness-disclosure",
        rows=build_harness_disclosure_table(),
        note=(
            "Per-model generation-harness configuration. The prompt text and repeated-run design are identical "
            "across models; the structured-output mechanism, completion budget, sampling regime, and reasoning "
            "configuration follow each provider's API surface and are therefore confounded with model identity. "
            "Completion budgets are truncation guards: reasoning tokens count against them on models that reason, "
            "so budgets were raised where required to avoid truncation. Identifiers marked alias float with "
            "provider updates; dated snapshots are pinned."
        ),
    )
    write_table_bundle(
        stem="variance-decomposition",
        rows=build_variance_decomposition_table(canonical_rows),
        note=(
            f"Split of the pooled predictive variance over the canonical {canonical_quantity_count}-quantity subpanel into the "
            "within-run component (the model's own stated p05-p95 quantiles) and the between-run component "
            "(variation of run means across the 15 repeated draws). The between-run share is the only component "
            "affected by the sampling regime, which differs for the three no-sampling-parameter Claude models."
        ),
    )
    write_table_bundle(
        stem="resampling-stability",
        rows=build_resampling_stability_table(canonical_rows),
        note=(
            f"Monte Carlo uncertainty in the pooled summaries from resampling the 15 runs within each canonical cell "
            "(200 bootstrap resamples, fixed seed). Center MC SE is the standard error of the pooled center; relative "
            "width MC SE is the standard error of the pooled 90% width divided by its mean. The final columns show the "
            "distribution of each model's average width rank across resamples; narrow intervals indicate the "
            "predictive-uncertainty ordering is stable to run-level resampling at R = 15."
        ),
    )
    write_table_bundle(
        stem="support-bounds",
        rows=build_support_bounds_table(),
        note=(
            "Registry support bounds used for quantile-bin reconstruction, tail extrapolation, and the transforms "
            "behind the REML and Bayesian estimators. Unbounded sides use a tail-extrapolation rule that extends "
            "0.25 x the adjacent inter-quantile gap beyond the elicited p05/p95."
        ),
    )
    mechanism_ablation = build_mechanism_ablation_table(canonical_rows)
    if mechanism_ablation:
        write_table_bundle(
            stem="mechanism-ablation",
            rows=mechanism_ablation,
            note=(
                "Same model (Claude Opus 4.7), same v4 prompts, same repeated-run design, two harness mechanisms: "
                "the April LiteLLM forced-function-call path (temperature 1.0, 1200-token budget) versus the July "
                "native Anthropic strict-JSON-schema path (no sampling parameters, 32000-token budget). Each cell "
                "pools 15 fresh runs elicited in July 2026 under the native mechanism against the April panel cell. "
                "Centers are stable (max absolute change 0.03); widths move from -15 to +7 percent across quantities with no consistent direction. "
                "Opus 4.7 runs without extended reasoning on both paths, so the reasoning-mode axis is not covered."
            ),
        )
    wording_rows, wording_tau_rows = build_wording_comparison_tables()
    write_table_bundle(
        stem="wording-comparison",
        rows=wording_rows,
        note=(
            "Same four models (the April premium tier), same quantities, same repeated-run design, two v4 "
            "clarifier wordings: the superseded April 19 elicitation (original wording — plain conditionals "
            "with the conventional direction first; the net-of-tax sibling's definition line states the "
            "conversion identity backwards) versus the published April 21 re-elicitation (revised wording — "
            "symmetric if-and-only-if clauses, worked magnitude example, corrected identity). April 19 runs "
            f"are read from git history (commit {WORDING_ARCHIVE_COMMIT[:8]}); every cell's prompt text is "
            "byte-verified against the original wording preserved in the seven holdout models' committed "
            "archives and against the current prompt builder's revised wording. Each cell pools 15 runs with "
            "the headline piecewise-uniform mixture. The comparison is not a pure wording experiment: the "
            "April 21 re-elicitation also moved to the per-quantity fallback harness that added request "
            "logging, and two days elapsed between elicitations, so wording is confounded with harness path "
            "and time."
        ),
    )
    write_table_bundle(
        stem="wording-comparison-tau",
        rows=wording_tau_rows,
        note=(
            "The Appendix A9 implied-tau bootstrap (1,000 draws, fixed seed, tau = -rho / (1 - rho) with "
            "rho = epsilon_taxrate / epsilon_netoftax) applied separately to each wording's 15 tax-rate and "
            "15 net-of-tax-rate runs for the four models elicited under both v4 clarifier wordings. Bands as "
            "in Appendix Table A9: LTCG-rate [0.15, 0.37], ordinary-income-rate (0.37, 0.55]. All eight "
            "model-wording cells are sign-consistent (tax-rate median < 0 < net-of-tax median), so the "
            "identity's premise holds throughout. See the Table A18 note for the harness-path confound."
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

    correlates_summary, correlates_spearman, policybench_release = (
        build_correlates_tables()
    )
    if correlates_summary:
        write_table_bundle(
            stem="correlates-model-summary",
            rows=correlates_summary,
            note=(
                "Per-model summary joining the elicitation panel to the published PolicyBench "
                f"leaderboard ({policybench_release}, US, no-tools, household-weighted within-$1 rate — the leaderboard headline). Width rank averages "
                "tie-averaged 90 percent interval-width ranks across the canonical panel "
                "(1 = tightest). Implied top rates come from Table 4's threshold-normalized "
                "log-utility mapping. Dashes mark models outside the PolicyBench panel."
            ),
        )
    if correlates_spearman:
        write_table_bundle(
            stem="policybench-correlates",
            rows=correlates_spearman,
            note=(
                "Spearman rank correlations between the PolicyBench within-$1 rate "
                f"({policybench_release}, US, no-tools) and per-model elicitation outcomes, "
                "with raw two-sided permutation p-values (20,000 draws, fixed seed; exact when "
                "n <= 8) plus Holm and Benjamini-Hochberg adjustments across the eight "
                "non-derived tests spanning both predictors. The top-rate row is the same "
                "ETI hypothesis under a monotone transformation, not an additional test. "
                "Descriptive: models differ across every axis at once, so these are "
                "cross-family associations, not causal effects."
            ),
        )

    country_rows = build_country_table()
    if country_rows:
        write_table_bundle(
            stem="correlates-country",
            rows=country_rows,
            note=(
                "US-lab versus Chinese-lab medians with exact group-label "
                "permutation p-values on the difference in medians, and Holm "
                "and Benjamini-Hochberg adjustment over the four-outcome "
                "family (* = the top-rate row is a derived transform of the "
                "ETI row and mirrors its adjusted values rather than "
                "entering the family). Exploratory: lab country is perfectly "
                "confounded with the serving path (every Chinese-lab model "
                "ran through OpenRouter JSON mode) and with elicitation "
                "wave, and co-varies with completion budget and reasoning "
                "configuration; prompts are English-language, and the group "
                "sizes (5 versus 20) put a floor near 0.02-0.05 on "
                "attainable p-values."
            ),
        )

    cap_gains_convention = build_cap_gains_convention_audit_table(comparison_rows)
    if cap_gains_convention:
        write_table_bundle(
            stem="cap-gains-convention-audit",
            rows=cap_gains_convention,
            note=(
                "Both capital-gains-realizations conventions elicited independently under prompt v4. "
                "Under the identity epsilon_taxrate = -(tau / (1 - tau)) * epsilon_netoftax, any "
                "model whose two answers are jointly consistent with a single tau prior implies one "
                "specific tau. The implied-tau column reports the median and 90 percent interval of "
                "1000 bootstrap draws that independently sample one tax-rate run and one "
                "net-of-tax-rate run from the 15 runs of each; this is not the plug-in ratio of "
                "medians, which differs from the bootstrap median because tau = -rho / (1 - rho) is "
                "nonlinear in rho (Jensen's inequality). The band flag is computed against the "
                "bootstrap median. 'LTCG-rate consistent' marks an implied median tau in [0.15, 0.37] "
                "(U.S. top-bracket long-term-capital-gains envelope, federal LTCG plus NIIT plus high "
                "state); 'ordinary-income-rate consistent' marks (0.37, 0.55] (federal ordinary-income "
                "top plus high state); 'plausible sign, outside bands' marks (0, 1) outside both "
                "windows; and 'out of band' marks a non-positive or > 1 median tau. The shared-tau "
                "coherence column flags the joint sign pattern of the two pooled medians: a model with "
                "(tax < 0, net > 0) is sign-consistent with the identity, whereas (both positive, both "
                "negative, or reversed) indicates that the two answers are not jointly consistent with "
                "a single tau prior — either because the model holds two independent literature "
                "anchors, or because one convention was answered with the opposite sign."
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
        costs_are_tracked = all(
            row.cost_per_run_usd is not None for row in model_rows
        )
        cost_per_successful_run: float | str = "—"
        if costs_are_tracked and total_successful_runs:
            total_estimated_cost = sum(
                row.cost_per_run_usd * row.n_successful_runs
                for row in model_rows
                if row.cost_per_run_usd is not None
            )
            cost_per_successful_run = round(
                total_estimated_cost / total_successful_runs, 4
            )
        table_rows.append(
            {
                "Model": model_label(model_name),
                "Organization": organization_label(model_name),
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
                "Cost / successful run": cost_per_successful_run,
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
    eti_rows = [
        row
        for row in comparison_rows
        if row.quantity_id == "tax.elasticity_of_taxable_income.top_earners"
    ]
    table_rows = []
    for row in sorted(eti_rows, key=lambda candidate: candidate.pooled_point_estimate):
        run_rows = read_csv(Path(row.source_dir) / "runs.csv")
        eti_quantiles = pooled_eti_quantiles_from_csv_rows(run_rows)
        if eti_quantiles is None:
            continue
        eti_q05, eti_q50, eti_q95 = eti_quantiles
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
        revenue_max_median = optimal_top_rate_from_eti(
            eti_q50,
            pareto_parameter=pareto_parameter,
            welfare_weight=0.0,
        )
        table_rows.append(
            {
                "Model": model_label(row.model_name),
                "ETI median [90%]": (
                    f"{eti_q50:.3f} "
                    f"[{eti_q05:.3f}, {eti_q95:.3f}]"
                ),
                "Top rate median [90%]": (
                    f"{100 * top_rate_median:.1f}% "
                    f"[{100 * top_rate_lower:.1f}%, {100 * top_rate_upper:.1f}%]"
                ),
                "Revenue-max median": f"{100 * revenue_max_median:.1f}%",
                "Top-rate 90% width (pp)": round(
                    100 * (top_rate_upper - top_rate_lower),
                    1,
                ),
            }
        )
    return table_rows


def build_top_rate_robustness_table(
    comparison_rows: list[ComparisonRow],
    *,
    baseline_a: float,
) -> list[dict[str, object]]:
    """Median implied top rate under alternative (Pareto a, CRRA gamma) pairs.

    Columns cover the baseline (a=baseline_a, gamma=1), three alternative Pareto
    parameters under log utility, and a CRRA gamma=2 column at baseline_a. The
    ETI median feeding each row is the same pooled-mixture median used in Table 4,
    so cross-column differences reflect only the formula's (a, gamma) pair.
    """
    eti_rows = [
        row
        for row in comparison_rows
        if row.quantity_id == "tax.elasticity_of_taxable_income.top_earners"
    ]

    parameterizations: list[tuple[str, float, float]] = [
        (f"Baseline top rate (a={baseline_a:.3f}, gamma=1)", baseline_a, 1.0),
        ("Top rate (a=1.3, gamma=1)", 1.3, 1.0),
        ("Top rate (a=1.5, gamma=1)", 1.5, 1.0),
        ("Top rate (a=1.7, gamma=1)", 1.7, 1.0),
        (f"Top rate (a={baseline_a:.3f}, gamma=2)", baseline_a, 2.0),
    ]

    table_rows: list[dict[str, object]] = []
    for row in eti_rows:
        run_rows = read_csv(Path(row.source_dir) / "runs.csv")
        eti_quantiles = pooled_eti_quantiles_from_csv_rows(run_rows)
        if eti_quantiles is None:
            continue
        _, eti_q50, _ = eti_quantiles

        out_row: dict[str, object] = {
            "Model": model_label(row.model_name),
            "ETI median": round(eti_q50, 3),
        }
        for label, a, gamma in parameterizations:
            tau_star = optimal_top_rate_from_eti(
                eti_q50,
                pareto_parameter=a,
                crra_gamma=gamma,
            )
            out_row[label] = round(100 * tau_star, 1)
        table_rows.append(out_row)

    baseline_column = f"Baseline top rate (a={baseline_a:.3f}, gamma=1)"
    table_rows.sort(
        key=lambda entry: (-float(entry[baseline_column]), str(entry["Model"]))
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
            "Cost / successful run": (
                round(float(row["new_cost_per_run_usd"]), 4)
                if row.get("new_cost_per_run_usd") not in (None, "")
                else "—"
            ),
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


def build_leave_one_organization_out_table(
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
        organizations = sorted({organization_label(row.model_name) for row in rows})
        for omitted_organization in organizations:
            retained_rows = [
                row
                for row in rows
                if organization_label(row.model_name) != omitted_organization
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
                    "Omitted organization": omitted_organization,
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
    point_diffs: dict[int, list[float]] = {5: [], 10: [], 15: []}
    width_diffs: dict[int, list[float]] = {5: [], 10: [], 15: []}
    cell_counts: dict[int, int] = {5: 0, 10: 0, 15: 0}

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
            for n in (5, 10, 15):
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


CAP_GAINS_LTCG_TAU_RANGE = (0.15, 0.37)
CAP_GAINS_ORDINARY_INCOME_TAU_RANGE = (0.37, 0.55)
CAP_GAINS_BOOTSTRAP_DRAWS = 1000
CAP_GAINS_BOOTSTRAP_SEED = 0


def _load_cap_gains_run_points(
    source_dir: str,
) -> tuple[list[float], list[float]]:
    """Return per-run point estimates for both cap-gains conventions.

    Reads the model's ``runs.jsonl`` and keeps only rows with
    ``parsed_ok=True`` and a finite ``point_estimate``.
    """
    runs_path = REPO_ROOT / source_dir / "runs.jsonl"
    tax_rate_points: list[float] = []
    net_points: list[float] = []
    if not runs_path.exists():
        return tax_rate_points, net_points
    with runs_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("parsed_ok"):
                continue
            qid = record.get("quantity_id")
            point = record.get("point_estimate")
            if point is None:
                continue
            point_value = float(point)
            if not math.isfinite(point_value):
                continue
            if qid == CAP_GAINS_TAX_RATE_ID:
                tax_rate_points.append(point_value)
            elif qid == CAP_GAINS_NET_OF_TAX_RATE_ID:
                net_points.append(point_value)
    return tax_rate_points, net_points


def _bootstrap_implied_tau_quantiles(
    tax_rate_points: list[float],
    net_points: list[float],
    *,
    n_draws: int = CAP_GAINS_BOOTSTRAP_DRAWS,
    seed: int = CAP_GAINS_BOOTSTRAP_SEED,
) -> tuple[float, float, float, float]:
    """Return (p05, p50, p95, share_in_unit_interval) of bootstrapped implied-tau draws.

    For each of ``n_draws`` iterations, sample one ``epsilon_taxrate`` and
    one ``epsilon_netoftax`` independently from the per-run lists, then
    invert ``epsilon_taxrate = -(tau / (1 - tau)) * epsilon_netoftax`` to
    ``tau = -rho / (1 - rho)`` where ``rho = epsilon_taxrate / epsilon_netoftax``.
    Pairs yielding a non-finite ``tau`` (``epsilon_netoftax == 0`` or
    ``rho == 1``) are dropped. NaN values are returned if fewer than two
    finite draws survive. The final element is the share of surviving draws
    that fall in the economically meaningful interval (0, 1); a low share
    signals a pole-straddling, weakly identified implied-tau distribution.
    """
    if not tax_rate_points or not net_points:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    taus: list[float] = []
    n_tax = len(tax_rate_points)
    n_net = len(net_points)
    for _ in range(n_draws):
        eps_tax = tax_rate_points[rng.randrange(n_tax)]
        eps_net = net_points[rng.randrange(n_net)]
        if eps_net == 0:
            continue
        rho = eps_tax / eps_net
        denominator = 1.0 - rho
        if denominator == 0:
            continue
        tau = -rho / denominator
        if math.isfinite(tau):
            taus.append(tau)
    if len(taus) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    taus.sort()
    share_in_unit = sum(1 for tau in taus if 0.0 < tau < 1.0) / len(taus)
    return (
        _bootstrap_quantile(taus, 0.05),
        _bootstrap_quantile(taus, 0.50),
        _bootstrap_quantile(taus, 0.95),
        share_in_unit,
    )


def _bootstrap_quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        (1 - weight) * sorted_values[lower_index]
        + weight * sorted_values[upper_index]
    )


def build_cap_gains_convention_audit_table(
    comparison_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    """One row per model showing both cap-gains parameterizations plus a
    bootstrap implied-tau column with uncertainty and a shared-tau coherence flag.

    Under the identity ``epsilon_taxrate = -(tau / (1 - tau)) * epsilon_netoftax``,
    a model whose two answers are jointly consistent with a single ``tau`` prior
    implies one specific ``tau``. We bootstrap the implied-tau distribution by
    drawing 1000 independent ``(epsilon_taxrate_i, epsilon_netoftax_j)`` pairs
    from each model's 15 tax-rate runs and 15 net-of-tax-rate runs, inverting
    the identity per pair, and reporting the median and 90 percent interval.
    The band flag is computed against the median implied ``tau``: ``[0.15, 0.37]``
    covers the U.S. top-bracket LTCG marginal rate envelope (federal LTCG plus
    NIIT plus the highest state layer) and ``[0.37, 0.55]`` covers the
    ordinary-income top-rate envelope (federal ordinary top plus state).
    """
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

    ltcg_lo, ltcg_hi = CAP_GAINS_LTCG_TAU_RANGE
    ord_lo, ord_hi = CAP_GAINS_ORDINARY_INCOME_TAU_RANGE
    band_column = (
        f"Band (LTCG [{ltcg_lo:.2f}, {ltcg_hi:.2f}], "
        f"ordinary-income [{ord_lo:.2f}, {ord_hi:.2f}])"
    )
    table_rows: list[dict[str, object]] = []
    for model_name in both:
        source_dir = tax_rate_rows[model_name].source_dir
        tax_rate_points, net_points = _load_cap_gains_run_points(source_dir)
        if not tax_rate_points or not net_points:
            continue

        eps_tax_median = median(tax_rate_points)
        eps_net_median = median(net_points)

        if eps_tax_median <= 0 and eps_net_median >= 0:
            coherence = "sign-consistent (tax<0, net>0)"
        elif eps_tax_median >= 0 and eps_net_median >= 0:
            coherence = "both positive"
        elif eps_tax_median <= 0 and eps_net_median <= 0:
            coherence = "both negative"
        else:
            coherence = "reversed (tax>0, net<0)"
        sign_consistent = coherence.startswith("sign-consistent")

        tau_p05, tau_p50, tau_p95, tau_share_unit = _bootstrap_implied_tau_quantiles(
            tax_rate_points, net_points
        )

        # The identity's premise (opposite signs) fails for non-sign-consistent
        # rows, so no implied tau is defined for them; and a bootstrap
        # distribution with substantial mass outside (0, 1) straddles the
        # rho = 1 pole, so its median is not a stable summary.
        if not sign_consistent:
            implied_tau_cell = "— (premise fails)"
            share_cell = "—"
            band_flag = "not identified"
        elif not math.isfinite(tau_p50):
            implied_tau_cell = "—"
            share_cell = "—"
            band_flag = "not identified"
        elif tau_share_unit < 0.8:
            implied_tau_cell = (
                f"{tau_p50:.3f} [{tau_p05:.3f}, {tau_p95:.3f}]"
            )
            share_cell = f"{100 * tau_share_unit:.0f}%"
            band_flag = "uninformative (pole-straddling)"
        else:
            implied_tau_cell = (
                f"{tau_p50:.3f} [{tau_p05:.3f}, {tau_p95:.3f}]"
            )
            share_cell = f"{100 * tau_share_unit:.0f}%"
            if ltcg_lo <= tau_p50 <= ltcg_hi:
                band_flag = "LTCG-rate consistent"
            elif ord_lo < tau_p50 <= ord_hi:
                band_flag = "ordinary-income-rate consistent"
            elif 0.0 < tau_p50 < 1.0:
                band_flag = "plausible sign, outside bands"
            else:
                band_flag = "out of band"

        table_rows.append(
            {
                "Model": model_label(model_name),
                "Organization": organization_label(model_name),
                "epsilon w.r.t. tax rate (median)": round(eps_tax_median, 3),
                "epsilon w.r.t. net-of-tax rate (median)": round(
                    eps_net_median, 3
                ),
                "Implied tau median [90%]": implied_tau_cell,
                "Share of draws in (0, 1)": share_cell,
                "Shared-tau coherence": coherence,
                band_column: band_flag,
            }
        )

    table_rows.sort(
        key=lambda row: (row["Shared-tau coherence"], str(row["Model"]))
    )
    return table_rows


def build_country_table() -> list[dict[str, object]]:
    """Display rows for the US-versus-China comparison table."""
    path = RESULTS_DIR / "correlates-country.csv"
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            derived = row.get("is_derived", "").strip().lower() == "true"
            rows.append(
                {
                    "Outcome": row["outcome"],
                    "US median (n)": f"{float(row['us_median']):.3f} ({row['us_n']})",
                    "China median (n)": (
                        f"{float(row['china_median']):.3f} ({row['china_n']})"
                    ),
                    "China - US": f"{float(row['china_minus_us']):+.3f}",
                    "Permutation p": f"{float(row['permutation_p']):.3f}",
                    "Holm p": (
                        f"{float(row['holm_adjusted_p']):.3f}" + ("*" if derived else "")
                        if row.get("holm_adjusted_p")
                        else "—"
                    ),
                    "BH p": (
                        f"{float(row['bh_adjusted_p']):.3f}" + ("*" if derived else "")
                        if row.get("bh_adjusted_p")
                        else "—"
                    ),
                }
            )
    return rows


def build_correlates_tables() -> tuple[
    list[dict[str, object]], list[dict[str, object]], str
]:
    """Display rows for the PolicyBench-correlates tables.

    Reads the artifacts scripts/build_correlates.py writes; returns
    ([], [], "") when they are absent so the build degrades cleanly.
    """
    summary_path = RESULTS_DIR / "correlates-model-summary.csv"
    spearman_path = RESULTS_DIR / "correlates-spearman.csv"
    scores_path = RESULTS_DIR / "policybench-scores.csv"
    if not summary_path.exists() or not spearman_path.exists():
        return [], [], ""

    required_spearman_columns = {
        "predictor",
        "outcome",
        "n_models",
        "spearman_rho",
        "raw_p",
        "holm_adjusted_p",
        "bh_adjusted_p",
        "family_size",
        "is_derived",
    }
    with spearman_path.open(newline="") as handle:
        fieldnames = set(csv.DictReader(handle).fieldnames or ())
    missing_columns = sorted(required_spearman_columns - fieldnames)
    if missing_columns:
        print(
            "warning: ignoring stale correlates-spearman.csv; rerun "
            "scripts/build_correlates.py after calibration (missing columns: "
            + ", ".join(missing_columns)
            + ")",
            file=sys.stderr,
        )
        return [], [], ""

    release = ""
    if scores_path.exists():
        with scores_path.open() as handle:
            first = next(csv.DictReader(handle), None)
            if first:
                release = first.get("source_release", "")

    def fmt(value: str, digits: int = 2) -> str:
        if value in ("", None):
            return "—"
        return f"{float(value):.{digits}f}"

    summary_rows: list[dict[str, object]] = []
    with summary_path.open() as handle:
        for row in csv.DictReader(handle):
            metadata = get_panel_model(row["model"])
            summary_rows.append(
                {
                    "Model": metadata.display_label,
                    "Organization": organization_display_label(
                        metadata.organization
                    ),
                    "Wave": WAVE_DISPLAY_LABELS[metadata.wave],
                    "PolicyBench within-$1": fmt(row["policybench_within_dollar"], 1),
                    "ETI median": fmt(row["eti_median"], 3),
                    "Implied top rate": (
                        f"{fmt(row['tau_star_pct'], 1)}%"
                        if row["tau_star_pct"]
                        else "—"
                    ),
                    "Avg width rank": fmt(row["avg_width_rank"], 1),
                }
            )
    summary_rows.sort(
        key=lambda r: (
            r["Implied top rate"] == "—",
            -float(str(r["Implied top rate"]).rstrip("%"))
            if r["Implied top rate"] != "—"
            else 0,
        )
    )

    spearman_rows: list[dict[str, object]] = []
    with spearman_path.open() as handle:
        for row in csv.DictReader(handle):
            spearman_rows.append(
                {
                    "Predictor": row["predictor"],
                    "Outcome": row["outcome"],
                    "Models": row["n_models"],
                    "Spearman rho": fmt(row["spearman_rho"], 3),
                    "Raw permutation p": fmt(row["raw_p"], 3),
                    "Holm-adjusted p": fmt(row["holm_adjusted_p"], 3),
                    "BH-adjusted p": fmt(row["bh_adjusted_p"], 3),
                    "Family size": row["family_size"],
                    "Derived": "yes" if row["is_derived"] == "True" else "no",
                }
            )

    return summary_rows, spearman_rows, release


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


def pooled_eti_quantiles_from_csv_rows(
    run_rows: list[dict[str, str]],
) -> tuple[float, float, float] | None:
    """Apply the paper's equal-weight piecewise-uniform ETI construction."""
    eti_quantity_id = "tax.elasticity_of_taxable_income.top_earners"
    eti_quantity = get_quantity(eti_quantity_id)
    estimates = [
        _belief_estimate_from_row(run_row)
        for run_row in run_rows
        if run_row.get("parsed_ok") == "True"
        and run_row.get("quantity_id") == eti_quantity_id
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
        return None
    eti_mixture = mixture_distribution(run_distributions)
    return (
        eti_mixture.quantile(0.05),
        eti_mixture.quantile(0.50),
        eti_mixture.quantile(0.95),
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


def write_top_rate_calibration(
    calibration: dict[str, float],
    path: Path = TOP_RATE_CALIBRATION_PATH,
) -> dict[str, float | None]:
    """Write the microdata calibration used by downstream correlates."""

    def finite_or_none(value: float) -> float | None:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    payload: dict[str, float | None] = {
        "a": finite_or_none(calibration["a"]),
        "gbar": finite_or_none(calibration["welfare_weight"]),
        "threshold": finite_or_none(calibration["threshold"]),
        "tail_mean": finite_or_none(calibration["mean_above"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)
    return payload


def estimate_policyengine_top_tail_pareto_parameter() -> dict[str, float]:
    if not POLICYENGINE_US_PYTHON.exists() or not POLICYENGINE_US_REPO.exists():
        print(
            "WARNING: PolicyEngine venv not found — Table 4/A13 will use the "
            f"FALLBACK Pareto parameter a = {TOP_RATE_PARETO_A}, not the microdata "
            "calibration described in the paper text.",
            file=sys.stderr,
        )
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
            [str(POLICYENGINE_US_PYTHON), "-P", "-c", script],
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
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = " Subprocess stderr tail: " + exc.stderr.strip()[-300:]
        print(
            "WARNING: PolicyEngine microdata Pareto calibration FAILED — Table 4/A13 "
            f"will use the FALLBACK a = {TOP_RATE_PARETO_A}, not the microdata "
            f"calibration described in the paper text. ({exc.__class__.__name__}){detail}",
            file=sys.stderr,
        )
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
            [str(POLICYENGINE_US_PYTHON), "-P", "-c", script],
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
    return get_panel_model(model_name).display_label


def organization_label(model_name: str) -> str:
    model = get_panel_model(model_name)
    return organization_display_label(model.organization)


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





# ---------------------------------------------------------------------------
# Referee-round additions: harness disclosure, variance decomposition,
# resampling stability, support bounds, and the cross-mechanism ablation.
# ---------------------------------------------------------------------------

HARNESS_DISCLOSURE_ROWS: list[dict[str, str]] = [
    # Serving-provider path is registry-derived; rows hold the remaining harness fields.
    {"model": "gpt-5.5", "mechanism": "strict JSON schema", "budget": "1200 (8000 for the 40 re-elicited runs)", "sampling": "temperature 1.0, batched n <= 8", "reasoning": "provider default effort", "identifier": "gpt-5.5", "id_type": "alias"},
    {"model": "gpt-5.6-sol", "mechanism": "strict JSON schema", "budget": "8000", "sampling": "temperature 1.0, batched n <= 8", "reasoning": "provider default effort", "identifier": "gpt-5.6-sol", "id_type": "alias"},
    {"model": "gpt-5.6-luna", "mechanism": "strict JSON schema", "budget": "8000", "sampling": "temperature 1.0, batched n <= 8", "reasoning": "provider default effort", "identifier": "gpt-5.6-luna", "id_type": "alias"},
    {"model": "gpt-5.6-terra", "mechanism": "strict JSON schema", "budget": "8000", "sampling": "temperature 1.0, batched n <= 8", "reasoning": "provider default effort", "identifier": "gpt-5.6-terra", "id_type": "alias"},
    {"model": "gpt-5.4", "mechanism": "strict JSON schema", "budget": "1200", "sampling": "temperature 1.0, batched n <= 8", "reasoning": "provider default effort", "identifier": "gpt-5.4", "id_type": "alias"},
    {"model": "gpt-5.4-mini", "mechanism": "strict JSON schema", "budget": "1200", "sampling": "temperature 1.0, batched n <= 8", "reasoning": "provider default effort", "identifier": "gpt-5.4-mini", "id_type": "alias"},
    {"model": "gpt-5.4-nano", "mechanism": "strict JSON schema", "budget": "1200", "sampling": "temperature 1.0, batched n <= 8", "reasoning": "provider default effort", "identifier": "gpt-5.4-nano", "id_type": "alias"},
    {"model": "claude-fable-5", "mechanism": "strict JSON schema", "budget": "32000", "sampling": "none accepted (provider default)", "reasoning": "always-on reasoning", "identifier": "claude-fable-5", "id_type": "alias"},
    {"model": "claude-opus-4.8", "mechanism": "strict JSON schema", "budget": "32000", "sampling": "none accepted (provider default)", "reasoning": "off (provider default)", "identifier": "claude-opus-4-8", "id_type": "alias"},
    {"model": "claude-sonnet-5", "mechanism": "strict JSON schema", "budget": "32000", "sampling": "none accepted (provider default)", "reasoning": "adaptive (provider default)", "identifier": "claude-sonnet-5", "id_type": "alias"},
    {"model": "claude-opus-5", "mechanism": "strict JSON schema", "budget": "32000", "sampling": "none accepted (provider default)", "reasoning": "adaptive, on by default (provider default)", "identifier": "claude-opus-5", "id_type": "alias"},
    {"model": "claude-opus-4.7", "mechanism": "forced function call", "budget": "1200", "sampling": "temperature 1.0", "reasoning": "off (provider default)", "identifier": "claude-opus-4-7", "id_type": "alias"},
    {"model": "claude-sonnet-4.6", "mechanism": "forced function call", "budget": "1200", "sampling": "temperature 1.0", "reasoning": "off (provider default)", "identifier": "claude-sonnet-4-6", "id_type": "alias"},
    {"model": "claude-haiku-4.5", "mechanism": "forced function call", "budget": "1200", "sampling": "temperature 1.0", "reasoning": "off (provider default)", "identifier": "claude-haiku-4-5-20251001", "id_type": "dated snapshot"},
    {"model": "gemini-3.1-pro-preview", "mechanism": "forced JSON object", "budget": "1200", "sampling": "temperature 1.0", "reasoning": "provider default thinking", "identifier": "gemini-3.1-pro-preview", "id_type": "preview alias"},
    {"model": "gemini-3.5-flash", "mechanism": "forced JSON object", "budget": "4000", "sampling": "temperature 1.0", "reasoning": "provider default thinking", "identifier": "gemini-3.5-flash", "id_type": "alias"},
    {"model": "gemini-3.6-flash", "mechanism": "forced JSON object", "budget": "8000", "sampling": "temperature 1.0", "reasoning": "provider default thinking", "identifier": "gemini-3.6-flash", "id_type": "alias"},
    {"model": "gemini-3-flash-preview", "mechanism": "forced JSON object", "budget": "1200", "sampling": "temperature 1.0", "reasoning": "provider default thinking", "identifier": "gemini-3-flash-preview", "id_type": "preview alias"},
    {"model": "gemini-3.1-flash-lite-preview", "mechanism": "forced JSON object", "budget": "1200", "sampling": "temperature 1.0", "reasoning": "provider default thinking", "identifier": "gemini-3.1-flash-lite-preview", "id_type": "preview alias"},
    {"model": "grok-4.20", "mechanism": "forced function call", "budget": "1200", "sampling": "temperature 1.0", "reasoning": "reasoning variant", "identifier": "xai/grok-4.20-reasoning", "id_type": "alias"},
    {"model": "grok-4.3", "mechanism": "forced function call", "budget": "4000", "sampling": "temperature 1.0", "reasoning": "provider default", "identifier": "xai/grok-4.3", "id_type": "alias"},
    {"model": "grok-4.5", "mechanism": "forced function call", "budget": "8000", "sampling": "temperature 1.0", "reasoning": "provider default", "identifier": "xai/grok-4.5", "id_type": "alias"},
    {"model": "deepseek-v4-pro", "mechanism": "forced JSON object (schema validated locally)", "budget": "8000", "sampling": "temperature 1.0", "reasoning": "provider default", "identifier": "openrouter/deepseek/deepseek-v4-pro", "id_type": "alias"},
    {"model": "qwen-3.7-max", "mechanism": "forced JSON object (schema validated locally)", "budget": "8000", "sampling": "temperature 1.0", "reasoning": "provider default", "identifier": "openrouter/qwen/qwen3.7-max", "id_type": "alias"},
    {"model": "kimi-k2.6", "mechanism": "forced JSON object (schema validated locally)", "budget": "8000", "sampling": "temperature 1.0", "reasoning": "provider default", "identifier": "openrouter/moonshotai/kimi-k2.6", "id_type": "alias"},
    {"model": "kimi-k3", "mechanism": "forced JSON object (schema validated locally)", "budget": "8000", "sampling": "temperature 1.0", "reasoning": "provider default", "identifier": "openrouter/moonshotai/kimi-k3", "id_type": "alias"},
    {"model": "glm-5.2", "mechanism": "forced JSON object (schema validated locally)", "budget": "16000", "sampling": "temperature 1.0", "reasoning": "provider default", "identifier": "openrouter/z-ai/glm-5.2", "id_type": "alias"},
    {"model": "minimax-m3", "mechanism": "forced JSON object (schema validated locally)", "budget": "8000", "sampling": "temperature 1.0", "reasoning": "provider default", "identifier": "openrouter/minimax/minimax-m3", "id_type": "alias"},
    {"model": "grok-4.1-fast", "mechanism": "forced function call", "budget": "1200", "sampling": "temperature 1.0", "reasoning": "non-reasoning variant", "identifier": "xai/grok-4-1-fast-non-reasoning", "id_type": "alias"},
]


def _canonical_completion_budget(row: dict[str, str]) -> int:
    """Resolve the budget the harness would use for this row's serving path."""
    from llm_econ_beliefs.providers import (
        ANTHROPIC_MAX_OUTPUT_TOKENS,
        ANTHROPIC_MODEL_ALIASES,
        LITELLM_MAX_COMPLETION_TOKENS_BY_MODEL,
        OPENAI_MAX_COMPLETION_TOKENS_BY_MODEL,
    )

    model = row["model"]
    if row["mechanism"] == "strict JSON schema":
        if model in ANTHROPIC_MODEL_ALIASES:
            return ANTHROPIC_MAX_OUTPUT_TOKENS
        return OPENAI_MAX_COMPLETION_TOKENS_BY_MODEL.get(model, 1200)
    return LITELLM_MAX_COMPLETION_TOKENS_BY_MODEL.get(model, 1200)


def _validate_harness_budgets() -> None:
    """The disclosed budget must include the canonical harness budget.

    Rows may document historical protocols alongside it (gpt-5.5 discloses
    both the 1200-token main pass and the 8000-token re-elicitation), so the
    check is membership over the integers in the budget field, not equality.
    """
    for row in HARNESS_DISCLOSURE_ROWS:
        disclosed = {int(tok) for tok in re.findall(r"\d+", row["budget"])}
        canonical = _canonical_completion_budget(row)
        if canonical not in disclosed:
            raise ValueError(
                f"harness disclosure for {row['model']} lists budget "
                f"{sorted(disclosed)} but providers.py resolves {canonical}"
            )


_validate_harness_budgets()


def build_harness_disclosure_table() -> list[dict[str, object]]:
    return [
        {
            "Model": model_label(row["model"]),
            "Provider path": serving_provider_display_label(
                get_panel_model(row["model"]).serving_provider_path
            ),
            "Output mechanism": row["mechanism"],
            "Completion budget": row["budget"],
            "Sampling": row["sampling"],
            "Reasoning config": row["reasoning"],
            "API identifier": row["identifier"],
            "Identifier type": row["id_type"],
        }
        for row in HARNESS_DISCLOSURE_ROWS
    ]


def build_variance_decomposition_table(
    comparison_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    """Per-model split of pooled predictive variance into within-run (stated
    quantile) and between-run (sampling) components over the canonical panel."""
    by_model: dict[str, list[tuple[float, float]]] = {}
    for row in comparison_rows:
        summary_path = Path(row.source_dir) / "summary.csv"
        if not summary_path.exists():
            continue
        for summary_row in read_csv(summary_path):
            if summary_row["quantity_id"] != row.quantity_id:
                continue
            within = summary_row.get("within_run_sd") or ""
            between = summary_row.get("between_run_sd") or ""
            if not within or not between:
                continue
            by_model.setdefault(row.model_name, []).append(
                (float(within), float(between))
            )

    table_rows: list[dict[str, object]] = []
    for model_name in sorted(by_model, key=model_label):
        pairs = by_model[model_name]
        shares = [
            (b * b) / (w * w + b * b)
            for w, b in pairs
            if (w * w + b * b) > 0
        ]
        if not shares:
            continue
        table_rows.append(
            {
                "Model": model_label(model_name),
                "Cells": len(pairs),
                "Median within-run SD": round(median(w for w, _ in pairs), 3),
                "Median between-run SD": round(median(b for _, b in pairs), 3),
                "Median between-run variance share": f"{100 * median(shares):.0f}%",
                "Max between-run variance share": f"{100 * max(shares):.0f}%",
            }
        )
    return table_rows


def build_resampling_stability_table(
    comparison_rows: list[ComparisonRow],
    *,
    n_resamples: int = 200,
    seed: int = 20260707,
) -> list[dict[str, object]]:
    """Bootstrap the 15 runs within each canonical cell to attach Monte Carlo
    standard errors to the pooled center and width and a stability interval to
    each model's average width rank."""
    cells: dict[tuple[str, str], list[BeliefEstimate]] = {}
    supports: dict[str, tuple[float | None, float | None]] = {}
    for row in comparison_rows:
        quantity = get_quantity(row.quantity_id)
        supports[row.quantity_id] = (quantity.lower_support, quantity.upper_support)
        run_rows = read_csv(Path(row.source_dir) / "runs.csv")
        estimates = [
            _belief_estimate_from_row(run_row)
            for run_row in run_rows
            if run_row["parsed_ok"] == "True"
            and run_row["quantity_id"] == row.quantity_id
        ]
        if estimates:
            cells[(row.model_name, row.quantity_id)] = estimates

    model_names = sorted({model for model, _ in cells}, key=model_label)
    quantity_ids = sorted({quantity for _, quantity in cells})
    rng = random.Random(seed)

    center_ses: dict[str, list[float]] = {model: [] for model in model_names}
    width_rel_ses: dict[str, list[float]] = {model: [] for model in model_names}
    width_draws: dict[tuple[str, str], list[float]] = {}

    for (model_name, quantity_id), estimates in sorted(cells.items()):
        lower_support, upper_support = supports[quantity_id]
        centers: list[float] = []
        widths: list[float] = []
        n = len(estimates)
        for _ in range(n_resamples):
            resample = [estimates[rng.randrange(n)] for _ in range(n)]
            aggregated = aggregate_beliefs(
                resample,
                lower_support=lower_support,
                upper_support=upper_support,
            )
            centers.append(aggregated.point_estimate)
            widths.append(
                (aggregated.upper_bound or 0.0) - (aggregated.lower_bound or 0.0)
            )
        width_draws[(model_name, quantity_id)] = widths
        center_se = _stdev(centers)
        width_se = _stdev(widths)
        mean_width = sum(widths) / len(widths)
        center_ses[model_name].append(center_se)
        if mean_width > 0:
            width_rel_ses[model_name].append(width_se / mean_width)

    # Average width rank per resample draw, per model.
    rank_draws: dict[str, list[float]] = {model: [] for model in model_names}
    for draw_index in range(n_resamples):
        per_model_ranks: dict[str, list[float]] = {model: [] for model in model_names}
        for quantity_id in quantity_ids:
            widths_here = [
                (model, width_draws[(model, quantity_id)][draw_index])
                for model in model_names
                if (model, quantity_id) in width_draws
            ]
            widths_here.sort(key=lambda item: item[1])
            for rank_position, (model, _) in enumerate(widths_here, start=1):
                per_model_ranks[model].append(float(rank_position))
        for model in model_names:
            if per_model_ranks[model]:
                rank_draws[model].append(
                    sum(per_model_ranks[model]) / len(per_model_ranks[model])
                )

    table_rows: list[dict[str, object]] = []
    for model in model_names:
        ranks = sorted(rank_draws[model])
        table_rows.append(
            {
                "Model": model_label(model),
                "Median center MC SE": round(median(center_ses[model]), 4),
                "Median relative width MC SE": f"{100 * median(width_rel_ses[model]):.0f}%",
                "Avg width rank (mean)": round(sum(ranks) / len(ranks), 2),
                "Avg width rank 90% interval": (
                    f"[{_bootstrap_quantile(ranks, 0.05):.2f}, "
                    f"{_bootstrap_quantile(ranks, 0.95):.2f}]"
                ),
            }
        )
    table_rows.sort(key=lambda row: row["Avg width rank (mean)"])
    return table_rows


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    return math.sqrt(sum((v - mean_value) ** 2 for v in values) / (len(values) - 1))


def build_support_bounds_table() -> list[dict[str, object]]:
    from llm_econ_beliefs.registry import list_quantities

    rows = []
    for quantity in list_quantities():
        rows.append(
            {
                "Quantity": quantity.name,
                "Quantity id": quantity.id,
                "Lower support": (
                    "unbounded" if quantity.lower_support is None else quantity.lower_support
                ),
                "Upper support": (
                    "unbounded" if quantity.upper_support is None else quantity.upper_support
                ),
            }
        )
    rows.sort(key=lambda row: str(row["Quantity id"]))
    return rows


MECHANISM_ABLATION_STEM = "mechanism-ablation-batch15"


def build_mechanism_ablation_table(
    comparison_rows: list[ComparisonRow],
) -> list[dict[str, object]]:
    """Same model (Claude Opus 4.7), same prompts, two harness mechanisms:
    the April LiteLLM forced-function-call path (1200-token budget) versus the
    July native Anthropic strict-JSON-schema path (32000-token budget)."""
    ablation_path = (
        RESULTS_DIR / f"claude-opus-4.7-{MECHANISM_ABLATION_STEM}" / "summary.csv"
    )
    if not ablation_path.exists():
        return []
    native_rows = {row["quantity_id"]: row for row in read_csv(ablation_path)}
    april_rows = {
        row.quantity_id: row
        for row in comparison_rows
        if row.model_name == "claude-opus-4.7"
    }
    table_rows: list[dict[str, object]] = []
    for quantity_id, native_row in sorted(native_rows.items()):
        april_row = april_rows.get(quantity_id)
        if april_row is None:
            continue
        native_center = float(native_row["pooled_point_estimate"])
        native_width = float(native_row["pooled_upper_bound"]) - float(
            native_row["pooled_lower_bound"]
        )
        april_width = april_row.pooled_upper - april_row.pooled_lower
        table_rows.append(
            {
                "Quantity": quantity_label(quantity_id),
                "LiteLLM center": round(april_row.pooled_point_estimate, 3),
                "Native center": round(native_center, 3),
                "Center change": round(
                    native_center - april_row.pooled_point_estimate, 3
                ),
                "LiteLLM 90% width": round(april_width, 3),
                "Native 90% width": round(native_width, 3),
            }
        )
    table_rows.sort(key=lambda row: str(row["Quantity"]))
    return table_rows


# ---------------------------------------------------------------------------
# Clarifier-wording ablation: the four April premium-tier models are the only
# cells elicited under both v4 clarifier wordings (original April 19 versus
# revised April 21). Their superseded April 19 runs live in git history.

WORDING_ARCHIVE_COMMIT = "ddca237501ffa3ea658b86fa6546aa98307bb3be"
DUAL_WORDING_MODELS = (
    "claude-opus-4.7",
    "claude-sonnet-4.6",
    "gemini-3.1-pro-preview",
    "grok-4.20",
)
# Any of the seven never-re-elicited April models preserves the original
# wording verbatim in its committed archive; gpt-5.4 stands in for the group.
WORDING_HOLDOUT_MODEL = "gpt-5.4"
SIGN_CLARIFIED_QUANTITY_IDS = (
    "labor_supply.income_elasticity.prime_age",
    CAP_GAINS_TAX_RATE_ID,
    CAP_GAINS_NET_OF_TAX_RATE_ID,
)


def _read_runs_jsonl_records(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_archived_runs_jsonl(commit: str, git_path: str) -> list[dict[str, object]]:
    try:
        content = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{git_path}"],
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"cannot read {git_path} at {commit}: {error.stderr.strip()} "
            "(the wording ablation needs full git history; run "
            "`git fetch --unshallow` in shallow clones)"
        ) from error
    return _read_runs_jsonl_records(content)


def _wording_cell(
    records: list[dict[str, object]], quantity_id: str
) -> tuple[list[BeliefEstimate], set[str]]:
    """Successful-run estimates plus the set of verbatim prompt texts."""
    estimates: list[BeliefEstimate] = []
    prompt_texts: set[str] = set()
    for record in records:
        if record.get("quantity_id") != quantity_id:
            continue
        prompt_texts.add(str(record["prompt"]))
        if not record.get("parsed_ok") or record.get("point_estimate") is None:
            continue
        point_value = float(record["point_estimate"])
        if not math.isfinite(point_value):
            continue
        quantiles = record.get("quantiles") or {}
        estimates.append(
            BeliefEstimate(
                point_estimate=point_value,
                quantity_id=quantity_id,
                quantiles={str(key): float(value) for key, value in quantiles.items()},
            )
        )
    return estimates, prompt_texts


def _pooled_wording_cell(
    estimates: list[BeliefEstimate], quantity_id: str
) -> dict[str, object]:
    quantity = get_quantity(quantity_id)
    aggregated = aggregate_beliefs(
        estimates,
        confidence_level=0.9,
        lower_support=quantity.lower_support,
        upper_support=quantity.upper_support,
    )
    return {
        "center": aggregated.point_estimate,
        "width": aggregated.upper_bound - aggregated.lower_bound,
        "points": [estimate.point_estimate for estimate in estimates],
    }


def _implied_tau_band_flag(tau_p50: float, tau_share_unit: float) -> str:
    """The Appendix A9 band flag for a sign-consistent cell."""
    ltcg_lo, ltcg_hi = CAP_GAINS_LTCG_TAU_RANGE
    ord_lo, ord_hi = CAP_GAINS_ORDINARY_INCOME_TAU_RANGE
    if not math.isfinite(tau_p50):
        return "not identified"
    if tau_share_unit < 0.8:
        return "uninformative (pole-straddling)"
    if ltcg_lo <= tau_p50 <= ltcg_hi:
        return "LTCG-rate consistent"
    if ord_lo < tau_p50 <= ord_hi:
        return "ordinary-income-rate consistent"
    if 0.0 < tau_p50 < 1.0:
        return "plausible sign, outside bands"
    return "out of band"


def build_wording_comparison_tables() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    """Within-model comparison of the two v4 clarifier wordings.

    For the four April premium-tier models — the only cells elicited under
    both wordings — pool the superseded April 19 runs (git history, commit
    ``WORDING_ARCHIVE_COMMIT``) and the published April 21 runs with the
    paper's piecewise-uniform construction, per sign-clarified quantity, and
    rerun the Appendix A9 implied-tau bootstrap per wording. Prompt text is
    byte-verified on every cell: April 19 text must equal the original
    wording preserved in the holdout model's committed archive, April 21
    text must equal the current prompt builder's revised wording. Raises if
    any identity fails, so the table cannot silently describe a different
    contrast than the Design section discloses.
    """
    holdout_path = (
        RESULTS_DIR / f"{WORDING_HOLDOUT_MODEL}-elasticities-batch15" / "runs.jsonl"
    )
    holdout_records = _read_runs_jsonl_records(holdout_path.read_text())
    original_texts: dict[str, str] = {}
    revised_texts: dict[str, str] = {}
    for quantity_id in SIGN_CLARIFIED_QUANTITY_IDS:
        _, holdout_prompts = _wording_cell(holdout_records, quantity_id)
        if len(holdout_prompts) != 1:
            raise RuntimeError(
                f"holdout {WORDING_HOLDOUT_MODEL}:{quantity_id} carries "
                f"{len(holdout_prompts)} prompt texts, expected 1"
            )
        original_texts[quantity_id] = next(iter(holdout_prompts))
        revised_texts[quantity_id] = create_belief_prompt(get_quantity(quantity_id))
        if original_texts[quantity_id] == revised_texts[quantity_id]:
            raise RuntimeError(
                f"{quantity_id}: holdout archive text equals the current prompt "
                "builder; the two-wording premise no longer holds"
            )

    table_rows: list[dict[str, object]] = []
    tau_rows: list[dict[str, object]] = []
    for model_name in DUAL_WORDING_MODELS:
        archived_records = _read_archived_runs_jsonl(
            WORDING_ARCHIVE_COMMIT,
            f"results/{model_name}-elasticities-batch15/runs.jsonl",
        )
        current_path = (
            RESULTS_DIR / f"{model_name}-elasticities-batch15" / "runs.jsonl"
        )
        current_records = _read_runs_jsonl_records(current_path.read_text())

        cells: dict[tuple[str, str], dict[str, object]] = {}
        for quantity_id in SIGN_CLARIFIED_QUANTITY_IDS:
            original_estimates, original_prompts = _wording_cell(
                archived_records, quantity_id
            )
            revised_estimates, revised_prompts = _wording_cell(
                current_records, quantity_id
            )
            if original_prompts != {original_texts[quantity_id]}:
                raise RuntimeError(
                    f"{model_name}:{quantity_id} April 19 prompt text does not "
                    "match the holdout original wording"
                )
            if revised_prompts != {revised_texts[quantity_id]}:
                raise RuntimeError(
                    f"{model_name}:{quantity_id} April 21 prompt text does not "
                    "match the current prompt builder"
                )
            if len(original_estimates) != 15 or len(revised_estimates) != 15:
                raise RuntimeError(
                    f"{model_name}:{quantity_id} expected 15 successful runs "
                    f"per wording, got {len(original_estimates)} original / "
                    f"{len(revised_estimates)} revised"
                )
            original_cell = _pooled_wording_cell(original_estimates, quantity_id)
            revised_cell = _pooled_wording_cell(revised_estimates, quantity_id)
            cells[(quantity_id, "original")] = original_cell
            cells[(quantity_id, "revised")] = revised_cell
            table_rows.append(
                {
                    "Model": model_label(model_name),
                    "Quantity": quantity_label(quantity_id),
                    "Original center": round(float(original_cell["center"]), 3),
                    "Revised center": round(float(revised_cell["center"]), 3),
                    "Center change": round(
                        float(revised_cell["center"]) - float(original_cell["center"]),
                        3,
                    ),
                    "Original 90% width": round(float(original_cell["width"]), 3),
                    "Revised 90% width": round(float(revised_cell["width"]), 3),
                }
            )

        tau_row: dict[str, object] = {"Model": model_label(model_name)}
        tau_medians: dict[str, float] = {}
        for wording, label in (("original", "Original"), ("revised", "Revised")):
            tax_points = list(cells[(CAP_GAINS_TAX_RATE_ID, wording)]["points"])
            net_points = list(cells[(CAP_GAINS_NET_OF_TAX_RATE_ID, wording)]["points"])
            if not (median(tax_points) <= 0 <= median(net_points)):
                raise RuntimeError(
                    f"{model_name} {wording}-wording medians are not "
                    "sign-consistent; the implied-tau table assumes the "
                    "identity's premise holds"
                )
            tau_p05, tau_p50, tau_p95, tau_share_unit = (
                _bootstrap_implied_tau_quantiles(tax_points, net_points)
            )
            tau_medians[wording] = tau_p50
            tau_row[f"{label} implied tau median [90%]"] = (
                f"{tau_p50:.3f} [{tau_p05:.3f}, {tau_p95:.3f}]"
            )
            tau_row[f"{label} share in (0, 1)"] = f"{100 * tau_share_unit:.0f}%"
            tau_row[f"{label} band"] = _implied_tau_band_flag(tau_p50, tau_share_unit)
        tau_row["Tau median change"] = round(
            tau_medians["revised"] - tau_medians["original"], 3
        )
        tau_rows.append(tau_row)

    return table_rows, tau_rows


if __name__ == "__main__":
    raise SystemExit(main())
