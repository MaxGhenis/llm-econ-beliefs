"""Utilities for eliciting LLM beliefs about economic quantities."""

from .aggregate import (
    aggregate_beliefs,
    bayesian_hierarchical_meta_analysis,
    random_effects_meta_analysis,
)
from .calibration import (
    CalibratedDistribution,
    CalibrationExample,
    CalibrationMetrics,
    EmpiricalCDFCalibrator,
    evaluate_calibration,
    fit_pit_calibrator,
)
from .distributions import (
    MixtureDistribution,
    PiecewiseDistribution,
    distribution_from_belief_estimate,
    empirical_cdf,
    empirical_quantile,
    has_full_quantiles,
    mixture_distribution,
    piecewise_distribution_from_quantiles,
)
from .models import (
    AggregatedBelief,
    BayesianBeliefSummary,
    BeliefEstimate,
    EconomicQuantity,
    ParameterMapping,
    ProviderBatchResult,
    PromptRun,
    RandomEffectsSummary,
    RequestLog,
    RunResult,
)
from .mappings import get_parameter_mapping, list_mapping_systems, list_parameter_mappings
from .parse import parse_belief_response
from .pricing import (
    OPENAI_MODEL_PRICING,
    OPENAI_PRICING_AS_OF,
    OPENAI_PRICING_SOURCE_URL,
    ModelPricing,
    estimate_request_cost,
    lookup_model_pricing,
)
from .prompts import create_belief_prompt
from .providers import (
    build_claude_command,
    build_litellm_belief_tool,
    build_openai_chat_payload,
    build_openai_response_payload,
    resolve_litellm_model_name,
    run_claude_prompt,
    run_litellm_prompt_logged,
    run_openai_prompt_batch,
    run_openai_prompt_batch_logged,
    run_openai_response_logged,
)
from .registry import get_quantity, list_quantities, list_tags
from .runner import build_run_grid, write_run_grid_csv

__version__ = "0.1.0"


def resolve_quantity_ids(*args, **kwargs):
    from .experiment import resolve_quantity_ids as _resolve_quantity_ids

    return _resolve_quantity_ids(*args, **kwargs)


def run_claude_experiment(*args, **kwargs):
    from .experiment import run_claude_experiment as _run_claude_experiment

    return _run_claude_experiment(*args, **kwargs)


def run_openai_experiment(*args, **kwargs):
    from .experiment import run_openai_experiment as _run_openai_experiment

    return _run_openai_experiment(*args, **kwargs)


def run_litellm_experiment(*args, **kwargs):
    from .experiment import run_litellm_experiment as _run_litellm_experiment

    return _run_litellm_experiment(*args, **kwargs)


def summarize_run_results(*args, **kwargs):
    from .experiment import summarize_run_results as _summarize_run_results

    return _summarize_run_results(*args, **kwargs)


def build_comparison_rows(*args, **kwargs):
    from .compare import build_comparison_rows as _build_comparison_rows

    return _build_comparison_rows(*args, **kwargs)


def read_summary_rows(*args, **kwargs):
    from .compare import read_summary_rows as _read_summary_rows

    return _read_summary_rows(*args, **kwargs)


def write_comparison_csv(*args, **kwargs):
    from .compare import write_comparison_csv as _write_comparison_csv

    return _write_comparison_csv(*args, **kwargs)


__all__ = [
    "__version__",
    "AggregatedBelief",
    "CalibratedDistribution",
    "CalibrationExample",
    "CalibrationMetrics",
    "BayesianBeliefSummary",
    "BeliefEstimate",
    "EmpiricalCDFCalibrator",
    "EconomicQuantity",
    "MixtureDistribution",
    "ModelPricing",
    "OPENAI_MODEL_PRICING",
    "OPENAI_PRICING_AS_OF",
    "OPENAI_PRICING_SOURCE_URL",
    "ParameterMapping",
    "PiecewiseDistribution",
    "PromptRun",
    "ProviderBatchResult",
    "RandomEffectsSummary",
    "RequestLog",
    "RunResult",
    "aggregate_beliefs",
    "bayesian_hierarchical_meta_analysis",
    "build_comparison_rows",
    "build_claude_command",
    "build_litellm_belief_tool",
    "build_openai_chat_payload",
    "build_openai_response_payload",
    "build_run_grid",
    "create_belief_prompt",
    "distribution_from_belief_estimate",
    "empirical_cdf",
    "empirical_quantile",
    "estimate_request_cost",
    "evaluate_calibration",
    "fit_pit_calibrator",
    "get_parameter_mapping",
    "get_quantity",
    "has_full_quantiles",
    "list_mapping_systems",
    "list_parameter_mappings",
    "list_quantities",
    "list_tags",
    "lookup_model_pricing",
    "mixture_distribution",
    "parse_belief_response",
    "piecewise_distribution_from_quantiles",
    "resolve_quantity_ids",
    "resolve_litellm_model_name",
    "random_effects_meta_analysis",
    "read_summary_rows",
    "run_claude_experiment",
    "run_claude_prompt",
    "run_litellm_experiment",
    "run_litellm_prompt_logged",
    "run_openai_experiment",
    "run_openai_prompt_batch",
    "run_openai_prompt_batch_logged",
    "run_openai_response_logged",
    "summarize_run_results",
    "write_comparison_csv",
    "write_run_grid_csv",
]
