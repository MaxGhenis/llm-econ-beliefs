"""Core data models for belief elicitation."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EconomicQuantity:
    """A named economic quantity to elicit from a model."""

    id: str
    name: str
    domain: str
    description: str
    population: str | None = None
    unit: str | None = None
    preferred_interpretation: str | None = None
    lower_support: float | None = None
    upper_support: float | None = None
    benchmark_summary: str | None = None
    benchmark_source: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptRun:
    """One prompt instantiation in an experiment grid."""

    model_name: str
    quantity_id: str
    run_index: int
    prompt_version: str
    prompt: str


@dataclass
class BeliefEstimate:
    """Parsed response for one model run."""

    point_estimate: float
    quantity_id: str | None = None
    interpretation: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    confidence_level: float | None = None
    quantiles: dict[str, float] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    reasoning_summary: str | None = None
    raw_response: str | None = None


@dataclass
class AggregatedBelief:
    """Pooled summary across repeated runs."""

    point_estimate: float
    confidence_level: float
    n_runs: int
    method: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    within_run_sd: float | None = None
    between_run_sd: float | None = None
    total_sd: float | None = None


@dataclass
class RandomEffectsSummary:
    """Random-effects estimate of latent belief and predictive uncertainty."""

    method: str
    transform: str
    n_runs: int
    latent_location: float
    latent_lower: float | None = None
    latent_upper: float | None = None
    predictive_lower: float | None = None
    predictive_upper: float | None = None
    tau: float | None = None
    typical_within_sd: float | None = None


@dataclass
class BayesianBeliefSummary:
    """Bayesian hierarchical estimate of latent and predictive uncertainty."""

    method: str
    transform: str
    n_runs: int
    latent_location: float
    latent_lower: float | None = None
    latent_upper: float | None = None
    predictive_lower: float | None = None
    predictive_upper: float | None = None
    tau_mean: float | None = None
    interval_scale_mean: float | None = None
    typical_within_sd: float | None = None


@dataclass
class RunResult:
    """One completed model run with parsed fields when available."""

    provider: str
    model_name: str
    quantity_id: str
    run_index: int
    prompt_version: str
    prompt: str
    raw_response: str | None
    parsed_ok: bool
    point_estimate: float | None = None
    interpretation: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    confidence_level: float | None = None
    quantiles: dict[str, float] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    reasoning_summary: str | None = None
    error: str | None = None


@dataclass
class ProviderBatchResult:
    """Outputs and optional request metadata from one provider batch call."""

    outputs: list[str]
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestLog:
    """One provider request with shared usage metadata."""

    provider: str
    model_name: str
    quantity_id: str
    request_index: int
    prompt_version: str
    batch_size: int
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    reasoning_tokens: int | None = None
    estimated_input_cost_usd: float | None = None
    estimated_cached_input_cost_usd: float | None = None
    estimated_output_cost_usd: float | None = None
    estimated_total_cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
