"""Aggregate uncertainty across repeated elicitation runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist, fmean, pvariance
from typing import Callable, Sequence

from .distributions import (
    QUANTILE_ORDER,
    distribution_from_belief_estimate,
    has_full_quantiles,
    mixture_distribution,
    piecewise_distribution_from_quantiles,
)
from .models import (
    AggregatedBelief,
    BayesianBeliefSummary,
    BeliefEstimate,
    RandomEffectsSummary,
)

NORMAL = NormalDist()
EPSILON = 1e-12


@dataclass(frozen=True)
class _TransformSpec:
    name: str
    forward: Callable[[float], float]
    inverse: Callable[[float], float]
    local_sd_to_original: Callable[[float, float], float]


def aggregate_beliefs(
    estimates: Sequence[BeliefEstimate],
    *,
    confidence_level: float = 0.9,
    lower_support: float | None = None,
    upper_support: float | None = None,
) -> AggregatedBelief:
    """Pool repeated runs using the law of total variance on the raw scale."""
    if not estimates:
        raise ValueError("At least one estimate is required")

    points = [estimate.point_estimate for estimate in estimates]
    point_estimate = fmean(points)

    between_run_var = pvariance(points) if len(points) > 1 else 0.0
    within_vars = [
        variance
        for estimate in estimates
        if (variance := _within_variance_raw(estimate, lower_support, upper_support)) is not None
        and math.isfinite(variance)
    ]
    within_run_var = fmean(within_vars) if within_vars else 0.0
    total_var = between_run_var + within_run_var

    interval = _mixture_interval_from_quantiles(
        estimates,
        confidence_level=confidence_level,
        lower_support=lower_support,
        upper_support=upper_support,
    )
    lower_bound = interval[0] if interval else None
    upper_bound = interval[1] if interval else None
    total_sd = math.sqrt(total_var) if total_var > 0 else 0.0
    if total_sd > 0 and interval is None:
        z_value = NORMAL.inv_cdf(0.5 + confidence_level / 2)
        lower_bound = point_estimate - z_value * total_sd
        upper_bound = point_estimate + z_value * total_sd
        if lower_support is not None:
            lower_bound = max(lower_bound, lower_support)
        if upper_support is not None:
            upper_bound = min(upper_bound, upper_support)

    return AggregatedBelief(
        point_estimate=point_estimate,
        confidence_level=confidence_level,
        n_runs=len(estimates),
        method="law_of_total_variance",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        within_run_sd=math.sqrt(within_run_var) if within_run_var > 0 else 0.0,
        between_run_sd=math.sqrt(between_run_var) if between_run_var > 0 else 0.0,
        total_sd=total_sd,
    )


def random_effects_meta_analysis(
    estimates: Sequence[BeliefEstimate],
    *,
    confidence_level: float = 0.9,
    lower_support: float | None = None,
    upper_support: float | None = None,
) -> RandomEffectsSummary:
    """Estimate a latent central belief and predictive uncertainty via REML."""
    if not estimates:
        raise ValueError("At least one estimate is required")

    transform, y_values, variances, typical_variance = _prepare_transformed_data(
        estimates,
        lower_support=lower_support,
        upper_support=upper_support,
    )

    tau2_hat = _estimate_reml_tau2(y_values, variances)
    weights = [1.0 / max(variance + tau2_hat, EPSILON) for variance in variances]
    weighted_sum = sum(weight * y for weight, y in zip(weights, y_values))
    total_weight = sum(weights)
    mu_hat = weighted_sum / total_weight

    z_value = NORMAL.inv_cdf(0.5 + confidence_level / 2)
    mu_sd = math.sqrt(1.0 / total_weight) if total_weight > 0 else 0.0
    latent_lower_z = mu_hat - z_value * mu_sd
    latent_upper_z = mu_hat + z_value * mu_sd

    predictive_sd = math.sqrt(max(tau2_hat + typical_variance, 0.0))
    predictive_lower_z = mu_hat - z_value * predictive_sd
    predictive_upper_z = mu_hat + z_value * predictive_sd

    latent_location = transform.inverse(mu_hat)
    return RandomEffectsSummary(
        method="random_effects_reml",
        transform=transform.name,
        n_runs=len(estimates),
        latent_location=latent_location,
        latent_lower=transform.inverse(latent_lower_z),
        latent_upper=transform.inverse(latent_upper_z),
        predictive_lower=transform.inverse(predictive_lower_z),
        predictive_upper=transform.inverse(predictive_upper_z),
        tau=transform.local_sd_to_original(math.sqrt(max(tau2_hat, 0.0)), latent_location),
        typical_within_sd=transform.local_sd_to_original(
            math.sqrt(max(typical_variance, 0.0)),
            latent_location,
        ),
    )


def bayesian_hierarchical_meta_analysis(
    estimates: Sequence[BeliefEstimate],
    *,
    confidence_level: float = 0.9,
    lower_support: float | None = None,
    upper_support: float | None = None,
    estimate_interval_scale: bool = True,
) -> BayesianBeliefSummary:
    """Estimate latent and predictive uncertainty with a small-grid Bayes model."""
    if not estimates:
        raise ValueError("At least one estimate is required")

    transform, y_values, variances, typical_variance = _prepare_transformed_data(
        estimates,
        lower_support=lower_support,
        upper_support=upper_support,
    )

    y_mean = fmean(y_values)
    y_sd = math.sqrt(pvariance(y_values)) if len(y_values) > 1 else 0.0
    within_sd = math.sqrt(max(typical_variance, 0.0))
    scale = max(y_sd, within_sd, 0.25)

    mu_prior_mean = y_mean
    mu_prior_sd = max(5.0 * scale, 1.0)
    tau_prior_sd = max(scale, 0.25)

    tau_grid = [0.0] + _log_grid(max(scale * 1e-3, 1e-4), max(scale * 10.0, 0.5), 80)
    if estimate_interval_scale and typical_variance > 0:
        lambda_grid = _log_grid(0.25, 4.0, 41)
    else:
        lambda_grid = [1.0]

    components = []
    for tau in tau_grid:
        tau2 = tau * tau
        for interval_scale in lambda_grid:
            log_weight, posterior_mean, posterior_var = _collapsed_log_posterior(
                y_values=y_values,
                variances=variances,
                tau2=tau2,
                interval_scale=interval_scale,
                mu_prior_mean=mu_prior_mean,
                mu_prior_sd=mu_prior_sd,
                tau_prior_sd=tau_prior_sd,
            )
            components.append(
                (
                    log_weight,
                    posterior_mean,
                    posterior_var,
                    tau,
                    interval_scale,
                )
            )

    normalized = _normalize_components(components)

    latent_means = [component[1] for component in normalized]
    latent_sds = [math.sqrt(component[2]) for component in normalized]
    weights = [component[0] for component in normalized]

    latent_location, latent_lower, latent_upper = _mixture_quantiles(
        means=latent_means,
        sds=latent_sds,
        weights=weights,
        transform=transform,
        confidence_level=confidence_level,
    )

    predictive_sds = [
        math.sqrt(
            max(
                component[2] + component[3] ** 2 + component[4] * typical_variance,
                0.0,
            )
        )
        for component in normalized
    ]
    predictive_location, predictive_lower, predictive_upper = _mixture_quantiles(
        means=latent_means,
        sds=predictive_sds,
        weights=weights,
        transform=transform,
        confidence_level=confidence_level,
    )

    tau_mean_z = sum(weight * component[3] for weight, component in zip(weights, normalized))
    interval_scale_mean = sum(
        weight * component[4] for weight, component in zip(weights, normalized)
    )
    typical_within_sd_orig = transform.local_sd_to_original(
        math.sqrt(max(interval_scale_mean * typical_variance, 0.0)),
        latent_location,
    )

    return BayesianBeliefSummary(
        method="bayesian_hierarchical_grid",
        transform=transform.name,
        n_runs=len(estimates),
        latent_location=latent_location,
        latent_lower=latent_lower,
        latent_upper=latent_upper,
        predictive_lower=predictive_lower,
        predictive_upper=predictive_upper,
        tau_mean=transform.local_sd_to_original(tau_mean_z, latent_location),
        interval_scale_mean=interval_scale_mean,
        typical_within_sd=typical_within_sd_orig,
    )


def _prepare_transformed_data(
    estimates: Sequence[BeliefEstimate],
    *,
    lower_support: float | None,
    upper_support: float | None,
) -> tuple[_TransformSpec, list[float], list[float], float]:
    transform = _make_transform(lower_support, upper_support)

    y_values = [transform.forward(estimate.point_estimate) for estimate in estimates]
    observed_variances = [
        variance
        for estimate in estimates
        if (variance := _within_variance_transformed(estimate, transform)) is not None
        and math.isfinite(variance)
        and variance >= 0
    ]
    typical_variance = fmean(observed_variances) if observed_variances else 0.0

    filled_variances = []
    for estimate in estimates:
        variance = _within_variance_transformed(estimate, transform)
        if variance is None or not math.isfinite(variance) or variance < 0:
            variance = typical_variance
        filled_variances.append(max(variance, 0.0))

    return transform, y_values, filled_variances, typical_variance


def _estimate_reml_tau2(y_values: Sequence[float], variances: Sequence[float]) -> float:
    if len(y_values) <= 1:
        return 0.0

    scale = max(
        math.sqrt(pvariance(y_values)) if len(y_values) > 1 else 0.0,
        math.sqrt(fmean(variances)) if variances else 0.0,
        0.25,
    )
    upper = max(scale * scale * 100.0, 1.0)
    grid = [0.0] + _log_grid(1e-8, upper, 160)

    best_tau2 = 0.0
    best_value = _reml_objective(0.0, y_values, variances)
    for tau2 in grid:
        objective = _reml_objective(tau2, y_values, variances)
        if objective < best_value:
            best_value = objective
            best_tau2 = tau2

    return best_tau2


def _reml_objective(
    tau2: float,
    y_values: Sequence[float],
    variances: Sequence[float],
) -> float:
    weights = [1.0 / max(variance + tau2, EPSILON) for variance in variances]
    total_weight = sum(weights)
    mu_hat = sum(weight * y for weight, y in zip(weights, y_values)) / total_weight

    return 0.5 * (
        sum(math.log(max(variance + tau2, EPSILON)) for variance in variances)
        + math.log(max(total_weight, EPSILON))
        + sum(
            weight * (y_value - mu_hat) ** 2
            for weight, y_value in zip(weights, y_values)
        )
    )


def _collapsed_log_posterior(
    *,
    y_values: Sequence[float],
    variances: Sequence[float],
    tau2: float,
    interval_scale: float,
    mu_prior_mean: float,
    mu_prior_sd: float,
    tau_prior_sd: float,
) -> tuple[float, float, float]:
    effective_variances = [
        max(tau2 + interval_scale * variance, EPSILON) for variance in variances
    ]
    prior_precision = 1.0 / max(mu_prior_sd * mu_prior_sd, EPSILON)
    likelihood_precision = sum(1.0 / variance for variance in effective_variances)
    posterior_precision = prior_precision + likelihood_precision
    posterior_var = 1.0 / posterior_precision
    posterior_mean = posterior_var * (
        mu_prior_mean * prior_precision
        + sum(y / variance for y, variance in zip(y_values, effective_variances))
    )

    quadratic = (
        mu_prior_mean * mu_prior_mean * prior_precision
        + sum(y * y / variance for y, variance in zip(y_values, effective_variances))
        - posterior_mean * posterior_mean * posterior_precision
    )
    log_likelihood = -0.5 * (
        sum(math.log(variance) for variance in effective_variances)
        + math.log(max(mu_prior_sd * mu_prior_sd, EPSILON))
        - math.log(max(posterior_var, EPSILON))
        + quadratic
    )

    tau = math.sqrt(max(tau2, 0.0))
    log_tau_prior = (
        math.log(math.sqrt(2.0 / math.pi))
        - math.log(max(tau_prior_sd, EPSILON))
        - (tau * tau) / (2.0 * tau_prior_sd * tau_prior_sd)
    )

    log_lambda_prior = 0.0
    if interval_scale > 0:
        lambda_log_sd = 0.75
        log_lambda = math.log(interval_scale)
        log_lambda_prior = (
            -math.log(max(interval_scale, EPSILON))
            - math.log(lambda_log_sd * math.sqrt(2.0 * math.pi))
            - (log_lambda * log_lambda) / (2.0 * lambda_log_sd * lambda_log_sd)
        )

    return log_likelihood + log_tau_prior + log_lambda_prior, posterior_mean, posterior_var


def _normalize_components(
    components: Sequence[tuple[float, float, float, float, float]]
) -> list[tuple[float, float, float, float, float]]:
    max_log_weight = max(component[0] for component in components)
    raw_weights = [math.exp(component[0] - max_log_weight) for component in components]
    total_weight = sum(raw_weights)

    normalized = []
    for component, weight in zip(components, raw_weights):
        normalized.append((weight / total_weight, *component[1:]))
    return normalized


def _mixture_quantiles(
    *,
    means: Sequence[float],
    sds: Sequence[float],
    weights: Sequence[float],
    transform: _TransformSpec,
    confidence_level: float,
) -> tuple[float, float, float]:
    lower_prob = (1.0 - confidence_level) / 2.0
    upper_prob = 1.0 - lower_prob

    lower_bound = min(mean - 8.0 * sd for mean, sd in zip(means, sds, strict=False))
    upper_bound = max(mean + 8.0 * sd for mean, sd in zip(means, sds, strict=False))
    if not math.isfinite(lower_bound):
        lower_bound = min(means) - 10.0
    if not math.isfinite(upper_bound):
        upper_bound = max(means) + 10.0

    def mixture_cdf(value: float) -> float:
        total = 0.0
        for weight, mean, sd in zip(weights, means, sds, strict=False):
            if sd <= 0:
                total += weight * (1.0 if value >= mean else 0.0)
            else:
                total += weight * NORMAL.cdf((value - mean) / sd)
        return total

    median_z = _bisect_quantile(mixture_cdf, 0.5, lower_bound, upper_bound)
    lower_z = _bisect_quantile(mixture_cdf, lower_prob, lower_bound, upper_bound)
    upper_z = _bisect_quantile(mixture_cdf, upper_prob, lower_bound, upper_bound)

    return (
        transform.inverse(median_z),
        transform.inverse(lower_z),
        transform.inverse(upper_z),
    )


def _bisect_quantile(
    cdf: Callable[[float], float],
    probability: float,
    lower: float,
    upper: float,
) -> float:
    left = lower
    right = upper
    for _ in range(120):
        midpoint = (left + right) / 2.0
        if cdf(midpoint) < probability:
            left = midpoint
        else:
            right = midpoint
    return (left + right) / 2.0


def _log_grid(start: float, stop: float, num: int) -> list[float]:
    if num <= 1:
        return [start]
    log_start = math.log(max(start, EPSILON))
    log_stop = math.log(max(stop, start + EPSILON))
    step = (log_stop - log_start) / (num - 1)
    return [math.exp(log_start + index * step) for index in range(num)]


def _make_transform(
    lower_support: float | None,
    upper_support: float | None,
) -> _TransformSpec:
    if lower_support is not None and upper_support is not None:
        span = upper_support - lower_support
        epsilon = max(abs(span), 1.0) * 1e-9

        def forward(value: float) -> float:
            clipped = min(max(value, lower_support + epsilon), upper_support - epsilon)
            scaled = (clipped - lower_support) / span
            return math.log(scaled / (1.0 - scaled))

        def inverse(value: float) -> float:
            logistic = 1.0 / (1.0 + math.exp(-value))
            return lower_support + span * logistic

        def local_sd_to_original(sd_z: float, location_x: float) -> float:
            return abs((location_x - lower_support) * (upper_support - location_x) / span) * sd_z

        return _TransformSpec("bounded_logit", forward, inverse, local_sd_to_original)

    if lower_support is not None and upper_support is None:
        epsilon = max(abs(lower_support), 1.0) * 1e-9

        def forward(value: float) -> float:
            clipped = max(value, lower_support + epsilon)
            return math.log(clipped - lower_support)

        def inverse(value: float) -> float:
            return lower_support + math.exp(value)

        def local_sd_to_original(sd_z: float, location_x: float) -> float:
            return max(location_x - lower_support, 0.0) * sd_z

        return _TransformSpec("lower_bounded_log", forward, inverse, local_sd_to_original)

    if lower_support is None and upper_support is not None:
        epsilon = max(abs(upper_support), 1.0) * 1e-9

        def forward(value: float) -> float:
            clipped = min(value, upper_support - epsilon)
            return -math.log(upper_support - clipped)

        def inverse(value: float) -> float:
            return upper_support - math.exp(-value)

        def local_sd_to_original(sd_z: float, location_x: float) -> float:
            return max(upper_support - location_x, 0.0) * sd_z

        return _TransformSpec("upper_bounded_log", forward, inverse, local_sd_to_original)

    def forward_identity(value: float) -> float:
        return value

    def inverse_identity(value: float) -> float:
        return value

    def local_sd_identity(sd_z: float, _: float) -> float:
        return sd_z

    return _TransformSpec("identity", forward_identity, inverse_identity, local_sd_identity)


def _within_variance_transformed(
    estimate: BeliefEstimate,
    transform: _TransformSpec,
) -> float | None:
    transformed_quantiles = _transformed_quantiles(estimate, transform)
    if transformed_quantiles:
        return piecewise_distribution_from_quantiles(transformed_quantiles).variance()

    if (
        estimate.lower_bound is None
        or estimate.upper_bound is None
        or estimate.confidence_level is None
        or not 0 < estimate.confidence_level < 1
    ):
        return None

    z_value = NORMAL.inv_cdf(0.5 + estimate.confidence_level / 2)
    if z_value <= 0:
        return None

    lower_z = transform.forward(estimate.lower_bound)
    upper_z = transform.forward(estimate.upper_bound)
    sigma = abs(upper_z - lower_z) / (2.0 * z_value)
    return sigma * sigma


def _within_variance_raw(
    estimate: BeliefEstimate,
    lower_support: float | None,
    upper_support: float | None,
) -> float | None:
    if has_full_quantiles(estimate):
        distribution = distribution_from_belief_estimate(
            estimate,
            lower_support=lower_support,
            upper_support=upper_support,
        )
        if distribution is not None:
            return distribution.variance()

    if (
        estimate.lower_bound is None
        or estimate.upper_bound is None
        or estimate.confidence_level is None
        or not 0 < estimate.confidence_level < 1
    ):
        return None

    z_value = NORMAL.inv_cdf(0.5 + estimate.confidence_level / 2)
    if z_value <= 0:
        return None

    sigma = abs(estimate.upper_bound - estimate.lower_bound) / (2.0 * z_value)
    return sigma * sigma


def _mixture_interval_from_quantiles(
    estimates: Sequence[BeliefEstimate],
    *,
    confidence_level: float,
    lower_support: float | None,
    upper_support: float | None,
) -> tuple[float, float] | None:
    if not estimates or not all(has_full_quantiles(estimate) for estimate in estimates):
        return None

    distributions = [
        distribution_from_belief_estimate(
            estimate,
            lower_support=lower_support,
            upper_support=upper_support,
        )
        for estimate in estimates
    ]
    mixture = mixture_distribution([distribution for distribution in distributions if distribution])
    return mixture.central_interval(confidence_level)


def _transformed_quantiles(
    estimate: BeliefEstimate,
    transform: _TransformSpec,
) -> dict[str, float]:
    if not has_full_quantiles(estimate):
        return {}
    return {
        key: transform.forward(estimate.quantiles[key])
        for key in QUANTILE_ORDER
    }
