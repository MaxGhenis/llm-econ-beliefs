"""Reconstruct and combine predictive distributions from elicited quantiles."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import BeliefEstimate

QUANTILE_ORDER = ("p05", "p25", "p50", "p75", "p95")
EPSILON = 1e-12
Segment = tuple[float, float, float]


def has_full_quantiles(
    value: BeliefEstimate | Mapping[str, float],
) -> bool:
    """Return whether an estimate or quantile mapping contains the full five-quantile schema."""
    quantiles = value.quantiles if isinstance(value, BeliefEstimate) else value
    return all(key in quantiles for key in QUANTILE_ORDER)


@dataclass(frozen=True)
class PiecewiseDistribution:
    """Approximate predictive distribution using piecewise-uniform quantile bins."""

    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("At least one segment is required")

        total_mass = sum(mass for _, _, mass in self.segments)
        if total_mass <= 0:
            raise ValueError("Segment masses must sum to a positive value")

        normalized_segments = []
        for left, right, mass in self.segments:
            if right < left:
                raise ValueError("Segment right endpoint must be >= left endpoint")
            normalized_segments.append((left, right, mass / total_mass))

        object.__setattr__(self, "segments", tuple(normalized_segments))

    @property
    def lower_support(self) -> float:
        return self.segments[0][0]

    @property
    def upper_support(self) -> float:
        return self.segments[-1][1]

    def cdf(self, value: float) -> float:
        cumulative = 0.0
        for left, right, mass in self.segments:
            if value < left:
                return cumulative
            width = right - left
            if value >= right:
                cumulative += mass
                continue
            if width <= 0:
                return cumulative + mass
            fraction = (value - left) / width
            return cumulative + mass * min(max(fraction, 0.0), 1.0)
        return 1.0

    def quantile(self, probability: float) -> float:
        probability = min(max(probability, 0.0), 1.0)
        if probability <= 0.0:
            return self.lower_support
        if probability >= 1.0:
            return self.upper_support

        cumulative = 0.0
        for left, right, mass in self.segments:
            next_cumulative = cumulative + mass
            if probability <= next_cumulative + EPSILON:
                width = right - left
                if mass <= 0 or width <= 0:
                    return right
                fraction = (probability - cumulative) / mass
                return left + min(max(fraction, 0.0), 1.0) * width
            cumulative = next_cumulative
        return self.upper_support

    def central_interval(self, confidence_level: float) -> tuple[float, float]:
        lower_prob = (1.0 - confidence_level) / 2.0
        upper_prob = 1.0 - lower_prob
        return self.quantile(lower_prob), self.quantile(upper_prob)

    def mean(self) -> float:
        return sum(mass * (left + right) / 2.0 for left, right, mass in self.segments)

    def variance(self) -> float:
        mean = self.mean()
        total = 0.0
        for left, right, mass in self.segments:
            width = right - left
            location = (left + right) / 2.0
            local_variance = (width * width) / 12.0 if width > 0 else 0.0
            total += mass * (local_variance + (location - mean) ** 2)
        return total


@dataclass(frozen=True)
class MixtureDistribution:
    """Finite mixture of piecewise predictive distributions."""

    components: tuple[PiecewiseDistribution, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("At least one component is required")
        if len(self.components) != len(self.weights):
            raise ValueError("Component and weight counts must match")

        total_weight = sum(self.weights)
        if total_weight <= 0:
            raise ValueError("Weights must sum to a positive value")

        object.__setattr__(
            self,
            "weights",
            tuple(weight / total_weight for weight in self.weights),
        )

    @property
    def lower_support(self) -> float:
        return min(component.lower_support for component in self.components)

    @property
    def upper_support(self) -> float:
        return max(component.upper_support for component in self.components)

    def cdf(self, value: float) -> float:
        return sum(
            weight * component.cdf(value)
            for component, weight in zip(self.components, self.weights, strict=True)
        )

    def quantile(self, probability: float) -> float:
        probability = min(max(probability, 0.0), 1.0)
        if probability <= 0.0:
            return self.lower_support
        if probability >= 1.0:
            return self.upper_support
        return _bisect_quantile(
            self.cdf,
            probability,
            self.lower_support,
            self.upper_support,
        )

    def central_interval(self, confidence_level: float) -> tuple[float, float]:
        lower_prob = (1.0 - confidence_level) / 2.0
        upper_prob = 1.0 - lower_prob
        return self.quantile(lower_prob), self.quantile(upper_prob)

    def mean(self) -> float:
        return sum(
            weight * component.mean()
            for component, weight in zip(self.components, self.weights, strict=True)
        )

    def variance(self) -> float:
        mean = self.mean()
        return sum(
            weight * (component.variance() + (component.mean() - mean) ** 2)
            for component, weight in zip(self.components, self.weights, strict=True)
        )


def piecewise_distribution_from_quantiles(
    quantiles: Mapping[str, float],
    *,
    lower_support: float | None = None,
    upper_support: float | None = None,
) -> PiecewiseDistribution:
    """Build a piecewise-uniform approximation from five elicited quantiles."""
    if not has_full_quantiles(quantiles):
        raise ValueError("Full p05/p25/p50/p75/p95 quantiles are required")

    q05 = quantiles["p05"]
    q25 = quantiles["p25"]
    q50 = quantiles["p50"]
    q75 = quantiles["p75"]
    q95 = quantiles["p95"]

    left_width = q25 - q05
    right_width = q95 - q75
    fallback_width = max(q95 - q05, 1.0)

    lower_endpoint = (
        lower_support
        if lower_support is not None
        else q05 - 0.25 * (left_width if left_width > 0 else fallback_width)
    )
    upper_endpoint = (
        upper_support
        if upper_support is not None
        else q95 + 0.25 * (right_width if right_width > 0 else fallback_width)
    )

    points = [
        min(lower_endpoint, q05),
        q05,
        q25,
        q50,
        q75,
        q95,
        max(upper_endpoint, q95),
    ]
    masses = [0.05, 0.20, 0.25, 0.25, 0.20, 0.05]
    return PiecewiseDistribution(tuple(zip(points[:-1], points[1:], masses, strict=True)))


def distribution_from_belief_estimate(
    estimate: BeliefEstimate,
    *,
    lower_support: float | None = None,
    upper_support: float | None = None,
) -> PiecewiseDistribution | None:
    """Build a run-level predictive distribution when five quantiles are available."""
    if not has_full_quantiles(estimate):
        return None
    return piecewise_distribution_from_quantiles(
        estimate.quantiles,
        lower_support=lower_support,
        upper_support=upper_support,
    )


def mixture_distribution(
    components: Sequence[PiecewiseDistribution],
    *,
    weights: Sequence[float] | None = None,
) -> MixtureDistribution:
    """Construct a normalized finite mixture distribution."""
    if weights is None:
        weights = [1.0] * len(components)
    return MixtureDistribution(tuple(components), tuple(weights))


def empirical_quantile(values: Sequence[float], probability: float) -> float:
    """Return the interpolated empirical quantile of a sorted or unsorted sample."""
    if not values:
        raise ValueError("At least one value is required")

    sorted_values = sorted(values)
    probability = min(max(probability, 0.0), 1.0)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    fraction = position - lower_index
    return lower_value + fraction * (upper_value - lower_value)


def empirical_cdf(values: Sequence[float], value: float) -> float:
    """Return the empirical CDF evaluated at a value."""
    if not values:
        raise ValueError("At least one value is required")
    sorted_values = sorted(values)
    return bisect_right(sorted_values, value) / len(sorted_values)


def _bisect_quantile(
    cdf,
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
