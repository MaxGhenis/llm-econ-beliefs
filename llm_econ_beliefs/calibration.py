"""Evaluate and recalibrate elicited predictive distributions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pvariance
from typing import Sequence

from .distributions import (
    MixtureDistribution,
    PiecewiseDistribution,
    empirical_cdf,
    empirical_quantile,
)


@dataclass(frozen=True)
class CalibrationExample:
    """One resolved outcome paired with a predictive distribution."""

    distribution: PiecewiseDistribution | MixtureDistribution | CalibratedDistribution
    observed_value: float
    label: str | None = None


@dataclass(frozen=True)
class CalibrationMetrics:
    """Summary metrics for predictive calibration on resolved tasks."""

    n_examples: int
    quantile_levels: tuple[float, ...]
    interval_levels: tuple[float, ...]
    mean_pinball_loss: float
    weighted_interval_score: float
    pit_mean: float
    pit_variance: float
    coverage_by_interval: dict[float, float]


@dataclass(frozen=True)
class EmpiricalCDFCalibrator:
    """Empirical PIT recalibrator."""

    sorted_pits: tuple[float, ...]
    method: str = "empirical_pit"

    @classmethod
    def fit(cls, examples: Sequence[CalibrationExample]) -> "EmpiricalCDFCalibrator":
        if not examples:
            raise ValueError("At least one calibration example is required")
        pits = sorted(
            min(max(example.distribution.cdf(example.observed_value), 0.0), 1.0)
            for example in examples
        )
        return cls(tuple(pits))

    def map_probability(self, probability: float) -> float:
        """Map an uncalibrated CDF value through the empirical PIT CDF."""
        probability = min(max(probability, 0.0), 1.0)
        return empirical_cdf(self.sorted_pits, probability)

    def inverse_probability(self, probability: float) -> float:
        """Map a target quantile level back to the uncalibrated probability scale."""
        probability = min(max(probability, 0.0), 1.0)
        return empirical_quantile(self.sorted_pits, probability)

    def calibrate_distribution(
        self,
        distribution: PiecewiseDistribution | MixtureDistribution,
    ) -> "CalibratedDistribution":
        return CalibratedDistribution(
            base_distribution=distribution,
            calibrator=self,
        )


@dataclass(frozen=True)
class CalibratedDistribution:
    """Distribution wrapper produced by empirical PIT recalibration."""

    base_distribution: PiecewiseDistribution | MixtureDistribution
    calibrator: EmpiricalCDFCalibrator

    def cdf(self, value: float) -> float:
        return self.calibrator.map_probability(self.base_distribution.cdf(value))

    def quantile(self, probability: float) -> float:
        raw_probability = self.calibrator.inverse_probability(probability)
        return self.base_distribution.quantile(raw_probability)

    def central_interval(self, confidence_level: float) -> tuple[float, float]:
        lower_prob = (1.0 - confidence_level) / 2.0
        upper_prob = 1.0 - lower_prob
        return self.quantile(lower_prob), self.quantile(upper_prob)


def fit_pit_calibrator(examples: Sequence[CalibrationExample]) -> EmpiricalCDFCalibrator:
    """Fit an empirical PIT calibrator on resolved examples."""
    return EmpiricalCDFCalibrator.fit(examples)


def evaluate_calibration(
    examples: Sequence[CalibrationExample],
    *,
    quantile_levels: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    interval_levels: Sequence[float] = (0.5, 0.9),
) -> CalibrationMetrics:
    """Evaluate predictive calibration on resolved numeric targets."""
    if not examples:
        raise ValueError("At least one calibration example is required")

    quantile_levels = tuple(quantile_levels)
    interval_levels = tuple(interval_levels)

    pinball_losses = []
    wis_scores = []
    pits = []
    coverage_by_interval = {level: 0.0 for level in interval_levels}

    for example in examples:
        pits.append(example.distribution.cdf(example.observed_value))
        for level in quantile_levels:
            predicted_quantile = example.distribution.quantile(level)
            pinball_losses.append(
                _pinball_loss(
                    example.observed_value,
                    predicted_quantile,
                    level,
                )
            )

        wis_scores.append(
            _weighted_interval_score(
                example.distribution,
                example.observed_value,
                interval_levels,
            )
        )

        for level in interval_levels:
            lower, upper = example.distribution.central_interval(level)
            if lower <= example.observed_value <= upper:
                coverage_by_interval[level] += 1.0

    n_examples = len(examples)
    return CalibrationMetrics(
        n_examples=n_examples,
        quantile_levels=quantile_levels,
        interval_levels=interval_levels,
        mean_pinball_loss=fmean(pinball_losses),
        weighted_interval_score=fmean(wis_scores),
        pit_mean=fmean(pits),
        pit_variance=pvariance(pits) if len(pits) > 1 else 0.0,
        coverage_by_interval={
            level: covered / n_examples
            for level, covered in coverage_by_interval.items()
        },
    )


def _pinball_loss(observed_value: float, predicted_quantile: float, level: float) -> float:
    error = observed_value - predicted_quantile
    return max(level * error, (level - 1.0) * error)


def _interval_score(
    observed_value: float,
    lower: float,
    upper: float,
    alpha: float,
) -> float:
    score = upper - lower
    if observed_value < lower:
        score += (2.0 / alpha) * (lower - observed_value)
    elif observed_value > upper:
        score += (2.0 / alpha) * (observed_value - upper)
    return score


def _weighted_interval_score(
    distribution: PiecewiseDistribution | MixtureDistribution | CalibratedDistribution,
    observed_value: float,
    interval_levels: Sequence[float],
) -> float:
    median = distribution.quantile(0.5)
    total = 0.5 * abs(observed_value - median)
    normalizer = 0.5

    for level in interval_levels:
        alpha = 1.0 - level
        lower, upper = distribution.central_interval(level)
        total += (alpha / 2.0) * _interval_score(
            observed_value,
            lower,
            upper,
            alpha,
        )
        normalizer += alpha / 2.0

    return total / normalizer
