from llm_econ_beliefs import (
    CalibrationExample,
    evaluate_calibration,
    fit_pit_calibrator,
    piecewise_distribution_from_quantiles,
)


def test_evaluate_calibration_reports_basic_metrics():
    distribution = piecewise_distribution_from_quantiles(
        {
            "p05": 0.0,
            "p25": 0.25,
            "p50": 0.5,
            "p75": 0.75,
            "p95": 1.0,
        }
    )
    examples = [
        CalibrationExample(distribution=distribution, observed_value=0.4, label="a"),
        CalibrationExample(distribution=distribution, observed_value=0.9, label="b"),
    ]

    metrics = evaluate_calibration(
        examples,
        quantile_levels=(0.05, 0.5, 0.95),
        interval_levels=(0.5, 0.9),
    )

    assert metrics.n_examples == 2
    assert metrics.mean_pinball_loss >= 0.0
    assert metrics.weighted_interval_score >= 0.0
    assert 0.0 <= metrics.pit_mean <= 1.0
    assert metrics.coverage_by_interval[0.9] == 1.0


def test_empirical_pit_calibration_can_shift_distribution_outward():
    base_distribution = piecewise_distribution_from_quantiles(
        {
            "p05": -1.0,
            "p25": -0.5,
            "p50": 0.0,
            "p75": 0.5,
            "p95": 1.0,
        }
    )
    examples = [
        CalibrationExample(distribution=base_distribution, observed_value=0.8),
        CalibrationExample(distribution=base_distribution, observed_value=0.9),
        CalibrationExample(distribution=base_distribution, observed_value=1.0),
    ]

    calibrator = fit_pit_calibrator(examples)
    calibrated_distribution = calibrator.calibrate_distribution(base_distribution)

    assert calibrated_distribution.quantile(0.5) > base_distribution.quantile(0.5)
    assert calibrated_distribution.quantile(0.9) >= base_distribution.quantile(0.9)
