import pytest

from llm_econ_beliefs import mixture_distribution, piecewise_distribution_from_quantiles


def test_piecewise_distribution_recovers_elicited_quantiles():
    distribution = piecewise_distribution_from_quantiles(
        {
            "p05": 0.2,
            "p25": 0.35,
            "p50": 0.5,
            "p75": 0.8,
            "p95": 1.5,
        },
        lower_support=0.0,
    )

    assert distribution.quantile(0.05) == pytest.approx(0.2)
    assert distribution.quantile(0.50) == pytest.approx(0.5)
    assert distribution.quantile(0.95) == pytest.approx(1.5)
    assert distribution.central_interval(0.9) == pytest.approx((0.2, 1.5))
    assert distribution.variance() > 0.0


def test_mixture_distribution_reflects_between_component_dispersion():
    distribution_low = piecewise_distribution_from_quantiles(
        {
            "p05": -1.0,
            "p25": -0.5,
            "p50": 0.0,
            "p75": 0.5,
            "p95": 1.0,
        }
    )
    distribution_high = piecewise_distribution_from_quantiles(
        {
            "p05": 1.0,
            "p25": 1.5,
            "p50": 2.0,
            "p75": 2.5,
            "p95": 3.0,
        }
    )

    mixture = mixture_distribution([distribution_low, distribution_high])

    assert mixture.mean() == pytest.approx(1.0)
    assert 0.5 < mixture.quantile(0.5) < 1.5
    assert mixture.variance() > distribution_low.variance()
    assert mixture.variance() > distribution_high.variance()
