from llm_econ_beliefs import aggregate_beliefs, parse_belief_response


def test_aggregate_beliefs_combines_uncertainty_layers():
    estimates = [
        parse_belief_response(
            """
            {
              "point_estimate": 0.4,
              "lower_bound": 0.2,
              "upper_bound": 0.6,
              "confidence_level": 0.9
            }
            """
        ),
        parse_belief_response(
            """
            {
              "point_estimate": 0.5,
              "lower_bound": 0.3,
              "upper_bound": 0.7,
              "confidence_level": 0.9
            }
            """
        ),
    ]

    aggregated = aggregate_beliefs(estimates, confidence_level=0.9)

    assert aggregated.method == "law_of_total_variance"
    assert aggregated.n_runs == 2
    assert aggregated.point_estimate == 0.45
    assert aggregated.lower_bound is not None
    assert aggregated.upper_bound is not None
    assert aggregated.lower_bound < aggregated.point_estimate < aggregated.upper_bound


def test_aggregate_beliefs_uses_quantile_mixture_when_available():
    estimates = [
        parse_belief_response(
            """
            {
              "point_estimate": 0.5,
              "quantiles": {
                "p05": 0.2,
                "p25": 0.35,
                "p50": 0.5,
                "p75": 0.8,
                "p95": 1.5
              }
            }
            """
        ),
        parse_belief_response(
            """
            {
              "point_estimate": 0.75,
              "quantiles": {
                "p05": 0.3,
                "p25": 0.45,
                "p50": 0.75,
                "p75": 1.0,
                "p95": 1.5
              }
            }
            """
        ),
    ]

    aggregated = aggregate_beliefs(
        estimates,
        confidence_level=0.9,
        lower_support=0.0,
    )

    assert aggregated.point_estimate == 0.625
    assert aggregated.lower_bound is not None
    assert aggregated.upper_bound is not None
    assert aggregated.lower_bound >= 0.0
    assert aggregated.upper_bound > 1.0
