import pytest

from llm_econ_beliefs import parse_belief_response


def test_parse_json_response():
    response = """
    {
      "interpretation": "Intensive-margin income elasticity",
      "point_estimate": -0.10,
      "quantiles": {
        "p05": -0.18,
        "p25": -0.12,
        "p50": -0.10,
        "p75": -0.06,
        "p95": -0.02
      },
      "citations": ["Imbens, Rubin, and Sacerdote (2001)", "Blundell and MaCurdy (1999)"],
      "reasoning_summary": "Lottery and review evidence imply a modest negative income effect."
    }
    """

    parsed = parse_belief_response(
        response,
        quantity_id="labor_supply.income_elasticity.prime_age",
    )

    assert parsed.quantity_id == "labor_supply.income_elasticity.prime_age"
    assert parsed.point_estimate == -0.10
    assert parsed.lower_bound == -0.18
    assert parsed.upper_bound == -0.02
    assert parsed.confidence_level == 0.90
    assert parsed.quantiles["p50"] == -0.10
    assert len(parsed.citations) == 2


def test_parse_json_response_uses_p50_as_point_estimate_when_present():
    response = """
    {
      "interpretation": "Lifecycle Frisch elasticity",
      "point_estimate": 0.75,
      "quantiles": {
        "p05": 0.2,
        "p25": 0.35,
        "p50": 0.5,
        "p75": 0.9,
        "p95": 1.5
      },
      "citations": [],
      "reasoning_summary": "Short summary."
    }
    """

    parsed = parse_belief_response(response)

    assert parsed.point_estimate == 0.5


def test_parse_free_text_response_without_quantiles_fails():
    # The quantile-complete contract: prose with only a point estimate is a
    # failed run, since the pooled mixture needs all five quantiles.
    with pytest.raises(ValueError, match="missing quantiles"):
        parse_belief_response("About 0.4 for a macro-calibration Frisch elasticity.")


def test_parse_free_text_with_interval_but_no_quantiles_fails():
    with pytest.raises(ValueError, match="missing quantiles"):
        parse_belief_response(
            """
            **Summary:** I chose the lifecycle interpretation.
            A point estimate of **0.5** is reasonable.
            The 90% interval [0.2, 1.5] spans the micro-to-macro gap.
            """
        )


def test_parse_free_text_with_beta_symbol_but_no_quantiles_fails():
    with pytest.raises(ValueError, match="missing quantiles"):
        parse_belief_response("Annual discount factor β ≈ 0.96, 90% CI: [0.94, 0.99].")


def test_parse_quantile_labels_from_free_text():
    parsed = parse_belief_response(
        """
        Point estimate: 2.0
        p05 = 0.5
        p25 = 1.0
        p50 = 2.0
        p75 = 3.0
        p95 = 8.0
        """
    )

    assert parsed.point_estimate == 2.0
    assert parsed.quantiles == {
        "p05": 0.5,
        "p25": 1.0,
        "p50": 2.0,
        "p75": 3.0,
        "p95": 8.0,
    }
    assert parsed.lower_bound == 0.5
    assert parsed.upper_bound == 8.0


def test_monotone_quantiles_leave_repair_flag_false():
    parsed = parse_belief_response(
        """
        {
          "point_estimate": 0.5,
          "quantiles": {"p05": 0.1, "p25": 0.3, "p50": 0.5, "p75": 0.7, "p95": 0.9},
          "citations": [],
          "reasoning_summary": ""
        }
        """
    )
    assert parsed.quantiles_repaired is False


def test_disordered_quantiles_trigger_running_max_repair():
    parsed = parse_belief_response(
        """
        {
          "point_estimate": 0.5,
          "quantiles": {"p05": 0.1, "p25": 0.3, "p50": 0.5, "p75": 0.4, "p95": 0.9},
          "citations": [],
          "reasoning_summary": ""
        }
        """
    )
    assert parsed.quantiles_repaired is True
    assert parsed.quantiles["p75"] == 0.5


def test_structured_payload_missing_quantiles_fails_parse():
    with pytest.raises(ValueError, match="missing quantiles: p05, p25"):
        parse_belief_response(
            """
            {
              "point_estimate": 0.4,
              "quantiles": {"p50": 0.4, "p75": 0.6, "p95": 0.8},
              "citations": [],
              "reasoning_summary": ""
            }
            """
        )


def test_free_text_without_quantiles_fails_parse():
    # Inkling once echoed the JSON-Schema URL and the text fallback plucked
    # "2020" out of "draft/2020-12" as a point estimate; the quantile-complete
    # contract turns that into a failed run instead of a parsed belief.
    with pytest.raises(ValueError, match="missing quantiles"):
        parse_belief_response(
            '{\n"https://json-schema.org/draft/2020-12/schema"\n'
        )
