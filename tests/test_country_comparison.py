"""US-lab versus Chinese-lab comparison in build_correlates."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_correlates import (  # noqa: E402
    build_country_comparison_rows,
    country_permutation_p,
)
from llm_econ_beliefs.model_registry import (  # noqa: E402
    MODEL_REGISTRY,
    ORGANIZATION_COUNTRY,
    country_for_organization,
)


def test_every_registry_organization_has_a_country():
    for model in MODEL_REGISTRY:
        assert country_for_organization(model.organization) in {"us", "china"}


def test_registry_has_exactly_seven_chinese_models():
    chinese = [
        model
        for model in MODEL_REGISTRY
        if ORGANIZATION_COUNTRY[model.organization] == "china"
    ]
    assert sorted(model.model_id for model in chinese) == [
        "deepseek-v4-pro",
        "glm-5.2",
        "kimi-k2.6",
        "kimi-k3",
        "minimax-m3",
        "qwen-3.7-max",
        "qwen3.8-max",
    ]


def _summary_row(model_id: str, organization: str, tau: float) -> dict:
    return {
        "model": model_id,
        "organization": organization,
        "tau_star_pct": tau,
        "eti_median": tau / 100,
        "avg_width_rank": 5.0,
        "mean_abs_center_labor_tax": 0.4,
        "mean_abs_center_macro_trade": 1.2,
    }


def test_country_comparison_medians_and_diff():
    rows = [
        _summary_row("gpt-a", "openai", 30.0),
        _summary_row("gpt-b", "openai", 34.0),
        _summary_row("claude-a", "anthropic", 38.0),
        _summary_row("kimi-a", "moonshot", 40.0),
        _summary_row("glm-a", "zhipu", 44.0),
    ]
    result = build_country_comparison_rows(rows)
    tau_row = next(r for r in result if "top rate" in str(r["outcome"]))
    assert tau_row["us_n"] == 3
    assert tau_row["china_n"] == 2
    assert tau_row["us_median"] == 34.0
    assert tau_row["china_median"] == 42.0
    assert tau_row["china_minus_us"] == 8.0
    assert 0.0 <= float(tau_row["permutation_p"]) <= 1.0
    assert "confounded" in str(tau_row["disclosure"])
    assert "completion budget" in str(tau_row["disclosure"])


def test_country_multiplicity_mirrors_capability_cut():
    rows = [
        _summary_row("gpt-a", "openai", 30.0),
        _summary_row("gpt-b", "openai", 34.0),
        _summary_row("claude-a", "anthropic", 38.0),
        _summary_row("kimi-a", "moonshot", 40.0),
        _summary_row("glm-a", "zhipu", 44.0),
    ]
    result = build_country_comparison_rows(rows)
    tau_row = next(r for r in result if "top rate" in str(r["outcome"]))
    eti_row = next(r for r in result if r["outcome"] == "ETI pooled median")
    tested = [r for r in result if not r["is_derived"]]
    assert tau_row["is_derived"] is True
    assert all(r["family_size"] == len(tested) for r in tested)
    # The derived top-rate row mirrors ETI's adjusted values (one hypothesis).
    assert tau_row["holm_adjusted_p"] == eti_row["holm_adjusted_p"]
    assert tau_row["bh_adjusted_p"] == eti_row["bh_adjusted_p"]
    for row in tested:
        assert float(row["holm_adjusted_p"]) >= float(row["permutation_p"]) - 1e-12
        assert float(row["bh_adjusted_p"]) >= float(row["permutation_p"]) - 1e-12


def test_permutation_p_is_one_for_identical_groups():
    p = country_permutation_p([1.0, 1.0, 1.0], [1.0, 1.0])
    assert p == 1.0


def test_permutation_p_detects_separated_groups():
    us = [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0]
    china = [50.0, 51.0, 52.0, 53.0, 54.0]
    p = country_permutation_p(us, china)
    # Exact enumeration over C(13, 5) = 1287 assignments: 20 concentrate
    # enough top values in the china group to reach the observed gap.
    # Medians are coarse at these group sizes, so this is the exact floor.
    assert abs(p - 20 / 1287) < 1e-12
