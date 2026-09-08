from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from llm_econ_beliefs.model_registry import (
    FRONTIER_MODEL_IDS,
    MODEL_REGISTRY,
    ORGANIZATION_DISPLAY_LABELS,
    ORGANIZATIONS,
    PANEL_MODEL_IDS,
    SERVING_PROVIDER_PATHS,
    WAVE_DISPLAY_LABELS,
    WAVES,
    write_model_registry_csv,
)
from paper import build_tables as paper_build_tables
from paper.build_tables import (
    ComparisonRow,
    build_leave_one_organization_out_table,
    pooled_eti_quantiles_from_csv_rows,
)
from scripts import build_comparison_artifacts, build_correlates, panel_grid_gate
from scripts.build_correlates import (
    DERIVED_TAU_LABEL,
    EXPECTED_PANEL_ONLY,
    EXPECTED_POLICYBENCH_ONLY,
    bh_adjusted_pvalues,
    holm_adjusted_pvalues,
    pooled_eti_median,
    validate_policybench_crosswalk,
)
from scripts.check_panel_grid import GridCheckResult


def test_registry_has_exact_taxonomy_and_round_trips_to_csv(tmp_path: Path) -> None:
    assert len(MODEL_REGISTRY) == 31
    assert len(PANEL_MODEL_IDS) == len(set(PANEL_MODEL_IDS)) == 31
    assert {model.organization for model in MODEL_REGISTRY} == set(ORGANIZATIONS)
    assert {model.serving_provider_path for model in MODEL_REGISTRY} == set(
        SERVING_PROVIDER_PATHS
    )
    assert {model.wave for model in MODEL_REGISTRY} == set(WAVES)

    output = write_model_registry_csv(tmp_path / "model-registry.csv")
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert tuple(rows[0]) == (
        "model_id",
        "display_label",
        "organization",
        "serving_provider_path",
        "model_family",
        "wave",
        "organization_label",
        "wave_label",
        "is_frontier",
    )
    assert [row["model_id"] for row in rows] == list(PANEL_MODEL_IDS)
    assert rows[0] == {
        "model_id": MODEL_REGISTRY[0].model_id,
        "display_label": MODEL_REGISTRY[0].display_label,
        "organization": MODEL_REGISTRY[0].organization,
        "serving_provider_path": MODEL_REGISTRY[0].serving_provider_path,
        "model_family": MODEL_REGISTRY[0].model_family,
        "wave": MODEL_REGISTRY[0].wave,
        "organization_label": ORGANIZATION_DISPLAY_LABELS[
            MODEL_REGISTRY[0].organization
        ],
        "wave_label": WAVE_DISPLAY_LABELS[MODEL_REGISTRY[0].wave],
        "is_frontier": str(MODEL_REGISTRY[0].model_id in FRONTIER_MODEL_IDS).lower(),
    }
    frontier_rows = [row for row in rows if row["is_frontier"] == "true"]
    assert {row["model_id"] for row in frontier_rows} == set(FRONTIER_MODEL_IDS)
    assert len({row["organization"] for row in frontier_rows}) == len(frontier_rows)


def _valid_policybench_rows() -> list[dict[str, str]]:
    models = sorted(
        (set(PANEL_MODEL_IDS) - EXPECTED_PANEL_ONLY) | EXPECTED_POLICYBENCH_ONLY
    )
    return [
        {
            "model": model,
            "source_release": "fixture-release",
            "condition": "no_tools",
            "country": "us",
        }
        for model in models
    ]


def test_crosswalk_prints_both_antijoins_and_accepts_only_expected_mismatches() -> None:
    stream = io.StringIO()

    validate_policybench_crosswalk(_valid_policybench_rows(), stream=stream)

    line = stream.getvalue().strip()
    assert line.startswith("POLICYBENCH_CROSSWALK=")
    payload = json.loads(line.removeprefix("POLICYBENCH_CROSSWALK="))
    assert set(payload["panel_only"]) == EXPECTED_PANEL_ONLY
    assert set(payload["policybench_only"]) == EXPECTED_POLICYBENCH_ONLY


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows + [dict(rows[0])], "Duplicate PolicyBench model IDs"),
        (
            lambda rows: [
                *rows[:-1],
                {**rows[-1], "source_release": "other-release"},
            ],
            "source_release must have exactly one",
        ),
        (
            lambda rows: [{**row, "condition": "with_tools"} for row in rows],
            "Expected PolicyBench condition",
        ),
        (
            lambda rows: [{**row, "country": "uk"} for row in rows],
            "Expected PolicyBench country",
        ),
        (
            lambda rows: [row for row in rows if row["model"] != "gpt-5.4-mini"],
            "Unexpected panel-only",
        ),
        (
            lambda rows: rows + [{**rows[0], "model": "unexpected-model"}],
            "Unexpected PolicyBench-only",
        ),
    ],
)
def test_crosswalk_fails_loudly_on_duplicates_or_unexpected_mismatches(
    mutate, message: str
) -> None:
    rows = _valid_policybench_rows()

    with pytest.raises(ValueError, match=message):
        validate_policybench_crosswalk(mutate(rows), stream=io.StringIO())


def test_grid_gate_excludes_incomplete_models_and_allow_partial_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_check(_results_root, *, model_name, batch, require_parsed):
        assert batch == "elasticities-batch15"
        calls.append((model_name, require_parsed))
        errors = () if model_name == "complete" else ("missing: fixture",)
        return GridCheckResult(390, 390 if not errors else 389, errors)

    monkeypatch.setattr(panel_grid_gate, "check_batch_directory", fake_check)
    stream = io.StringIO()
    gated = panel_grid_gate.gate_panel_models(
        Path("unused"), ["complete", "partial"], stream=stream
    )

    assert gated.included_models == ("complete",)
    assert gated.excluded_models == ("partial",)
    assert gated.incomplete_models == ("partial",)
    assert calls == [("complete", True), ("partial", True)]
    payload = json.loads(
        stream.getvalue().strip().removeprefix("PANEL_GRID_EXCLUSIONS=")
    )
    assert payload["excluded_models"] == ["partial"]
    assert payload["errors_by_model"] == {"partial": ["missing: fixture"]}

    override = panel_grid_gate.gate_panel_models(
        Path("unused"),
        ["complete", "partial"],
        allow_partial=True,
        stream=io.StringIO(),
    )
    assert override.included_models == ("complete", "partial")
    assert override.excluded_models == ()
    assert override.incomplete_models == ("partial",)


def test_comparison_summary_integrity_requires_one_complete_row_per_quantity() -> None:
    rows = [
        {
            "model_name": "gpt-5.4",
            "quantity_id": quantity.id,
            "n_successful_runs": 15,
            "pooled_point_estimate": 0.5,
            "pooled_lower_bound": 0.25,
            "pooled_upper_bound": 0.75,
        }
        for quantity in build_comparison_artifacts.list_quantities()
    ]
    assert build_comparison_artifacts._summary_integrity_errors(rows, "gpt-5.4") == ()

    broken = rows[:-1] + [dict(rows[0])]
    errors = build_comparison_artifacts._summary_integrity_errors(broken, "gpt-5.4")
    assert any(error.startswith("duplicate quantity_id") for error in errors)
    assert any(error.startswith("missing quantity_id") for error in errors)


def test_holm_and_bh_adjustments_match_known_stepwise_values() -> None:
    pvalues = [0.01, 0.04, 0.03, 0.002]

    assert holm_adjusted_pvalues(pvalues) == pytest.approx([0.03, 0.06, 0.06, 0.008])
    assert bh_adjusted_pvalues(pvalues) == pytest.approx([0.02, 0.04, 0.04, 0.008])


def test_derived_tau_row_reuses_eti_test_and_family_has_eight_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_correlates, "permutation_p", lambda xs, ys: 0.05)
    rows = [
        {
            "policybench_tax_within_dollar": float(index),
            "policybench_within_dollar": float(index * 2),
            "mean_abs_center_labor_tax": float(index + 1),
            "mean_abs_center_macro_trade": float(5 - index),
            "avg_width_rank": float(index % 3),
            "eti_median": float(index),
            "tau_star_pct": float(100 - index),
        }
        for index in range(1, 6)
    ]

    output = build_correlates.build_correlation_rows(rows)

    assert sum(not row["is_derived"] for row in output) == 8
    assert {row["family_size"] for row in output} == {8}
    assert DERIVED_TAU_LABEL.endswith(
        "derived, monotone transform of ETI — not an additional test"
    )
    for predictor in {row["predictor"] for row in output}:
        eti = next(
            row
            for row in output
            if row["predictor"] == predictor
            and row["outcome"] == build_correlates.PRIMARY_OUTCOME_LABEL
        )
        tau = next(
            row for row in output if row["predictor"] == predictor and row["is_derived"]
        )
        assert tau["raw_p"] == eti["raw_p"]
        assert tau["holm_adjusted_p"] == eti["holm_adjusted_p"]
        assert tau["bh_adjusted_p"] == eti["bh_adjusted_p"]
        assert tau["spearman_rho"] == pytest.approx(-eti["spearman_rho"])


def test_sensitivities_cover_each_organization_both_wave_groups_and_clusters() -> None:
    overlap = [
        model for model in MODEL_REGISTRY if model.model_id not in EXPECTED_PANEL_ONLY
    ]
    summary_rows = [
        {
            "model": model.model_id,
            "organization": model.organization,
            "wave": model.wave,
            build_correlates.PRIMARY_PREDICTOR_KEY: float(index),
            build_correlates.PRIMARY_OUTCOME_KEY: float(index * 2 + index % 3),
        }
        for index, model in enumerate(overlap, 1)
    ]

    output = build_correlates.build_sensitivity_rows(summary_rows)
    leave_out = [
        row for row in output if row["analysis"] == "leave_one_organization_out"
    ]
    within_wave = [row for row in output if row["analysis"] == "within_wave"]
    cluster = next(
        row for row in output if row["analysis"] == "organization_cluster_permutation"
    )

    assert {row["omitted_organization"] for row in leave_out} == set(ORGANIZATIONS)
    assert {row["wave"] for row in within_wave} == {"april_only", "july_only"}
    assert cluster["n_models"] == len(overlap)
    assert cluster["n_organizations"] == len(ORGANIZATIONS)
    assert cluster["statistic_level"] == "model"
    assert (
        cluster["permutation_unit"]
        == "organization_score_block_within_equal_size_strata"
    )
    # Equal-size strata over the 28-model overlap: anthropic at 7, openai
    # at 6, google at 5 (1! each), xai/alibaba/moonshot at 2 (3!), and
    # four single-model labs (4!) -> 6 * 24 = 144 block assignments.
    assert cluster["n_permutations"] == 144
    assert 0 < cluster["permutation_p"] <= 1


def test_runs_jsonl_eti_median_matches_build_tables_piecewise_mixture(
    tmp_path: Path,
) -> None:
    quantile_sets = [
        {"p05": 0.10, "p25": 0.20, "p50": 0.35, "p75": 0.55, "p95": 0.80},
        {"p05": 0.20, "p25": 0.30, "p50": 0.45, "p75": 0.70, "p95": 1.10},
        {"p05": 0.05, "p25": 0.15, "p50": 0.25, "p75": 0.40, "p95": 0.65},
    ]
    json_rows = [
        {
            "model_name": "gpt-5.4",
            "prompt_version": "v4",
            "quantity_id": build_correlates.ETI_QUANTITY_ID,
            "run_index": index,
            "parsed_ok": True,
            "point_estimate": quantiles["p50"],
            "quantiles": quantiles,
        }
        for index, quantiles in enumerate(quantile_sets, 1)
    ]
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text("".join(json.dumps(row) + "\n" for row in json_rows))
    loaded_json_rows = [json.loads(line) for line in runs_path.read_text().splitlines()]
    csv_rows = [
        {
            **{key: str(value) for key, value in row.items() if key != "quantiles"},
            "parsed_ok": "True",
            "quantiles": json.dumps(row["quantiles"]),
        }
        for row in json_rows
    ]

    paper_quantiles = pooled_eti_quantiles_from_csv_rows(csv_rows)

    assert paper_quantiles is not None
    assert pooled_eti_median(loaded_json_rows) == pytest.approx(
        paper_quantiles[1], abs=1e-14
    )


def test_build_tables_rejects_legacy_correlates_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "correlates-model-summary.csv").write_text("model,eti_median\na,0.3\n")
    (tmp_path / "correlates-spearman.csv").write_text(
        "predictor,outcome,n_models,spearman_rho,permutation_p\nx,y,4,0.5,0.2\n"
    )
    monkeypatch.setattr(paper_build_tables, "RESULTS_DIR", tmp_path)

    with pytest.raises(ValueError, match="Stale correlates-spearman.csv"):
        paper_build_tables.build_correlates_tables()


def test_leave_one_organization_out_keeps_independent_labs_separate() -> None:
    independent_models = (
        "deepseek-v4-pro",
        "qwen-3.7-max",
        "kimi-k2.6",
        "glm-5.2",
        "minimax-m3",
    )
    rows = [
        ComparisonRow(
            model_name=model_id,
            quantity_id=build_correlates.ETI_QUANTITY_ID,
            n_successful_runs=15,
            pooled_point_estimate=float(index),
            pooled_lower=float(index) - 0.1,
            pooled_upper=float(index) + 0.1,
            pooled_width=0.2,
            cost_per_run_usd=None,
            source_dir="unused",
        )
        for index, model_id in enumerate(independent_models, 1)
    ]

    output = build_leave_one_organization_out_table(
        labor_tax_rows=rows,
        macro_trade_rows=[],
    )
    omitted = {row["Omitted organization"] for row in output}

    assert omitted == {"DeepSeek", "Alibaba", "Moonshot AI", "Zhipu AI", "MiniMax"}
    assert "Other" not in omitted


def test_consistency_metrics_share_math() -> None:
    from scripts.build_correlates import consistency_metrics

    cells = [(1.0, 0.0), (1.0, 1.0), (0.0, 2.0)]
    metrics = consistency_metrics(cells)
    # Shares: 0, 0.5, 1.0 -> max 1.0; between SDs 0, 1, 2 -> median 1.
    assert metrics["max_between_run_share"] == 1.0
    assert metrics["median_between_run_sd"] == 1.0
    with pytest.raises(ValueError):
        consistency_metrics([(0.0, 0.0)])
