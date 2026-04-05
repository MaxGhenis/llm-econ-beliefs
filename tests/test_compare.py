from pathlib import Path

from llm_econ_beliefs.compare import (
    build_comparison_rows,
    read_summary_rows,
    write_comparison_csv,
)


def test_build_comparison_rows_filters_and_formats_intervals(tmp_path: Path):
    result_dir = tmp_path / "gpt-5.4-frisch"
    result_dir.mkdir()
    (result_dir / "summary.csv").write_text(
        "\n".join(
            [
                "model_name,quantity_id,n_successful_runs,pooled_point_estimate,pooled_lower_bound,pooled_upper_bound,reml_predictive_lower,reml_predictive_upper,bayes_predictive_lower,bayes_predictive_upper,usage_total_tokens_per_successful_run,usage_estimated_total_cost_usd_per_successful_run",
                "gpt-5.4,labor_supply.frisch_elasticity.prime_age,5,0.5,0.2,1.5,0.18,1.31,0.24,1.03,336.4,0.0037385",
                "gpt-5.4,household.annual_discount_factor,5,0.96,0.94,0.99,0.94,0.99,0.95,0.98,150.0,0.001",
            ]
        )
    )

    rows = build_comparison_rows(
        [result_dir],
        quantity_id="labor_supply.frisch_elasticity.prime_age",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["source_label"] == "gpt-5.4-frisch"
    assert row["pooled_90_interval"] == "[0.2, 1.5]"
    assert row["reml_predictive_90_interval"] == "[0.18, 1.31]"
    assert row["bayes_predictive_90_interval"] == "[0.24, 1.03]"


def test_write_comparison_csv_round_trips_rows(tmp_path: Path):
    result_dir = tmp_path / "mini"
    result_dir.mkdir()
    (result_dir / "summary.csv").write_text(
        "\n".join(
            [
                "model_name,quantity_id,n_successful_runs,pooled_point_estimate,pooled_lower_bound,pooled_upper_bound,reml_predictive_lower,reml_predictive_upper,bayes_predictive_lower,bayes_predictive_upper,usage_total_tokens_per_successful_run,usage_estimated_total_cost_usd_per_successful_run",
                "gpt-5.4-mini,labor_supply.frisch_elasticity.prime_age,5,0.54,0.1,1.95,0.13,1.92,0.19,1.41,363.4,0.00124305",
            ]
        )
    )

    rows = build_comparison_rows([result_dir])
    output_path = tmp_path / "comparison.csv"
    write_comparison_csv(output_path, rows)

    content = output_path.read_text()
    assert "source_label,model_name,quantity_id" in content
    assert "gpt-5.4-mini" in content
    assert "[0.1, 1.95]" in content
    assert "0.00124305" in content


def test_read_summary_rows_parses_numeric_cells(tmp_path: Path):
    result_dir = tmp_path / "nano"
    result_dir.mkdir()
    (result_dir / "summary.csv").write_text(
        "\n".join(
            [
                "model_name,quantity_id,n_successful_runs,pooled_point_estimate",
                "gpt-5.4-nano,labor_supply.frisch_elasticity.prime_age,5,0.72",
            ]
        )
    )

    rows = read_summary_rows(result_dir)

    assert rows[0]["n_successful_runs"] == 5
    assert rows[0]["pooled_point_estimate"] == 0.72
