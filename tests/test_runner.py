from pathlib import Path

from llm_econ_beliefs import build_run_grid, write_run_grid_csv


def test_build_run_grid_and_write_csv(tmp_path: Path):
    runs = build_run_grid(
        model_names=["gpt-5", "claude-sonnet"],
        quantity_ids=["labor_supply.frisch_elasticity.prime_age"],
        n_runs=2,
    )

    assert len(runs) == 4
    assert runs[0].quantity_id == "labor_supply.frisch_elasticity.prime_age"

    output_path = tmp_path / "grid.csv"
    write_run_grid_csv(output_path, runs)

    assert output_path.exists()
    content = output_path.read_text()
    assert "model_name,quantity_id,run_index,prompt_version,prompt" in content

