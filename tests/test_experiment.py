from pathlib import Path

from llm_econ_beliefs import ProviderBatchResult
from llm_econ_beliefs.experiment import (
    resolve_quantity_ids,
    run_claude_experiment,
    run_openai_experiment,
)


def test_resolve_quantity_ids_combines_explicit_ids_and_tags():
    quantity_ids = resolve_quantity_ids(
        ["household.annual_discount_factor"],
        ["og_usa"],
    )

    assert "household.annual_discount_factor" in quantity_ids
    assert "labor_supply.frisch_elasticity.prime_age" in quantity_ids


def test_run_claude_experiment_writes_outputs(tmp_path: Path):
    responses = iter(
        [
            """
            {
              "interpretation": "Annual discount factor",
              "point_estimate": 0.96,
              "quantiles": {
                "p05": 0.94,
                "p25": 0.95,
                "p50": 0.96,
                "p75": 0.97,
                "p95": 0.99
              },
              "citations": ["OG-Core docs"],
              "reasoning_summary": "Typical annual calibration value."
            }
            """,
            """
            {
              "interpretation": "Annual discount factor",
              "point_estimate": 0.97,
              "quantiles": {
                "p05": 0.95,
                "p25": 0.96,
                "p50": 0.97,
                "p75": 0.98,
                "p95": 0.99
              },
              "citations": ["OG-Core docs"],
              "reasoning_summary": "Typical annual calibration value."
            }
            """,
        ]
    )

    def fake_invoke(prompt: str, model_name: str) -> str:
        return next(responses)

    records, summaries = run_claude_experiment(
        quantity_ids=["household.annual_discount_factor"],
        n_runs=2,
        output_dir=tmp_path,
        model_name="sonnet",
        invoke=fake_invoke,
    )

    assert len(records) == 2
    assert records[0].quantiles["p50"] == 0.96
    assert summaries[0]["quantity_id"] == "household.annual_discount_factor"
    assert (tmp_path / "runs.jsonl").exists()
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "prompt_grid.csv").exists()


def test_run_openai_experiment_batches_samples_per_request(tmp_path: Path):
    batches = []

    def fake_invoke_batch(prompt: str, model_name: str, n: int) -> ProviderBatchResult:
        batches.append((prompt, model_name, n))
        return ProviderBatchResult(
            outputs=[
                f"""
                {{
                  "interpretation": "Annual discount factor",
                  "point_estimate": {0.95 + 0.01 * index:.2f},
                  "quantiles": {{
                    "p05": 0.94,
                    "p25": 0.95,
                    "p50": {0.95 + 0.01 * index:.2f},
                    "p75": 0.98,
                    "p95": 0.99
                  }},
                  "citations": ["OG-Core docs"],
                  "reasoning_summary": "Typical annual calibration value."
                }}
                """
                for index in range(n)
            ],
            request_id=f"req_{len(batches)}",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 25 * n,
                "total_tokens": 100 + 25 * n,
                "prompt_tokens_details": {"cached_tokens": 10},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        )

    records, summaries = run_openai_experiment(
        quantity_ids=["household.annual_discount_factor"],
        n_runs=5,
        output_dir=tmp_path,
        model_name="gpt-5.4-mini",
        batch_size=3,
        invoke_batch=fake_invoke_batch,
    )

    assert len(records) == 5
    assert summaries[0]["n_successful_runs"] == 5
    assert batches[0][2] == 3
    assert batches[1][2] == 2
    assert summaries[0]["n_requests"] == 2
    assert summaries[0]["usage_prompt_tokens_total"] == 200
    assert summaries[0]["usage_completion_tokens_total"] == 125
    assert summaries[0]["usage_estimated_total_cost_usd_total"] is not None
    assert (tmp_path / "requests.jsonl").exists()
    assert (tmp_path / "requests.csv").exists()


def test_run_openai_experiment_caps_batch_size_at_openai_limit(tmp_path: Path):
    batches = []

    def fake_invoke_batch(prompt: str, model_name: str, n: int) -> ProviderBatchResult:
        batches.append((prompt, model_name, n))
        return ProviderBatchResult(
            outputs=[
                """
                {
                  "interpretation": "Annual discount factor",
                  "point_estimate": 0.96,
                  "quantiles": {
                    "p05": 0.94,
                    "p25": 0.95,
                    "p50": 0.96,
                    "p75": 0.97,
                    "p95": 0.99
                  },
                  "citations": ["OG-Core docs"],
                  "reasoning_summary": "Typical annual calibration value."
                }
                """
                for _ in range(n)
            ],
            request_id=f"req_{len(batches)}",
            usage={},
        )

    records, summaries = run_openai_experiment(
        quantity_ids=["household.annual_discount_factor"],
        n_runs=15,
        output_dir=tmp_path,
        model_name="gpt-5.4-mini",
        batch_size=15,
        invoke_batch=fake_invoke_batch,
    )

    assert len(records) == 15
    assert summaries[0]["n_successful_runs"] == 15
    assert [batch[2] for batch in batches] == [8, 7]
