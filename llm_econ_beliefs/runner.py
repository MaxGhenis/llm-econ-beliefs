"""Build repeatable prompt grids for batch elicitation."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .models import PromptRun
from .prompts import create_belief_prompt
from .registry import get_quantity


def build_run_grid(
    *,
    model_names: Sequence[str],
    quantity_ids: Sequence[str],
    n_runs: int,
    prompt_version: str = "v1",
    include_uncertainty: bool = True,
) -> list[PromptRun]:
    """Create a grid of prompts ready to hand to an inference client."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    runs: list[PromptRun] = []
    for model_name in model_names:
        for quantity_id in quantity_ids:
            quantity = get_quantity(quantity_id)
            prompt = create_belief_prompt(
                quantity,
                include_uncertainty=include_uncertainty,
            )
            for run_index in range(1, n_runs + 1):
                runs.append(
                    PromptRun(
                        model_name=model_name,
                        quantity_id=quantity_id,
                        run_index=run_index,
                        prompt_version=prompt_version,
                        prompt=prompt,
                    )
                )
    return runs


def write_run_grid_csv(path: str | Path, runs: Sequence[PromptRun]) -> None:
    """Write a prompt grid to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "quantity_id",
                "run_index",
                "prompt_version",
                "prompt",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))
