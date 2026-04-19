"""Drive the v4 full-panel rerun across all 11 models.

Batches run sequentially so we can watch for cost explosions and kill early.
Each model-batch writes to its own results dir, overwriting previous runs.

Usage:
    python3 scripts/run_v4_full_panel.py [--dry-run] [--smoke]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llm_econ_beliefs import (
    list_quantities,
    run_litellm_experiment,
    run_openai_experiment,
)

MODELS = [
    ("openai", "gpt-5.4-nano"),
    ("openai", "gpt-5.4-mini"),
    ("openai", "gpt-5.4"),
    ("litellm", "claude-haiku-4.5"),
    ("litellm", "claude-sonnet-4.6"),
    ("litellm", "claude-opus-4.7"),
    ("litellm", "gemini-3.1-flash-lite-preview"),
    ("litellm", "gemini-3-flash-preview"),
    ("litellm", "gemini-3.1-pro-preview"),
    ("litellm", "grok-4.1-fast"),
    ("litellm", "grok-4.20"),
]

BATCHES = {
    "elasticities-batch15": {
        "quantity_filter": lambda q: True,
        "prompt_version": "v4",
    },
    "armington-clarify-batch15": {
        "quantity_filter": lambda q: q.id == "trade.armington_elasticity.import_domestic",
        "prompt_version": "armington-clarify",
    },
    "ies-clarify-batch15": {
        "quantity_filter": lambda q: q.id == "household.intertemporal_elasticity_of_substitution",
        "prompt_version": "ies-clarify",
    },
}


def run_cell(
    provider: str,
    model_name: str,
    quantity_ids: list[str],
    prompt_version: str,
    output_dir: Path,
    dry_run: bool,
) -> tuple[int, int]:
    """Run one model-batch cell. Returns (n_successful, n_total)."""
    if dry_run:
        print(f"  DRY RUN: would run {len(quantity_ids)} quantities x 15 runs")
        return 0, 0

    kwargs = dict(
        quantity_ids=quantity_ids,
        n_runs=15,
        output_dir=str(output_dir),
        model_name=model_name,
        prompt_version=prompt_version,
        tool_regime="none",
    )
    if provider == "openai":
        kwargs["batch_size"] = 5
        records, _ = run_openai_experiment(**kwargs)
    else:
        records, _ = run_litellm_experiment(**kwargs)

    n_ok = sum(1 for r in records if r.parsed_ok)
    return n_ok, len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="only gpt-5.4-nano main batch")
    parser.add_argument("--only-model", help="run a single model")
    parser.add_argument(
        "--only-batch",
        choices=list(BATCHES.keys()),
        help="run a single batch type",
    )
    args = parser.parse_args()

    results_root = REPO_ROOT / "results"
    results_root.mkdir(exist_ok=True)

    all_quantities = list_quantities()
    models = MODELS
    if args.smoke:
        models = [("openai", "gpt-5.4-nano")]
    if args.only_model:
        models = [(p, m) for p, m in models if m == args.only_model]
        if not models:
            print(f"No such model: {args.only_model}")
            return 2

    batches = BATCHES
    if args.smoke or args.only_batch:
        key = args.only_batch or "elasticities-batch15"
        batches = {key: BATCHES[key]}

    total_ok = 0
    total_runs = 0
    start = time.time()

    for provider, model_name in models:
        for batch_key, batch_spec in batches.items():
            quantity_ids = [
                q.id for q in all_quantities if batch_spec["quantity_filter"](q)
            ]
            if not quantity_ids:
                continue

            output_dir = results_root / f"{model_name}-{batch_key}"
            print(
                f"\n[{time.strftime('%H:%M:%S')}] {model_name} / {batch_key} "
                f"({len(quantity_ids)} quantities, v={batch_spec['prompt_version']})"
            )

            try:
                n_ok, n_total = run_cell(
                    provider=provider,
                    model_name=model_name,
                    quantity_ids=quantity_ids,
                    prompt_version=batch_spec["prompt_version"],
                    output_dir=output_dir,
                    dry_run=args.dry_run,
                )
                total_ok += n_ok
                total_runs += n_total
                if n_total:
                    rate = 100.0 * n_ok / n_total
                    elapsed = time.time() - start
                    print(
                        f"  {n_ok}/{n_total} parsed ({rate:.1f}%), "
                        f"elapsed {elapsed:.0f}s"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                continue

    if total_runs:
        print(
            f"\nDone. {total_ok}/{total_runs} parsed "
            f"({100.0 * total_ok / total_runs:.1f}%) in "
            f"{time.time() - start:.0f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
