"""Drive the v4 full-panel rerun across all 11 models.

Each model-batch runs in its own subprocess so a hung LiteLLM call
can be killed at the OS level without wedging the whole run.

Usage:
    python3 scripts/run_v4_full_panel.py [--dry-run] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CELL_TIMEOUT_SECONDS = 2700  # 45 min — any single cell past this is SIGKILLed

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


def exec_cell(
    provider: str,
    model_name: str,
    quantity_ids: list[str],
    prompt_version: str,
    output_dir: Path,
) -> int:
    """Run one cell in-process. This is invoked as the child in --exec-cell mode."""
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
    print(f"  {n_ok}/{len(records)} parsed")
    return 0


def run_cell_in_subprocess(
    provider: str,
    model_name: str,
    quantity_ids: list[str],
    prompt_version: str,
    output_dir: Path,
) -> tuple[int, int]:
    """Spawn the driver as a child in --exec-cell mode and enforce CELL_TIMEOUT_SECONDS."""
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--exec-cell",
        "--provider",
        provider,
        "--model",
        model_name,
        "--prompt-version",
        prompt_version,
        "--output-dir",
        str(output_dir),
        "--quantity-ids",
        ",".join(quantity_ids),
    ]
    try:
        proc = subprocess.run(
            cmd,
            timeout=CELL_TIMEOUT_SECONDS,
            capture_output=False,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if proc.returncode != 0:
            return 0, 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {CELL_TIMEOUT_SECONDS}s — child SIGKILLed, moving to next")
        return 0, 0

    return count_records(output_dir, prompt_version)


def count_records(output_dir: Path, expected_version: str) -> tuple[int, int]:
    runs_path = output_dir / "runs.jsonl"
    if not runs_path.exists():
        return 0, 0
    total = 0
    ok = 0
    with runs_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("prompt_version") != expected_version:
                continue
            total += 1
            if record.get("parsed_ok"):
                ok += 1
    return ok, total


def cell_already_complete(
    output_dir: Path, expected_version: str, expected_runs: int
) -> bool:
    ok, total = count_records(output_dir, expected_version)
    return total >= expected_runs


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
    parser.add_argument("--skip-complete", action="store_true")

    # Child-mode arguments — used when this script is re-invoked as a subprocess
    parser.add_argument("--exec-cell", action="store_true")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--prompt-version")
    parser.add_argument("--output-dir")
    parser.add_argument("--quantity-ids")

    args = parser.parse_args()

    if args.exec_cell:
        return exec_cell(
            provider=args.provider,
            model_name=args.model,
            quantity_ids=args.quantity_ids.split(","),
            prompt_version=args.prompt_version,
            output_dir=Path(args.output_dir),
        )

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
            expected_runs = len(quantity_ids) * 15
            if args.skip_complete and cell_already_complete(
                output_dir, batch_spec["prompt_version"], expected_runs
            ):
                print(
                    f"\n[{time.strftime('%H:%M:%S')}] {model_name} / {batch_key} "
                    f"SKIP (already complete under v={batch_spec['prompt_version']})"
                )
                continue

            print(
                f"\n[{time.strftime('%H:%M:%S')}] {model_name} / {batch_key} "
                f"({len(quantity_ids)} quantities, v={batch_spec['prompt_version']})"
            )

            if args.dry_run:
                print(f"  DRY RUN: would run {len(quantity_ids)} quantities x 15 runs")
                continue

            n_ok, n_total = run_cell_in_subprocess(
                provider=provider,
                model_name=model_name,
                quantity_ids=quantity_ids,
                prompt_version=batch_spec["prompt_version"],
                output_dir=output_dir,
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

    if total_runs:
        print(
            f"\nDone. {total_ok}/{total_runs} parsed "
            f"({100.0 * total_ok / total_runs:.1f}%) in "
            f"{time.time() - start:.0f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
