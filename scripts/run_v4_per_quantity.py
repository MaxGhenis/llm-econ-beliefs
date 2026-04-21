"""Per-quantity rerun fallback for models where multi-quantity cells hang.

For a given model, spawns one subprocess per quantity (each doing 15 runs),
then merges all per-quantity outputs into a single canonical batch dir
matching what a normal cell run would produce.

Usage:
    python3 scripts/run_v4_per_quantity.py --model claude-sonnet-4.6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llm_econ_beliefs import list_quantities
from llm_econ_beliefs.experiment import (
    _write_jsonl,
    _write_requests_csv,
    _write_runs_csv,
    _write_summary_csv,
    summarize_run_results,
)
from llm_econ_beliefs.models import RequestLog, RunResult
from llm_econ_beliefs.runner import build_run_grid, write_run_grid_csv

PROVIDER_FOR_MODEL = {
    "claude-sonnet-4.6": "litellm",
    "claude-opus-4.7": "litellm",
    "gemini-3.1-pro-preview": "litellm",
    "grok-4.20": "litellm",
    "claude-haiku-4.5": "litellm",
    "gemini-3-flash-preview": "litellm",
    "gemini-3.1-flash-lite-preview": "litellm",
    "grok-4.1-fast": "litellm",
    "gpt-5.4": "openai",
    "gpt-5.4-mini": "openai",
    "gpt-5.4-nano": "openai",
}


def run_one_quantity(
    model_name: str,
    quantity_id: str,
    prompt_version: str,
    target_dir: Path,
    per_quantity_timeout: int = 900,
) -> bool:
    """Spawn subprocess for a single quantity, return True on success."""
    target_dir.mkdir(parents=True, exist_ok=True)
    provider = PROVIDER_FOR_MODEL[model_name]
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "run_v4_full_panel.py"),
        "--exec-cell",
        "--provider",
        provider,
        "--model",
        model_name,
        "--prompt-version",
        prompt_version,
        "--output-dir",
        str(target_dir),
        "--quantity-ids",
        quantity_id,
    ]
    try:
        proc = subprocess.run(
            cmd,
            timeout=per_quantity_timeout,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if proc.returncode != 0:
            print(f"  FAIL {quantity_id}: exit {proc.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {quantity_id}")
        return False


def load_runs(path: Path) -> list[RunResult]:
    if not path.exists():
        return []
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(RunResult(**data))
    return records


def load_requests(path: Path) -> list[RequestLog]:
    if not path.exists():
        return []
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(RequestLog(**data))
    return records


def merge_per_quantity(
    staging_root: Path, target_dir: Path, model_name: str, prompt_version: str
) -> None:
    """Gather per-quantity results into target_dir in canonical batch form."""
    target_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[RunResult] = []
    all_request_logs: list[RequestLog] = []
    next_request_index = 1
    for sub in sorted(staging_root.iterdir()):
        if not sub.is_dir():
            continue
        runs_path = sub / "runs.jsonl"
        if runs_path.exists():
            all_records.extend(load_runs(runs_path))
        requests_path = sub / "requests.jsonl"
        for log in load_requests(requests_path):
            log.request_index = next_request_index
            all_request_logs.append(log)
            next_request_index += 1

    if not all_records:
        print("No records to merge.")
        return

    quantity_ids = sorted({r.quantity_id for r in all_records})
    runs = build_run_grid(
        model_names=[model_name],
        quantity_ids=quantity_ids,
        n_runs=15,
        prompt_version=prompt_version,
        tool_regime="none",
    )
    write_run_grid_csv(target_dir / "prompt_grid.csv", runs)
    _write_jsonl(target_dir / "runs.jsonl", all_records)
    _write_runs_csv(target_dir / "runs.csv", all_records)
    if all_request_logs:
        _write_jsonl(target_dir / "requests.jsonl", all_request_logs)
        _write_requests_csv(target_dir / "requests.csv", all_request_logs)
    summaries = summarize_run_results(all_records, request_logs=all_request_logs)
    _write_summary_csv(target_dir / "summary.csv", summaries)
    print(
        f"Merged {len(all_records)} records and {len(all_request_logs)} "
        f"request logs into {target_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-version", default="v4")
    parser.add_argument(
        "--batch",
        choices=["main", "armington-clarify", "ies-clarify"],
        default="main",
    )
    parser.add_argument("--per-quantity-timeout", type=int, default=900)
    args = parser.parse_args()

    if args.batch == "main":
        qids = [
            q.id
            for q in list_quantities()
            # everything except simulation-facing sub-deciles/secondary:
            # those are part of the registry and get included
            if True
        ]
        target_dir = REPO_ROOT / "results" / f"{args.model}-elasticities-batch15"
    elif args.batch == "armington-clarify":
        qids = ["trade.armington_elasticity.import_domestic"]
        target_dir = REPO_ROOT / "results" / f"{args.model}-armington-clarify-batch15"
    elif args.batch == "ies-clarify":
        qids = ["household.intertemporal_elasticity_of_substitution"]
        target_dir = REPO_ROOT / "results" / f"{args.model}-ies-clarify-batch15"

    staging_root = REPO_ROOT / "results" / f"_perquantity_{args.model}_{args.batch}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    print(f"Running {args.model} / {args.batch} per-quantity ({len(qids)} quantities)")
    start = time.time()
    successes = 0
    for i, qid in enumerate(qids, 1):
        print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(qids)} {qid}")
        sub = staging_root / qid.replace(".", "_")
        if run_one_quantity(
            args.model,
            qid,
            args.prompt_version,
            sub,
            per_quantity_timeout=args.per_quantity_timeout,
        ):
            successes += 1

    elapsed = time.time() - start
    print(f"\nPer-quantity phase done: {successes}/{len(qids)} in {elapsed:.0f}s")

    merge_per_quantity(staging_root, target_dir, args.model, args.prompt_version)
    shutil.rmtree(staging_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
