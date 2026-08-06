"""Per-quantity rerun fallback for models where multi-quantity cells hang.

For a given model, spawns one subprocess per quantity (each doing 15 runs),
then merges all per-quantity outputs into a single canonical batch dir
matching what a normal cell run would produce.

Reruns resume from the staging directory by default: quantities whose staged
`runs.jsonl` already holds a full set of records are skipped, so an
interrupted panel run picks up where it left off. Pass `--fresh` to discard
staged results and re-elicit everything.

Usage:
    python3 scripts/run_v4_per_quantity.py --model claude-sonnet-4.6
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_panel_grid import check_run_files, validate_grid_rows

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
    "gpt-5.5": "openai",
    "claude-sonnet-5": "anthropic",
    "claude-opus-4.8": "anthropic",
    "claude-opus-5": "anthropic",
    "claude-fable-5": "anthropic",
    "gemini-3.5-flash": "litellm",
    "gemini-3.6-flash": "litellm",
    "grok-4.3": "litellm",
    "grok-4.5": "litellm",
    "deepseek-v4-pro": "litellm",
    "qwen-3.7-max": "litellm",
    "kimi-k2.6": "litellm",
    "kimi-k3": "litellm",
    "glm-5.2": "litellm",
    "minimax-m3": "litellm",
    "qwen3.8-max": "litellm",
    "inkling": "litellm",
    "gpt-5.6-sol": "openai",
    "gpt-5.6-luna": "openai",
    "gpt-5.6-terra": "openai",
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
    staging_root: Path,
    target_dir: Path,
    model_name: str,
    prompt_version: str,
    *,
    expected_quantity_ids: Sequence[str] | None = None,
    n_runs: int = 15,
) -> bool:
    """Validate staging and atomically publish canonical batch artifacts.

    Production callers pass ``expected_quantity_ids`` so the exact grid is
    checked before any target artifact is touched.  Returning ``False`` leaves
    both the staging tree and the canonical target unchanged.
    """
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
        return False

    quantity_ids = (
        list(expected_quantity_ids)
        if expected_quantity_ids is not None
        else sorted({record.quantity_id for record in all_records})
    )
    if expected_quantity_ids is not None:
        grid_result = validate_grid_rows(
            (
                {
                    "model_name": record.model_name,
                    "prompt_version": record.prompt_version,
                    "quantity_id": record.quantity_id,
                    "run_index": record.run_index,
                    "parsed_ok": record.parsed_ok,
                }
                for record in all_records
            ),
            model_name=model_name,
            prompt_version=prompt_version,
            quantity_ids=quantity_ids,
            n_runs=n_runs,
        )
        if not grid_result.ok:
            print("Staging grid is incomplete or invalid; publication refused.")
            for error in grid_result.errors:
                print(f"  {error}")
            return False

    quantity_order = {
        quantity_id: index for index, quantity_id in enumerate(quantity_ids)
    }
    all_records.sort(
        key=lambda record: (
            quantity_order.get(record.quantity_id, len(quantity_order)),
            record.run_index,
        )
    )
    runs = build_run_grid(
        model_names=[model_name],
        quantity_ids=quantity_ids,
        n_runs=n_runs,
        prompt_version=prompt_version,
        tool_regime="none",
    )
    summaries = summarize_run_results(all_records, request_logs=all_request_logs)

    # Materialize every artifact away from the canonical directory first.
    # Each final os.replace is atomic. Existing artifacts are backed up before
    # publication so an I/O failure partway through can roll the whole set back
    # instead of leaving a mixed old/new canonical batch.
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}.publish-",
            dir=target_dir.parent,
        )
    )
    try:
        artifacts = (
            "prompt_grid.csv",
            "runs.csv",
            "requests.jsonl",
            "requests.csv",
            "summary.csv",
            # Publish runs.jsonl last: it is the authoritative completeness
            # artifact consumed by the exact-grid checker.
            "runs.jsonl",
        )
        write_run_grid_csv(temp_dir / "prompt_grid.csv", runs)
        _write_jsonl(temp_dir / "runs.jsonl", all_records)
        _write_runs_csv(temp_dir / "runs.csv", all_records)
        _write_jsonl(temp_dir / "requests.jsonl", all_request_logs)
        _write_requests_csv(temp_dir / "requests.csv", all_request_logs)
        _write_summary_csv(temp_dir / "summary.csv", summaries)
        if not (temp_dir / "summary.csv").exists():
            (temp_dir / "summary.csv").touch()

        target_existed = target_dir.exists()
        backup_dir = temp_dir / "backup"
        backup_dir.mkdir()
        if target_existed:
            if not target_dir.is_dir():
                print(f"Canonical target is not a directory: {target_dir}")
                return False
            for artifact in artifacts:
                existing = target_dir / artifact
                if existing.exists():
                    shutil.copy2(existing, backup_dir / artifact)

        target_dir.mkdir(parents=True, exist_ok=True)
        published: list[str] = []
        try:
            for artifact in artifacts:
                os.replace(temp_dir / artifact, target_dir / artifact)
                published.append(artifact)
        except OSError as exc:
            rollback_errors: list[str] = []
            for artifact in reversed(published):
                target = target_dir / artifact
                backup = backup_dir / artifact
                try:
                    if backup.exists():
                        os.replace(backup, target)
                    else:
                        target.unlink(missing_ok=True)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{artifact}: {rollback_exc}")
            if not target_existed:
                try:
                    target_dir.rmdir()
                except OSError as rollback_exc:
                    rollback_errors.append(f"target directory: {rollback_exc}")
            if rollback_errors:
                details = "; ".join(rollback_errors)
                raise RuntimeError(
                    f"Publication failed ({exc}); rollback also failed: {details}"
                ) from exc
            print(f"Publication failed and was rolled back: {exc}")
            return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(
        f"Merged {len(all_records)} records and {len(all_request_logs)} "
        f"request logs into {target_dir}"
    )
    return True


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
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard any staged per-quantity results instead of resuming",
    )
    args = parser.parse_args()

    if args.model not in PROVIDER_FOR_MODEL:
        print(f"Unknown model: {args.model}", file=sys.stderr)
        return 2

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
    if staging_root.exists() and args.fresh:
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.model} / {args.batch} per-quantity ({len(qids)} quantities)")
    start = time.time()
    successes = 0
    for i, qid in enumerate(qids, 1):
        sub = staging_root / qid.replace(".", "_")
        staged_records = load_runs(sub / "runs.jsonl")
        staged_grid = validate_grid_rows(
            (
                {
                    "model_name": record.model_name,
                    "prompt_version": record.prompt_version,
                    "quantity_id": record.quantity_id,
                    "run_index": record.run_index,
                }
                for record in staged_records
            ),
            model_name=args.model,
            prompt_version=args.prompt_version,
            quantity_ids=[qid],
            n_runs=15,
        )
        if staged_grid.ok:
            print(
                f"[{time.strftime('%H:%M:%S')}] {i}/{len(qids)} {qid} "
                "SKIP (exact staged grid)"
            )
            successes += 1
            continue
        if sub.exists():
            shutil.rmtree(sub)  # partial cell from an interrupted run — redo cleanly
        print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(qids)} {qid}")
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

    if successes != len(qids):
        print(
            "One or more quantity children failed; preserving staging and "
            "leaving the canonical target untouched."
        )
        return 1

    published = merge_per_quantity(
        staging_root,
        target_dir,
        args.model,
        args.prompt_version,
        expected_quantity_ids=qids,
        n_runs=15,
    )
    if not published:
        print("Preserved staging; canonical target was not published.")
        return 1
    shutil.rmtree(staging_root, ignore_errors=True)
    parsed_grid = check_run_files(
        [target_dir / "runs.jsonl"],
        model_name=args.model,
        prompt_version=args.prompt_version,
        quantity_ids=qids,
        n_runs=15,
        require_parsed=True,
    )
    if not parsed_grid.ok:
        print("Canonical grid contains unresolved runs.")
        for error in parsed_grid.errors:
            print(f"  {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
