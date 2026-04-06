"""End-to-end experiment runner for belief elicitation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .aggregate import (
    aggregate_beliefs,
    bayesian_hierarchical_meta_analysis,
    random_effects_meta_analysis,
)
from .models import BeliefEstimate, ProviderBatchResult, RequestLog, RunResult
from .parse import parse_belief_response
from .pricing import estimate_request_cost
from .providers import (
    OPENAI_CHAT_COMPLETIONS_MAX_N,
    run_claude_prompt,
    run_litellm_prompt_logged,
    run_openai_prompt_batch_logged,
    run_openai_response_logged,
)
from .registry import get_quantity, list_quantities
from .runner import build_run_grid, write_run_grid_csv


def run_claude_experiment(
    *,
    quantity_ids: Sequence[str],
    n_runs: int,
    output_dir: str | Path,
    model_name: str = "sonnet",
    prompt_version: str = "v2",
    tool_regime: str = "none",
    invoke: Callable[[str, str], str] | None = None,
) -> tuple[list[RunResult], list[dict[str, object]]]:
    """Run a repeated-prompt experiment through the Claude CLI."""
    if invoke is None:
        def invoke(prompt: str, current_model_name: str) -> str:
            return run_claude_prompt(
                prompt,
                model_name=current_model_name,
                cwd=str(output_dir),
            )

    def invoke_batch(prompt: str, current_model_name: str, n: int) -> list[str]:
        return [invoke(prompt, current_model_name) for _ in range(n)]

    return _run_batched_experiment(
        provider="claude_cli",
        quantity_ids=quantity_ids,
        n_runs=n_runs,
        output_dir=output_dir,
        model_name=model_name,
        prompt_version=prompt_version,
        tool_regime=tool_regime,
        invoke_batch=invoke_batch,
        batch_size=1,
    )


def run_openai_experiment(
    *,
    quantity_ids: Sequence[str],
    n_runs: int,
    output_dir: str | Path,
    model_name: str = "gpt-5.4-mini",
    prompt_version: str = "v2",
    tool_regime: str = "none",
    api_mode: str = "chat",
    batch_size: int | None = None,
    temperature: float = 1.0,
    invoke_batch: Callable[[str, str, int], ProviderBatchResult | list[str]] | None = None,
) -> tuple[list[RunResult], list[dict[str, object]]]:
    """Run a repeated-prompt experiment through the OpenAI Chat Completions API."""
    if invoke_batch is None:
        if api_mode == "chat":
            def invoke_batch(
                prompt: str,
                current_model_name: str,
                n: int,
            ) -> ProviderBatchResult:
                return run_openai_prompt_batch_logged(
                    prompt,
                    model_name=current_model_name,
                    n=n,
                    temperature=temperature,
                )
        elif api_mode == "responses":
            def invoke_batch(
                prompt: str,
                current_model_name: str,
                n: int,
            ) -> ProviderBatchResult:
                if n != 1:
                    raise ValueError("Responses API runner expects n=1 per request")
                return run_openai_response_logged(
                    prompt,
                    model_name=current_model_name,
                    tool_regime=tool_regime,
                )
        else:
            raise ValueError(f"Unsupported api_mode: {api_mode}")

    return _run_batched_experiment(
        provider=(
            "openai_chat_completions"
            if api_mode == "chat"
            else "openai_responses"
        ),
        quantity_ids=quantity_ids,
        n_runs=n_runs,
        output_dir=output_dir,
        model_name=model_name,
        prompt_version=prompt_version,
        tool_regime=tool_regime,
        invoke_batch=invoke_batch,
        batch_size=(
            min(batch_size or n_runs, OPENAI_CHAT_COMPLETIONS_MAX_N)
            if api_mode == "chat"
            else 1
        ),
    )


def run_litellm_experiment(
    *,
    quantity_ids: Sequence[str],
    n_runs: int,
    output_dir: str | Path,
    model_name: str,
    prompt_version: str = "v2",
    tool_regime: str = "none",
    temperature: float = 1.0,
    invoke_batch: Callable[[str, str, int], ProviderBatchResult | list[str]] | None = None,
) -> tuple[list[RunResult], list[dict[str, object]]]:
    """Run a repeated-prompt experiment through LiteLLM-backed providers."""
    if tool_regime != "none":
        raise ValueError("LiteLLM provider path currently supports tool_regime='none' only")

    if invoke_batch is None:
        def invoke_batch(
            prompt: str,
            current_model_name: str,
            n: int,
        ) -> ProviderBatchResult:
            if n != 1:
                raise ValueError("LiteLLM runner expects n=1 per request")
            return run_litellm_prompt_logged(
                prompt,
                model_name=current_model_name,
                temperature=temperature,
            )

    return _run_batched_experiment(
        provider="litellm_completion",
        quantity_ids=quantity_ids,
        n_runs=n_runs,
        output_dir=output_dir,
        model_name=model_name,
        prompt_version=prompt_version,
        tool_regime=tool_regime,
        invoke_batch=invoke_batch,
        batch_size=1,
    )


def _run_batched_experiment(
    *,
    provider: str,
    quantity_ids: Sequence[str],
    n_runs: int,
    output_dir: str | Path,
    model_name: str,
    prompt_version: str,
    tool_regime: str,
    invoke_batch: Callable[[str, str, int], ProviderBatchResult | list[str]],
    batch_size: int,
) -> tuple[list[RunResult], list[dict[str, object]]]:
    """Run an experiment where a provider may return multiple draws per request."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    runs = build_run_grid(
        model_names=[model_name],
        quantity_ids=quantity_ids,
        n_runs=n_runs,
        prompt_version=prompt_version,
        tool_regime=tool_regime,
    )
    write_run_grid_csv(output_dir / "prompt_grid.csv", runs)

    records: list[RunResult] = []
    request_logs: list[RequestLog] = []
    grouped_runs: dict[tuple[str, str], list] = {}
    for run in runs:
        grouped_runs.setdefault((run.model_name, run.quantity_id), []).append(run)

    request_index = 0
    for (_, quantity_id), run_group in sorted(grouped_runs.items()):
        for start in range(0, len(run_group), batch_size):
            batch_runs = run_group[start : start + batch_size]
            prompt = batch_runs[0].prompt
            try:
                raw_batch_result = invoke_batch(prompt, batch_runs[0].model_name, len(batch_runs))
                batch_result = _normalize_batch_result(raw_batch_result)
                raw_responses = batch_result.outputs
                if len(raw_responses) != len(batch_runs):
                    raise RuntimeError(
                        f"Provider returned {len(raw_responses)} responses for {len(batch_runs)} runs"
                    )
                if batch_result.request_id is not None or batch_result.usage:
                    request_index += 1
                    request_logs.append(
                        _request_log_from_batch_result(
                            provider=provider,
                            model_name=batch_runs[0].model_name,
                            quantity_id=quantity_id,
                            prompt_version=batch_runs[0].prompt_version,
                            tool_regime=batch_runs[0].tool_regime,
                            batch_size=len(batch_runs),
                            request_index=request_index,
                            batch_result=batch_result,
                        )
                    )
                for run, raw_response in zip(batch_runs, raw_responses, strict=True):
                    parsed = parse_belief_response(raw_response, quantity_id=run.quantity_id)
                    records.append(
                        _record_from_parsed(
                            run,
                            parsed,
                            raw_response,
                            provider=provider,
                        )
                    )
            except Exception as exc:
                for run in batch_runs:
                    records.append(
                        RunResult(
                            provider=provider,
                            model_name=run.model_name,
                            quantity_id=run.quantity_id,
                            run_index=run.run_index,
                            prompt_version=run.prompt_version,
                            tool_regime=run.tool_regime,
                            prompt=run.prompt,
                            raw_response=None,
                            parsed_ok=False,
                            error=str(exc),
                        )
                    )

    _write_jsonl(output_dir / "runs.jsonl", records)
    _write_runs_csv(output_dir / "runs.csv", records)
    if request_logs:
        _write_jsonl(output_dir / "requests.jsonl", request_logs)
        _write_requests_csv(output_dir / "requests.csv", request_logs)

    summaries = summarize_run_results(records, request_logs=request_logs)
    _write_summary_csv(output_dir / "summary.csv", summaries)

    return records, summaries


def summarize_run_results(
    records: Sequence[RunResult],
    *,
    request_logs: Sequence[RequestLog] = (),
) -> list[dict[str, object]]:
    """Aggregate successful runs by quantity."""
    grouped: dict[tuple[str, str], list[BeliefEstimate]] = {}
    tool_regimes: dict[tuple[str, str], str] = {}
    for record in records:
        tool_regimes[(record.model_name, record.quantity_id)] = record.tool_regime
        if not record.parsed_ok or record.point_estimate is None:
            continue
        estimate = BeliefEstimate(
            point_estimate=record.point_estimate,
            quantity_id=record.quantity_id,
            interpretation=record.interpretation,
            lower_bound=record.lower_bound,
            upper_bound=record.upper_bound,
            confidence_level=record.confidence_level,
            quantiles=dict(record.quantiles),
            citations=list(record.citations),
            reasoning_summary=record.reasoning_summary,
            raw_response=record.raw_response,
        )
        grouped.setdefault((record.model_name, record.quantity_id), []).append(estimate)

    grouped_logs: dict[tuple[str, str], list[RequestLog]] = {}
    for request_log in request_logs:
        grouped_logs.setdefault((request_log.model_name, request_log.quantity_id), []).append(
            request_log
        )

    summaries: list[dict[str, object]] = []
    for (model_name, quantity_id), estimates in sorted(grouped.items()):
        quantity = get_quantity(quantity_id)
        logs = grouped_logs.get((model_name, quantity_id), [])
        aggregated = aggregate_beliefs(
            estimates,
            confidence_level=0.9,
            lower_support=quantity.lower_support,
            upper_support=quantity.upper_support,
        )
        random_effects = random_effects_meta_analysis(
            estimates,
            confidence_level=0.9,
            lower_support=quantity.lower_support,
            upper_support=quantity.upper_support,
        )
        bayesian = bayesian_hierarchical_meta_analysis(
            estimates,
            confidence_level=0.9,
            lower_support=quantity.lower_support,
            upper_support=quantity.upper_support,
        )
        summaries.append(
            {
                "model_name": model_name,
                "quantity_id": quantity_id,
                "quantity_name": quantity.name,
                "tool_regime": tool_regimes.get((model_name, quantity_id), "none"),
                "n_successful_runs": len(estimates),
                "pooled_point_estimate": aggregated.point_estimate,
                "pooled_lower_bound": aggregated.lower_bound,
                "pooled_upper_bound": aggregated.upper_bound,
                "within_run_sd": aggregated.within_run_sd,
                "between_run_sd": aggregated.between_run_sd,
                "total_sd": aggregated.total_sd,
                "n_requests": len(logs) if logs else None,
                "mean_batch_size": _mean_or_none([log.batch_size for log in logs]),
                "usage_prompt_tokens_total": _sum_or_none(log.prompt_tokens for log in logs),
                "usage_completion_tokens_total": _sum_or_none(log.completion_tokens for log in logs),
                "usage_total_tokens_total": _sum_or_none(log.total_tokens for log in logs),
                "usage_cached_prompt_tokens_total": _sum_or_none(
                    log.cached_prompt_tokens for log in logs
                ),
                "usage_reasoning_tokens_total": _sum_or_none(log.reasoning_tokens for log in logs),
                "usage_estimated_input_cost_usd_total": _sum_or_none(
                    log.estimated_input_cost_usd for log in logs
                ),
                "usage_estimated_cached_input_cost_usd_total": _sum_or_none(
                    log.estimated_cached_input_cost_usd for log in logs
                ),
                "usage_estimated_output_cost_usd_total": _sum_or_none(
                    log.estimated_output_cost_usd for log in logs
                ),
                "usage_estimated_tool_cost_usd_total": _sum_or_none(
                    log.estimated_tool_cost_usd for log in logs
                ),
                "usage_estimated_total_cost_usd_total": _sum_or_none(
                    log.estimated_total_cost_usd for log in logs
                ),
                "usage_total_tokens_per_successful_run": (
                    _sum_or_none(log.total_tokens for log in logs) / len(estimates)
                    if logs and _sum_or_none(log.total_tokens for log in logs) is not None
                    else None
                ),
                "usage_estimated_total_cost_usd_per_successful_run": (
                    _sum_or_none(log.estimated_total_cost_usd for log in logs) / len(estimates)
                    if logs
                    and _sum_or_none(log.estimated_total_cost_usd for log in logs) is not None
                    else None
                ),
                "tool_call_count_total": _sum_or_none(log.tool_call_count for log in logs),
                "web_search_call_count_total": _sum_or_none(
                    log.web_search_call_count for log in logs
                ),
                "code_interpreter_call_count_total": _sum_or_none(
                    log.code_interpreter_call_count for log in logs
                ),
                "pool_transform": random_effects.transform,
                "reml_latent_location": random_effects.latent_location,
                "reml_latent_lower": random_effects.latent_lower,
                "reml_latent_upper": random_effects.latent_upper,
                "reml_predictive_lower": random_effects.predictive_lower,
                "reml_predictive_upper": random_effects.predictive_upper,
                "reml_tau": random_effects.tau,
                "reml_typical_within_sd": random_effects.typical_within_sd,
                "bayes_latent_location": bayesian.latent_location,
                "bayes_latent_lower": bayesian.latent_lower,
                "bayes_latent_upper": bayesian.latent_upper,
                "bayes_predictive_lower": bayesian.predictive_lower,
                "bayes_predictive_upper": bayesian.predictive_upper,
                "bayes_tau_mean": bayesian.tau_mean,
                "bayes_interval_scale_mean": bayesian.interval_scale_mean,
                "bayes_typical_within_sd": bayesian.typical_within_sd,
            }
        )
    return summaries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for an experiment."""
    parser = argparse.ArgumentParser(description="Run an LLM economic-belief experiment.")
    parser.add_argument("--provider", choices=["claude", "openai", "litellm"], default="claude")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--samples-per-request", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--tool-regime", choices=["none", "full"], default="none")
    parser.add_argument("--openai-api", choices=["chat", "responses"], default="chat")
    parser.add_argument("--quantity", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def resolve_quantity_ids(explicit_ids: Sequence[str], tags: Sequence[str]) -> list[str]:
    """Resolve quantity IDs from explicit IDs plus tag filters."""
    selected = list(explicit_ids)
    for tag in tags:
        selected.extend(quantity.id for quantity in list_quantities(tag=tag))

    deduped = []
    seen = set()
    for quantity_id in selected:
        if quantity_id not in seen:
            seen.add(quantity_id)
            deduped.append(quantity_id)
    if not deduped:
        raise ValueError("No quantities selected")
    return deduped


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI experiment entrypoint."""
    args = parse_args(argv)
    quantity_ids = resolve_quantity_ids(args.quantity, args.tag)

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("results") / f"{args.model}-{timestamp}"

    if args.provider == "claude":
        _, summaries = run_claude_experiment(
            quantity_ids=quantity_ids,
            n_runs=args.runs,
            output_dir=output_dir,
            model_name=args.model,
            prompt_version=args.prompt_version,
            tool_regime=args.tool_regime,
        )
    elif args.provider == "openai":
        _, summaries = run_openai_experiment(
            quantity_ids=quantity_ids,
            n_runs=args.runs,
            output_dir=output_dir,
            model_name=args.model,
            prompt_version=args.prompt_version,
            tool_regime=args.tool_regime,
            api_mode=args.openai_api,
            batch_size=args.samples_per_request,
            temperature=args.temperature,
        )
    else:
        _, summaries = run_litellm_experiment(
            quantity_ids=quantity_ids,
            n_runs=args.runs,
            output_dir=output_dir,
            model_name=args.model,
            prompt_version=args.prompt_version,
            tool_regime=args.tool_regime,
            temperature=args.temperature,
        )

    print(f"Wrote results to {output_dir}")
    for summary in summaries:
        lower = summary["pooled_lower_bound"]
        upper = summary["pooled_upper_bound"]
        if lower is not None and upper is not None:
            interval_text = f"[{lower:.4g}, {upper:.4g}]"
        else:
            interval_text = "unavailable"
        print(
            f"{summary['quantity_id']}: {summary['pooled_point_estimate']:.4g} "
            f"{interval_text} from {summary['n_successful_runs']} runs"
        )

    return 0


def _record_from_parsed(
    run,
    parsed: BeliefEstimate,
    raw_response: str,
    *,
    provider: str = "claude_cli",
) -> RunResult:
    return RunResult(
        provider=provider,
        model_name=run.model_name,
        quantity_id=run.quantity_id,
        run_index=run.run_index,
        prompt_version=run.prompt_version,
        tool_regime=run.tool_regime,
        prompt=run.prompt,
        raw_response=raw_response,
        parsed_ok=True,
        point_estimate=parsed.point_estimate,
        interpretation=parsed.interpretation,
        lower_bound=parsed.lower_bound,
        upper_bound=parsed.upper_bound,
        confidence_level=parsed.confidence_level,
        quantiles=dict(parsed.quantiles),
        citations=list(parsed.citations),
        reasoning_summary=parsed.reasoning_summary,
    )


def _normalize_batch_result(
    raw_batch_result: ProviderBatchResult | list[str],
) -> ProviderBatchResult:
    if isinstance(raw_batch_result, ProviderBatchResult):
        return raw_batch_result
    return ProviderBatchResult(outputs=list(raw_batch_result))


def _request_log_from_batch_result(
    *,
    provider: str,
    model_name: str,
    quantity_id: str,
    prompt_version: str,
    tool_regime: str,
    batch_size: int,
    request_index: int,
    batch_result: ProviderBatchResult,
) -> RequestLog:
    usage = dict(batch_result.usage)
    prompt_details = (
        usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or {}
    )
    completion_details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    precomputed_total_cost = usage.get("litellm_cost_usd")
    if precomputed_total_cost is not None:
        precomputed_total_cost = float(precomputed_total_cost)

    return estimate_request_cost(RequestLog(
        provider=provider,
        model_name=model_name,
        quantity_id=quantity_id,
        request_index=request_index,
        prompt_version=prompt_version,
        tool_regime=tool_regime,
        batch_size=batch_size,
        request_id=batch_result.request_id,
        prompt_tokens=usage.get("prompt_tokens", usage.get("input_tokens")),
        completion_tokens=usage.get("completion_tokens", usage.get("output_tokens")),
        total_tokens=usage.get("total_tokens"),
        cached_prompt_tokens=prompt_details.get("cached_tokens"),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        estimated_total_cost_usd=precomputed_total_cost,
        tool_call_count=len(batch_result.tool_trace) or None,
        web_search_call_count=sum(
            1 for item in batch_result.tool_trace if item.get("type") == "web_search_call"
        ) or None,
        code_interpreter_call_count=sum(
            1
            for item in batch_result.tool_trace
            if item.get("type") == "code_interpreter_call"
        ) or None,
        tool_sources=list(batch_result.tool_sources),
        tool_trace=list(batch_result.tool_trace),
        usage=usage,
    ))


def _sum_or_none(values: Iterable[int | float | None]) -> int | float | None:
    filtered = [value for value in values if value is not None]
    return sum(filtered) if filtered else None


def _mean_or_none(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _write_jsonl(path: Path, records: Iterable[RunResult]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")


def _write_runs_csv(path: Path, records: Sequence[RunResult]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "provider",
                "model_name",
                "quantity_id",
                "run_index",
                "prompt_version",
                "tool_regime",
                "parsed_ok",
                "point_estimate",
                "interpretation",
                "lower_bound",
                "upper_bound",
                "confidence_level",
                "quantiles",
                "citations",
                "reasoning_summary",
                "error",
                "raw_response",
                "prompt",
            ],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["quantiles"] = json.dumps(record.quantiles, sort_keys=True)
            row["citations"] = " | ".join(record.citations)
            writer.writerow(row)


def _write_requests_csv(path: Path, request_logs: Sequence[RequestLog]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "provider",
                "model_name",
                "quantity_id",
                "request_index",
                "prompt_version",
                "tool_regime",
                "batch_size",
                "request_id",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_prompt_tokens",
                "reasoning_tokens",
                "estimated_input_cost_usd",
                "estimated_cached_input_cost_usd",
                "estimated_output_cost_usd",
                "estimated_tool_cost_usd",
                "estimated_total_cost_usd",
                "tool_call_count",
                "web_search_call_count",
                "code_interpreter_call_count",
                "tool_sources",
                "tool_trace",
                "usage",
            ],
        )
        writer.writeheader()
        for request_log in request_logs:
            row = asdict(request_log)
            row["tool_sources"] = json.dumps(request_log.tool_sources, sort_keys=True)
            row["tool_trace"] = json.dumps(request_log.tool_trace, sort_keys=True)
            row["usage"] = json.dumps(request_log.usage, sort_keys=True)
            writer.writerow(row)


def _write_summary_csv(path: Path, summaries: Sequence[dict[str, object]]) -> None:
    if not summaries:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    raise SystemExit(main())
