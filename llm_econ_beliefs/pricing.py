"""Estimate request costs from provider usage logs."""

from __future__ import annotations

from dataclasses import dataclass

from .models import RequestLog

TOKENS_PER_MILLION = 1_000_000
OPENAI_PRICING_SOURCE_URL = "https://openai.com/api/pricing/"
OPENAI_PRICING_AS_OF = "2026-04-05"
OPENAI_WEB_SEARCH_CALL_USD = 0.01
OPENAI_CODE_INTERPRETER_SESSION_USD = 0.03


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing for one model family."""

    input_per_million_usd: float
    cached_input_per_million_usd: float | None
    output_per_million_usd: float
    source_url: str = OPENAI_PRICING_SOURCE_URL
    as_of_date: str = OPENAI_PRICING_AS_OF


OPENAI_MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-5.4": ModelPricing(2.50, 0.25, 15.00),
    "gpt-5.4-mini": ModelPricing(0.750, 0.075, 4.500),
    "gpt-5.4-nano": ModelPricing(0.20, 0.02, 1.25),
    "gpt-5": ModelPricing(1.25, 0.125, 10.00),
    "gpt-5-mini": ModelPricing(0.25, 0.025, 2.00),
    "gpt-5-nano": ModelPricing(0.05, 0.005, 0.40),
}


def lookup_model_pricing(provider: str, model_name: str) -> ModelPricing | None:
    """Return pricing for a provider/model pair when known."""
    if provider not in {"openai_chat_completions", "openai_responses"}:
        return None

    for base_name, pricing in sorted(
        OPENAI_MODEL_PRICING.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if model_name == base_name or model_name.startswith(f"{base_name}-"):
            return pricing
    return None


def estimate_request_cost(request_log: RequestLog) -> RequestLog:
    """Return a request log with estimated costs filled when pricing is known."""
    if any(
        value is not None
        for value in (
            request_log.estimated_input_cost_usd,
            request_log.estimated_cached_input_cost_usd,
            request_log.estimated_output_cost_usd,
            request_log.estimated_tool_cost_usd,
            request_log.estimated_total_cost_usd,
        )
    ):
        return request_log

    pricing = lookup_model_pricing(request_log.provider, request_log.model_name)
    if pricing is None:
        return request_log

    prompt_tokens = request_log.prompt_tokens or 0
    cached_prompt_tokens = min(request_log.cached_prompt_tokens or 0, prompt_tokens)
    uncached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    completion_tokens = request_log.completion_tokens or 0

    estimated_input_cost = _usd_for_tokens(
        uncached_prompt_tokens,
        pricing.input_per_million_usd,
    )
    cached_rate = (
        pricing.cached_input_per_million_usd
        if pricing.cached_input_per_million_usd is not None
        else pricing.input_per_million_usd
    )
    estimated_cached_input_cost = _usd_for_tokens(
        cached_prompt_tokens,
        cached_rate,
    )
    estimated_output_cost = _usd_for_tokens(
        completion_tokens,
        pricing.output_per_million_usd,
    )
    estimated_tool_cost = (
        (request_log.web_search_call_count or 0) * OPENAI_WEB_SEARCH_CALL_USD
        + (OPENAI_CODE_INTERPRETER_SESSION_USD if (request_log.code_interpreter_call_count or 0) > 0 else 0.0)
    )
    estimated_total_cost = (
        estimated_input_cost
        + estimated_cached_input_cost
        + estimated_output_cost
        + estimated_tool_cost
    )

    return RequestLog(
        provider=request_log.provider,
        model_name=request_log.model_name,
        quantity_id=request_log.quantity_id,
        request_index=request_log.request_index,
        prompt_version=request_log.prompt_version,
        batch_size=request_log.batch_size,
        request_id=request_log.request_id,
        prompt_tokens=request_log.prompt_tokens,
        completion_tokens=request_log.completion_tokens,
        total_tokens=request_log.total_tokens,
        cached_prompt_tokens=request_log.cached_prompt_tokens,
        reasoning_tokens=request_log.reasoning_tokens,
        estimated_input_cost_usd=estimated_input_cost,
        estimated_cached_input_cost_usd=estimated_cached_input_cost,
        estimated_output_cost_usd=estimated_output_cost,
        estimated_tool_cost_usd=estimated_tool_cost,
        estimated_total_cost_usd=estimated_total_cost,
        tool_regime=request_log.tool_regime,
        tool_call_count=request_log.tool_call_count,
        web_search_call_count=request_log.web_search_call_count,
        code_interpreter_call_count=request_log.code_interpreter_call_count,
        tool_sources=list(request_log.tool_sources),
        tool_trace=list(request_log.tool_trace),
        usage=dict(request_log.usage),
    )


def _usd_for_tokens(tokens: int, price_per_million_usd: float) -> float:
    return tokens * price_per_million_usd / TOKENS_PER_MILLION
