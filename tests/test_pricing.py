from llm_econ_beliefs import RequestLog, estimate_request_cost, lookup_model_pricing


def test_lookup_model_pricing_supports_snapshot_names():
    pricing = lookup_model_pricing("openai_chat_completions", "gpt-5.4-mini-2026-03-17")

    assert pricing is not None
    assert pricing.input_per_million_usd == 0.75
    assert pricing.cached_input_per_million_usd == 0.075
    assert pricing.output_per_million_usd == 4.5


def test_estimate_request_cost_uses_cached_and_uncached_prompt_rates():
    request_log = RequestLog(
        provider="openai_chat_completions",
        model_name="gpt-5.4-mini",
        quantity_id="labor_supply.frisch_elasticity.prime_age",
        request_index=1,
        prompt_version="v1",
        batch_size=5,
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cached_prompt_tokens=200,
    )

    enriched = estimate_request_cost(request_log)

    assert enriched.estimated_input_cost_usd == 800 * 0.75 / 1_000_000
    assert enriched.estimated_cached_input_cost_usd == 200 * 0.075 / 1_000_000
    assert enriched.estimated_output_cost_usd == 500 * 4.5 / 1_000_000
    assert enriched.estimated_total_cost_usd == (
        enriched.estimated_input_cost_usd
        + enriched.estimated_cached_input_cost_usd
        + enriched.estimated_output_cost_usd
    )
