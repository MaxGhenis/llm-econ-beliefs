from llm_econ_beliefs import create_belief_prompt, get_quantity


def test_prompt_includes_core_metadata_and_json_instruction():
    quantity = get_quantity("labor_supply.frisch_elasticity.prime_age")
    prompt = create_belief_prompt(quantity)

    assert "Frisch elasticity of labor supply" in prompt
    assert "Return valid JSON only" in prompt
    assert '"point_estimate"' in prompt
    assert '"quantiles"' in prompt
    assert '"p05"' in prompt
    assert '"p95"' in prompt
    assert "Plausible support" in prompt
