from llm_econ_beliefs import create_belief_prompt, get_quantity


def test_prompt_includes_core_metadata_and_json_instruction():
    quantity = get_quantity("labor_supply.frisch_elasticity.prime_age")
    prompt = create_belief_prompt(quantity)

    assert "Frisch elasticity of labor supply" in prompt
    assert "Return valid JSON only with exactly this shape:" in prompt
    assert '"point_estimate"' in prompt
    assert '"quantiles"' in prompt
    assert '"p05"' in prompt
    assert '"p95"' in prompt
    assert "Answer from your current memory and background knowledge only." in prompt
    assert "- ID:" not in prompt
    assert "Benchmark note" not in prompt
    assert "Plausible support" not in prompt


def test_tool_enabled_prompt_allows_tools():
    quantity = get_quantity("labor_supply.frisch_elasticity.prime_age")
    prompt = create_belief_prompt(quantity, tool_regime="full")

    assert "You may use any available tools" in prompt
    assert "Do not use tools" not in prompt


def test_income_elasticity_prompt_clarifies_sign_convention():
    quantity = get_quantity("labor_supply.income_elasticity.prime_age")
    prompt = create_belief_prompt(quantity)

    assert "Sign convention for this quantity:" in prompt
    assert (
        "\u03b5 > 0 if and only if additional non-labor income raises annual hours worked"
        in prompt
    )
    assert (
        "\u03b5 < 0 if and only if additional non-labor income reduces annual hours worked"
        in prompt
    )
    assert (
        "if \u03b5 = 0.5, a 1 percent increase in non-labor income changes annual hours worked by 0.5 percent (not 50 percent)"
        in prompt
    )


def test_capital_gains_prompt_clarifies_sign_convention_against_tax_rate():
    quantity = get_quantity("tax.capital_gains_realizations.elasticity")
    prompt = create_belief_prompt(quantity)

    assert "Sign convention for this quantity:" in prompt
    assert "\u03b5 > 0 if and only if a higher capital-gains tax rate raises realizations" in prompt
    assert "\u03b5 < 0 if and only if a higher capital-gains tax rate reduces realizations" in prompt
    assert "not with respect to the net-of-tax rate" in prompt
    assert (
        "if \u03b5 = 0.5, a 1 percent increase in the marginal capital-gains tax rate changes long-term realizations by 0.5 percent (not 50 percent)"
        in prompt
    )


def test_capital_gains_net_of_tax_rate_prompt_clarifies_sign_convention():
    quantity = get_quantity(
        "tax.capital_gains_realizations.elasticity.net_of_tax_rate"
    )
    prompt = create_belief_prompt(quantity)

    assert "Sign convention for this quantity:" in prompt
    assert "\u03b5 > 0 if and only if a higher net-of-tax rate raises realizations" in prompt
    assert "\u03b5 < 0 if and only if a higher net-of-tax rate reduces realizations" in prompt
    assert "not with respect to the tax rate" in prompt
    assert (
        "if \u03b5 = 0.5, a 1 percent increase in the net-of-tax rate changes long-term realizations by 0.5 percent (not 50 percent)"
        in prompt
    )


def test_ies_clarification_prompt_targets_macro_calibration_object():
    quantity = get_quantity("household.intertemporal_elasticity_of_substitution")
    prompt = create_belief_prompt(quantity, prompt_version="ies-clarify")

    assert "Clarification for this quantity:" in prompt
    assert "representative-household annual macro-calibration setting" in prompt
    assert "Do not answer with the inverse of CRRA" in prompt
    assert "Do not switch to an asset-pricing or recursive-preferences interpretation." in prompt
