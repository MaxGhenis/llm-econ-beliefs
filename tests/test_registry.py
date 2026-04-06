from llm_econ_beliefs import get_quantity, list_quantities, list_tags


def test_registry_loads_quantities():
    quantities = list_quantities()

    assert len(quantities) >= 10
    assert any(quantity.id == "labor_supply.frisch_elasticity.prime_age" for quantity in quantities)


def test_can_filter_registry_by_tag():
    og_quantities = list_quantities(tag="og_usa")

    assert og_quantities
    assert all("og_usa" in quantity.tags for quantity in og_quantities)


def test_can_filter_registry_by_policyengine_tag():
    pe_quantities = list_quantities(tag="policyengine_us")

    assert pe_quantities
    assert all("policyengine_us" in quantity.tags for quantity in pe_quantities)


def test_get_quantity_returns_expected_entry():
    quantity = get_quantity("household.annual_discount_factor")

    assert quantity.name == "Annual discount factor"
    assert quantity.lower_support == 0.0
    assert quantity.upper_support == 1.0


def test_get_policyengine_quantity_returns_expected_entry():
    quantity = get_quantity("tax.capital_gains_realizations.elasticity")

    assert quantity.name == "Capital gains realizations elasticity"
    assert quantity.lower_support == -10.0
    assert quantity.upper_support == 2.0


def test_list_tags_contains_expected_values():
    tags = list_tags()

    assert "labor_supply" in tags
    assert "og_usa" in tags
    assert "policyengine_us" in tags
