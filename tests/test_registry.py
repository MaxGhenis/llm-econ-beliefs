import pytest

from llm_econ_beliefs import get_quantity, list_quantities, list_tags
from llm_econ_beliefs.models import CONVENTION_LITERALS
from llm_econ_beliefs.registry import _to_quantity


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


def test_convention_sibling_is_bidirectional_for_capital_gains():
    canonical = get_quantity("tax.capital_gains_realizations.elasticity")
    sibling = get_quantity(
        "tax.capital_gains_realizations.elasticity.net_of_tax_rate"
    )

    assert canonical.convention == "tax_rate"
    assert sibling.convention == "net_of_tax_rate"
    assert (
        canonical.convention_sibling_id
        == "tax.capital_gains_realizations.elasticity.net_of_tax_rate"
    )
    assert (
        sibling.convention_sibling_id
        == "tax.capital_gains_realizations.elasticity"
    )


def test_registry_known_conventions_are_all_in_literal_set():
    for quantity in list_quantities():
        if quantity.convention is not None:
            assert quantity.convention in CONVENTION_LITERALS


def test_unknown_convention_raises_value_error():
    payload = {
        "id": "dummy.quantity",
        "name": "Dummy",
        "domain": "dummy",
        "description": "Dummy description.",
        "convention": "not_a_real_convention",
    }
    with pytest.raises(ValueError, match="Unknown convention"):
        _to_quantity(payload)
