from llm_econ_beliefs import (
    get_parameter_mapping,
    get_quantity,
    list_mapping_systems,
    list_parameter_mappings,
)


def test_can_list_policyengine_mappings():
    mappings = list_parameter_mappings(system="policyengine_us")

    assert len(mappings) == 14
    assert all(mapping.system == "policyengine_us" for mapping in mappings)


def test_get_parameter_mapping_returns_expected_entry():
    mapping = get_parameter_mapping(
        "policyengine_us",
        "gov.simulation.labor_supply_responses.elasticities.income",
    )

    assert mapping.quantity_id == "labor_supply.income_elasticity.prime_age"


def test_all_policyengine_mapping_targets_exist_in_registry():
    for mapping in list_parameter_mappings(system="policyengine_us"):
        quantity = get_quantity(mapping.quantity_id)
        assert quantity.id == mapping.quantity_id


def test_list_mapping_systems_contains_policyengine_us():
    systems = list_mapping_systems()

    assert "policyengine_us" in systems
