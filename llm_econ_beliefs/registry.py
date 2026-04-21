"""Quantity registry utilities."""

from __future__ import annotations

import tomllib
from importlib.resources import files

from .models import CONVENTION_LITERALS, EconomicQuantity


def _load_registry_text() -> str:
    return files("llm_econ_beliefs.data").joinpath("quantities.toml").read_text()


def list_quantities(*, domain: str | None = None, tag: str | None = None) -> list[EconomicQuantity]:
    """Load quantities from the packaged registry."""
    payload = tomllib.loads(_load_registry_text())
    quantities = [_to_quantity(item) for item in payload.get("quantity", [])]

    if domain is not None:
        quantities = [quantity for quantity in quantities if quantity.domain == domain]
    if tag is not None:
        quantities = [quantity for quantity in quantities if tag in quantity.tags]

    return quantities


def get_quantity(quantity_id: str) -> EconomicQuantity:
    """Return one quantity by ID."""
    for quantity in list_quantities():
        if quantity.id == quantity_id:
            return quantity
    raise KeyError(f"Unknown quantity: {quantity_id}")


def list_tags() -> list[str]:
    """List all tags used in the registry."""
    tags = {tag for quantity in list_quantities() for tag in quantity.tags}
    return sorted(tags)


def _to_quantity(payload: dict) -> EconomicQuantity:
    convention = payload.get("convention")
    if convention is not None and convention not in CONVENTION_LITERALS:
        raise ValueError(
            f"Unknown convention {convention!r} for quantity {payload.get('id')!r}; "
            f"expected one of {CONVENTION_LITERALS} or None"
        )
    return EconomicQuantity(
        id=payload["id"],
        name=payload["name"],
        domain=payload["domain"],
        description=payload["description"],
        population=payload.get("population"),
        unit=payload.get("unit"),
        preferred_interpretation=payload.get("preferred_interpretation"),
        lower_support=payload.get("lower_support"),
        upper_support=payload.get("upper_support"),
        benchmark_summary=payload.get("benchmark_summary"),
        benchmark_source=payload.get("benchmark_source"),
        tags=tuple(payload.get("tags", [])),
        convention=convention,
        convention_sibling_id=payload.get("convention_sibling_id"),
    )

