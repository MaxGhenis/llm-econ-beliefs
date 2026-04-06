"""Parameter-mapping utilities for policy models."""

from __future__ import annotations

import tomllib
from importlib.resources import files

from .models import ParameterMapping


def _load_mapping_text() -> str:
    return files("llm_econ_beliefs.data").joinpath("parameter_mappings.toml").read_text()


def list_parameter_mappings(*, system: str | None = None) -> list[ParameterMapping]:
    """Load parameter mappings from packaged data."""
    payload = tomllib.loads(_load_mapping_text())
    mappings = [_to_mapping(item) for item in payload.get("mapping", [])]
    if system is not None:
        mappings = [mapping for mapping in mappings if mapping.system == system]
    return mappings


def get_parameter_mapping(system: str, parameter_path: str) -> ParameterMapping:
    """Return one parameter mapping by system and parameter path."""
    for mapping in list_parameter_mappings(system=system):
        if mapping.parameter_path == parameter_path:
            return mapping
    raise KeyError(f"Unknown parameter mapping: {system}:{parameter_path}")


def list_mapping_systems() -> list[str]:
    """List all systems used in the parameter-mapping table."""
    systems = {mapping.system for mapping in list_parameter_mappings()}
    return sorted(systems)


def _to_mapping(payload: dict) -> ParameterMapping:
    return ParameterMapping(
        system=payload["system"],
        parameter_path=payload["parameter_path"],
        quantity_id=payload["quantity_id"],
        label=payload.get("label"),
        notes=payload.get("notes"),
    )
