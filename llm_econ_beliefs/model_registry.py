"""Canonical metadata for every model in the elicitation panel."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path


ORGANIZATIONS = (
    "anthropic",
    "openai",
    "google",
    "xai",
    "deepseek",
    "alibaba",
    "moonshot",
    "zhipu",
    "minimax",
)

# Lab home country, derived from organization. Every July 2026
# independent-lab addition is a Chinese lab, so country is perfectly
# confounded with the OpenRouter serving path and elicitation wave —
# analyses that group by country must disclose that.
ORGANIZATION_COUNTRY = {
    "anthropic": "us",
    "openai": "us",
    "google": "us",
    "xai": "us",
    "deepseek": "china",
    "alibaba": "china",
    "moonshot": "china",
    "zhipu": "china",
    "minimax": "china",
}


def country_for_organization(organization: str) -> str:
    """Lab home country for an organization key."""
    return ORGANIZATION_COUNTRY[organization]


SERVING_PROVIDER_PATHS = (
    "openai_chat_completions",
    "litellm_completion",
    "anthropic_native",
    "openrouter_via_litellm",
)

WAVES = (
    "april_2026",
    "july_2026_frontier",
    "july_2026_independent",
    "july_2026_gpt56",
    "july_2026_late",
)


@dataclass(frozen=True, slots=True)
class PanelModel:
    """Stable model identity and analysis metadata."""

    model_id: str
    display_label: str
    organization: str
    serving_provider_path: str
    model_family: str
    wave: str


MODEL_REGISTRY: tuple[PanelModel, ...] = (
    PanelModel(
        "gpt-5.4",
        "GPT-5.4",
        "openai",
        "openai_chat_completions",
        "gpt",
        "april_2026",
    ),
    PanelModel(
        "gpt-5.4-mini",
        "GPT-5.4 mini",
        "openai",
        "openai_chat_completions",
        "gpt",
        "april_2026",
    ),
    PanelModel(
        "gpt-5.4-nano",
        "GPT-5.4 nano",
        "openai",
        "openai_chat_completions",
        "gpt",
        "april_2026",
    ),
    PanelModel(
        "claude-opus-4.7",
        "Claude Opus 4.7",
        "anthropic",
        "litellm_completion",
        "claude",
        "april_2026",
    ),
    PanelModel(
        "claude-sonnet-4.6",
        "Claude Sonnet 4.6",
        "anthropic",
        "litellm_completion",
        "claude",
        "april_2026",
    ),
    PanelModel(
        "claude-haiku-4.5",
        "Claude Haiku 4.5",
        "anthropic",
        "litellm_completion",
        "claude",
        "april_2026",
    ),
    PanelModel(
        "gemini-3.1-pro-preview",
        "Gemini 3.1 Pro",
        "google",
        "litellm_completion",
        "gemini",
        "april_2026",
    ),
    PanelModel(
        "gemini-3-flash-preview",
        "Gemini 3 Flash",
        "google",
        "litellm_completion",
        "gemini",
        "april_2026",
    ),
    PanelModel(
        "gemini-3.1-flash-lite-preview",
        "Gemini 3.1 Flash-Lite",
        "google",
        "litellm_completion",
        "gemini",
        "april_2026",
    ),
    PanelModel(
        "grok-4.20",
        "Grok 4.20",
        "xai",
        "litellm_completion",
        "grok",
        "april_2026",
    ),
    PanelModel(
        "grok-4.1-fast",
        "Grok 4.1 Fast",
        "xai",
        "litellm_completion",
        "grok",
        "april_2026",
    ),
    PanelModel(
        "gpt-5.5",
        "GPT-5.5",
        "openai",
        "openai_chat_completions",
        "gpt",
        "july_2026_frontier",
    ),
    PanelModel(
        "claude-fable-5",
        "Claude Fable 5",
        "anthropic",
        "anthropic_native",
        "claude",
        "july_2026_frontier",
    ),
    PanelModel(
        "claude-opus-4.8",
        "Claude Opus 4.8",
        "anthropic",
        "anthropic_native",
        "claude",
        "july_2026_frontier",
    ),
    PanelModel(
        "claude-sonnet-5",
        "Claude Sonnet 5",
        "anthropic",
        "anthropic_native",
        "claude",
        "july_2026_frontier",
    ),
    PanelModel(
        "gemini-3.5-flash",
        "Gemini 3.5 Flash",
        "google",
        "litellm_completion",
        "gemini",
        "july_2026_frontier",
    ),
    PanelModel(
        "grok-4.3",
        "Grok 4.3",
        "xai",
        "litellm_completion",
        "grok",
        "july_2026_frontier",
    ),
    PanelModel(
        "deepseek-v4-pro",
        "DeepSeek V4 Pro",
        "deepseek",
        "openrouter_via_litellm",
        "deepseek",
        "july_2026_independent",
    ),
    PanelModel(
        "qwen-3.7-max",
        "Qwen 3.7 Max",
        "alibaba",
        "openrouter_via_litellm",
        "qwen",
        "july_2026_independent",
    ),
    PanelModel(
        "kimi-k2.6",
        "Kimi K2.6",
        "moonshot",
        "openrouter_via_litellm",
        "kimi",
        "july_2026_independent",
    ),
    PanelModel(
        "glm-5.2",
        "GLM-5.2",
        "zhipu",
        "openrouter_via_litellm",
        "glm",
        "july_2026_independent",
    ),
    PanelModel(
        "minimax-m3",
        "MiniMax M3",
        "minimax",
        "openrouter_via_litellm",
        "minimax",
        "july_2026_independent",
    ),
    PanelModel(
        "gpt-5.6-sol",
        "GPT-5.6 Sol",
        "openai",
        "openai_chat_completions",
        "gpt",
        "july_2026_gpt56",
    ),
    PanelModel(
        "gpt-5.6-luna",
        "GPT-5.6 Luna",
        "openai",
        "openai_chat_completions",
        "gpt",
        "july_2026_gpt56",
    ),
    PanelModel(
        "gpt-5.6-terra",
        "GPT-5.6 Terra",
        "openai",
        "openai_chat_completions",
        "gpt",
        "july_2026_gpt56",
    ),
    PanelModel(
        "grok-4.5",
        "Grok 4.5",
        "xai",
        "litellm_completion",
        "grok",
        "july_2026_late",
    ),
    PanelModel(
        "kimi-k3",
        "Kimi K3",
        "moonshot",
        "openrouter_via_litellm",
        "kimi",
        "july_2026_late",
    ),
    PanelModel(
        "gemini-3.6-flash",
        "Gemini 3.6 Flash",
        "google",
        "litellm_completion",
        "gemini",
        "july_2026_late",
    ),
    # Fable 5 keeps Anthropic's frontier slot: Opus 5 is the Opus-line
    # successor, not the flagship.
    PanelModel(
        "claude-opus-5",
        "Claude Opus 5",
        "anthropic",
        "anthropic_native",
        "claude",
        "july_2026_late",
    ),
)

ORGANIZATION_DISPLAY_LABELS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "xai": "xAI",
    "deepseek": "DeepSeek",
    "alibaba": "Alibaba",
    "moonshot": "Moonshot AI",
    "zhipu": "Zhipu AI",
    "minimax": "MiniMax",
}

SERVING_PROVIDER_DISPLAY_LABELS = {
    "openai_chat_completions": "OpenAI Chat Completions",
    "litellm_completion": "LiteLLM",
    "anthropic_native": "native Anthropic API",
    "openrouter_via_litellm": "LiteLLM via OpenRouter",
}

WAVE_DISPLAY_LABELS = {
    "april_2026": "April 2026",
    "july_2026_frontier": "July 2026 frontier",
    "july_2026_independent": "July 2026 independent labs",
    "july_2026_gpt56": "July 2026 GPT-5.6",
    "july_2026_late": "July 2026 late",
}

MODEL_REGISTRY_BY_ID = {model.model_id: model for model in MODEL_REGISTRY}
PANEL_MODEL_IDS = tuple(model.model_id for model in MODEL_REGISTRY)
CSV_FIELDNAMES = tuple(PanelModel.__dataclass_fields__)

# Each organization's newest flagship in the panel, for "frontier" views.
# One model per organization; validated below.
FRONTIER_MODEL_IDS = frozenset(
    {
        "claude-fable-5",
        "gpt-5.6-sol",
        "gemini-3.6-flash",
        "grok-4.5",
        "deepseek-v4-pro",
        "qwen-3.7-max",
        "kimi-k3",
        "glm-5.2",
        "minimax-m3",
    }
)


def _validate_registry() -> None:
    if len(MODEL_REGISTRY) != 29:
        raise ValueError(f"The panel registry must contain 29 models, got {len(MODEL_REGISTRY)}")
    if len(MODEL_REGISTRY_BY_ID) != len(MODEL_REGISTRY):
        raise ValueError("Panel model IDs must be unique")
    organizations = {model.organization for model in MODEL_REGISTRY}
    if organizations != set(ORGANIZATIONS):
        raise ValueError(
            "Panel organizations differ from the canonical set: "
            f"expected {sorted(ORGANIZATIONS)!r}, found {sorted(organizations)!r}"
        )
    serving_paths = {model.serving_provider_path for model in MODEL_REGISTRY}
    if serving_paths != set(SERVING_PROVIDER_PATHS):
        raise ValueError(
            "Panel serving-provider paths differ from the canonical set: "
            f"expected {sorted(SERVING_PROVIDER_PATHS)!r}, "
            f"found {sorted(serving_paths)!r}"
        )
    waves = {model.wave for model in MODEL_REGISTRY}
    if waves != set(WAVES):
        raise ValueError(
            "Panel waves differ from the canonical set: "
            f"expected {sorted(WAVES)!r}, found {sorted(waves)!r}"
        )
    if any(
        not value
        for model in MODEL_REGISTRY
        for value in (
            model.model_id,
            model.display_label,
            model.organization,
            model.serving_provider_path,
            model.model_family,
            model.wave,
        )
    ):
        raise ValueError("Panel registry fields must not be blank")
    if len({model.display_label for model in MODEL_REGISTRY}) != len(MODEL_REGISTRY):
        raise ValueError("Panel display labels must be unique")
    unknown_frontier = FRONTIER_MODEL_IDS - set(PANEL_MODEL_IDS)
    if unknown_frontier:
        raise ValueError(f"Unknown frontier model IDs: {sorted(unknown_frontier)}")
    frontier_organizations = [
        MODEL_REGISTRY_BY_ID[model_id].organization
        for model_id in FRONTIER_MODEL_IDS
    ]
    if sorted(frontier_organizations) != sorted(ORGANIZATIONS):
        raise ValueError(
            "Frontier set must name exactly one model per organization, got "
            f"{sorted(frontier_organizations)!r}"
        )


_validate_registry()


def get_panel_model(model_id: str) -> PanelModel:
    """Return one registered panel model, failing on unknown IDs."""
    try:
        return MODEL_REGISTRY_BY_ID[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown panel model: {model_id}") from exc


def organization_display_label(organization: str) -> str:
    """Return the canonical human-readable organization label."""
    try:
        return ORGANIZATION_DISPLAY_LABELS[organization]
    except KeyError as exc:
        raise KeyError(f"Unknown panel organization: {organization}") from exc


def serving_provider_display_label(serving_provider_path: str) -> str:
    """Return the canonical human-readable serving-provider path."""
    try:
        return SERVING_PROVIDER_DISPLAY_LABELS[serving_provider_path]
    except KeyError as exc:
        raise KeyError(
            f"Unknown serving-provider path: {serving_provider_path}"
        ) from exc


def default_registry_csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "model-registry.csv"


def write_model_registry_csv(path: Path | None = None) -> Path:
    """Write the canonical registry CSV atomically and return its path."""
    target = path or default_registry_csv_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    # Display labels and the frontier flag ride along so downstream
    # consumers (the dashboard) never re-encode them by hand.
    fieldnames = (
        *CSV_FIELDNAMES,
        "organization_label",
        "wave_label",
        "is_frontier",
    )
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in MODEL_REGISTRY:
            row = asdict(model)
            row["organization_label"] = ORGANIZATION_DISPLAY_LABELS[model.organization]
            row["wave_label"] = WAVE_DISPLAY_LABELS[model.wave]
            row["is_frontier"] = str(model.model_id in FRONTIER_MODEL_IDS).lower()
            writer.writerow(row)
    os.replace(temporary, target)
    return target


if __name__ == "__main__":
    print(write_model_registry_csv())
