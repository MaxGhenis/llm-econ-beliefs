"""Prompt builders for economic-belief elicitation."""

from __future__ import annotations

import json

from .models import EconomicQuantity


def create_belief_prompt(
    quantity: EconomicQuantity,
    *,
    include_uncertainty: bool = True,
    ask_for_citations: bool = True,
    horizon_note: str = "Use your best current sense of the literature, not a historical estimate from a specific decade unless you say so.",
) -> str:
    """Create a JSON-first elicitation prompt for one quantity."""

    lines = [
        "You are a careful empirical economist.",
        "",
        "Quantity of interest:",
        f"- ID: {quantity.id}",
        f"- Name: {quantity.name}",
        f"- Description: {quantity.description}",
    ]

    if quantity.population:
        lines.append(f"- Population/context: {quantity.population}")
    if quantity.unit:
        lines.append(f"- Units: {quantity.unit}")
    if quantity.preferred_interpretation:
        lines.append(
            f"- Preferred interpretation: {quantity.preferred_interpretation}"
        )
    if quantity.lower_support is not None or quantity.upper_support is not None:
        lower = quantity.lower_support if quantity.lower_support is not None else "-infinity"
        upper = quantity.upper_support if quantity.upper_support is not None else "infinity"
        lines.append(f"- Plausible support: [{lower}, {upper}]")
    if quantity.benchmark_summary:
        lines.append(f"- Benchmark note: {quantity.benchmark_summary}")

    lines.extend(
        [
            "",
            "Task:",
            "1. Use the most standard interpretation in applied economics.",
            "2. If the quantity is ambiguous, choose one interpretation and say exactly what you chose.",
            "3. Give your own best central estimate.",
        ]
    )

    if include_uncertainty:
        lines.append(
            "4. Report your subjective quantiles p05, p25, p50, p75, and p95 for this quantity."
        )
        lines.append("5. Make the quantiles weakly increasing and numerically coherent.")
    if ask_for_citations:
        lines.append("6. Give 2 to 4 literature anchors or model-documentation anchors.")
    lines.append("7. Keep the explanation brief and substantive.")
    lines.extend(["", horizon_note])

    schema: dict[str, object] = {
        "interpretation": "short string",
        "point_estimate": 0.0,
        "reasoning_summary": "short string",
    }

    if include_uncertainty:
        schema["quantiles"] = {
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }
    if ask_for_citations:
        schema["citations"] = ["Author (Year)", "Author (Year)"]

    lines.extend(
        [
            "",
            "Return valid JSON only. Use numbers, not strings, for numeric fields.",
            json.dumps(schema, indent=2),
        ]
    )

    return "\n".join(lines)
