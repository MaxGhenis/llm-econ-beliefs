"""Prompt builders for economic-belief elicitation."""

from __future__ import annotations

from .models import EconomicQuantity


def create_belief_prompt(
    quantity: EconomicQuantity,
    *,
    include_uncertainty: bool = True,
    ask_for_citations: bool = True,
    tool_regime: str = "none",
) -> str:
    """Create a JSON-first elicitation prompt for one quantity."""

    if tool_regime == "none":
        lines = [
            "Answer from your current memory and background knowledge only.",
            "Do not use tools, files, the web, code, or external resources.",
            "Do not try to reconstruct a literature review or search for a consensus estimate.",
            "Report the belief you currently endorse.",
            "",
            "Quantity of interest:",
            f"- Name: {quantity.name}",
            f"- Definition: {quantity.description}",
        ]
    elif tool_regime == "full":
        lines = [
            "Start from your current background knowledge.",
            "You may use any available tools, including web or code tools, if they materially improve your estimate.",
            "Use tools only as needed; do not turn this into an exhaustive literature review.",
            "Report the belief you endorse after using any tools you choose to use.",
            "",
            "Quantity of interest:",
            f"- Name: {quantity.name}",
            f"- Definition: {quantity.description}",
        ]
    else:
        raise ValueError(f"Unsupported tool_regime: {tool_regime}")

    if quantity.preferred_interpretation:
        lines.append(f"- Target interpretation: {quantity.preferred_interpretation}")
    if quantity.population:
        lines.append(f"- Population/context: {quantity.population}")
    if quantity.unit:
        lines.append(f"- Units: {quantity.unit}")

    lines.extend(
        [
            "",
            "Task:",
            "1. Use exactly the target interpretation above. In `interpretation`, restate it briefly.",
        ]
    )

    if include_uncertainty:
        lines.append("2. Give your subjective quantiles p05, p25, p50, p75, and p95 for this quantity.")
        lines.append("3. Set `point_estimate` equal to `p50`.")
        lines.append("4. Make the quantiles weakly increasing and numerically coherent.")
    else:
        lines.append("2. Give one numeric best guess for this quantity.")
    if ask_for_citations:
        lines.append(
            "5. In `citations`, list up to 3 source anchors from memory that influenced your belief. "
            "These are recall anchors only. If none come to mind confidently, return `[]`."
        )
    lines.append("6. Keep `reasoning_summary` brief and substantive.")

    lines.extend(
        [
            "",
            "Return valid JSON only with exactly this shape:",
        ]
    )

    template_lines = [
        "{",
        '  "interpretation": "...",',
        '  "point_estimate": <number>,',
    ]
    if include_uncertainty:
        template_lines.extend(
            [
                '  "quantiles": {',
                '    "p05": <number>,',
                '    "p25": <number>,',
                '    "p50": <number>,',
                '    "p75": <number>,',
                '    "p95": <number>',
                "  },",
            ]
        )
    if ask_for_citations:
        template_lines.append('  "citations": ["..."],')
    template_lines.append('  "reasoning_summary": "..."')
    template_lines.append("}")
    lines.extend(template_lines)

    return "\n".join(lines)
