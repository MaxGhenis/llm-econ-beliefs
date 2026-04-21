"""Prompt builders for economic-belief elicitation."""

from __future__ import annotations

from .models import EconomicQuantity


def create_belief_prompt(
    quantity: EconomicQuantity,
    *,
    include_uncertainty: bool = True,
    ask_for_citations: bool = True,
    tool_regime: str = "none",
    prompt_version: str = "v4",
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
    if quantity.id == "labor_supply.income_elasticity.prime_age":
        lines.extend(
            [
                "",
                "Sign convention for this quantity:",
                "- An elasticity of \u03b5 means that a 1 percent increase in non-labor income changes annual hours worked by \u03b5 percent; for example, if \u03b5 = 0.5, a 1 percent increase in non-labor income changes annual hours worked by 0.5 percent (not 50 percent).",
                "- \u03b5 > 0 if and only if additional non-labor income raises annual hours worked.",
                "- \u03b5 < 0 if and only if additional non-labor income reduces annual hours worked.",
            ]
        )
    if quantity.id == "tax.capital_gains_realizations.elasticity":
        lines.extend(
            [
                "",
                "Sign convention for this quantity:",
                "- An elasticity of \u03b5 means that a 1 percent increase in the marginal capital-gains tax rate changes long-term realizations by \u03b5 percent; for example, if \u03b5 = 0.5, a 1 percent increase in the marginal capital-gains tax rate changes long-term realizations by 0.5 percent (not 50 percent).",
                "- \u03b5 > 0 if and only if a higher capital-gains tax rate raises realizations.",
                "- \u03b5 < 0 if and only if a higher capital-gains tax rate reduces realizations.",
                "- This is the elasticity with respect to the tax rate itself, not with respect to the net-of-tax rate (1 - \u03c4).",
            ]
        )
    if quantity.id == "tax.capital_gains_realizations.elasticity.net_of_tax_rate":
        lines.extend(
            [
                "",
                "Sign convention for this quantity:",
                "- An elasticity of \u03b5 means that a 1 percent increase in the net-of-tax rate (1 - \u03c4) changes long-term realizations by \u03b5 percent; for example, if \u03b5 = 0.5, a 1 percent increase in the net-of-tax rate changes long-term realizations by 0.5 percent (not 50 percent).",
                "- \u03b5 > 0 if and only if a higher net-of-tax rate raises realizations.",
                "- \u03b5 < 0 if and only if a higher net-of-tax rate reduces realizations.",
                "- This is the elasticity with respect to the net-of-tax rate, not with respect to the tax rate \u03c4 itself.",
            ]
        )
    if (
        quantity.id == "trade.armington_elasticity.import_domestic"
        and prompt_version == "armington-clarify"
    ):
        lines.extend(
            [
                "",
                "Clarification for this quantity:",
                "- This is the top-level elasticity between the aggregate import composite and domestically produced goods.",
                "- It is not the elasticity across different foreign source countries.",
                "- It is not a sector-level or product-level import-demand elasticity.",
                "- Think of the value typically used in aggregate U.S. CGE or macro trade calibration.",
            ]
        )
    if (
        quantity.id == "household.intertemporal_elasticity_of_substitution"
        and prompt_version == "ies-clarify"
    ):
        lines.extend(
            [
                "",
                "Clarification for this quantity:",
                "- This is the elasticity of intertemporal substitution for nondurable consumption in a representative-household annual macro-calibration setting.",
                "- Treat it as a consumption-growth response to the intertemporal price of consumption, not as a generic willingness-to-take-risk parameter.",
                "- Do not answer with the inverse of CRRA unless you independently think that is the right value for this calibration target.",
                "- Do not switch to an asset-pricing or recursive-preferences interpretation.",
            ]
        )

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
