"""Parse structured or semi-structured LLM belief responses."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Sequence

from .models import BeliefEstimate


POINT_KEYS = (
    "point_estimate",
    "best_estimate",
    "central_estimate",
    "estimate",
    "median",
    "p50",
)
LOWER_KEYS = ("lower_bound", "lower_90", "p05", "p10", "lower")
UPPER_KEYS = ("upper_bound", "upper_90", "p95", "p90", "upper")
CONFIDENCE_KEYS = ("confidence_level", "interval_probability", "coverage")
INTERPRETATION_KEYS = ("interpretation", "definition", "quantity_interpretation")
REASONING_KEYS = ("reasoning_summary", "reasoning", "summary", "notes")
CITATION_KEYS = ("citations", "references", "literature_anchors")
QUANTILE_ORDER = ("p05", "p25", "p50", "p75", "p95")
QUANTILE_ALIASES = {
    "p05": ("p05", "p5", "q05", "q5", "5th percentile", "5th quantile"),
    "p25": ("p25", "q25", "25th percentile", "first quartile", "q1"),
    "p50": ("p50", "q50", "50th percentile", "median"),
    "p75": ("p75", "q75", "75th percentile", "third quartile", "q3"),
    "p95": ("p95", "q95", "95th percentile", "95th quantile"),
}


def parse_belief_response(
    response_text: str,
    *,
    quantity_id: str | None = None,
) -> BeliefEstimate:
    """Parse either JSON-first output or a free-form fallback."""
    if not response_text or not response_text.strip():
        raise ValueError("Response text is empty")

    payload = _extract_payload(response_text)
    if payload is not None:
        return _parse_structured_payload(
            payload,
            raw_response=response_text,
            quantity_id=quantity_id,
        )

    quantiles = _extract_quantiles_from_text(response_text)
    point_estimate = _extract_point_estimate_from_text(response_text, quantiles=quantiles)
    if point_estimate is None:
        raise ValueError("Could not parse a point estimate from the response")

    interval = _extract_interval_from_text(response_text)
    lower_bound = quantiles.get("p05") if "p05" in quantiles else (interval[0] if interval else None)
    upper_bound = quantiles.get("p95") if "p95" in quantiles else (interval[1] if interval else None)
    confidence_level = 0.9 if lower_bound is not None and upper_bound is not None else None

    return BeliefEstimate(
        quantity_id=quantity_id,
        point_estimate=point_estimate,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level,
        quantiles=quantiles,
        reasoning_summary=response_text.strip(),
        raw_response=response_text,
    )


def _extract_payload(response_text: str) -> dict[str, Any] | None:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
        if match:
            stripped = match.group(1)

    for candidate in (stripped, _first_braced_block(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _parse_structured_payload(
    payload: dict[str, Any],
    *,
    raw_response: str,
    quantity_id: str | None,
) -> BeliefEstimate:
    quantiles = _lookup_quantiles(payload)
    point_estimate = _lookup_numeric(payload, POINT_KEYS)
    if point_estimate is None:
        point_estimate = quantiles.get("p50")
    if point_estimate is None:
        raise ValueError("Structured response is missing a point estimate")

    lower_bound = _lookup_numeric(payload, LOWER_KEYS)
    upper_bound = _lookup_numeric(payload, UPPER_KEYS)
    if lower_bound is None:
        lower_bound = quantiles.get("p05")
    if upper_bound is None:
        upper_bound = quantiles.get("p95")
    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    confidence_level = _lookup_confidence(payload)
    if confidence_level is None and lower_bound is not None and upper_bound is not None:
        confidence_level = 0.9

    return BeliefEstimate(
        quantity_id=quantity_id,
        point_estimate=point_estimate,
        interpretation=_lookup_string(payload, INTERPRETATION_KEYS),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level,
        quantiles=quantiles,
        citations=_lookup_string_list(payload, CITATION_KEYS),
        reasoning_summary=_lookup_string(payload, REASONING_KEYS),
        raw_response=raw_response,
    )


def _lookup_quantiles(payload: dict[str, Any]) -> dict[str, float]:
    quantiles: dict[str, float] = {}
    candidate = payload.get("quantiles")
    if isinstance(candidate, dict):
        for key in QUANTILE_ORDER:
            if key in candidate:
                value = _coerce_float(candidate[key])
                if value is not None:
                    quantiles[key] = value

    for key in QUANTILE_ORDER:
        value = _lookup_numeric(payload, (key,))
        if value is not None:
            quantiles[key] = value

    return _sorted_quantiles(quantiles)


def _lookup_numeric(payload: dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in payload:
            value = _coerce_float(payload[key])
            if value is not None:
                return value
    return None


def _lookup_string(payload: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return str(payload[key]).strip()
    return None


def _lookup_string_list(payload: dict[str, Any], keys: Sequence[str]) -> list[str]:
    for key in keys:
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            parts = re.split(r"\n|;|, (?=[A-Z])", value)
            return [part.strip(" -") for part in parts if part.strip()]
    return []


def _lookup_confidence(payload: dict[str, Any]) -> float | None:
    for key in CONFIDENCE_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, str) and value.endswith("%"):
            value = value[:-1]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 1:
            numeric /= 100
        if 0 < numeric < 1:
            return numeric
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    text = re.sub(r"[^0-9eE+.\-]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_point_estimate_from_text(
    response_text: str,
    *,
    quantiles: dict[str, float],
) -> float | None:
    patterns = [
        r"(?is)(?:point estimate|best estimate|central estimate)\D{0,40}([-+]?\d*\.?\d+)",
        r"(?is)(?:point estimate|best estimate|central estimate).*?(?:=|≈|~)\s*([-+]?\d*\.?\d+)",
        r"(?is)(?:\bbeta\b|β|sigma|σ)\s*(?:=|≈|~)\s*([-+]?\d*\.?\d+)",
        r"(?i)(?:about|around|roughly|approximately)\s*([-+]?\d*\.?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return float(match.group(1))

    if "p50" in quantiles:
        return quantiles["p50"]

    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if lines:
        first_line_value = _extract_first_number(lines[0])
        if first_line_value is not None:
            return first_line_value

    return _extract_first_number(response_text)


def _extract_quantiles_from_text(response_text: str) -> dict[str, float]:
    quantiles: dict[str, float] = {}
    for key, aliases in QUANTILE_ALIASES.items():
        patterns = [
            rf"(?i)(?:{'|'.join(re.escape(alias) for alias in aliases)})\D{{0,20}}([-+]?\d*\.?\d+)",
            rf"(?i)(?:{'|'.join(re.escape(alias) for alias in aliases)}).*?(?:=|≈|~)\s*([-+]?\d*\.?\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                quantiles[key] = float(match.group(1))
                break
    return _sorted_quantiles(quantiles)


def _extract_interval_from_text(response_text: str) -> tuple[float, float] | None:
    patterns = [
        r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
        r"(?i)(?:confidence|credible|uncertainty)\s+interval[^-\d+]*"
        r"([-+]?\d*\.?\d+)\s*(?:to|–|—|-)\s*([-+]?\d*\.?\d+)",
        r"(?i)(?:90%\s*(?:ci|interval)|ci)\D{0,20}([-+]?\d*\.?\d+)\s*(?:to|–|—|-)\s*([-+]?\d*\.?\d+)",
        r"(?i)(?:between|from)\s*([-+]?\d*\.?\d+)\s*(?:and|to)\s*([-+]?\d*\.?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if not match:
            continue
        lower = float(match.group(1))
        upper = float(match.group(2))
        return (lower, upper) if lower <= upper else (upper, lower)
    return None


def _extract_first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def _first_braced_block(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _sorted_quantiles(quantiles: dict[str, float]) -> dict[str, float]:
    if not quantiles:
        return {}

    sorted_values = []
    running_max = None
    for key in QUANTILE_ORDER:
        if key not in quantiles:
            continue
        value = quantiles[key]
        if running_max is None:
            running_max = value
        else:
            running_max = max(running_max, value)
        sorted_values.append((key, running_max))

    return dict(sorted_values)
