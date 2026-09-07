"""Read retained calibration values without importing or running PolicyEngine.

These are inputs to deterministic replay, not new microdata estimates or evidence
that the historical data identity has been established.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "results/top-rate-calibration.json"


def load_frozen_calibration(path: Path = CALIBRATION) -> dict[str, float]:
    """Require finite, internally consistent top-tail values; never substitute."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen calibration: {path}. Restore the retained artifact; "
            "fresh calibration is a separate scientific operation."
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Top-rate calibration must be a JSON object")
    # Preserve the explicit historical fallback diagnostic.
    if payload.get("a") == 1.5:
        raise ValueError("Top-rate calibration carries fallback a=1.5")
    values = {}
    for key in ("a", "gbar", "threshold", "tail_mean"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"Top-rate calibration {key} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"Top-rate calibration {key} must be finite")
        values[key] = float(value)
    a, gbar = values["a"], values["gbar"]
    threshold, tail_mean = values["threshold"], values["tail_mean"]
    if not (a > 1 and 0 <= gbar < 1 and 0 < threshold < tail_mean):
        raise ValueError("Invalid top-tail calibration domain")
    if not math.isclose(a, tail_mean / (tail_mean - threshold), rel_tol=1e-12):
        raise ValueError("Calibration a is inconsistent with threshold and tail mean")
    if not math.isclose(gbar, a / (a + 1), rel_tol=1e-12):
        raise ValueError("Calibration gbar is inconsistent with frozen gamma=1")
    return values


def verify_calibration_provenance(root: Path = ROOT) -> dict:
    """Bind the explicit historical provenance limits to unchanged file bytes."""
    provenance = json.loads((root / "results/calibration-provenance.json").read_text())
    for artifact in (provenance, provenance["flat_tax_table"]):
        path = root / artifact["artifact"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(
                f"Frozen calibration hash mismatch: {artifact['artifact']}"
            )
    if provenance["status"] != "frozen_historical_artifact":
        raise ValueError("Cached replay requires the named historical calibration")
    if provenance["mapping_assumptions"] != {"percentile": 0.99, "crra_gamma": 1.0}:
        raise ValueError(
            "Cached mapping assumptions differ from the retained manuscript"
        )
    load_frozen_calibration(root / provenance["artifact"])
    return provenance
