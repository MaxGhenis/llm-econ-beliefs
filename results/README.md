# Results

This directory contains the elicited belief artifacts for the paper.

## Naming convention

Model × prompt-family output directories are named:

    <model>-<prompt-family>-batch15

Despite the `batch15` suffix being historical (it refers to the R = 15
repeated-runs design), the current v4 panel also lives in directories
matching `*-elasticities-batch15`. The canonical marker of which
prompt-version a result set belongs to is the `prompt_version` field
inside each `runs.jsonl` record, **not** the directory name. Concretely:

- `*-elasticities-batch15/` → `prompt_version = "v4"` (main panel, 26 quantities)
- `*-armington-clarify-batch15/` → `prompt_version = "armington-clarify"` (Armington robustness probe)
- `*-ies-clarify-batch15/` → `prompt_version = "ies-clarify"` (IES robustness probe)

Always read `runs.jsonl` if you need to be sure which prompt family a
directory came from.

## Generation scripts

- `scripts/run_v4_full_panel.py` — the main driver. Runs every
  (model, prompt-family) cell in an isolated subprocess so a hung
  provider call can be SIGKILLed without taking down the rest of the
  panel. This is the recommended entry point for a full re-elicitation.
- `scripts/run_v4_per_quantity.py` — the per-quantity fallback. For
  models where multi-quantity cells hang under the main driver, this
  script runs one subprocess per quantity and then merges the
  per-quantity outputs into the canonical `*-elasticities-batch15/`
  directory shape via `merge_per_quantity(...)`.

## Per-quantity fallback models

Four models went through the per-quantity fallback during the v4 panel
rerun because their multi-quantity cells hung under the main driver:

- `claude-sonnet-4.6`
- `claude-opus-4.7`
- `gemini-3.1-pro-preview`
- `grok-4.20`

Their final merged directories look identical to the other seven
models, but their `summary.csv` rows show `$0.0000` in the total-cost
column because the per-quantity merge did not previously aggregate
request-log usage metadata. Downstream consumers should either
recompute cost from `requests.jsonl` directly or treat the cost column
as missing for those four rows. (The merge logic has since been
updated to sum usage across per-quantity staging dirs; existing
artifacts were produced before that fix.)

Transient per-quantity staging dirs live at
`results/_perquantity_<model>_<batch>/` during a run and are consumed
by the merge step.
