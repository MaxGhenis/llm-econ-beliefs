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

## July 2026 model additions

Six models released after the April rerun were elicited under the same
v4 prompts in July 2026 via `scripts/run_v4_new_models.py`:

- `gpt-5.5` (OpenAI Chat Completions path)
- `claude-fable-5`, `claude-opus-4.8`, `claude-sonnet-5` (native
  Anthropic SDK path — structured outputs, no sampling parameters,
  model-default thinking)
- `gemini-3.5-flash`, `grok-4.3` (LiteLLM path)

All six ran through the per-quantity fallback with the fixed merge, so
their `summary.csv` cost columns are aggregated end to end (no
`$0.0000` caveat). The Anthropic-path runs used four concurrent
workers per quantity; draws remain independent single-request
elicitations.

## Failed-run re-elicitation manifest

`failure-manifest.csv` records every July 2026 run slot whose first
attempt failed on an infrastructure error and was re-elicited as a
fresh draw: 58 runs across 15 (model, batch, quantity) cells, with the
error class and the replacement request IDs (traceable in each
directory's `requests.jsonl`). The two gpt-5.5 capital-gains cells were
re-elicited in full at a raised completion cap (1200 -> 8000 tokens)
after every first-attempt run returned empty text; all other cells
replaced at most 5 of 15 runs under identical settings.
`scripts/rerun_failed_runs.py` now also archives the failed records to
`failed-runs-archive.jsonl` inside the affected directory before
replacing them (the July round predates this and is reconstructed in
the manifest from appended request logs).

## Retained archive inventory

`archive-manifest.json` is generated from observed model IDs, prompt-version tags,
parse outcomes and prompt-text hashes in every retained run file. It locks all
retained CSV/JSON/JSONL inputs by SHA-256, including raw responses, request logs,
failed attempts, scientific summaries, the PolicyBench pin and frozen calibration.
`python3 scripts/archive_manifest.py` verifies it; `--write` is an explicit archive
update that requires reviewing the evidence diff, never an ordinary replay step.

The retained main grid is 31 x 26 x 15 = 12,090 successful records. Separate
inventories retain 930 follow-up records (903 parsed, 27 failed), 526 auxiliary
records (375 parsed, 151 failed), and 5,896 archived failed-attempt records. Counts
of archived attempts are not counts of distinct failed slots or paid requests.
The older 58-slot recovery manifest above has its original July scope; it is not
a count of all subsequent recoveries. Exact historical comparison snapshots under
`archive/` are listed separately and are not added to these counts.

A directory name alone does not prove prompt identity. In particular the retained
April Claude Opus 4.7 Armington follow-up is tagged `v4`, not `armington-clarify`;
the inventory reports the observed tag without relabeling it. Main-panel v4
wording is uniform within each model-quantity cell; 23 quantities share one text
across all 31 models, while three sign-clarified quantities split 24/7 between
the disclosed revised/original wordings. The manifest preserves each variant's
SHA-256 and model membership.

`calibration-provenance.json` documents the frozen top-tail values and rounded
flat-tax display table. Historical model/data source identity remains unknown;
replaying those files is not a fresh microsimulation or a validation against
resolved economic truths.
