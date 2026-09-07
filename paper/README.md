# Paper

Canonical manuscript: `paper.qmd`.

Supporting files:

- bibliography: `references.bib`
- Quarto project file: `_quarto.yml`
- build script for all data-derived tables: `build_tables.py`
- main-text tables: `tables/model-overview-labor-tax.md`, `tables/model-overview-macro-trade.md`, `tables/benchmark-comparison-labor-tax.md`, `tables/toy-top-rate-labor-tax.md`, `tables/income-signfix-delta.md`
- appendix tables: `tables/stability-appendix.md`, `tables/pooling-robustness-appendix.md`, `tables/leave-one-provider-out-appendix.md`, `tables/quantile-rule-appendix.md`, `tables/tool-use-appendix.md`, `tables/grok-failures-appendix.md`, `tables/quantity-disagreement.md`, `tables/armington-clarify-delta.md`, `tables/ies-clarify-delta.md`, `tables/flat-tax-demogrant-appendix.md`
- simulation-facing tables: `tables/model-overview-simulation.md`, `tables/quantity-disagreement-simulation.md`
- referee reports used during revision: `referee-reports/`

## Cached manuscript reproduction

From the repository root:

```bash
python3 scripts/reproduce_cached.py --check
```

The dependency-free check requires a clean Git checkout. It deletes and rebuilds
the declared analysis outputs in a temporary source copy with no Git history and
compares bytes. Run `--write` to update outputs intentionally, then inspect and
commit the diff before running `--check`. The lower-level `python3
paper/build_tables.py` also uses only cached inputs and fails on missing or
changed evidence. It never invokes a PolicyEngine interpreter.

Table 4 and A13 use the frozen `results/top-rate-calibration.json` exactly.
A12 replays the retained rounded CSV in `results/frozen/`; unrounded values do
not survive. `results/calibration-provenance.json` binds these files to their
SHA-256 hashes and records unknown historical source identity explicitly.
The existing manuscript describes Enhanced CPS 2024, but no retained record
verifies the model version, dataset build, simulation period or generation time.
This replay does not establish those facts.

## Fresh calibration is separate

`python3 paper/build_tables.py --fresh-calibration` is an explicit error, before
any output is written. The former automatic subprocess path used unspecified
local defaults and manual microdata weights; it has been removed. A future,
explicitly requested recalibration requires a reviewed implementation using
PolicyEngine's MicroSeries and `map_to`, an explicitly selected certified model
and dataset release, a simulation period, and a new provenance artifact recording
package versions, dataset identity/hash and execution time. It must write to a
new artifact, compare with the frozen baseline, and receive scientific review
before any manuscript calibration changes. A fresh run cannot retroactively fill
in unknown historical provenance. No fresh calibration was run for this archive.

## Render verification

From the repository root, after any manuscript prose or table-note edit:

```bash
quarto render paper/paper.qmd --to html
quarto render paper/paper.qmd --to pdf
bash scripts/sync_paper_embed.sh
```

Inspect the changed PDF pages with `pdftoppm`, and bump `PAPER_VERSION` in
`dashboard/src/app/paper/page.tsx` when syncing the render. These commands create
local artifacts only; do not dispatch the deployment workflow. HTML/PDF rendering
requires Quarto and TeX and is separate from the dependency-free scientific
reproduction gate. Binary render identity depends on those tools/fonts/metadata;
the gate promises byte equality for analysis CSV/Markdown and release metadata,
not for PDF binaries.

## Core question

If you ask a frontier LLM for a canonical economic parameter, what prompt-conditioned response distribution does it produce?

That distribution has at least four parts:

- a central estimate
- an uncertainty interval
- a choice of interpretation when the parameter name is ambiguous
- a set of literature anchors it claims to rely on

## Initial parameter families

- labor supply
- household preferences
- production
- taxation
- trade
- macro persistence and growth

## Calibration appendix

Calibration lives in the repo but is secondary to raw elicitation.

- main text: raw elicited distributions over economic quantities
- methods appendix or secondary section: post-hoc calibration on resolved numeric tasks

The default calibration object is the pooled predictive distribution, not the uncertainty around a latent consensus mean. Good default losses and diagnostics are pinball loss for elicited quantiles, weighted interval score for central intervals, and PIT diagnostics with empirical-PIT recalibration for full pooled CDFs.

Raw distributions stay primary; calibrated distributions are a secondary, externally corrected object.
