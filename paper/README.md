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

Rebuild tables from the current `results/` CSVs:

```bash
PYTHONPATH=$(pwd)/.. /opt/homebrew/opt/python@3.14/bin/python3.14 build_tables.py
```

The flat-tax and optimal-top-rate tables call into the PolicyEngine-US venv at `~/PolicyEngine/policyengine-us/.venv/bin/python`. If that venv is missing the script falls back to stylized Pareto parameters and emits a note.

Render the manuscript:

```bash
/Users/maxghenis/quarto/bin/quarto render paper.qmd
```

Quarto is installed at `/Users/maxghenis/quarto/bin/quarto` locally and is not on the default shell `PATH`. The manuscript renders to `paper.html`.

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
