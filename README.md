# LLM Econ Beliefs

This repository studies what large language models say they believe about canonical economic quantities.

The target is not simulated policy behavior. The target is elicited beliefs: point estimates, uncertainty, interpretation, and literature anchors for parameters that economists actually use in calibration, estimation, and policy analysis.

That includes quantities that show up in model environments such as OG-USA and OG-Core:

- Frisch elasticity of labor supply
- coefficient of relative risk aversion
- annual discount factor
- capital share
- elasticity of substitution between capital and labor
- tax-function or tax-response parameters

It also includes adjacent public-finance and macro parameters:

- elasticity of taxable income
- income elasticity of labor supply
- Armington elasticities
- TFP persistence

## Repo Layout

```text
.
├── llm_econ_beliefs/
│   ├── aggregate.py
│   ├── calibration.py
│   ├── distributions.py
│   ├── models.py
│   ├── parse.py
│   ├── prompts.py
│   ├── registry.py
│   ├── runner.py
│   └── data/quantities.toml
├── paper/
│   └── README.md
└── tests/
```

## Initial Design

Each model run is one elicited posterior over a named quantity. The current default prompt is a memory-only elicitation prompt with a fixed target interpretation, and it returns structured JSON with:

- `interpretation`
- `point_estimate`
- `quantiles.p05`
- `quantiles.p25`
- `quantiles.p50`
- `quantiles.p75`
- `quantiles.p95`
- `citations`
- `reasoning_summary`

The current default prompt version is `v2`. Older runs in the repo used earlier prompt variants and remain useful for comparison, but they should not be pooled with `v2` runs without an explicit robustness check.

Repeated runs are then pooled using the law of total variance:

`Var(theta) = E[Var(theta | run)] + Var(E[theta | run])`

That cleanly separates:

- stated within-run uncertainty
- instability across runs
- total pooled uncertainty

When run-level quantiles are available, the code reconstructs an approximate within-run distribution from those quantiles instead of assuming a symmetric interval around the point estimate.

Calibration lives in a separate module so the repo can support a second object without conflating it with raw elicitation:

- raw belief distributions from the model
- post-hoc recalibrated predictive distributions on resolved numeric tasks

The intended workflow is:

1. reconstruct a run-level distribution from elicited quantiles
2. pool runs into a model-level mixture distribution
3. evaluate that pooled distribution on held-out resolved numeric targets
4. optionally fit a PIT-based recalibrator and report calibrated predictive performance as a secondary result

## Quick Start

```bash
uv venv .venv
uv pip install -p .venv/bin/python -e ".[dev,providers]"
.venv/bin/python -m pytest
```

The core package has no runtime dependencies; the `providers` extra pulls in
`anthropic` (native Claude path) and `litellm` (Gemini/Grok paths).

```python
from llm_econ_beliefs import aggregate_beliefs, create_belief_prompt, get_quantity

quantity = get_quantity("labor_supply.frisch_elasticity.prime_age")
prompt = create_belief_prompt(quantity)
print(prompt)
```

```bash
python3 -m llm_econ_beliefs \
  --model sonnet \
  --runs 2 \
  --quantity labor_supply.frisch_elasticity.prime_age \
  --quantity household.relative_risk_aversion.crra \
  --output-dir results/sonnet-poc
```

```bash
python3 -m llm_econ_beliefs \
  --provider openai \
  --model gpt-5.4-mini \
  --runs 5 \
  --samples-per-request 5 \
  --temperature 1.0 \
  --quantity labor_supply.frisch_elasticity.prime_age \
  --output-dir results/gpt-5.4-mini-frisch-batch5
```

For OpenAI Chat Completions, `--samples-per-request` maps to the API's `n` parameter so repeated draws can share the prompt cost within a single request.

```bash
.venv/bin/python -m llm_econ_beliefs \
  --provider anthropic \
  --model claude-fable-5 \
  --runs 5 \
  --max-workers 4 \
  --quantity labor_supply.frisch_elasticity.prime_age \
  --output-dir results/claude-fable-5-frisch-batch5
```

The `anthropic` provider serves Claude Fable 5, Claude Opus 5, Claude Opus
4.8, and Claude Sonnet 5 through the official Anthropic SDK with structured
outputs (`output_config.format`). These models reject sampling parameters, so
requests carry no temperature; each model keeps its default thinking behavior
(always-on for Fable 5, adaptive for Opus 5 and Sonnet 5, off for Opus 4.8), and
`--max-workers` parallelizes repeated draws. Older Claude models
(Opus 4.7 and earlier) continue to run through the LiteLLM path used for the
April 2026 panel.

```bash
python3 -m llm_econ_beliefs.compare \
  --quantity labor_supply.frisch_elasticity.prime_age \
  --result-dir results/gpt-5.4-frisch-batch5-v1 \
  --result-dir results/gpt-5.4-mini-frisch-batch5-v4 \
  --result-dir results/gpt-5.4-nano-frisch-batch5-v1 \
  --output results/frisch-model-comparison.csv
```

When provider metadata is available, the experiment runner also writes request-level usage logs:

- `requests.jsonl`
- `requests.csv`

and appends aggregated usage columns to `summary.csv`, including prompt tokens, completion tokens, total tokens, and average total tokens per successful run.

For supported OpenAI models, the runner also estimates USD cost from logged token usage and writes request-level and summary-level cost columns. The current local pricing table is sourced from [OpenAI API Pricing](https://openai.com/api/pricing/) as of April 5, 2026.

```python
from llm_econ_beliefs import (
    CalibrationExample,
    evaluate_calibration,
    fit_pit_calibrator,
    mixture_distribution,
    piecewise_distribution_from_quantiles,
)

distribution = piecewise_distribution_from_quantiles(
    {"p05": 0.2, "p25": 0.35, "p50": 0.5, "p75": 0.8, "p95": 1.5}
)
examples = [CalibrationExample(distribution=distribution, observed_value=0.7)]
metrics = evaluate_calibration(examples)
calibrator = fit_pit_calibrator(examples)
calibrated_distribution = calibrator.calibrate_distribution(distribution)
print(metrics)
print(calibrated_distribution.quantile(0.5))
```

## Reproducing the v4 panel

Two reproduction paths are supported.

### (a) From cached results

Use the artifacts already committed under `results/` to rebuild the
analysis layer and all paper tables, then verify the test suite and
prose pins. This incurs no provider API cost.

```bash
python3 -m pytest
python3 scripts/build_correlates.py
cd paper && PYTHONPATH=$(pwd)/.. python3 build_tables.py
cd .. && python3 scripts/verify_paper_prose.py
```

The canonical order matters: `build_tables.py` writes the microdata
calibration to `results/top-rate-calibration.json`, which
`scripts/build_correlates.py` consumes to regenerate
`results/correlates-*.csv`, which `build_tables.py` in turn reformats
into `paper/tables/`. Starting from the committed artifacts, the
sequence above is a fixed point — `git status` stays clean — and
`verify_paper_prose.py` then asserts every pinned number in
`paper/paper.qmd` against the regenerated tables.

Two tables (Table 4 and Appendix A13) calibrate a Pareto tail from
PolicyEngine US microdata. Point `POLICYENGINE_US_REPO` at a
policyengine-us checkout with an installed `.venv` (or set
`POLICYENGINE_US_PYTHON` to its interpreter) to reproduce the
committed `a = 1.621` build; without it, `build_tables.py` prints a
loud warning, uses the fallback `a = 1.5` for those two tables only,
and leaves the committed `results/top-rate-calibration.json`
untouched so `build_correlates.py` keeps working from the microdata
values.

### (b) Re-elicit from scratch

Run the April v4 panel from scratch. This calls out to every provider
and takes roughly 1–2 hours of wall time. The full committed panel is
31 models × 26 quantities × 15 runs (12,090 main-panel runs) elicited
in six waves: the 11-model April wave via this driver (~$23), the
6-model July frontier wave (~$38 including clarify probes), the
5-model Chinese-lab wave through OpenRouter (~$29 at the account
level, including recovered failures), the 3-model GPT-5.6 wave
(request-level costs untracked), the late Grok 4.5, Kimi K3,
Gemini 3.6 Flash, and Claude Opus 5 additions ($2.71 and $2.99 tracked
for Grok 4.5 and Gemini 3.6 Flash; ~$10.69 and ~$11.30 estimated from
logged tokens for Kimi K3 and Claude Opus 5), and the August Qwen 3.8
Max and Inkling additions ($10.61 and $7.97 tracked — OpenRouter now
reports per-request costs, so their logs carry them, including the
failed attempts their recovery replaced). Set provider API keys in your
environment first (`OPENROUTER_API_KEY` for the Chinese-lab models).
If any runs fail on infrastructure errors,
`scripts/rerun_failed_runs.py` re-elicits the failed slots in place
and archives the failed records; `scripts/check_panel_grid.py` is the
authoritative completeness check, and
`scripts/supervise_openweights.sh` loops recovery passes to a full
grid under a single-instance lock.

```bash
python3 scripts/run_v4_full_panel.py
```

Four premium models (`claude-sonnet-4.6`, `claude-opus-4.7`,
`gemini-3.1-pro-preview`, and `grok-4.20`) may need the per-quantity
fallback when multi-quantity cells hang under the main driver. Re-run
those four explicitly with:

```bash
python3 scripts/run_v4_per_quantity.py --model claude-sonnet-4.6
python3 scripts/run_v4_per_quantity.py --model claude-opus-4.7
python3 scripts/run_v4_per_quantity.py --model gemini-3.1-pro-preview
python3 scripts/run_v4_per_quantity.py --model grok-4.20
```

See `results/README.md` for the result-directory naming convention and
a note on how the per-quantity fallback affects `summary.csv` cost
aggregation.

### (c) July 2026 model additions

Fourteen models released after the April rerun extend the panel under
the same v4 prompts, in three waves: the frontier updates (`gpt-5.5`,
`claude-fable-5`, `claude-opus-4.8`, `claude-sonnet-5`,
`gemini-3.5-flash`, `grok-4.3`), the Chinese-lab wave
(`deepseek-v4-pro`, `qwen-3.7-max`, `kimi-k2.6`, `glm-5.2`,
`minimax-m3`, served through OpenRouter), and the GPT-5.6 family
(`gpt-5.6-sol`, `gpt-5.6-luna`, `gpt-5.6-terra`). Re-elicit any subset
with:

```bash
.venv/bin/python scripts/run_v4_new_models.py --models <comma-separated-ids>
```

The driver uses the per-quantity fallback path for every cell (results
persist incrementally; completed cells are skipped on rerun) and
validates model ids against the registry
(`llm_econ_beliefs/model_registry.py`). The July Claude models run
through the native Anthropic SDK provider; the Chinese-lab models run
through OpenRouter in JSON mode with local schema validation.

## Initial Quantity Set

The first registry is intentionally broad enough to support a paper that starts with labor-supply review parameters but can expand to model-calibration inputs used in OG-USA-style work.

Official OG-USA and OG-Core documentation that informed the initial quantity list:

- [Calibration of Macroeconomic Parameters — OG-USA](https://pslmodels.github.io/OG-USA/content/calibration/macro.html)
- [Exogenous Parameters — OG-USA](https://pslmodels.github.io/OG-USA/content/calibration/exogenous_parameters.html)
- [Model Parameters — OG-Core](https://pslmodels.github.io/OG-Core/content/intro/parameters.html)

## Dashboard

A small Next.js + Tailwind inspection app now lives in [dashboard/README.md](/Users/maxghenis/llm-econ-beliefs/dashboard/README.md).

It reads the existing `results/` artifacts and lets you:

- switch among elasticity quantities
- compare model centers and intervals under pooled, REML, and Bayesian methods
- inspect run-level raw responses, quantiles, and citation anchors

Run it with:

```bash
cd dashboard
PATH=/opt/homebrew/bin:$PATH npm run dev
```
