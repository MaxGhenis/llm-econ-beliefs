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

Each model run is one elicited posterior over a named quantity. The prompt pins down the object, asks the model to resolve ambiguity explicitly, and returns structured JSON with:

- `interpretation`
- `point_estimate`
- `quantiles.p05`
- `quantiles.p25`
- `quantiles.p50`
- `quantiles.p75`
- `quantiles.p95`
- `citations`
- `reasoning_summary`

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
python3 -m pytest
```

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
