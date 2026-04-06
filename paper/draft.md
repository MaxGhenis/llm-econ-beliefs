# What Do Large Language Models Believe About Economic Elasticities?

## Abstract

This paper studies the beliefs that frontier large language models appear to express about canonical economic elasticities. Rather than asking whether the models can forecast realized outcomes or simulate policy behavior, we ask a simpler question: if a model is directly asked for an economic parameter, what distribution does it reveal? We elicit point estimates, five subjective quantiles, a fixed interpretation, and brief literature anchors for eight elasticity quantities spanning labor supply, household preferences, taxation, production, and trade. We run repeated elicitation prompts across 11 models from OpenAI, Anthropic, Google, and xAI, then pool within-run uncertainty and across-run variation into model-level predictive distributions.

In the current elasticity panel, we collect 1,305 successful runs out of 1,320 attempted runs across 11 models, 8 quantities, and 15 runs per model-quantity cell, at a total API cost of about `$6.73`. The main findings are descriptive. First, models differ systematically in the responsiveness they imply. In this panel, Gemini models tend to report lower elasticities, Grok models tend to report higher elasticities, and Claude and GPT models usually sit in between. Second, the models differ in stated uncertainty. Gemini and Claude models tend to express narrower uncertainty intervals, while Grok and smaller GPT variants tend to express wider ones. Third, disagreement is highly quantity-specific: cross-model spread is small for the elasticity of taxable income but large for the Armington elasticity and extensive-margin labor supply.

The contribution is methodological as much as substantive. The paper proposes a simple way to treat repeated LLM elicitation runs as distributions rather than point forecasts, and it shows that models differ not only in central estimates but also in uncertainty and in the structure of disagreement across parameter classes. A secondary contribution is practical: for this kind of elicitation, low-cost models already reveal substantial cross-provider heterogeneity at very low dollar cost.

## 1. Introduction

Economists use elasticities constantly. They enter sufficient-statistics formulas, optimal tax calculations, quantitative macro calibrations, and applied policy debates. Yet many of the elasticities that matter most are uncertain, interpretation-sensitive, and contested. If large language models are going to be used as policy assistants, research aids, or informal synthetic experts, it matters what elasticities they appear to believe.

The key object in this paper is not forecasting accuracy and not simulated policy behavior. It is manifested belief. We ask a model for a quantity such as the Frisch elasticity of labor supply, the elasticity of taxable income, or the Armington elasticity. We require a fixed interpretation, a point estimate, and a distributional summary. Repeating that elicitation many times lets us measure three distinct objects:

- the central estimate the model tends to report
- the uncertainty the model states within a run
- the variation the model exhibits across repeated runs

This is conceptually close to expert elicitation, but with machine respondents rather than human experts. It is also close to the recent literature on LLM uncertainty elicitation, but the target is different. We are not asking for confidence about a ground-truth quiz answer. We are asking what the model appears to endorse about contested economic quantities. In that setting, the disagreement itself is often part of the result.

The initial empirical question is narrow: what do leading models believe about a common panel of economic elasticities? But the broader project is to build a general method for eliciting LLM beliefs about economic parameters. The elasticity panel is useful because it includes canonical parameters with clear policy relevance and well-known disagreements in the literature. It also makes the interpretation of model differences straightforward: holding welfare weights fixed, lower behavioral elasticities generally imply more room for redistribution, while higher elasticities generally imply larger efficiency costs of redistribution.

This paper therefore contributes on three margins. First, it provides a reproducible design for eliciting probabilistic beliefs from LLMs over economic quantities. Second, it documents systematic cross-model differences in central estimates and stated uncertainty. Third, it shows that these differences are economically meaningful: the models do not merely disagree in noise, but along margins that would map into different policy conclusions in standard optimal-tax and calibrated-macro environments.

## 2. Design

### 2.1 Prompting target

The main prompt is a memory-only elicitation prompt. It tells the model to answer from background knowledge only, not to use tools or external resources, and not to reconstruct a consensus estimate by conducting a literature review. The prompt fixes the target interpretation of the quantity and requests JSON with:

- `interpretation`
- `point_estimate`
- `quantiles.p05`
- `quantiles.p25`
- `quantiles.p50`
- `quantiles.p75`
- `quantiles.p95`
- `citations`
- `reasoning_summary`

The current default prompt version is `v2`. Earlier prompt variants remain in the repository as robustness material but are not pooled with the `v2` main panel.

### 2.2 Quantities

The current paper draft focuses on 8 elasticities:

- intertemporal elasticity of substitution
- extensive-margin labor supply elasticity for single mothers
- Frisch elasticity of labor supply for prime-age workers
- income elasticity of labor supply for prime-age workers
- Marshallian wage elasticity of labor supply for prime-age workers
- elasticity of substitution between capital and labor
- elasticity of taxable income for top earners
- Armington elasticity between imported and domestic goods

These quantities were selected because they are economically interpretable, policy-relevant, and present real room for disagreement across literatures.

### 2.3 Repeated elicitation and pooling

Each model-quantity cell is elicited 15 times. We then construct a run-level belief distribution from the elicited quantiles and pool those runs into a model-level predictive distribution. This keeps two uncertainty objects separate:

- within-run uncertainty, reflected in the model’s own quantiles
- across-run variation, reflected in repeated elicitation instability

The paper’s default interval object is the pooled predictive interval, not a confidence interval for the mean response. That choice is deliberate. The goal is to measure how uncertain the model appears to be, not how precisely we know the mean of its repeated answers.

### 2.4 Models

The current main panel includes 11 models:

- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.4-nano`
- `claude-opus-4.6`
- `claude-sonnet-4.6`
- `claude-haiku-4.5`
- `gemini-3.1-pro-preview`
- `gemini-3-flash-preview`
- `gemini-3.1-flash-lite-preview`
- `grok-4.20`
- `grok-4.1-fast`

The design is intentionally symmetric across providers: same quantities, same prompt family, same number of repeated runs.

## 3. Data Collected So Far

The current no-tools panel contains:

- 11 models
- 8 quantities
- 15 runs per model-quantity cell
- 1,320 attempted runs
- 1,305 successful runs
- total observed API cost of about `$6.73`

Success rates are high overall. All models except one completed every planned run. The only notable failure rate comes from `grok-4.20`, which completed `105` of `120` runs.

Total model-level costs vary substantially:

- `claude-opus-4.6`: `$2.133`
- `claude-sonnet-4.6`: `$1.333`
- `gemini-3.1-pro-preview`: `$1.123`
- `grok-4.20`: `$1.019`
- `claude-haiku-4.5`: `$0.367`
- `gemini-3-flash-preview`: `$0.109`
- `gemini-3.1-flash-lite-preview`: `$0.084`
- `grok-4.1-fast`: `$0.018`

This cost dispersion matters for study design. The cheap models are not merely a convenience; they make it feasible to estimate repeated-run uncertainty at scale.

## 4. Main Descriptive Results

### 4.1 Models differ in implied responsiveness

The main descriptive pattern is that models systematically differ in how elastic they think the world is. To summarize this without comparing quantities on incomparable scales, we rank models within each quantity by the magnitude of the pooled point estimate and then average those within-quantity ranks.

On that measure, the most responsiveness-heavy models in the current panel are:

- `gpt-5.4-nano`
- `grok-4.20`
- `gpt-5.4-mini`
- `gpt-5.4`

The least responsiveness-heavy are:

- `gemini-3.1-flash-lite-preview`
- `gemini-3-flash-preview`
- `gemini-3.1-pro-preview`
- `claude-opus-4.6`

The same pattern is visible in the policy-relevant subset of people and tax elasticities. There, `grok-4.20` and `gpt-5.4-nano` sit toward the high-elasticity end, while Gemini models sit toward the low-elasticity end. Holding social welfare weights fixed, this pattern maps naturally into a policy interpretation: models with lower behavioral elasticities would support more redistribution ceteris paribus, while models with higher elasticities would support less.

This is not an ideological result in any rigorous sense, but it is a meaningful descriptive result about model priors over behavioral responsiveness.

### 4.2 Models differ in stated confidence

Models also differ in how tight their pooled uncertainty bands are. Ranking models within each quantity by pooled 90 percent interval width gives a consistent confidence ordering.

The most confident models in the current panel are:

- `gemini-3.1-pro-preview`
- `gemini-3.1-flash-lite-preview`
- `claude-haiku-4.5`
- `gemini-3-flash-preview`

The least confident are:

- `grok-4.20`
- `grok-4.1-fast`
- `gpt-5.4-mini`
- `gpt-5.4-nano`

This is a statement about expressed uncertainty, not calibrated uncertainty. A model with narrow intervals may simply be overconfident. But the cross-model pattern is stable enough to be worth treating as an empirical finding in its own right.

### 4.3 Disagreement is concentrated in a few quantities

Cross-model disagreement is not uniform across the elasticity panel.

The largest spread in pooled point estimates is for the Armington elasticity:

- lowest pooled center: `1.5` for `claude-opus-4.6`
- highest pooled center: `3.07` for `grok-4.1-fast`

The next largest spreads are:

- extensive-margin labor supply elasticity for single mothers
- intertemporal elasticity of substitution

By contrast, the smallest spread is for the elasticity of taxable income:

- lowest pooled center: about `0.37`
- highest pooled center: about `0.51`

This is useful because it shows that the models are not just randomly shifted versions of one another. Instead, disagreement appears to be concentrated in parameters where the underlying literature is more interpretation-sensitive or where macro and micro traditions diverge more sharply.

### 4.4 A few concrete examples

For the Frisch elasticity of labor supply, most models cluster around `0.5`, but `gpt-5.4-nano` is noticeably higher at about `0.73` and `grok-4.20` is also relatively high at about `0.64`. For the income elasticity of labor supply, nearly every model places the pooled center below zero, but `grok-4.1-fast` is a notable outlier with a positive pooled center. For the Armington elasticity, the panel ranges from the low `1.5` values of `claude-opus-4.6` and `claude-haiku-4.5` to the much higher values of `gpt-5.4-mini` and `grok-4.1-fast`.

These are not small numeric differences. In a calibrated model or an optimal-policy sufficient-statistics exercise, they would matter.

## 5. Interpretation

The paper’s main claim is not that one model is correct. There is no single benchmark truth for many of these quantities, and some quantities are themselves interpretation-sensitive. The result is that frontier LLMs appear to carry different priors over the elasticity structure of the economy.

That matters for at least three reasons.

First, if users treat LLMs as informal research assistants, these prior differences can influence what kinds of policy arguments feel “natural” to the model. A model with low ETI and low labor-supply elasticities will tend to make redistribution look cheaper than a model with higher elasticity beliefs.

Second, stated uncertainty is itself informative. Some models behave as if the relevant literature is tight and settled; others behave as if the same quantities are wide-open. Users interacting with these models will experience that difference directly.

Third, disagreement is structured. The current panel suggests that some models are especially conservative on trade and production elasticities, while others are especially responsiveness-heavy on labor and tax elasticities. A natural next step is to test whether those differences remain stable when the panel expands beyond elasticities.

## 6. Limitations

This draft should be read as an initial results paper, not a final measurement of “true” model beliefs.

The current limitations are straightforward:

- the prompt remains a prompt, not a perfect readout of latent beliefs
- the main panel is memory-only; retrieval-augmented and tool-using arms are still secondary
- the current paper focuses on elasticities, not the broader OG-USA-style parameter set
- the interval analysis is descriptive; calibration against resolved numeric tasks is still modular but secondary
- citation strings are noisy and need normalization before they can support serious bibliometric analysis

The current design nevertheless identifies a meaningful object: the distribution the model manifests under a fixed elicitation protocol.

## 7. Next Sections To Add

The most immediate next writing tasks are:

1. A literature review on expert elicitation, LLM uncertainty elicitation, and probabilistic aggregation.
2. A methods subsection formalizing the pooled predictive distribution and the hierarchical alternatives.
3. A results table in publication form, likely with one row per quantity and one column per model family.
4. A calibration section using resolved numeric tasks as an external benchmark.
5. A tools appendix comparing memory-only, web-assisted, and tool-using elicitation.

## References To Add

The current draft still needs formal citations for at least the following literatures and sources:

- expert elicitation and parameter uncertainty
- LLM confidence and interval elicitation
- Metaculus and related distribution aggregation work
- CBO and handbook reviews for labor-supply elasticities
- public-finance references for ETI
- trade and production references for Armington and capital-labor substitution

Until those are added, this file should be treated as a results draft rather than a near-final manuscript.
