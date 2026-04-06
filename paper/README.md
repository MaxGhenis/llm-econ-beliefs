# Paper Framing

Current draft:

- `draft.md`

Working title:

`What Do Large Language Models Believe About Economic Parameters?`

## Core Question

If you ask a frontier LLM for a canonical economic parameter, what posterior does it reveal?

That posterior has at least four objects:

- a central estimate
- an uncertainty interval
- a choice of interpretation when the parameter name is ambiguous
- a set of literature anchors it claims to rely on

## Initial Parameter Families

- labor supply
- household preferences
- production
- taxation
- trade
- macro persistence and growth

## Natural First Table

One row per quantity, one column per model:

- pooled point estimate
- pooled 90 percent interval
- between-run standard deviation
- benchmark range from a review paper, handbook, or model documentation

## First Empirical Cut

Start with a registry of roughly 10 to 15 quantities and run 20 to 100 samples per model-quantity pair.

Prioritize quantities where at least one of the following is true:

- there is a review article or handbook range
- the quantity appears in OG-USA or OG-Core documentation
- economists disagree materially, making uncertainty elicitation informative

## Calibration Appendix

Keep calibration inside the same paper for now, but modular in the repo and in the writeup.

The clean division is:

- main text: raw elicited beliefs over economic quantities
- methods appendix or secondary section: post-hoc calibration on resolved numeric tasks

Calibration should answer a different question from the main paper:

- main paper question: what distribution does the model appear to express?
- calibration question: how should that manifested distribution be adjusted if we want better out-of-sample predictive reliability?

The default calibration object should be the pooled predictive distribution, not the uncertainty around a latent consensus mean.

Good default losses and diagnostics:

- pinball loss for elicited quantiles
- weighted interval score for central intervals
- PIT diagnostics and empirical-PIT recalibration for full pooled CDFs

Avoid making calibration the headline estimand. Raw beliefs stay primary; calibrated distributions are a secondary, externally corrected object.
