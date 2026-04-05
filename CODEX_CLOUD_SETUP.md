# Codex Cloud Setup

Use this setup when you want a Codex cloud run that is as close as possible to a clean belief elicitation rather than a tool-using research task.

## Goal

Estimate Codex's manifested beliefs about an economic quantity without contamination from:

- the local machine
- project-specific instructions
- internet lookups
- autonomous code execution

## Recommended Design

Use a separate minimal GitHub repo for Codex cloud tasks.

Repo contents:

- no `AGENTS.md`
- no literature notes
- no experiment outputs
- no calibration files
- no scripts beyond the bare minimum

If possible, make the repo nearly empty. The task prompt should carry the elicitation instructions.

## Codex Environment

In Codex web environment settings:

- base image: `universal`
- setup script: empty
- maintenance script: empty
- internet access: off
- environment variables: none
- secrets: none

Also reset the environment cache between batches when you want fresh runs. Codex cloud can reuse cached container state, which is convenient for engineering tasks but not ideal for repeated elicitation.

## Task Prompt

Use one fresh cloud task per run.

Recommended prompt:

```text
Do not use shell commands, files, git history, the web, MCP tools, or any external resources.
Do not inspect the repository contents.
Answer only from your current background knowledge and internal beliefs.

Quantity of interest:
- ID: labor_supply.frisch_elasticity.prime_age
- Name: Frisch elasticity of labor supply
- Description: Percent change in hours worked from a 1 percent change in the net-of-tax wage, holding the marginal utility of wealth constant.
- Population/context: Prime-age workers in the United States
- Units: elasticity
- Preferred interpretation: Lifecycle or macro-calibration Frisch elasticity for hours worked
- Plausible support: [0.0, 10.0]

Task:
1. Use the most standard interpretation in applied economics.
2. If the quantity is ambiguous, choose one interpretation and say exactly what you chose.
3. Give your own best central estimate.
4. Give the 5th, 25th, 50th, 75th, and 95th percentiles of your uncertainty distribution.
5. Give 2 to 4 literature anchors from memory.
6. Keep the explanation brief and substantive.

Return valid JSON only.
{
  "interpretation": "short string",
  "point_estimate": 0.0,
  "quantiles": {
    "p05": 0.0,
    "p25": 0.0,
    "p50": 0.0,
    "p75": 0.0,
    "p95": 0.0
  },
  "citations": [
    "Author (Year)",
    "Author (Year)"
  ],
  "reasoning_summary": "short string"
}
```

## Contamination Checks

After each run:

- check the work log for tool use
- discard any run that used shell, file reads, web, or repo inspection
- record whether the response followed the JSON schema exactly

If Codex ignores the "do not inspect" instruction and uses tools anyway, treat that as a different experimental arm:

- `codex-cloud-no-tools`
- `codex-cloud-agentic`

Do not pool those together.

## Run Protocol

For each quantity:

1. Start a new cloud task.
2. Paste the prompt.
3. Save the raw JSON.
4. Save whether any tools were used.
5. Repeat.

Suggested first pass:

- `n = 10` for debugging
- `n = 15` for a usable comparison to the API models

## Interpretation

This setup does not identify Codex's "true internal beliefs." It identifies the beliefs Codex manifests in a clean cloud task when instructed not to research.

That is still useful, but it should be labeled clearly in the paper.
