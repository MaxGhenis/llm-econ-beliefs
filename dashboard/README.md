# Dashboard

This is a small Next.js + Tailwind viewer for the experiment artifacts in the sibling `../results` directory.

## What It Shows

- quantity-by-quantity model comparisons
- interval swaps across pooled, REML, and Bayesian methods
- run-level response inspection
- citation anchors and raw responses for specific runs

The app reads:

- `../results/*/summary.csv`
- `../results/*/runs.jsonl`

and picks the preferred experiment for each `(model, quantity)` pair by:

1. highest `n_successful_runs`
2. highest logged token usage
3. latest experiment timestamp

## Run

```bash
PATH=/opt/homebrew/bin:$PATH npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

## Build

```bash
PATH=/opt/homebrew/bin:$PATH npm run build
```

## Notes

- The app is dependency-light on purpose. It uses `csv-parse` to read summary files and Node file-system access for the run artifacts.
- The route `/api/responses` fetches run-level payloads on demand so the initial page does not need to inline every raw response.
