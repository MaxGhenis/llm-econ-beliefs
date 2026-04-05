import fs from "node:fs";
import path from "node:path";

import { parse as parseCsv } from "csv-parse/sync";

import type {
  DashboardSummaryData,
  IntervalMethodDefinition,
  IntervalMethodId,
  ModelRunPayload,
  ModelSummary,
  QuantitySummary,
  RunDetail,
  SourceAnchor,
  SourceSummary,
} from "@/lib/dashboard-types";

interface SummaryRecord {
  modelName: string;
  quantityId: string;
  quantityName: string;
  domain: string;
  nSuccessfulRuns: number;
  pooledPointEstimate: number | null;
  pooledLowerBound: number | null;
  pooledUpperBound: number | null;
  remlLatentLocation: number | null;
  remlLatentLower: number | null;
  remlLatentUpper: number | null;
  remlPredictiveLower: number | null;
  remlPredictiveUpper: number | null;
  bayesLatentLocation: number | null;
  bayesLatentLower: number | null;
  bayesLatentUpper: number | null;
  bayesPredictiveLower: number | null;
  bayesPredictiveUpper: number | null;
  usageEstimatedTotalCostUsdTotal: number | null;
  usageEstimatedTotalCostUsdPerSuccessfulRun: number | null;
  usageTotalTokensTotal: number | null;
  usageTotalTokensPerSuccessfulRun: number | null;
  experimentDir: string;
  experimentUpdatedAt: number;
}

interface StoredRunRecord {
  provider: string;
  model_name: string;
  quantity_id: string;
  run_index: number;
  prompt_version: string;
  prompt: string | null;
  raw_response: string | null;
  parsed_ok: boolean;
  point_estimate: number | null;
  interpretation: string | null;
  lower_bound: number | null;
  upper_bound: number | null;
  confidence_level: number | null;
  quantiles: Record<string, number>;
  citations: string[];
  reasoning_summary: string | null;
  error: string | null;
}

const MODEL_ORDER = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"];

const METHOD_DEFINITIONS: IntervalMethodDefinition[] = [
  {
    id: "pooled",
    label: "Pooled Mixture 90% Interval",
    shortLabel: "Pooled",
    kind: "mixture",
    description:
      "Mixture-style pooled interval using the elicited within-run distributions plus across-run variation.",
  },
  {
    id: "remlPredictive",
    label: "REML Predictive 90% Interval",
    shortLabel: "REML Predictive",
    kind: "predictive",
    description:
      "Random-effects predictive interval for a fresh run, using the REML-style meta-analytic approximation.",
  },
  {
    id: "remlLatent",
    label: "REML Latent 90% Interval",
    shortLabel: "REML Latent",
    kind: "latent",
    description:
      "Random-effects interval for the latent central belief rather than a fresh future run.",
  },
  {
    id: "bayesPredictive",
    label: "Bayesian Predictive 90% Interval",
    shortLabel: "Bayes Predictive",
    kind: "predictive",
    description:
      "Bayesian hierarchical predictive interval for a fresh run after pooling within-run and across-run uncertainty.",
  },
  {
    id: "bayesLatent",
    label: "Bayesian Latent 90% Interval",
    shortLabel: "Bayes Latent",
    kind: "latent",
    description:
      "Bayesian hierarchical interval for the latent central belief rather than a fresh future run.",
  },
];

const runCache = new Map<string, StoredRunRecord[]>();

export function loadDashboardSummaryData(
  resultsDir: string = resolveResultsDir(),
): DashboardSummaryData {
  const summaries = selectPreferredSummaries(loadSummaryRows(resultsDir));
  const grouped = new Map<string, SummaryRecord[]>();

  for (const summary of summaries) {
    const group = grouped.get(summary.quantityId) ?? [];
    group.push(summary);
    grouped.set(summary.quantityId, group);
  }

  const quantities = Array.from(grouped.values())
    .map((quantityRows) => buildQuantitySummary(quantityRows))
    .sort((left, right) => left.quantityName.localeCompare(right.quantityName));

  const selectedModelNames = Array.from(
    new Set(
      quantities.flatMap((quantity) => quantity.availableModels),
    ),
  ).sort(compareModelNames);

  const totalCostUsd = sumNullable(
    quantities.flatMap((quantity) =>
      quantity.modelSummaries.map((summary) => summary.totalCostUsd),
    ),
  );

  return {
    generatedAt: new Date().toISOString(),
    methods: METHOD_DEFINITIONS,
    modelNames: selectedModelNames,
    quantities,
    stats: {
      quantityCount: quantities.length,
      modelCount: selectedModelNames.length,
      selectedResultCount: quantities.reduce(
        (count, quantity) => count + quantity.modelSummaries.length,
        0,
      ),
      totalCostUsd,
    },
  };
}

export function loadRunPayload(
  quantityId: string,
  modelName: string,
  resultsDir: string = resolveResultsDir(),
): ModelRunPayload | null {
  const summaries = selectPreferredSummaries(loadSummaryRows(resultsDir));
  const match = summaries.find(
    (summary) =>
      summary.quantityId === quantityId && summary.modelName === modelName,
  );

  if (!match) {
    return null;
  }

  const runs = loadRunsForExperiment(match.experimentDir)
    .filter(
      (run) =>
        run.quantity_id === quantityId && run.model_name === modelName,
    )
    .sort((left, right) => left.run_index - right.run_index)
    .map<RunDetail>((run) => ({
      runIndex: run.run_index,
      promptVersion: run.prompt_version,
      parsedOk: run.parsed_ok,
      pointEstimate: run.point_estimate,
      lowerBound: run.lower_bound,
      upperBound: run.upper_bound,
      confidenceLevel: run.confidence_level,
      quantiles: run.quantiles ?? {},
      interpretation: run.interpretation,
      citations: run.citations ?? [],
      reasoningSummary: run.reasoning_summary,
      rawResponse: run.raw_response,
      prompt: run.prompt,
      error: run.error,
    }));

  return {
    quantityId,
    modelName,
    experimentDir: path.basename(match.experimentDir),
    experimentUpdatedAt: new Date(match.experimentUpdatedAt).toISOString(),
    runs,
  };
}

export function resolveResultsDir(): string {
  // Local dev: results sit next to the dashboard directory
  const localPath = path.resolve(process.cwd(), "..", "results");
  if (fs.existsSync(localPath)) return localPath;
  // Vercel build: results are copied into the dashboard directory
  const bundledPath = path.resolve(process.cwd(), "results");
  if (fs.existsSync(bundledPath)) return bundledPath;
  return localPath;
}

function buildQuantitySummary(quantityRows: SummaryRecord[]): QuantitySummary {
  const modelSummaries = quantityRows
    .map((row) => buildModelSummary(row))
    .sort((left, right) => compareModelNames(left.modelName, right.modelName));

  return {
    quantityId: quantityRows[0].quantityId,
    quantityName: quantityRows[0].quantityName,
    domain: quantityRows[0].domain,
    availableModels: modelSummaries.map((summary) => summary.modelName),
    modelSummaries,
  };
}

function buildModelSummary(row: SummaryRecord): ModelSummary {
  const runs = loadRunsForExperiment(row.experimentDir).filter(
    (run) =>
      run.quantity_id === row.quantityId &&
      run.model_name === row.modelName &&
      run.parsed_ok,
  );

  return {
    modelName: row.modelName,
    experimentDir: path.basename(row.experimentDir),
    experimentUpdatedAt: new Date(row.experimentUpdatedAt).toISOString(),
    nSuccessfulRuns: row.nSuccessfulRuns,
    totalCostUsd: row.usageEstimatedTotalCostUsdTotal,
    costPerRunUsd: row.usageEstimatedTotalCostUsdPerSuccessfulRun,
    totalTokens: row.usageTotalTokensTotal,
    tokensPerRun: row.usageTotalTokensPerSuccessfulRun,
    intervals: {
      pooled: {
        center: row.pooledPointEstimate,
        lower: row.pooledLowerBound,
        upper: row.pooledUpperBound,
      },
      remlPredictive: {
        center: row.remlLatentLocation,
        lower: row.remlPredictiveLower,
        upper: row.remlPredictiveUpper,
      },
      remlLatent: {
        center: row.remlLatentLocation,
        lower: row.remlLatentLower,
        upper: row.remlLatentUpper,
      },
      bayesPredictive: {
        center: row.bayesLatentLocation,
        lower: row.bayesPredictiveLower,
        upper: row.bayesPredictiveUpper,
      },
      bayesLatent: {
        center: row.bayesLatentLocation,
        lower: row.bayesLatentLower,
        upper: row.bayesLatentUpper,
      },
    },
    sourceSummary: summarizeSources(runs),
  };
}

function summarizeSources(runs: StoredRunRecord[]): SourceSummary {
  const counter = new Map<string, number>();

  for (const run of runs) {
    for (const citation of run.citations ?? []) {
      counter.set(citation, (counter.get(citation) ?? 0) + 1);
    }
  }

  const total = Array.from(counter.values()).reduce((sum, value) => sum + value, 0);
  const topAnchors = Array.from(counter.entries())
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 8)
    .map<SourceAnchor>(([citation, count]) => ({
      citation,
      count,
      share: total > 0 ? count / total : 0,
    }));

  return {
    uniqueCitations: counter.size,
    top1Share: topAnchors[0] ? topAnchors[0].share : null,
    top3Share:
      total > 0
        ? topAnchors.slice(0, 3).reduce((sum, anchor) => sum + anchor.count, 0) /
          total
        : null,
    topAnchors,
  };
}

function selectPreferredSummaries(rows: SummaryRecord[]): SummaryRecord[] {
  const grouped = new Map<string, SummaryRecord[]>();

  for (const row of rows) {
    const key = `${row.modelName}::${row.quantityId}`;
    const group = grouped.get(key) ?? [];
    group.push(row);
    grouped.set(key, group);
  }

  return Array.from(grouped.values()).map((group) =>
    [...group].sort(compareSummaryRows)[0],
  );
}

function compareSummaryRows(left: SummaryRecord, right: SummaryRecord): number {
  return (
    right.nSuccessfulRuns - left.nSuccessfulRuns ||
    (right.usageTotalTokensTotal ?? -1) - (left.usageTotalTokensTotal ?? -1) ||
    right.experimentUpdatedAt - left.experimentUpdatedAt ||
    right.experimentDir.localeCompare(left.experimentDir)
  );
}

function loadSummaryRows(resultsDir: string): SummaryRecord[] {
  const experimentDirs = fs
    .readdirSync(resultsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(resultsDir, entry.name));

  const rows: SummaryRecord[] = [];

  for (const experimentDir of experimentDirs) {
    const summaryPath = path.join(experimentDir, "summary.csv");
    if (!fs.existsSync(summaryPath)) {
      continue;
    }

    const experimentUpdatedAt = fs.statSync(summaryPath).mtimeMs;
    const raw = fs.readFileSync(summaryPath, "utf8");
    const parsed = parseCsv(raw, {
      columns: true,
      skip_empty_lines: true,
      relax_quotes: true,
    }) as Record<string, string>[];

    for (const row of parsed) {
      const quantityId = row.quantity_id;
      if (!quantityId.includes("elasticity")) {
        continue;
      }

      rows.push({
        modelName: row.model_name,
        quantityId,
        quantityName: row.quantity_name,
        domain: quantityId.split(".")[0] ?? "other",
        nSuccessfulRuns: parseInteger(row.n_successful_runs) ?? 0,
        pooledPointEstimate: parseNumber(row.pooled_point_estimate),
        pooledLowerBound: parseNumber(row.pooled_lower_bound),
        pooledUpperBound: parseNumber(row.pooled_upper_bound),
        remlLatentLocation: parseNumber(row.reml_latent_location),
        remlLatentLower: parseNumber(row.reml_latent_lower),
        remlLatentUpper: parseNumber(row.reml_latent_upper),
        remlPredictiveLower: parseNumber(row.reml_predictive_lower),
        remlPredictiveUpper: parseNumber(row.reml_predictive_upper),
        bayesLatentLocation: parseNumber(row.bayes_latent_location),
        bayesLatentLower: parseNumber(row.bayes_latent_lower),
        bayesLatentUpper: parseNumber(row.bayes_latent_upper),
        bayesPredictiveLower: parseNumber(row.bayes_predictive_lower),
        bayesPredictiveUpper: parseNumber(row.bayes_predictive_upper),
        usageEstimatedTotalCostUsdTotal: parseNumber(
          row.usage_estimated_total_cost_usd_total,
        ),
        usageEstimatedTotalCostUsdPerSuccessfulRun: parseNumber(
          row.usage_estimated_total_cost_usd_per_successful_run,
        ),
        usageTotalTokensTotal: parseNumber(row.usage_total_tokens_total),
        usageTotalTokensPerSuccessfulRun: parseNumber(
          row.usage_total_tokens_per_successful_run,
        ),
        experimentDir,
        experimentUpdatedAt,
      });
    }
  }

  return rows;
}

function loadRunsForExperiment(experimentDir: string): StoredRunRecord[] {
  const cached = runCache.get(experimentDir);
  if (cached) {
    return cached;
  }

  const runsPath = path.join(experimentDir, "runs.jsonl");
  if (!fs.existsSync(runsPath)) {
    runCache.set(experimentDir, []);
    return [];
  }

  const parsed = fs
    .readFileSync(runsPath, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map<StoredRunRecord>((line) => JSON.parse(line));

  runCache.set(experimentDir, parsed);
  return parsed;
}

function parseNumber(value: string | undefined): number | null {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseInteger(value: string | undefined): number | null {
  if (!value) {
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function sumNullable(values: Array<number | null>): number | null {
  const observed = values.filter((value): value is number => value !== null);
  if (!observed.length) {
    return null;
  }
  return observed.reduce((sum, value) => sum + value, 0);
}

function compareModelNames(left: string, right: string): number {
  const leftIndex = MODEL_ORDER.indexOf(left);
  const rightIndex = MODEL_ORDER.indexOf(right);

  if (leftIndex >= 0 || rightIndex >= 0) {
    if (leftIndex === -1) {
      return 1;
    }
    if (rightIndex === -1) {
      return -1;
    }
    return leftIndex - rightIndex;
  }

  return left.localeCompare(right);
}

export function getMethodDefinition(
  methodId: IntervalMethodId,
): IntervalMethodDefinition {
  return (
    METHOD_DEFINITIONS.find((definition) => definition.id === methodId) ??
    METHOD_DEFINITIONS[0]
  );
}
