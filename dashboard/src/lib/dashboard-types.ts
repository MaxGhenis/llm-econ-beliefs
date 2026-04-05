export type IntervalMethodId =
  | "pooled"
  | "remlPredictive"
  | "remlLatent"
  | "bayesPredictive"
  | "bayesLatent";

export type IntervalKind = "mixture" | "predictive" | "latent";

export interface IntervalMethodDefinition {
  id: IntervalMethodId;
  label: string;
  shortLabel: string;
  kind: IntervalKind;
  description: string;
}

export interface IntervalSnapshot {
  center: number | null;
  lower: number | null;
  upper: number | null;
}

export interface SourceAnchor {
  citation: string;
  count: number;
  share: number;
}

export interface SourceSummary {
  uniqueCitations: number;
  top1Share: number | null;
  top3Share: number | null;
  topAnchors: SourceAnchor[];
}

export interface ModelSummary {
  modelName: string;
  experimentDir: string;
  experimentUpdatedAt: string;
  nSuccessfulRuns: number;
  totalCostUsd: number | null;
  costPerRunUsd: number | null;
  totalTokens: number | null;
  tokensPerRun: number | null;
  intervals: Record<IntervalMethodId, IntervalSnapshot>;
  sourceSummary: SourceSummary;
}

export interface QuantitySummary {
  quantityId: string;
  quantityName: string;
  domain: string;
  availableModels: string[];
  modelSummaries: ModelSummary[];
}

export interface DashboardStats {
  quantityCount: number;
  modelCount: number;
  selectedResultCount: number;
  totalCostUsd: number | null;
}

export interface DashboardSummaryData {
  generatedAt: string;
  methods: IntervalMethodDefinition[];
  modelNames: string[];
  quantities: QuantitySummary[];
  stats: DashboardStats;
}

export interface RunDetail {
  runIndex: number;
  promptVersion: string;
  parsedOk: boolean;
  pointEstimate: number | null;
  lowerBound: number | null;
  upperBound: number | null;
  confidenceLevel: number | null;
  quantiles: Record<string, number>;
  interpretation: string | null;
  citations: string[];
  reasoningSummary: string | null;
  rawResponse: string | null;
  prompt: string | null;
  error: string | null;
}

export interface ModelRunPayload {
  quantityId: string;
  modelName: string;
  experimentDir: string;
  experimentUpdatedAt: string;
  runs: RunDetail[];
}
