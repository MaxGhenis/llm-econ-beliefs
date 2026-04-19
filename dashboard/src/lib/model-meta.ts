export type ProviderKey = "anthropic" | "google" | "openai" | "xai";

export const MODEL_ORDER = [
  "gpt-5.4",
  "gpt-5.4-mini",
  "gpt-5.4-nano",
  "claude-opus-4.6",
  "claude-sonnet-4.6",
  "claude-haiku-4.5",
  "gemini-3.1-pro-preview",
  "gemini-3-flash-preview",
  "gemini-3.1-flash-lite-preview",
  "grok-4.20",
  "grok-4.1-fast",
] as const;

export const MODEL_LABELS: Record<string, string> = {
  "claude-opus-4.6": "Claude Opus 4.6",
  "claude-sonnet-4.6": "Claude Sonnet 4.6",
  "claude-haiku-4.5": "Claude Haiku 4.5",
  "grok-4.20": "Grok 4.20",
  "grok-4.1-fast": "Grok 4.1 Fast",
  "gpt-5.4": "GPT-5.4",
  "gpt-5.4-mini": "GPT-5.4 mini",
  "gpt-5.4-nano": "GPT-5.4 nano",
  "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
  "gemini-3-flash-preview": "Gemini 3 Flash",
  "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
};

export const PROVIDER_LABELS: Record<ProviderKey, string> = {
  anthropic: "Anthropic",
  google: "Google",
  openai: "OpenAI",
  xai: "xAI",
};

export function getProviderForModel(model: string): ProviderKey | null {
  if (model.startsWith("claude-")) return "anthropic";
  if (model.startsWith("gemini-")) return "google";
  if (model.startsWith("gpt-")) return "openai";
  if (model.startsWith("grok-")) return "xai";
  return null;
}

export function getModelLabel(model: string): string {
  return MODEL_LABELS[model] ?? model;
}

export function compareModelNames(left: string, right: string): number {
  const leftIndex = MODEL_ORDER.indexOf(left as (typeof MODEL_ORDER)[number]);
  const rightIndex = MODEL_ORDER.indexOf(right as (typeof MODEL_ORDER)[number]);

  if (leftIndex >= 0 || rightIndex >= 0) {
    if (leftIndex === -1) return 1;
    if (rightIndex === -1) return -1;
    return leftIndex - rightIndex;
  }

  return left.localeCompare(right);
}
