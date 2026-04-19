"use client";

import { Anthropic, Google, OpenAI, XAI } from "@lobehub/icons";

import { PROVIDER_LABELS, type ProviderKey } from "@/lib/model-meta";

export function ProviderMark({
  provider,
  size = 14,
  className = "",
}: {
  provider: ProviderKey | null;
  size?: number;
  className?: string;
}) {
  if (!provider) {
    return (
      <span
        aria-hidden="true"
        className={`block rounded-full bg-current ${className}`}
        style={{ width: size, height: size, color: "var(--text-tertiary)" }}
      />
    );
  }

  const label = PROVIDER_LABELS[provider];

  return (
    <span
      className={`inline-flex items-center justify-center ${className}`}
      aria-label={label}
      title={label}
    >
      {provider === "anthropic" && (
        <Anthropic
          size={size}
          style={{ color: "var(--text-primary)" }}
          aria-hidden="true"
        />
      )}
      {provider === "google" && <Google.Color size={size} aria-hidden="true" />}
      {provider === "openai" && (
        <OpenAI
          size={size}
          style={{ color: "var(--text-primary)" }}
          aria-hidden="true"
        />
      )}
      {provider === "xai" && (
        <XAI
          size={size}
          style={{ color: "var(--text-primary)" }}
          aria-hidden="true"
        />
      )}
    </span>
  );
}
