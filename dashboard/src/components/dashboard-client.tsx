"use client";

import {
  startTransition,
  useDeferredValue,
  useEffect,
  useState,
} from "react";

import { IntervalPlot } from "@/components/interval-plot";
import type {
  DashboardSummaryData,
  IntervalMethodDefinition,
  IntervalMethodId,
  ModelRunPayload,
  ModelSummary,
  RunDetail,
} from "@/lib/dashboard-types";

interface DashboardClientProps {
  data: DashboardSummaryData;
}

export function DashboardClient({ data }: DashboardClientProps) {
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search);
  const [selectedQuantityId, setSelectedQuantityId] = useState(
    data.quantities[0]?.quantityId ?? "",
  );
  const [selectedMethodId, setSelectedMethodId] =
    useState<IntervalMethodId>("pooled");
  const [selectedModelName, setSelectedModelName] = useState(
    data.quantities[0]?.availableModels[0] ?? "",
  );
  const [selectedRunIndex, setSelectedRunIndex] = useState<number | null>(null);
  const [runCache, setRunCache] = useState<Record<string, ModelRunPayload>>({});

  const filteredQuantities = data.quantities.filter((quantity) => {
    const haystack =
      `${quantity.quantityName} ${quantity.quantityId}`.toLowerCase();
    return haystack.includes(deferredSearch.trim().toLowerCase());
  });

  const selectedQuantity =
    filteredQuantities.find(
      (quantity) => quantity.quantityId === selectedQuantityId,
    ) ??
    filteredQuantities[0] ??
    data.quantities.find(
      (quantity) => quantity.quantityId === selectedQuantityId,
    ) ??
    data.quantities[0] ??
    null;

  const selectedMethod =
    data.methods.find((method) => method.id === selectedMethodId) ??
    data.methods[0];

  useEffect(() => {
    if (!selectedQuantity) return;
    if (!selectedQuantity.availableModels.includes(selectedModelName)) {
      startTransition(() => {
        setSelectedModelName(selectedQuantity.availableModels[0] ?? "");
        setSelectedRunIndex(null);
      });
    }
  }, [selectedModelName, selectedQuantity]);

  useEffect(() => {
    if (!selectedQuantity || !selectedModelName) return;

    const cacheKey = `${selectedQuantity.quantityId}::${selectedModelName}`;
    if (runCache[cacheKey]) return;

    let cancelled = false;

    void fetch(
      `/api/responses?quantityId=${encodeURIComponent(
        selectedQuantity.quantityId,
      )}&modelName=${encodeURIComponent(selectedModelName)}`,
      { cache: "no-store" },
    )
      .then(async (response) => {
        if (!response.ok) throw new Error(`Request failed with ${response.status}`);
        return (await response.json()) as ModelRunPayload;
      })
      .then((payload) => {
        if (cancelled) return;
        setRunCache((current) => ({ ...current, [cacheKey]: payload }));
      })
      .catch(() => {
        if (!cancelled) {
          setRunCache((current) => ({
            ...current,
            [cacheKey]: {
              quantityId: selectedQuantity.quantityId,
              modelName: selectedModelName,
              experimentDir: "unavailable",
              experimentUpdatedAt: new Date().toISOString(),
              runs: [],
            },
          }));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [runCache, selectedModelName, selectedQuantity, selectedRunIndex]);

  const activeCacheKey =
    selectedQuantity && selectedModelName
      ? `${selectedQuantity.quantityId}::${selectedModelName}`
      : "";
  const activePayload = activeCacheKey ? runCache[activeCacheKey] : undefined;
  const activeRuns = activePayload?.runs ?? [];
  const loadingActiveRuns =
    Boolean(activeCacheKey) && activePayload === undefined;
  const selectedRun =
    activeRuns.find((run) => run.runIndex === selectedRunIndex) ??
    activeRuns[0] ??
    null;

  if (!selectedQuantity) {
    return (
      <div className="relative z-10 mx-auto flex min-h-screen max-w-4xl items-center justify-center px-6 py-12">
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          No elasticity results are available yet.
        </p>
      </div>
    );
  }

  const selectedModelSummary =
    selectedQuantity.modelSummaries.find(
      (summary) => summary.modelName === selectedModelName,
    ) ?? selectedQuantity.modelSummaries[0];

  /* Group quantities by domain for the sidebar */
  const domainGroups = new Map<string, typeof filteredQuantities>();
  for (const q of filteredQuantities) {
    const group = domainGroups.get(q.domain) ?? [];
    group.push(q);
    domainGroups.set(q.domain, group);
  }

  return (
    <div className="relative z-10 min-h-screen">
      {/* Top bar */}
      <header
        className="reveal sticky top-0 z-30 border-b px-5 py-3 backdrop-blur-xl"
        style={{
          background: "rgba(8, 11, 17, 0.82)",
          borderColor: "var(--border)",
        }}
      >
        <div className="mx-auto flex max-w-[1800px] items-center justify-between">
          <div className="flex items-center gap-4">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-md"
              style={{ background: "var(--gold-dim)" }}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="none"
                aria-hidden="true"
              >
                <circle
                  cx="8"
                  cy="8"
                  r="6"
                  stroke="var(--gold)"
                  strokeWidth="1.5"
                  fill="none"
                />
                <circle
                  cx="8"
                  cy="8"
                  r="2.5"
                  fill="var(--gold)"
                />
                <line
                  x1="8"
                  y1="1"
                  x2="8"
                  y2="4"
                  stroke="var(--gold)"
                  strokeWidth="1"
                />
                <line
                  x1="8"
                  y1="12"
                  x2="8"
                  y2="15"
                  stroke="var(--gold)"
                  strokeWidth="1"
                />
                <line
                  x1="1"
                  y1="8"
                  x2="4"
                  y2="8"
                  stroke="var(--gold)"
                  strokeWidth="1"
                />
                <line
                  x1="12"
                  y1="8"
                  x2="15"
                  y2="8"
                  stroke="var(--gold)"
                  strokeWidth="1"
                />
              </svg>
            </div>
            <h1
              className="font-serif text-lg font-semibold tracking-tight"
              style={{ color: "var(--text-primary)" }}
            >
              Elasticity Beliefs Atlas
            </h1>
          </div>
          <div className="flex items-center gap-5">
            <Stat label="Quantities" value={`${data.stats.quantityCount}`} />
            <Stat label="Models" value={`${data.stats.modelCount}`} />
            <Stat
              label="Total spend"
              value={formatCurrency(data.stats.totalCostUsd)}
            />
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1800px] gap-0 xl:grid-cols-[280px_minmax(0,1fr)_380px]">
        {/* Left sidebar: quantities */}
        <aside
          className="reveal border-r xl:sticky xl:top-[53px] xl:h-[calc(100svh-53px)] xl:overflow-hidden"
          style={{
            borderColor: "var(--border)",
            animationDelay: "60ms",
          }}
        >
          <div className="p-4">
            <div className="relative">
              <svg
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2"
                width="14"
                height="14"
                viewBox="0 0 16 16"
                fill="none"
                aria-hidden="true"
              >
                <circle
                  cx="7"
                  cy="7"
                  r="5.5"
                  stroke="var(--text-tertiary)"
                  strokeWidth="1.5"
                />
                <line
                  x1="11"
                  y1="11"
                  x2="14.5"
                  y2="14.5"
                  stroke="var(--text-tertiary)"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search quantities..."
                className="w-full rounded-lg border py-2.5 pl-9 pr-3 text-sm outline-none transition focus:border-[color:var(--gold)]"
                style={{
                  background: "var(--bg-surface)",
                  borderColor: "var(--border)",
                  color: "var(--text-primary)",
                }}
              />
            </div>
          </div>

          <div className="flex flex-col gap-0.5 overflow-y-auto px-2 pb-4 xl:max-h-[calc(100svh-53px-72px)]">
            {Array.from(domainGroups.entries()).map(
              ([domain, quantities]) => (
                <div key={domain} className="mb-1">
                  <div
                    className="sticky top-0 z-10 px-2 pb-1 pt-3 font-mono text-[10px] font-semibold uppercase tracking-[0.2em]"
                    style={{
                      color: "var(--text-tertiary)",
                      background: "var(--bg-deep)",
                    }}
                  >
                    {domain}
                  </div>
                  {quantities.map((quantity) => {
                    const isSelected =
                      quantity.quantityId === selectedQuantity.quantityId;
                    return (
                      <button
                        key={quantity.quantityId}
                        type="button"
                        onClick={() =>
                          startTransition(() => {
                            setSelectedQuantityId(quantity.quantityId);
                            setSelectedModelName(
                              quantity.availableModels[0] ?? "",
                            );
                            setSelectedRunIndex(null);
                          })
                        }
                        className="group w-full rounded-lg px-3 py-2.5 text-left transition-all"
                        style={{
                          background: isSelected
                            ? "var(--gold-dim)"
                            : "transparent",
                          border: isSelected
                            ? "1px solid var(--border-active)"
                            : "1px solid transparent",
                        }}
                      >
                        <div
                          className="text-[13px] font-medium leading-snug transition-colors"
                          style={{
                            color: isSelected
                              ? "var(--gold)"
                              : "var(--text-primary)",
                          }}
                        >
                          {quantity.quantityName}
                        </div>
                        <div className="mt-1 flex items-center gap-2">
                          <span
                            className="font-mono text-[10px]"
                            style={{ color: "var(--text-tertiary)" }}
                          >
                            {quantity.availableModels.length} model
                            {quantity.availableModels.length !== 1 ? "s" : ""}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              ),
            )}
          </div>
        </aside>

        {/* Center: main content */}
        <section
          className="reveal min-w-0 border-r p-5"
          style={{
            borderColor: "var(--border)",
            animationDelay: "120ms",
          }}
        >
          {/* Quantity header */}
          <div className="mb-6">
            <div
              className="font-mono text-[10px] font-semibold uppercase tracking-[0.25em]"
              style={{ color: "var(--gold)" }}
            >
              {selectedQuantity.domain}
            </div>
            <h2
              className="mt-2 font-serif text-4xl font-semibold leading-tight tracking-tight lg:text-[2.6rem]"
              style={{ color: "var(--text-primary)" }}
            >
              {selectedQuantity.quantityName}
            </h2>
            <p
              className="mt-3 max-w-2xl text-sm leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              Compare manifested belief centers across models. Swap interval
              generators to see how pooled, REML, and Bayesian uncertainty
              bands shift.
            </p>
          </div>

          {/* Method selector */}
          <div
            className="mb-6 flex flex-wrap items-center gap-2 rounded-lg border p-1"
            style={{
              background: "var(--bg-surface)",
              borderColor: "var(--border)",
            }}
          >
            {data.methods.map((method) => {
              const isActive = method.id === selectedMethod.id;
              return (
                <button
                  key={method.id}
                  type="button"
                  onClick={() => setSelectedMethodId(method.id)}
                  className="rounded-md px-3 py-2 text-xs font-medium transition-all"
                  style={{
                    background: isActive ? "var(--bg-raised)" : "transparent",
                    color: isActive
                      ? "var(--text-primary)"
                      : "var(--text-secondary)",
                    boxShadow: isActive
                      ? "0 1px 3px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04)"
                      : "none",
                  }}
                >
                  {method.shortLabel}
                </button>
              );
            })}
          </div>

          {/* Method description */}
          <p
            className="mb-5 text-xs leading-relaxed"
            style={{ color: "var(--text-tertiary)" }}
          >
            {selectedMethod.description}
          </p>

          {/* Interval plot */}
          <IntervalPlot
            models={selectedQuantity.modelSummaries}
            method={selectedMethod}
          />

          {/* Model cards */}
          <div className="mt-6 grid gap-4 xl:grid-cols-2">
            {selectedQuantity.modelSummaries.map((summary) => (
              <ModelPanel
                key={`${selectedQuantity.quantityId}-${summary.modelName}`}
                model={summary}
                methods={data.methods}
                selectedMethodId={selectedMethod.id}
                onInspect={() =>
                  startTransition(() => {
                    setSelectedModelName(summary.modelName);
                    setSelectedRunIndex(null);
                  })
                }
              />
            ))}
          </div>
        </section>

        {/* Right sidebar: response inspector */}
        <aside
          className="reveal xl:sticky xl:top-[53px] xl:h-[calc(100svh-53px)] xl:overflow-hidden"
          style={{ animationDelay: "180ms" }}
        >
          <div
            className="border-b p-4"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="flex items-center justify-between">
              <div>
                <div
                  className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em]"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Response inspector
                </div>
                <h3
                  className="mt-1 font-serif text-xl font-semibold"
                  style={{ color: "var(--text-primary)" }}
                >
                  {selectedModelSummary?.modelName ?? "Select a model"}
                </h3>
              </div>
              <div
                className="rounded-md px-2.5 py-1 font-mono text-[11px]"
                style={{
                  background: "var(--blue-dim)",
                  color: "var(--blue)",
                }}
              >
                {selectedModelSummary?.nSuccessfulRuns ?? 0} runs
              </div>
            </div>

            {/* Model tabs */}
            <div className="mt-3 flex flex-wrap gap-1.5">
              {selectedQuantity.availableModels.map((modelName) => {
                const isActive = modelName === selectedModelName;
                return (
                  <button
                    key={modelName}
                    type="button"
                    onClick={() =>
                      startTransition(() => {
                        setSelectedModelName(modelName);
                        setSelectedRunIndex(null);
                      })
                    }
                    className="rounded-md px-2.5 py-1.5 font-mono text-[11px] font-medium transition-all"
                    style={{
                      background: isActive
                        ? "var(--gold-dim)"
                        : "var(--bg-surface)",
                      color: isActive
                        ? "var(--gold)"
                        : "var(--text-secondary)",
                      border: isActive
                        ? "1px solid var(--border-active)"
                        : "1px solid var(--border)",
                    }}
                  >
                    {modelName}
                  </button>
                );
              })}
            </div>
          </div>

          <div
            className="grid gap-0 xl:grid-cols-[160px_minmax(0,1fr)]"
            style={{ height: "calc(100% - 130px)" }}
          >
            {/* Run list */}
            <div
              className="border-r p-3 xl:overflow-y-auto"
              style={{ borderColor: "var(--border)" }}
            >
              <div
                className="mb-2 font-mono text-[10px] font-semibold uppercase tracking-[0.2em]"
                style={{ color: "var(--text-tertiary)" }}
              >
                Runs
              </div>
              <div className="flex flex-col gap-1.5">
                {loadingActiveRuns ? (
                  <div
                    className="shimmer rounded-md px-3 py-4 text-xs"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Loading...
                  </div>
                ) : activeRuns.length ? (
                  activeRuns.map((run) => {
                    const isActive = run.runIndex === selectedRun?.runIndex;
                    return (
                      <button
                        key={run.runIndex}
                        type="button"
                        onClick={() => setSelectedRunIndex(run.runIndex)}
                        className="rounded-md px-3 py-2.5 text-left transition-all"
                        style={{
                          background: isActive
                            ? "var(--bg-raised)"
                            : "transparent",
                          border: isActive
                            ? "1px solid var(--border-hover)"
                            : "1px solid transparent",
                        }}
                      >
                        <div
                          className="font-mono text-[10px] uppercase tracking-[0.15em]"
                          style={{
                            color: isActive
                              ? "var(--gold)"
                              : "var(--text-tertiary)",
                          }}
                        >
                          Run {run.runIndex}
                        </div>
                        <div
                          className="mt-1 font-mono text-sm font-semibold"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {run.pointEstimate !== null
                            ? formatNumber(run.pointEstimate)
                            : "—"}
                        </div>
                        <div
                          className="mt-0.5 font-mono text-[10px]"
                          style={{ color: "var(--text-tertiary)" }}
                        >
                          [{formatMaybeNumber(run.quantiles.p05)},{" "}
                          {formatMaybeNumber(run.quantiles.p95)}]
                        </div>
                      </button>
                    );
                  })
                ) : (
                  <div
                    className="rounded-md px-3 py-4 text-xs"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    No runs available.
                  </div>
                )}
              </div>
            </div>

            {/* Response detail */}
            <ResponseDetail
              model={selectedModelSummary}
              run={selectedRun}
              loading={loadingActiveRuns}
            />
          </div>
        </aside>
      </main>
    </div>
  );
}

/* ---------- Sub-components ---------- */

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span
        className="font-mono text-[10px] uppercase tracking-[0.15em]"
        style={{ color: "var(--text-tertiary)" }}
      >
        {label}
      </span>
      <span
        className="font-mono text-sm font-semibold"
        style={{ color: "var(--text-primary)" }}
      >
        {value}
      </span>
    </div>
  );
}

function ModelPanel({
  model,
  methods,
  selectedMethodId,
  onInspect,
}: {
  model: ModelSummary;
  methods: IntervalMethodDefinition[];
  selectedMethodId: IntervalMethodId;
  onInspect: () => void;
}) {
  return (
    <section
      className="reveal-scale rounded-lg border p-4"
      style={{
        background: "var(--bg-surface)",
        borderColor: "var(--border)",
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3
            className="font-serif text-lg font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            {model.modelName}
          </h3>
          <p
            className="mt-1 font-mono text-[10px]"
            style={{ color: "var(--text-tertiary)" }}
          >
            {model.experimentDir}
          </p>
        </div>
        <button
          type="button"
          onClick={onInspect}
          className="rounded-md border px-3 py-1.5 font-mono text-[11px] font-medium transition-all hover:border-[color:var(--gold)]"
          style={{
            borderColor: "var(--border-hover)",
            color: "var(--text-secondary)",
            background: "transparent",
          }}
        >
          Inspect
        </button>
      </div>

      {/* Metrics row */}
      <div className="mt-4 grid grid-cols-3 gap-2">
        <MetricTile label="Runs" value={`${model.nSuccessfulRuns}`} />
        <MetricTile
          label="$/run"
          value={formatCurrency(model.costPerRunUsd)}
        />
        <MetricTile
          label="Citations"
          value={`${model.sourceSummary.uniqueCitations}`}
        />
      </div>

      {/* Intervals table */}
      <div
        className="mt-4 overflow-hidden rounded-md border"
        style={{ borderColor: "var(--border)" }}
      >
        <table className="min-w-full text-xs">
          <thead>
            <tr style={{ background: "var(--bg-raised)" }}>
              <th
                className="px-3 py-2 text-left font-mono text-[10px] uppercase tracking-[0.15em]"
                style={{ color: "var(--text-tertiary)" }}
              >
                Method
              </th>
              <th
                className="px-3 py-2 text-right font-mono text-[10px] uppercase tracking-[0.15em]"
                style={{ color: "var(--text-tertiary)" }}
              >
                Center
              </th>
              <th
                className="px-3 py-2 text-right font-mono text-[10px] uppercase tracking-[0.15em]"
                style={{ color: "var(--text-tertiary)" }}
              >
                Interval
              </th>
            </tr>
          </thead>
          <tbody>
            {methods.map((method) => {
              const interval = model.intervals[method.id];
              const isSelected = method.id === selectedMethodId;
              return (
                <tr
                  key={`${model.modelName}-${method.id}`}
                  style={{
                    background: isSelected ? "var(--gold-dim)" : "transparent",
                  }}
                >
                  <td
                    className="border-t px-3 py-2 font-medium"
                    style={{
                      borderColor: "var(--border)",
                      color: isSelected
                        ? "var(--gold)"
                        : "var(--text-primary)",
                    }}
                  >
                    {method.shortLabel}
                  </td>
                  <td
                    className="border-t px-3 py-2 text-right font-mono"
                    style={{
                      borderColor: "var(--border)",
                      color: "var(--text-primary)",
                    }}
                  >
                    {formatMaybeNumber(interval.center)}
                  </td>
                  <td
                    className="border-t px-3 py-2 text-right font-mono"
                    style={{
                      borderColor: "var(--border)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    {formatInterval(interval.lower, interval.upper)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Top anchors */}
      <div className="mt-4">
        <div
          className="font-mono text-[10px] uppercase tracking-[0.15em]"
          style={{ color: "var(--text-tertiary)" }}
        >
          Top anchors
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {model.sourceSummary.topAnchors.slice(0, 5).map((anchor) => (
            <span
              key={`${model.modelName}-${anchor.citation}`}
              className="rounded-md border px-2 py-1 font-mono text-[10px]"
              style={{
                borderColor: "var(--border)",
                background: "var(--bg-raised)",
                color: "var(--text-secondary)",
              }}
            >
              {anchor.citation}
              <span
                className="ml-1.5"
                style={{ color: "var(--text-tertiary)" }}
              >
                {anchor.count}
              </span>
            </span>
          ))}
        </div>
        <p
          className="mt-2 text-[11px]"
          style={{ color: "var(--text-tertiary)" }}
        >
          Top 3 citations:{" "}
          {formatPercent(model.sourceSummary.top3Share)} of total
        </p>
      </div>
    </section>
  );
}

function ResponseDetail({
  model,
  run,
  loading,
}: {
  model: ModelSummary | undefined;
  run: RunDetail | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="p-4">
        <div
          className="shimmer h-6 w-40 rounded-md"
          style={{ background: "var(--bg-raised)" }}
        />
        <div
          className="shimmer mt-3 h-4 w-64 rounded-md"
          style={{ background: "var(--bg-raised)" }}
        />
      </div>
    );
  }

  if (!model || !run) {
    return (
      <div className="flex items-center justify-center p-6">
        <p
          className="text-center text-xs"
          style={{ color: "var(--text-tertiary)" }}
        >
          Select a model and run to inspect.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-y-auto p-4 xl:max-h-full">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div
            className="font-mono text-[10px] uppercase tracking-[0.15em]"
            style={{ color: "var(--text-tertiary)" }}
          >
            Run {run.runIndex}
          </div>
          <h4
            className="mt-1 font-serif text-base font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            {model.modelName}
          </h4>
        </div>
        <span
          className="rounded-md px-2 py-1 font-mono text-[10px]"
          style={{
            background: "var(--bg-raised)",
            color: "var(--text-secondary)",
          }}
        >
          {formatInterval(run.lowerBound, run.upperBound)}
        </span>
      </div>

      {/* Quick stats */}
      <div className="mt-4 grid grid-cols-3 gap-2">
        <MetricTile label="Point" value={formatMaybeNumber(run.pointEstimate)} />
        <MetricTile label="Prompt" value={run.promptVersion || "?"} />
        <MetricTile label="p50" value={formatMaybeNumber(run.quantiles.p50)} />
      </div>

      {/* Quantile strip */}
      <div className="mt-4">
        <div
          className="font-mono text-[10px] uppercase tracking-[0.15em]"
          style={{ color: "var(--text-tertiary)" }}
        >
          Quantiles
        </div>
        <div className="mt-2 grid grid-cols-5 gap-1.5">
          {["p05", "p25", "p50", "p75", "p95"].map((key) => (
            <div
              key={key}
              className="rounded-md border px-2 py-2 text-center"
              style={{
                borderColor: "var(--border)",
                background: "var(--bg-raised)",
              }}
            >
              <div
                className="font-mono text-[9px] uppercase"
                style={{ color: "var(--text-tertiary)" }}
              >
                {key}
              </div>
              <div
                className="mt-1 font-mono text-[12px] font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                {formatMaybeNumber(run.quantiles[key])}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interpretation */}
      <TextSection title="Interpretation" text={run.interpretation} />
      <TextSection title="Reasoning" text={run.reasoningSummary} />

      {/* Citations */}
      <div className="mt-4">
        <div
          className="font-mono text-[10px] uppercase tracking-[0.15em]"
          style={{ color: "var(--text-tertiary)" }}
        >
          Literature anchors
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {run.citations.length ? (
            run.citations.map((citation) => (
              <span
                key={`${run.runIndex}-${citation}`}
                className="rounded-md border px-2 py-1 font-mono text-[10px]"
                style={{
                  borderColor: "var(--border)",
                  background: "var(--bg-raised)",
                  color: "var(--text-secondary)",
                }}
              >
                {citation}
              </span>
            ))
          ) : (
            <span
              className="text-[11px]"
              style={{ color: "var(--text-tertiary)" }}
            >
              No citations captured.
            </span>
          )}
        </div>
      </div>

      {/* Collapsible raw sections */}
      <CollapsibleSection title="Raw response">
        {run.rawResponse || "No raw response captured."}
      </CollapsibleSection>
      <CollapsibleSection title="Prompt">
        {run.prompt || "No prompt captured."}
      </CollapsibleSection>
    </div>
  );
}

function TextSection({
  title,
  text,
}: {
  title: string;
  text: string | null;
}) {
  if (!text) return null;
  return (
    <div className="mt-4">
      <div
        className="font-mono text-[10px] uppercase tracking-[0.15em]"
        style={{ color: "var(--text-tertiary)" }}
      >
        {title}
      </div>
      <p
        className="mt-1.5 text-[13px] leading-relaxed"
        style={{ color: "var(--text-secondary)" }}
      >
        {text}
      </p>
    </div>
  );
}

function CollapsibleSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <details
      className="mt-3 rounded-md border"
      style={{
        borderColor: "var(--border)",
        background: "var(--bg-surface)",
      }}
    >
      <summary
        className="cursor-pointer px-3 py-2.5 font-mono text-[10px] uppercase tracking-[0.15em]"
        style={{ color: "var(--text-tertiary)" }}
      >
        {title}
      </summary>
      <pre
        className="overflow-x-auto whitespace-pre-wrap border-t px-3 py-3 font-mono text-[11px] leading-relaxed"
        style={{
          borderColor: "var(--border)",
          color: "var(--text-secondary)",
        }}
      >
        {children}
      </pre>
    </details>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="rounded-md border px-3 py-2"
      style={{
        borderColor: "var(--border)",
        background: "var(--bg-raised)",
      }}
    >
      <div
        className="font-mono text-[9px] uppercase tracking-[0.15em]"
        style={{ color: "var(--text-tertiary)" }}
      >
        {label}
      </div>
      <div
        className="mt-1 font-mono text-sm font-semibold"
        style={{ color: "var(--text-primary)" }}
      >
        {value}
      </div>
    </div>
  );
}

/* ---------- Formatters ---------- */

function formatCurrency(value: number | null): string {
  if (value === null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 4,
  }).format(value);
}

function formatPercent(value: number | null): string {
  if (value === null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatInterval(
  lower: number | null | undefined,
  upper: number | null | undefined,
): string {
  if (
    lower === null ||
    lower === undefined ||
    upper === null ||
    upper === undefined
  )
    return "—";
  return `[${formatNumber(lower)}, ${formatNumber(upper)}]`;
}

function formatMaybeNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return formatNumber(value);
}

function formatNumber(value: number): string {
  const abs = Math.abs(value);
  const fractionDigits = abs >= 10 ? 1 : abs >= 1 ? 2 : 3;
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value);
}
