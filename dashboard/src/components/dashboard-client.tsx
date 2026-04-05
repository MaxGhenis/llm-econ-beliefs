"use client";

import {
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
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

  /* Inspector drawer state */
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectedModelName, setInspectedModelName] = useState(
    data.quantities[0]?.availableModels[0] ?? "",
  );
  const [selectedRunIndex, setSelectedRunIndex] = useState<number | null>(null);
  const [runCache, setRunCache] = useState<Record<string, ModelRunPayload>>({});
  const drawerRef = useRef<HTMLDivElement>(null);

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

  /* Close drawer on outside click */
  useEffect(() => {
    if (!inspectorOpen) return;
    function handleClick(e: MouseEvent) {
      if (drawerRef.current && !drawerRef.current.contains(e.target as Node)) {
        setInspectorOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [inspectorOpen]);

  /* Close drawer on Escape */
  useEffect(() => {
    if (!inspectorOpen) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setInspectorOpen(false);
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [inspectorOpen]);

  /* Reset model when quantity changes */
  useEffect(() => {
    if (!selectedQuantity) return;
    if (
      inspectedModelName &&
      !selectedQuantity.availableModels.includes(inspectedModelName)
    ) {
      startTransition(() => {
        setInspectedModelName(selectedQuantity.availableModels[0] ?? "");
        setSelectedRunIndex(null);
      });
    }
  }, [inspectedModelName, selectedQuantity]);

  /* Lazy-load run data */
  useEffect(() => {
    if (!selectedQuantity || !inspectedModelName) return;

    const cacheKey = `${selectedQuantity.quantityId}::${inspectedModelName}`;
    if (runCache[cacheKey]) return;

    let cancelled = false;

    void fetch(
      `/api/responses?quantityId=${encodeURIComponent(
        selectedQuantity.quantityId,
      )}&modelName=${encodeURIComponent(inspectedModelName)}`,
      { cache: "no-store" },
    )
      .then(async (response) => {
        if (!response.ok)
          throw new Error(`Request failed with ${response.status}`);
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
              modelName: inspectedModelName,
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
  }, [runCache, inspectedModelName, selectedQuantity, inspectorOpen]);

  const activeCacheKey =
    selectedQuantity && inspectedModelName
      ? `${selectedQuantity.quantityId}::${inspectedModelName}`
      : "";
  const activePayload = activeCacheKey ? runCache[activeCacheKey] : undefined;
  const activeRuns = activePayload?.runs ?? [];
  const loadingActiveRuns =
    Boolean(activeCacheKey) && activePayload === undefined;
  const selectedRun =
    activeRuns.find((run) => run.runIndex === selectedRunIndex) ??
    activeRuns[0] ??
    null;

  const inspectedModelSummary = selectedQuantity?.modelSummaries.find(
    (s) => s.modelName === inspectedModelName,
  );

  if (!selectedQuantity) {
    return (
      <div className="relative z-10 mx-auto flex min-h-screen max-w-4xl items-center justify-center px-6 py-12">
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          No elasticity results are available yet.
        </p>
      </div>
    );
  }

  /* Group quantities by domain */
  const domainGroups = new Map<string, typeof filteredQuantities>();
  for (const q of filteredQuantities) {
    const group = domainGroups.get(q.domain) ?? [];
    group.push(q);
    domainGroups.set(q.domain, group);
  }

  function openInspector(modelName: string) {
    startTransition(() => {
      setInspectedModelName(modelName);
      setSelectedRunIndex(null);
      setInspectorOpen(true);
    });
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
        <div className="mx-auto flex max-w-[1400px] items-center justify-between">
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
                <circle cx="8" cy="8" r="2.5" fill="var(--gold)" />
                <line x1="8" y1="1" x2="8" y2="4" stroke="var(--gold)" strokeWidth="1" />
                <line x1="8" y1="12" x2="8" y2="15" stroke="var(--gold)" strokeWidth="1" />
                <line x1="1" y1="8" x2="4" y2="8" stroke="var(--gold)" strokeWidth="1" />
                <line x1="12" y1="8" x2="15" y2="8" stroke="var(--gold)" strokeWidth="1" />
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
          </div>
        </div>
      </header>

      {/* Two-column layout: sidebar + main */}
      <div className="mx-auto grid max-w-[1400px] xl:grid-cols-[280px_minmax(0,1fr)]">
        {/* Left sidebar: quantities */}
        <aside
          className="reveal border-r xl:sticky xl:top-[53px] xl:h-[calc(100svh-53px)] xl:overflow-hidden"
          style={{ borderColor: "var(--border)", animationDelay: "60ms" }}
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
                <circle cx="7" cy="7" r="5.5" stroke="var(--text-tertiary)" strokeWidth="1.5" />
                <line x1="11" y1="11" x2="14.5" y2="14.5" stroke="var(--text-tertiary)" strokeWidth="1.5" strokeLinecap="round" />
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
            {Array.from(domainGroups.entries()).map(([domain, quantities]) => (
              <div key={domain} className="mb-1">
                <div
                  className="sticky top-0 z-10 px-2 pb-1 pt-3 font-mono text-[10px] font-semibold uppercase tracking-[0.2em]"
                  style={{ color: "var(--text-tertiary)", background: "var(--bg-deep)" }}
                >
                  {domain}
                </div>
                {quantities.map((quantity) => {
                  const isSelected = quantity.quantityId === selectedQuantity.quantityId;
                  return (
                    <button
                      key={quantity.quantityId}
                      type="button"
                      onClick={() =>
                        startTransition(() => {
                          setSelectedQuantityId(quantity.quantityId);
                          setInspectedModelName(quantity.availableModels[0] ?? "");
                          setSelectedRunIndex(null);
                          setInspectorOpen(false);
                        })
                      }
                      className="group w-full rounded-lg px-3 py-2.5 text-left transition-all"
                      style={{
                        background: isSelected ? "var(--gold-dim)" : "transparent",
                        border: isSelected ? "1px solid var(--border-active)" : "1px solid transparent",
                      }}
                    >
                      <div
                        className="text-[13px] font-medium leading-snug"
                        style={{ color: isSelected ? "var(--gold)" : "var(--text-primary)" }}
                      >
                        {quantity.quantityName}
                      </div>
                      <span className="mt-1 font-mono text-[10px]" style={{ color: "var(--text-tertiary)" }}>
                        {quantity.availableModels.length} model{quantity.availableModels.length !== 1 ? "s" : ""}
                      </span>
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </aside>

        {/* Main content */}
        <section className="reveal min-w-0 p-6" style={{ animationDelay: "120ms" }}>
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
              generators to see how pooled, REML, and Bayesian uncertainty bands
              shift.
            </p>
          </div>

          {/* Method tabs */}
          <div
            className="mb-6 flex flex-wrap items-center gap-2 rounded-lg border p-1"
            style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
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
                    color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
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
          <p className="mb-5 text-xs leading-relaxed" style={{ color: "var(--text-tertiary)" }}>
            {selectedMethod.description}
          </p>

          {/* Interval plot */}
          <IntervalPlot
            models={selectedQuantity.modelSummaries}
            method={selectedMethod}
          />

          {/* Model cards grid — scales with model count */}
          <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {selectedQuantity.modelSummaries.map((summary) => (
              <ModelPanel
                key={`${selectedQuantity.quantityId}-${summary.modelName}`}
                model={summary}
                methods={data.methods}
                selectedMethodId={selectedMethod.id}
                onInspect={() => openInspector(summary.modelName)}
              />
            ))}
          </div>

          {/* Prompt section */}
          <PromptSection runs={activeRuns} selectedRun={selectedRun} />
        </section>
      </div>

      {/* Inspector drawer overlay */}
      {inspectorOpen && (
        <div className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm" style={{ animation: "reveal 200ms ease both" }}>
          <div
            ref={drawerRef}
            className="absolute right-0 top-0 h-full w-full max-w-[480px] overflow-y-auto border-l"
            style={{
              background: "var(--bg-deep)",
              borderColor: "var(--border)",
              animation: "slide-in 300ms cubic-bezier(0.16,1,0.3,1) both",
            }}
          >
            {/* Drawer header */}
            <div className="sticky top-0 z-10 border-b p-4 backdrop-blur-xl" style={{ background: "rgba(8,11,17,0.92)", borderColor: "var(--border)" }}>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em]" style={{ color: "var(--text-tertiary)" }}>
                    Response inspector
                  </div>
                  <h3 className="mt-1 font-serif text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
                    {inspectedModelSummary?.modelName ?? "—"}
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={() => setInspectorOpen(false)}
                  className="flex h-8 w-8 items-center justify-center rounded-md border transition hover:border-[color:var(--border-hover)]"
                  style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
                  aria-label="Close inspector"
                >
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <line x1="2" y1="2" x2="12" y2="12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                    <line x1="12" y1="2" x2="2" y2="12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                </button>
              </div>

              {/* Model tabs in drawer */}
              <div className="mt-3 flex flex-wrap gap-1.5">
                {selectedQuantity.availableModels.map((modelName) => {
                  const isActive = modelName === inspectedModelName;
                  return (
                    <button
                      key={modelName}
                      type="button"
                      onClick={() =>
                        startTransition(() => {
                          setInspectedModelName(modelName);
                          setSelectedRunIndex(null);
                        })
                      }
                      className="rounded-md px-2.5 py-1.5 font-mono text-[11px] font-medium transition-all"
                      style={{
                        background: isActive ? "var(--gold-dim)" : "var(--bg-surface)",
                        color: isActive ? "var(--gold)" : "var(--text-secondary)",
                        border: isActive ? "1px solid var(--border-active)" : "1px solid var(--border)",
                      }}
                    >
                      {modelName}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Run selector + detail */}
            <div className="p-4">
              {/* Run pills */}
              <div className="mb-4">
                <div className="mb-2 font-mono text-[10px] font-semibold uppercase tracking-[0.2em]" style={{ color: "var(--text-tertiary)" }}>
                  Runs ({activeRuns.length})
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {loadingActiveRuns ? (
                    <div className="shimmer rounded-md px-4 py-2 text-xs" style={{ color: "var(--text-tertiary)" }}>
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
                          className="rounded-md px-3 py-2 text-left transition-all"
                          style={{
                            background: isActive ? "var(--bg-raised)" : "var(--bg-surface)",
                            border: isActive ? "1px solid var(--border-hover)" : "1px solid var(--border)",
                          }}
                        >
                          <div className="font-mono text-[10px]" style={{ color: isActive ? "var(--gold)" : "var(--text-tertiary)" }}>
                            #{run.runIndex}
                          </div>
                          <div className="mt-0.5 font-mono text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
                            {run.pointEstimate !== null ? formatNumber(run.pointEstimate) : "—"}
                          </div>
                        </button>
                      );
                    })
                  ) : (
                    <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>No runs.</span>
                  )}
                </div>
              </div>

              {/* Response detail */}
              <ResponseDetail
                model={inspectedModelSummary}
                run={selectedRun}
                loading={loadingActiveRuns}
              />
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes slide-in {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>
    </div>
  );
}

/* ---------- Sub-components ---------- */

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
        {label}
      </span>
      <span className="font-mono text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
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
      style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-serif text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
            {model.modelName}
          </h3>
          <p className="mt-1 font-mono text-[10px]" style={{ color: "var(--text-tertiary)" }}>
            {model.experimentDir}
          </p>
        </div>
        <button
          type="button"
          onClick={onInspect}
          className="rounded-md border px-3 py-1.5 font-mono text-[11px] font-medium transition-all hover:border-[color:var(--gold)] hover:text-[color:var(--gold)]"
          style={{ borderColor: "var(--border-hover)", color: "var(--text-secondary)", background: "transparent" }}
        >
          Inspect
        </button>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <MetricTile label="Runs" value={`${model.nSuccessfulRuns}`} />
        <MetricTile label="Citations" value={`${model.sourceSummary.uniqueCitations}`} />
      </div>

      <div className="mt-3 overflow-hidden rounded-md border" style={{ borderColor: "var(--border)" }}>
        <table className="min-w-full text-xs">
          <thead>
            <tr style={{ background: "var(--bg-raised)" }}>
              <th className="px-3 py-2 text-left font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
                Method
              </th>
              <th className="px-3 py-2 text-right font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
                Center
              </th>
              <th className="px-3 py-2 text-right font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
                90% CI
              </th>
            </tr>
          </thead>
          <tbody>
            {methods.map((method) => {
              const interval = model.intervals[method.id];
              const isSelected = method.id === selectedMethodId;
              return (
                <tr key={`${model.modelName}-${method.id}`} style={{ background: isSelected ? "var(--gold-dim)" : "transparent" }}>
                  <td className="border-t px-3 py-1.5 font-medium" style={{ borderColor: "var(--border)", color: isSelected ? "var(--gold)" : "var(--text-primary)" }}>
                    {method.shortLabel}
                  </td>
                  <td className="border-t px-3 py-1.5 text-right font-mono" style={{ borderColor: "var(--border)", color: "var(--text-primary)" }}>
                    {formatMaybeNumber(interval.center)}
                  </td>
                  <td className="border-t px-3 py-1.5 text-right font-mono" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>
                    {formatInterval(interval.lower, interval.upper)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {model.sourceSummary.topAnchors.length > 0 && (
        <div className="mt-3">
          <div className="flex flex-wrap gap-1">
            {model.sourceSummary.topAnchors.slice(0, 3).map((anchor) => (
              <span
                key={`${model.modelName}-${anchor.citation}`}
                className="rounded border px-1.5 py-0.5 font-mono text-[9px]"
                style={{ borderColor: "var(--border)", background: "var(--bg-raised)", color: "var(--text-tertiary)" }}
              >
                {anchor.citation}
              </span>
            ))}
          </div>
        </div>
      )}

      {(model.costPerRunUsd !== null || model.tokensPerRun !== null) && (
        <details className="mt-3 rounded-md border" style={{ borderColor: "var(--border)", background: "var(--bg-raised)" }}>
          <summary className="cursor-pointer px-3 py-2 font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
            Cost &amp; usage
          </summary>
          <div className="grid grid-cols-2 gap-2 px-3 pb-3 pt-1">
            {model.costPerRunUsd !== null && (
              <div>
                <div className="font-mono text-[9px] uppercase" style={{ color: "var(--text-tertiary)" }}>$/run</div>
                <div className="mt-0.5 font-mono text-xs" style={{ color: "var(--text-secondary)" }}>{formatCurrency(model.costPerRunUsd)}</div>
              </div>
            )}
            {model.tokensPerRun !== null && (
              <div>
                <div className="font-mono text-[9px] uppercase" style={{ color: "var(--text-tertiary)" }}>Tokens/run</div>
                <div className="mt-0.5 font-mono text-xs" style={{ color: "var(--text-secondary)" }}>{model.tokensPerRun.toLocaleString()}</div>
              </div>
            )}
          </div>
        </details>
      )}
    </section>
  );
}

function PromptSection({
  runs,
  selectedRun,
}: {
  runs: RunDetail[];
  selectedRun: RunDetail | null;
}) {
  const prompt = selectedRun?.prompt ?? runs[0]?.prompt;
  if (!prompt) return null;

  return (
    <div className="mt-8 rounded-lg border p-5" style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}>
      <div className="mb-3 flex items-center gap-2">
        <div className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em]" style={{ color: "var(--gold)" }}>
          Elicitation prompt
        </div>
        <span className="font-mono text-[10px]" style={{ color: "var(--text-tertiary)" }}>
          (v{selectedRun?.promptVersion ?? runs[0]?.promptVersion ?? "?"})
        </span>
      </div>
      <pre
        className="max-h-[400px] overflow-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed"
        style={{ color: "var(--text-secondary)" }}
      >
        {prompt}
      </pre>
    </div>
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
      <div>
        <div className="shimmer h-6 w-40 rounded-md" style={{ background: "var(--bg-raised)" }} />
        <div className="shimmer mt-3 h-4 w-64 rounded-md" style={{ background: "var(--bg-raised)" }} />
      </div>
    );
  }

  if (!model || !run) {
    return (
      <p className="text-center text-xs" style={{ color: "var(--text-tertiary)" }}>
        Select a run to inspect.
      </p>
    );
  }

  return (
    <div>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
            Run {run.runIndex}
          </div>
          <h4 className="mt-1 font-serif text-base font-semibold" style={{ color: "var(--text-primary)" }}>
            {model.modelName}
          </h4>
        </div>
        <span className="rounded-md px-2 py-1 font-mono text-[10px]" style={{ background: "var(--bg-raised)", color: "var(--text-secondary)" }}>
          {formatInterval(run.lowerBound, run.upperBound)}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <MetricTile label="Point" value={formatMaybeNumber(run.pointEstimate)} />
        <MetricTile label="Prompt" value={run.promptVersion || "?"} />
        <MetricTile label="p50" value={formatMaybeNumber(run.quantiles.p50)} />
      </div>

      {/* Quantiles */}
      <div className="mt-4">
        <div className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
          Quantiles
        </div>
        <div className="mt-2 grid grid-cols-5 gap-1.5">
          {["p05", "p25", "p50", "p75", "p95"].map((key) => (
            <div key={key} className="rounded-md border px-2 py-2 text-center" style={{ borderColor: "var(--border)", background: "var(--bg-raised)" }}>
              <div className="font-mono text-[9px] uppercase" style={{ color: "var(--text-tertiary)" }}>{key}</div>
              <div className="mt-1 font-mono text-[12px] font-semibold" style={{ color: "var(--text-primary)" }}>
                {formatMaybeNumber(run.quantiles[key])}
              </div>
            </div>
          ))}
        </div>
      </div>

      <TextSection title="Interpretation" text={run.interpretation} />
      <TextSection title="Reasoning" text={run.reasoningSummary} />

      {/* Citations */}
      {run.citations.length > 0 && (
        <div className="mt-4">
          <div className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
            Literature anchors
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {run.citations.map((citation) => (
              <span
                key={`${run.runIndex}-${citation}`}
                className="rounded-md border px-2 py-1 font-mono text-[10px]"
                style={{ borderColor: "var(--border)", background: "var(--bg-raised)", color: "var(--text-secondary)" }}
              >
                {citation}
              </span>
            ))}
          </div>
        </div>
      )}

      <CollapsibleSection title="Raw response">
        {run.rawResponse || "No raw response captured."}
      </CollapsibleSection>
    </div>
  );
}

function TextSection({ title, text }: { title: string; text: string | null }) {
  if (!text) return null;
  return (
    <div className="mt-4">
      <div className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
        {title}
      </div>
      <p className="mt-1.5 text-[13px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>
        {text}
      </p>
    </div>
  );
}

function CollapsibleSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="mt-3 rounded-md border" style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}>
      <summary className="cursor-pointer px-3 py-2.5 font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>
        {title}
      </summary>
      <pre className="overflow-x-auto whitespace-pre-wrap border-t px-3 py-3 font-mono text-[11px] leading-relaxed" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>
        {children}
      </pre>
    </details>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border px-3 py-2" style={{ borderColor: "var(--border)", background: "var(--bg-raised)" }}>
      <div className="font-mono text-[9px] uppercase tracking-[0.15em]" style={{ color: "var(--text-tertiary)" }}>{label}</div>
      <div className="mt-1 font-mono text-sm font-semibold" style={{ color: "var(--text-primary)" }}>{value}</div>
    </div>
  );
}

/* ---------- Formatters ---------- */

function formatCurrency(value: number | null): string {
  if (value === null) return "—";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 4 }).format(value);
}

function formatInterval(lower: number | null | undefined, upper: number | null | undefined): string {
  if (lower === null || lower === undefined || upper === null || upper === undefined) return "—";
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
