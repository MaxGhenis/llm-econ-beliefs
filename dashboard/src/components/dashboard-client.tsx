"use client";

import {
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
  useState,
} from "react";

import { IntervalPlot } from "@/components/interval-plot";
import { ProviderMark } from "@/components/provider-mark";
import {
  PROVIDER_LABELS,
  compareModelNames,
  getModelLabel,
  getProviderForModel,
  isFlagshipModel,
  type ProviderKey,
} from "@/lib/model-meta";
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
  const [sortMode, setSortMode] = useState<"model" | "pointEstimate">("model");
  const [selectedProviders, setSelectedProviders] = useState<Set<ProviderKey>>(
    () => new Set<ProviderKey>(["anthropic", "google", "openai", "xai"]),
  );
  const [flagshipOnly, setFlagshipOnly] = useState(false);

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
  const sortedModelSummaries = selectedQuantity
    ? [...selectedQuantity.modelSummaries]
        .filter((summary) => {
          const provider = getProviderForModel(summary.modelName);
          if (!provider || !selectedProviders.has(provider)) return false;
          if (flagshipOnly && !isFlagshipModel(summary.modelName)) return false;
          return true;
        })
        .sort((left, right) =>
          sortMode === "pointEstimate"
            ? compareModelSummariesByCenter(left, right, selectedMethod.id)
            : compareModelNames(left.modelName, right.modelName),
        )
    : [];

  function toggleProvider(provider: ProviderKey) {
    setSelectedProviders((current) => {
      const next = new Set(current);
      if (next.has(provider)) {
        if (next.size === 1) return current;
        next.delete(provider);
      } else {
        next.add(provider);
      }
      return next;
    });
  }
  const quantityNote = selectedQuantity
    ? getQuantityNote(selectedQuantity.quantityId)
    : null;

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

  /* Reset model when quantity or filter changes and current inspected model drops out */
  useEffect(() => {
    if (!selectedQuantity) return;
    const visibleNames = sortedModelSummaries.map((summary) => summary.modelName);
    if (visibleNames.length === 0) return;
    if (!visibleNames.includes(inspectedModelName)) {
      startTransition(() => {
        setInspectedModelName(visibleNames[0]);
        setSelectedRunIndex(null);
      });
    }
  }, [inspectedModelName, selectedQuantity, sortedModelSummaries]);

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
        <p className="text-sm" style={{ color: "var(--muted-foreground)" }}>
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
          background: "color-mix(in oklab, var(--background) 85%, transparent)",
          borderColor: "var(--border)",
        }}
      >
        <div className="mx-auto flex max-w-[1400px] items-center justify-between">
          <div className="flex items-center gap-4">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-md"
              style={{ background: "color-mix(in oklab, var(--primary) 10%, transparent)" }}
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
                  stroke="var(--primary)"
                  strokeWidth="1.5"
                  fill="none"
                />
                <circle cx="8" cy="8" r="2.5" fill="var(--primary)" />
                <line x1="8" y1="1" x2="8" y2="4" stroke="var(--primary)" strokeWidth="1" />
                <line x1="8" y1="12" x2="8" y2="15" stroke="var(--primary)" strokeWidth="1" />
                <line x1="1" y1="8" x2="4" y2="8" stroke="var(--primary)" strokeWidth="1" />
                <line x1="12" y1="8" x2="15" y2="8" stroke="var(--primary)" strokeWidth="1" />
              </svg>
            </div>
            <h1
              className="font-sans text-lg font-semibold tracking-tight"
              style={{ color: "var(--foreground)" }}
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
                <circle cx="7" cy="7" r="5.5" stroke="var(--muted-foreground)" strokeWidth="1.5" />
                <line x1="11" y1="11" x2="14.5" y2="14.5" stroke="var(--muted-foreground)" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search quantities..."
                className="w-full rounded-lg border py-2.5 pl-9 pr-3 text-sm outline-none transition focus:border-[color:var(--primary)]"
                style={{
                  background: "var(--card)",
                  borderColor: "var(--border)",
                  color: "var(--foreground)",
                }}
              />
            </div>
          </div>
          <div className="flex flex-col gap-0.5 overflow-y-auto px-2 pb-4 xl:max-h-[calc(100svh-53px-72px)]">
            {Array.from(domainGroups.entries()).map(([domain, quantities]) => (
              <div key={domain} className="mb-1">
                <div
                  className="sticky top-0 z-10 px-2 pb-1 pt-3 font-mono text-[10px] font-semibold uppercase tracking-[0.2em]"
                  style={{ color: "var(--muted-foreground)", background: "var(--background)" }}
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
                        background: isSelected ? "color-mix(in oklab, var(--primary) 10%, transparent)" : "transparent",
                        border: isSelected ? "1px solid var(--primary)" : "1px solid transparent",
                      }}
                    >
                      <div
                        className="text-[13px] font-medium leading-snug"
                        style={{ color: isSelected ? "var(--primary)" : "var(--foreground)" }}
                      >
                        {quantity.quantityName}
                      </div>
                      <span className="mt-1 font-mono text-[10px]" style={{ color: "var(--muted-foreground)" }}>
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
              style={{ color: "var(--primary)" }}
            >
              {selectedQuantity.domain}
            </div>
            <h2
              className="mt-2 font-sans text-4xl font-semibold leading-tight tracking-tight lg:text-[2.6rem]"
              style={{ color: "var(--foreground)" }}
            >
              {selectedQuantity.quantityName}
            </h2>
            <p
              className="mt-3 max-w-2xl text-sm leading-relaxed"
              style={{ color: "var(--muted-foreground)" }}
            >
              Compare manifested belief centers across models. Swap interval
              generators to see how pooled, REML, and Bayesian uncertainty bands
              shift.
            </p>
            {quantityNote && (
              <div
                className="mt-4 max-w-3xl rounded-lg border px-4 py-3 text-sm leading-relaxed"
                style={{
                  background: "var(--card)",
                  borderColor: "var(--primary)",
                  color: "var(--muted-foreground)",
                }}
              >
                <div
                  className="mb-1 font-mono text-[10px] font-semibold uppercase tracking-[0.18em]"
                  style={{ color: "var(--primary)" }}
                >
                  Prompt note
                </div>
                {quantityNote}
              </div>
            )}
          </div>

          {/* Method and sort controls */}
          <div
            className="mb-3 rounded-lg border p-3"
            style={{ background: "var(--card)", borderColor: "var(--border)" }}
          >
            <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
              <div className="flex flex-wrap items-center gap-2">
                {data.methods.map((method) => {
                  const isActive = method.id === selectedMethod.id;
                  return (
                    <button
                      key={method.id}
                      type="button"
                      onClick={() => setSelectedMethodId(method.id)}
                      className="rounded-md px-3 py-2 text-xs font-medium transition-all"
                      style={{
                        background: isActive ? "var(--muted)" : "transparent",
                        color: isActive ? "var(--foreground)" : "var(--muted-foreground)",
                        boxShadow: isActive
                          ? "0 1px 3px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.6)"
                          : "none",
                      }}
                    >
                      {method.shortLabel}
                    </button>
                  );
                })}
              </div>
              <label className="flex items-center gap-3 text-xs">
                <span
                  className="font-mono uppercase tracking-[0.15em]"
                  style={{ color: "var(--muted-foreground)" }}
                >
                  Sort
                </span>
                <select
                  value={sortMode}
                  onChange={(event) =>
                    setSortMode(event.target.value as "model" | "pointEstimate")
                  }
                  className="rounded-md border px-3 py-2 outline-none transition"
                  style={{
                    background: "var(--muted)",
                    borderColor: "var(--border)",
                    color: "var(--foreground)",
                  }}
                >
                  <option value="model">Canonical model order</option>
                  <option value="pointEstimate">Point estimate, low to high</option>
                </select>
              </label>
            </div>
          </div>

          {/* Provider filter + flagship-only toggle */}
          <div
            className="mb-6 rounded-lg border p-3"
            style={{ background: "var(--card)", borderColor: "var(--border)" }}
          >
            <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className="font-mono text-[10px] uppercase tracking-[0.15em]"
                  style={{ color: "var(--muted-foreground)" }}
                >
                  Provider
                </span>
                {(["openai", "anthropic", "google", "xai"] as ProviderKey[]).map(
                  (provider) => {
                    const isActive = selectedProviders.has(provider);
                    return (
                      <button
                        key={provider}
                        type="button"
                        onClick={() => toggleProvider(provider)}
                        aria-pressed={isActive}
                        className="flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-all"
                        style={{
                          background: isActive ? "var(--muted)" : "transparent",
                          borderColor: isActive ? "var(--primary)" : "var(--border)",
                          color: isActive ? "var(--foreground)" : "var(--muted-foreground)",
                          opacity: isActive ? 1 : 0.55,
                        }}
                      >
                        <ProviderMark provider={provider} size={14} />
                        <span>{PROVIDER_LABELS[provider]}</span>
                      </button>
                    );
                  },
                )}
              </div>
              <label
                className="flex items-center gap-2 text-xs"
                style={{ color: "var(--muted-foreground)" }}
              >
                <input
                  type="checkbox"
                  checked={flagshipOnly}
                  onChange={(event) => setFlagshipOnly(event.target.checked)}
                  className="h-4 w-4 cursor-pointer accent-[color:var(--primary)]"
                />
                <span
                  className="font-mono uppercase tracking-[0.15em]"
                  style={{ color: flagshipOnly ? "var(--foreground)" : "var(--muted-foreground)" }}
                >
                  Flagship only
                </span>
                <span style={{ color: "var(--muted-foreground)" }}>
                  (Opus 4.7, GPT-5.4, Gemini 3.1 Pro, Grok 4.20)
                </span>
              </label>
            </div>
          </div>
          <p className="mb-5 text-xs leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
            {selectedMethod.description}
          </p>

          {/* Interval plot */}
          <IntervalPlot
            models={sortedModelSummaries}
            method={selectedMethod}
          />

          {/* Model cards grid — scales with model count */}
          <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {sortedModelSummaries.map((summary) => (
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
        <div className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm" style={{ animation: "reveal 200ms ease both" }}>
          <div
            ref={drawerRef}
            className="absolute right-0 top-0 h-full w-full max-w-[480px] overflow-y-auto border-l"
            style={{
              background: "var(--background)",
              borderColor: "var(--border)",
              animation: "slide-in 300ms cubic-bezier(0.16,1,0.3,1) both",
            }}
          >
            {/* Drawer header */}
            <div className="sticky top-0 z-10 border-b p-4 backdrop-blur-xl" style={{ background: "color-mix(in oklab, var(--background) 92%, transparent)", borderColor: "var(--border)" }}>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em]" style={{ color: "var(--muted-foreground)" }}>
                    Response inspector
                  </div>
                  <div className="mt-1 flex items-center gap-2">
                    <ProviderMark
                      provider={getProviderForModel(inspectedModelSummary?.modelName ?? "")}
                      size={16}
                    />
                    <h3 className="font-sans text-xl font-semibold" style={{ color: "var(--foreground)" }}>
                      {inspectedModelSummary
                        ? getModelLabel(inspectedModelSummary.modelName)
                        : "—"}
                    </h3>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setInspectorOpen(false)}
                  className="flex h-8 w-8 items-center justify-center rounded-md border transition hover:border-[color:var(--border-medium)]"
                  style={{ borderColor: "var(--border)", color: "var(--muted-foreground)" }}
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
                {sortedModelSummaries.map((summary) => {
                  const modelName = summary.modelName;
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
                      className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 font-mono text-[11px] font-medium transition-all"
                      style={{
                        background: isActive ? "color-mix(in oklab, var(--primary) 10%, transparent)" : "var(--card)",
                        color: isActive ? "var(--primary)" : "var(--muted-foreground)",
                        border: isActive ? "1px solid var(--primary)" : "1px solid var(--border)",
                      }}
                    >
                      <ProviderMark
                        provider={getProviderForModel(modelName)}
                        size={12}
                      />
                      <span>{getModelLabel(modelName)}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Run selector + detail */}
            <div className="p-4">
              {/* Run pills */}
              <div className="mb-4">
                <div className="mb-2 font-mono text-[10px] font-semibold uppercase tracking-[0.2em]" style={{ color: "var(--muted-foreground)" }}>
                  Runs ({activeRuns.length})
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {loadingActiveRuns ? (
                    <div className="shimmer rounded-md px-4 py-2 text-xs" style={{ color: "var(--muted-foreground)" }}>
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
                            background: isActive ? "var(--muted)" : "var(--card)",
                            border: isActive ? "1px solid var(--border-medium)" : "1px solid var(--border)",
                          }}
                        >
                          <div className="font-mono text-[10px]" style={{ color: isActive ? "var(--primary)" : "var(--muted-foreground)" }}>
                            #{run.runIndex}
                          </div>
                          <div className="mt-0.5 font-mono text-xs font-semibold" style={{ color: "var(--foreground)" }}>
                            {run.pointEstimate !== null ? formatNumber(run.pointEstimate) : "—"}
                          </div>
                        </button>
                      );
                    })
                  ) : (
                    <span className="text-xs" style={{ color: "var(--muted-foreground)" }}>No runs.</span>
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
      <span className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
        {label}
      </span>
      <span className="font-mono text-sm font-semibold" style={{ color: "var(--foreground)" }}>
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
      style={{ background: "var(--card)", borderColor: "var(--border)" }}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <ProviderMark
              provider={getProviderForModel(model.modelName)}
              size={16}
            />
            <h3 className="font-sans text-lg font-semibold" style={{ color: "var(--foreground)" }}>
              {getModelLabel(model.modelName)}
            </h3>
          </div>
          <p className="mt-1 font-mono text-[10px]" style={{ color: "var(--muted-foreground)" }}>
            {model.experimentDir}
          </p>
        </div>
        <button
          type="button"
          onClick={onInspect}
          className="rounded-md border px-3 py-1.5 font-mono text-[11px] font-medium transition-all hover:border-[color:var(--primary)] hover:text-[color:var(--primary)]"
          style={{ borderColor: "var(--border-medium)", color: "var(--muted-foreground)", background: "transparent" }}
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
            <tr style={{ background: "var(--muted)" }}>
              <th className="px-3 py-2 text-left font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
                Method
              </th>
              <th className="px-3 py-2 text-right font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
                Center
              </th>
              <th className="px-3 py-2 text-right font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
                90% CI
              </th>
            </tr>
          </thead>
          <tbody>
            {methods.map((method) => {
              const interval = model.intervals[method.id];
              const isSelected = method.id === selectedMethodId;
              return (
                <tr key={`${model.modelName}-${method.id}`} style={{ background: isSelected ? "color-mix(in oklab, var(--primary) 10%, transparent)" : "transparent" }}>
                  <td className="border-t px-3 py-1.5 font-medium" style={{ borderColor: "var(--border)", color: isSelected ? "var(--primary)" : "var(--foreground)" }}>
                    {method.shortLabel}
                  </td>
                  <td className="border-t px-3 py-1.5 text-right font-mono" style={{ borderColor: "var(--border)", color: "var(--foreground)" }}>
                    {formatMaybeNumber(interval.center)}
                  </td>
                  <td className="border-t px-3 py-1.5 text-right font-mono" style={{ borderColor: "var(--border)", color: "var(--muted-foreground)" }}>
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
                style={{ borderColor: "var(--border)", background: "var(--muted)", color: "var(--muted-foreground)" }}
              >
                {anchor.citation}
              </span>
            ))}
          </div>
        </div>
      )}

      {(model.costPerRunUsd !== null || model.tokensPerRun !== null) && (
        <details className="mt-3 rounded-md border" style={{ borderColor: "var(--border)", background: "var(--muted)" }}>
          <summary className="cursor-pointer px-3 py-2 font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
            Cost &amp; usage
          </summary>
          <div className="grid grid-cols-2 gap-2 px-3 pb-3 pt-1">
            {model.costPerRunUsd !== null && (
              <div>
                <div className="font-mono text-[9px] uppercase" style={{ color: "var(--muted-foreground)" }}>$/run</div>
                <div className="mt-0.5 font-mono text-xs" style={{ color: "var(--muted-foreground)" }}>{formatCurrency(model.costPerRunUsd)}</div>
              </div>
            )}
            {model.tokensPerRun !== null && (
              <div>
                <div className="font-mono text-[9px] uppercase" style={{ color: "var(--muted-foreground)" }}>Tokens/run</div>
                <div className="mt-0.5 font-mono text-xs" style={{ color: "var(--muted-foreground)" }}>{model.tokensPerRun.toLocaleString()}</div>
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
    <div className="mt-8 rounded-lg border p-5" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
      <div className="mb-3 flex items-center gap-2">
        <div className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em]" style={{ color: "var(--primary)" }}>
          Elicitation prompt
        </div>
        <span className="font-mono text-[10px]" style={{ color: "var(--muted-foreground)" }}>
          ({formatPromptVersion(selectedRun?.promptVersion ?? runs[0]?.promptVersion ?? "?")})
        </span>
      </div>
      <pre
        className="max-h-[400px] overflow-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed"
        style={{ color: "var(--muted-foreground)" }}
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
        <div className="shimmer h-6 w-40 rounded-md" style={{ background: "var(--muted)" }} />
        <div className="shimmer mt-3 h-4 w-64 rounded-md" style={{ background: "var(--muted)" }} />
      </div>
    );
  }

  if (!model || !run) {
    return (
      <p className="text-center text-xs" style={{ color: "var(--muted-foreground)" }}>
        Select a run to inspect.
      </p>
    );
  }

  return (
    <div>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
            Run {run.runIndex}
          </div>
          <div className="mt-1 flex items-center gap-2">
            <ProviderMark
              provider={getProviderForModel(model.modelName)}
              size={14}
            />
            <h4 className="font-sans text-base font-semibold" style={{ color: "var(--foreground)" }}>
              {getModelLabel(model.modelName)}
            </h4>
          </div>
        </div>
        <span className="rounded-md px-2 py-1 font-mono text-[10px]" style={{ background: "var(--muted)", color: "var(--muted-foreground)" }}>
          {formatInterval(run.lowerBound, run.upperBound)}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <MetricTile label="Point" value={formatMaybeNumber(run.pointEstimate)} />
        <MetricTile label="Prompt" value={formatPromptVersion(run.promptVersion || "?")} />
        <MetricTile label="p50" value={formatMaybeNumber(run.quantiles.p50)} />
      </div>

      {/* Quantiles */}
      <div className="mt-4">
        <div className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
          Quantiles
        </div>
        <div className="mt-2 grid grid-cols-5 gap-1.5">
          {["p05", "p25", "p50", "p75", "p95"].map((key) => (
            <div key={key} className="rounded-md border px-2 py-2 text-center" style={{ borderColor: "var(--border)", background: "var(--muted)" }}>
              <div className="font-mono text-[9px] uppercase" style={{ color: "var(--muted-foreground)" }}>{key}</div>
              <div className="mt-1 font-mono text-[12px] font-semibold" style={{ color: "var(--foreground)" }}>
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
          <div className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
            Literature anchors
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {run.citations.map((citation) => (
              <span
                key={`${run.runIndex}-${citation}`}
                className="rounded-md border px-2 py-1 font-mono text-[10px]"
                style={{ borderColor: "var(--border)", background: "var(--muted)", color: "var(--muted-foreground)" }}
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
      <div className="font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
        {title}
      </div>
      <p className="mt-1.5 text-[13px] leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
        {text}
      </p>
    </div>
  );
}

function CollapsibleSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="mt-3 rounded-md border" style={{ borderColor: "var(--border)", background: "var(--card)" }}>
      <summary className="cursor-pointer px-3 py-2.5 font-mono text-[10px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>
        {title}
      </summary>
      <pre className="overflow-x-auto whitespace-pre-wrap border-t px-3 py-3 font-mono text-[11px] leading-relaxed" style={{ borderColor: "var(--border)", color: "var(--muted-foreground)" }}>
        {children}
      </pre>
    </details>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border px-3 py-2" style={{ borderColor: "var(--border)", background: "var(--muted)" }}>
      <div className="font-mono text-[9px] uppercase tracking-[0.15em]" style={{ color: "var(--muted-foreground)" }}>{label}</div>
      <div className="mt-1 font-mono text-sm font-semibold" style={{ color: "var(--foreground)" }}>{value}</div>
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

function formatPromptVersion(value: string): string {
  if (!value) return "?";
  return value;
}

function compareModelSummariesByCenter(
  left: ModelSummary,
  right: ModelSummary,
  methodId: IntervalMethodId,
): number {
  const leftCenter = left.intervals[methodId].center;
  const rightCenter = right.intervals[methodId].center;
  if (leftCenter === null && rightCenter === null) {
    return compareModelNames(left.modelName, right.modelName);
  }
  if (leftCenter === null) return 1;
  if (rightCenter === null) return -1;
  return (
    leftCenter - rightCenter ||
    compareModelNames(left.modelName, right.modelName)
  );
}

function getQuantityNote(quantityId: string): string | null {
  if (quantityId === "labor_supply.income_elasticity.prime_age") {
    return "This quantity uses the later sign-clarified rerun. The explicit note that positive values mean people work more when they have more resources eliminated the earlier sign confusion in some models.";
  }
  if (quantityId === "labor_supply.policy_response.income_elasticity") {
    return "Legacy panel entry retained for reference. This quantity has since been merged into the canonical prime-age income elasticity and was not rerun under the latest sign-clarified prompt.";
  }
  return null;
}
