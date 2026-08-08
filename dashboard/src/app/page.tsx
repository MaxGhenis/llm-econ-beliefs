import Link from "next/link";

import { PageBand, ProvenanceFooter } from "@/components/site-chrome";
import { StripPlot, providerColor } from "@/components/strip-plot";
import { getModelLabel } from "@/lib/model-meta";
import {
  HEADLINE_QUANTITY_IDS,
  LABOR_TAX_QUANTITY_IDS,
  MACRO_TRADE_QUANTITY_IDS,
  getSummaryData,
  loadBenchmarkBands,
  loadSlimRuns,
  modelsByCenter,
  slugForQuantity,
  tieAveragedRanks,
  totalRunCount,
} from "@/lib/site-data";
import type { QuantitySummary } from "@/lib/dashboard-types";

export function generateMetadata() {
  const modelCount = getSummaryData().stats.modelCount;
  return {
    title: "AI beliefs · PolicyEngine",
    description: `What ${modelCount} frontier language models answer when asked for economic elasticities: interval charts, run-level responses, and a peer-review-hardened paper.`,
  };
}

/** Average within-quantity |center| rank (1 = most elastic), mirroring the
 *  paper's Table 1-2 construction — tie-averaged ranks, exactly as
 *  build_tables.average_ranks computes them. */
function topByAbsRank(
  quantities: QuantitySummary[],
  quantityIds: readonly string[],
  count: number,
): { modelName: string; avgRank: number }[] {
  const rankSums = new Map<string, { total: number; n: number }>();
  for (const quantityId of quantityIds) {
    const quantity = quantities.find((q) => q.quantityId === quantityId);
    if (!quantity) continue;
    const absoluteCenters = new Map(
      quantity.modelSummaries.flatMap((summary) => {
        const center = summary.intervals.pooled.center;
        return center !== null
          ? [[summary.modelName, Math.abs(center)] as const]
          : [];
      }),
    );
    for (const [modelName, rank] of tieAveragedRanks(absoluteCenters, true)) {
      const entry = rankSums.get(modelName) ?? { total: 0, n: 0 };
      entry.total += rank;
      entry.n += 1;
      rankSums.set(modelName, entry);
    }
  }
  return Array.from(rankSums.entries())
    .map(([modelName, { total, n }]) => ({ modelName, avgRank: total / n }))
    .sort((a, b) => a.avgRank - b.avgRank)
    .slice(0, count);
}

function centerRange(
  quantities: QuantitySummary[],
  quantityId: string,
): { name: string; slug: string; min: number; max: number } | null {
  const quantity = quantities.find((q) => q.quantityId === quantityId);
  if (!quantity) return null;
  const centers = quantity.modelSummaries
    .map((s) => s.intervals.pooled.center)
    .filter((value): value is number => value !== null);
  if (centers.length === 0) return null;
  return {
    name: quantity.quantityName,
    slug: slugForQuantity(quantity.quantityId),
    min: Math.min(...centers),
    max: Math.max(...centers),
  };
}

export default function Home() {
  const data = getSummaryData();
  const bands = loadBenchmarkBands();
  const runCount = totalRunCount();

  const laborTop = topByAbsRank(data.quantities, LABOR_TAX_QUANTITY_IDS, 3);
  const macroTop = topByAbsRank(data.quantities, MACRO_TRADE_QUANTITY_IDS, 3);

  const convergent = [
    "household.annual_discount_factor",
    "household.relative_risk_aversion.crra",
    "labor_supply.frisch_elasticity.prime_age",
  ]
    .map((quantityId) => centerRange(data.quantities, quantityId))
    .filter((entry): entry is NonNullable<typeof entry> => entry !== null);

  const capGains = data.quantities.find(
    (q) => q.quantityId === "tax.capital_gains_realizations.elasticity",
  );
  const geminiRuns = capGains
    ? loadSlimRuns(capGains.quantityId, "gemini-3.5-flash")
    : [];
  const geminiNegative = geminiRuns.filter(
    (run) => (run.pointEstimate ?? 0) < 0,
  ).length;

  const headline = HEADLINE_QUANTITY_IDS.map((quantityId) =>
    data.quantities.find((q) => q.quantityId === quantityId),
  ).filter((quantity): quantity is QuantitySummary => Boolean(quantity));

  return (
    <div>
      <PageBand
        title="AI beliefs about economic parameters"
        lede={
          <>
            When you ask a frontier language model for the elasticity of
            taxable income, it answers with a number and an uncertainty band.
            This project elicits those answers systematically —{" "}
            {data.stats.modelCount} models, 26 quantities, 15 independent runs
            each — and maps where the models agree, where they diverge, and
            what their answers imply for policy.
          </>
        }
        aside={
          <div
            className="flex items-center gap-5 text-sm"
            style={{ color: "var(--muted-foreground)" }}
          >
            <Stat value={`${data.stats.modelCount}`} label="models" />
            <Stat value={`${data.stats.quantityCount}`} label="quantities" />
            <Stat value={runCount.toLocaleString()} label="runs" />
          </div>
        }
      />

      <div className="mx-auto max-w-[1100px] px-5 py-8">
        {/* Findings */}
        <div className="grid gap-4 md:grid-cols-3">
          <FindingCard
            title="Rankings flip by domain"
            href="/quantities"
            body="No model family is uniformly more elastic. The most elastic models on labor-and-tax quantities differ from the most elastic on macro-and-trade."
          >
            <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
              <TopList title="Labor and tax" entries={laborTop} />
              <TopList title="Macro and trade" entries={macroTop} />
            </div>
          </FindingCard>

          <FindingCard
            title="Generations converge on core parameters"
            href={`/quantities/${convergent[0]?.slug ?? ""}`}
            body="Across six elicitation waves and ten organizations, pooled centers for the core preference parameters sit in tight cross-model ranges."
          >
            <ul className="mt-3 space-y-1.5 text-xs">
              {convergent.map((entry) => (
                <li
                  key={entry.slug}
                  className="flex items-center justify-between gap-2"
                >
                  <span style={{ color: "var(--foreground)" }}>
                    {entry.name}
                  </span>
                  <span
                    className="font-mono"
                    style={{ color: "var(--muted-foreground)" }}
                  >
                    {entry.min.toFixed(2)}–{entry.max.toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          </FindingCard>

          {geminiRuns.length > 0 ? (
            <FindingCard
              title="A near-zero mean can hide two camps"
              href={
                capGains
                  ? `/quantities/${slugForQuantity(capGains.quantityId)}`
                  : "/quantities"
              }
              body={`Gemini 3.5 Flash's capital-gains runs split ${geminiNegative} negative to ${
                geminiRuns.length - geminiNegative
              } positive — a bimodal answer that averages to nearly zero without any run saying zero.`}
            >
              <svg
                viewBox="0 0 100 16"
                className="mt-3 h-4 w-full"
                aria-label="Gemini 3.5 Flash capital-gains run estimates"
              >
                {(() => {
                  const points = geminiRuns
                    .map((run) => run.pointEstimate)
                    .filter((value): value is number => value !== null);
                  const min = Math.min(...points);
                  const max = Math.max(...points);
                  const span = max - min || 1;
                  return points.map((value, index) => (
                    <circle
                      key={index}
                      cx={((value - min) / span) * 92 + 4}
                      cy={8}
                      r={2.6}
                      fill={providerColor("gemini-3.5-flash")}
                      fillOpacity={0.75}
                    />
                  ));
                })()}
              </svg>
            </FindingCard>
          ) : null}
        </div>

        {/* Headline panel */}
        <h2
          className="mt-10 text-lg font-semibold"
          style={{ color: "var(--foreground)" }}
        >
          The nine headline elasticities
        </h2>
        <p
          className="mt-1 max-w-2xl text-sm"
          style={{ color: "var(--muted-foreground)" }}
        >
          Every panel: {data.stats.modelCount} models sorted by pooled center,
          dot at the center, bar spanning the pooled 90 percent interval. Shaded
          regions are review-based literature ranges. Click through for
          run-level detail.
        </p>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {headline.map((quantity) => {
            const rows = modelsByCenter(quantity).map((summary) => ({
              modelName: summary.modelName,
              center: summary.intervals.pooled.center,
              lower: summary.intervals.pooled.lower,
              upper: summary.intervals.pooled.upper,
            }));
            return (
              <Link
                key={quantity.quantityId}
                href={`/quantities/${slugForQuantity(quantity.quantityId)}`}
                className="rounded-lg border p-4 transition hover:border-[color:var(--ring)]"
                style={{
                  borderColor: "var(--border)",
                  background: "var(--card)",
                }}
              >
                <h3
                  className="text-sm font-medium"
                  style={{ color: "var(--foreground)" }}
                >
                  {quantity.quantityName}
                </h3>
                <div className="mt-2">
                  <StripPlot
                    rows={rows}
                    band={bands.get(quantity.quantityName) ?? null}
                    compact
                  />
                </div>
              </Link>
            );
          })}
        </div>

        <div className="mt-8 flex flex-wrap gap-3 text-sm">
          <Link
            href="/quantities"
            className="rounded-md border px-3 py-2 font-medium transition hover:opacity-80"
            style={{
              borderColor: "var(--border)",
              color: "var(--foreground)",
              background: "var(--card)",
            }}
          >
            All 26 quantities
          </Link>
          <Link
            href="/models"
            className="rounded-md border px-3 py-2 font-medium transition hover:opacity-80"
            style={{
              borderColor: "var(--border)",
              color: "var(--foreground)",
              background: "var(--card)",
            }}
          >
            All {data.stats.modelCount} models
          </Link>
        </div>

        <ProvenanceFooter runCount={runCount} />
      </div>
    </div>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <span>
      <span
        className="text-lg font-semibold"
        style={{ color: "var(--foreground)" }}
      >
        {value}
      </span>{" "}
      {label}
    </span>
  );
}

function FindingCard({
  title,
  body,
  href,
  children,
}: {
  title: string;
  body: string;
  href: string;
  children?: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className="rounded-lg border p-4 transition hover:border-[color:var(--ring)]"
      style={{ borderColor: "var(--border)", background: "var(--card)" }}
    >
      <h3
        className="text-sm font-semibold"
        style={{ color: "var(--foreground)" }}
      >
        {title}
      </h3>
      <p
        className="mt-1.5 text-sm leading-relaxed"
        style={{ color: "var(--muted-foreground)" }}
      >
        {body}
      </p>
      {children}
    </Link>
  );
}

function TopList({
  title,
  entries,
}: {
  title: string;
  entries: { modelName: string; avgRank: number }[];
}) {
  return (
    <div>
      <p
        className="font-medium uppercase tracking-wide"
        style={{ color: "var(--muted-foreground)" }}
      >
        {title}
      </p>
      <ol className="mt-1.5 space-y-1">
        {entries.map((entry) => (
          <li key={entry.modelName} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: providerColor(entry.modelName) }}
            />
            <span style={{ color: "var(--foreground)" }}>
              {getModelLabel(entry.modelName)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
