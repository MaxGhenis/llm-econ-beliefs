import { PageBand, ProvenanceFooter } from "@/components/site-chrome";
import {
  loadModelOverview,
  loadModelRegistry,
  slugForModel,
  totalRunCount,
} from "@/lib/site-data";
import Link from "next/link";

export const metadata = {
  title: "Generations · AI beliefs · PolicyEngine",
  description:
    "How elicited elasticities move across model generations: same-product-line successor pairs and within-lab trajectories, by subpanel.",
};

/** Same-product-line successor pairs (older, newer), by display label.
 *  Curated to direct product successions only — no cross-tier pairs. */
const SUCCESSOR_PAIRS: Array<[string, string]> = [
  ["Claude Opus 4.7", "Claude Opus 4.8"],
  ["Claude Opus 4.8", "Claude Opus 5"],
  ["Claude Sonnet 4.6", "Claude Sonnet 5"],
  ["GPT-5.4", "GPT-5.5"],
  ["GPT-5.5", "GPT-5.6 Sol"],
  ["Gemini 3 Flash", "Gemini 3.5 Flash"],
  ["Gemini 3.5 Flash", "Gemini 3.6 Flash"],
  ["Grok 4.3", "Grok 4.5"],
  ["Kimi K2.6", "Kimi K3"],
];

const WAVE_ORDER: Record<string, number> = {
  april_2026: 0,
  july_2026_frontier: 1,
  july_2026_independent: 2,
  july_2026_gpt56: 3,
  july_2026_late: 4,
};

interface ModelRanks {
  laborAbs: number | null;
  macroAbs: number | null;
  laborWidth: number | null;
}

function formatRank(value: number | null): string {
  return value === null || Number.isNaN(value) ? "—" : value.toFixed(1);
}

function formatDelta(older: number | null, newer: number | null): string {
  if (
    older === null ||
    newer === null ||
    Number.isNaN(older) ||
    Number.isNaN(newer)
  ) {
    return "—";
  }
  const delta = newer - older;
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}`;
}

export default function GenerationsPage() {
  const labor = loadModelOverview("labor_tax");
  const macro = loadModelOverview("macro_trade");
  const registry = loadModelRegistry();
  const ranks = new Map<string, ModelRanks>();
  for (const row of labor) {
    ranks.set(row.model, {
      laborAbs: row.absRank,
      laborWidth: row.widthRank,
      macroAbs: null,
    });
  }
  for (const row of macro) {
    const entry = ranks.get(row.model) ?? {
      laborAbs: null,
      laborWidth: null,
      macroAbs: null,
    };
    entry.macroAbs = row.absRank;
    ranks.set(row.model, entry);
  }
  const panelSize = ranks.size;

  const pairs = SUCCESSOR_PAIRS.flatMap(([older, newer]) => {
    const olderRanks = ranks.get(older);
    const newerRanks = ranks.get(newer);
    return olderRanks && newerRanks
      ? [{ older, newer, olderRanks, newerRanks }]
      : [];
  });

  const countDirection = (
    accessor: (entry: ModelRanks) => number | null,
  ): { less: number; more: number; total: number } => {
    let less = 0;
    let more = 0;
    let total = 0;
    for (const pair of pairs) {
      const olderValue = accessor(pair.olderRanks);
      const newerValue = accessor(pair.newerRanks);
      if (olderValue === null || newerValue === null) continue;
      total += 1;
      // Rank 1 = highest |elasticity|, so a larger rank = less elastic.
      if (newerValue > olderValue) less += 1;
      if (newerValue < olderValue) more += 1;
    }
    return { less, more, total };
  };
  const laborDirection = countDirection((entry) => entry.laborAbs);
  const macroDirection = countDirection((entry) => entry.macroAbs);
  const widthDirection = countDirection((entry) => entry.laborWidth);

  const organizations = new Map<
    string,
    {
      label: string;
      models: Array<{
        label: string;
        modelId: string;
        wave: string;
        waveLabel: string;
      }>;
    }
  >();
  for (const row of registry.values()) {
    const bucket = organizations.get(row.organization) ?? {
      label: row.organizationLabel,
      models: [],
    };
    bucket.models.push({
      label: row.displayLabel,
      modelId: row.modelId,
      wave: row.wave,
      waveLabel: row.waveLabel,
    });
    organizations.set(row.organization, bucket);
  }
  const multiModelOrgs = [...organizations.values()]
    .filter((bucket) => bucket.models.length >= 2)
    .map((bucket) => ({
      ...bucket,
      models: [...bucket.models].sort(
        (a, b) =>
          (WAVE_ORDER[a.wave] ?? 9) - (WAVE_ORDER[b.wave] ?? 9) ||
          a.label.localeCompare(b.label),
      ),
    }))
    .sort((a, b) => b.models.length - a.models.length);
  const singleModelOrgs = [...organizations.values()]
    .filter((bucket) => bucket.models.length === 1)
    .map((bucket) => bucket.label)
    .sort();

  const cellStyle = { color: "var(--foreground)" };
  const mutedStyle = { color: "var(--muted-foreground)" };

  return (
    <div>
      <PageBand
        title="Generations"
        lede="How elicited elasticities move as labs ship newer models: same-product-line successor pairs, then every multi-model lab's trajectory. Descriptive only — elicitation wave and harness configuration are confounded with model generation (see Methods), so read shifts as observations, not effects."
      />
      <div className="mx-auto max-w-[1100px] px-5 py-8">
        <section>
          <h2 className="text-lg font-semibold" style={cellStyle}>
            Same-product-line successors
          </h2>
          <p className="mt-1 max-w-3xl text-sm" style={mutedStyle}>
            Each pair compares a model with its direct successor in the same
            product line. Ranks run 1 (most elastic, or tightest) to{" "}
            {panelSize}; Δ is newer minus older, so positive Δ = the newer
            model ranks less elastic (or wider).
          </p>
          <div
            className="mt-3 overflow-x-auto rounded-lg border"
            style={{ borderColor: "var(--border)", background: "var(--card)" }}
          >
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--border)", ...mutedStyle }}>
                  <th className="px-3 py-2 font-medium">Pair</th>
                  <th className="px-3 py-2 font-medium">Labor-tax rank (old → new)</th>
                  <th className="px-3 py-2 font-medium">Δ</th>
                  <th className="px-3 py-2 font-medium">Macro-trade rank (old → new)</th>
                  <th className="px-3 py-2 font-medium">Δ</th>
                  <th className="px-3 py-2 font-medium">Width rank (old → new)</th>
                  <th className="px-3 py-2 font-medium">Δ</th>
                </tr>
              </thead>
              <tbody>
                {pairs.map((pair) => (
                  <tr
                    key={`${pair.older}-${pair.newer}`}
                    className="border-b last:border-b-0"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <td className="px-3 py-2" style={cellStyle}>
                      {pair.older} → {pair.newer}
                    </td>
                    <td className="px-3 py-2 font-mono" style={cellStyle}>
                      {formatRank(pair.olderRanks.laborAbs)} → {formatRank(pair.newerRanks.laborAbs)}
                    </td>
                    <td className="px-3 py-2 font-mono" style={mutedStyle}>
                      {formatDelta(pair.olderRanks.laborAbs, pair.newerRanks.laborAbs)}
                    </td>
                    <td className="px-3 py-2 font-mono" style={cellStyle}>
                      {formatRank(pair.olderRanks.macroAbs)} → {formatRank(pair.newerRanks.macroAbs)}
                    </td>
                    <td className="px-3 py-2 font-mono" style={mutedStyle}>
                      {formatDelta(pair.olderRanks.macroAbs, pair.newerRanks.macroAbs)}
                    </td>
                    <td className="px-3 py-2 font-mono" style={cellStyle}>
                      {formatRank(pair.olderRanks.laborWidth)} → {formatRank(pair.newerRanks.laborWidth)}
                    </td>
                    <td className="px-3 py-2 font-mono" style={mutedStyle}>
                      {formatDelta(pair.olderRanks.laborWidth, pair.newerRanks.laborWidth)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 max-w-3xl text-sm" style={mutedStyle}>
            Counted across the {laborDirection.total} pairs: the newer model
            ranks less elastic on labor-and-tax in {laborDirection.less}, more
            elastic in {laborDirection.more}; on macro-and-trade, less elastic
            in {macroDirection.less}, more elastic in {macroDirection.more};
            and wider in {widthDirection.less}, tighter in{" "}
            {widthDirection.more}. No adjustment for multiple comparisons —
            these are counts, not tests.
          </p>
        </section>

        <section className="mt-10">
          <h2 className="text-lg font-semibold" style={cellStyle}>
            Lab trajectories
          </h2>
          <p className="mt-1 max-w-3xl text-sm" style={mutedStyle}>
            Every lab with more than one model in the panel, ordered by
            elicitation wave. Single-model labs ({singleModelOrgs.join(", ")})
            have no within-lab comparison yet.
          </p>
          <div className="mt-3 grid gap-4 md:grid-cols-2">
            {multiModelOrgs.map((bucket) => (
              <div
                key={bucket.label}
                className="rounded-lg border p-4"
                style={{ borderColor: "var(--border)", background: "var(--card)" }}
              >
                <h3 className="text-sm font-semibold" style={cellStyle}>
                  {bucket.label}
                </h3>
                <table className="mt-2 w-full text-left text-xs">
                  <thead>
                    <tr className="border-b" style={{ borderColor: "var(--border)", ...mutedStyle }}>
                      <th className="py-1.5 pr-2 font-medium">Model</th>
                      <th className="py-1.5 pr-2 font-medium">Wave</th>
                      <th className="py-1.5 pr-2 font-medium">Labor-tax</th>
                      <th className="py-1.5 pr-2 font-medium">Macro</th>
                      <th className="py-1.5 font-medium">Width</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bucket.models.map((model) => {
                      const modelRanks = ranks.get(model.label);
                      return (
                        <tr
                          key={model.label}
                          className="border-b last:border-b-0"
                          style={{ borderColor: "var(--border)" }}
                        >
                          <td className="py-1.5 pr-2" style={cellStyle}>
                            <Link
                              href={`/models/${slugForModel(model.modelId)}`}
                              className="hover:underline"
                            >
                              {model.label}
                            </Link>
                          </td>
                          <td className="py-1.5 pr-2" style={mutedStyle}>
                            {model.waveLabel}
                          </td>
                          <td className="py-1.5 pr-2 font-mono" style={cellStyle}>
                            {formatRank(modelRanks?.laborAbs ?? null)}
                          </td>
                          <td className="py-1.5 pr-2 font-mono" style={cellStyle}>
                            {formatRank(modelRanks?.macroAbs ?? null)}
                          </td>
                          <td className="py-1.5 font-mono" style={cellStyle}>
                            {formatRank(modelRanks?.laborWidth ?? null)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
          <p className="mt-3 max-w-3xl text-xs" style={mutedStyle}>
            Ranks are averages of within-quantity ranks over each subpanel
            (labor-and-tax and macro-and-trade), 1 = most elastic; width rank
            1 = tightest pooled 90 percent interval. Within-lab comparisons
            share the lab&apos;s serving path in most cases, but elicitation
            wave, output mechanism, and completion budget still differ across
            generations — the paper&apos;s harness-disclosure and
            cross-mechanism ablation appendices bound those effects.
          </p>
        </section>
      </div>
      <ProvenanceFooter runCount={totalRunCount()} />
    </div>
  );
}
