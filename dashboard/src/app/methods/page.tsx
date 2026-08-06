import Link from "next/link";

import { PageBand, ProvenanceFooter } from "@/components/site-chrome";
import { loadHarnessRows, totalRunCount } from "@/lib/site-data";

export const metadata = {
  title: "Methods · AI beliefs · PolicyEngine",
  description:
    "Elicitation protocol, pooling, and per-model harness disclosure for the AI beliefs panel: memory-only prompts, 15 runs per cell, quantile-based distributions.",
};

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export default function MethodsPage() {
  const harness = loadHarnessRows();

  return (
    <div>
      <PageBand
        title="Methods"
        lede="The protocol identifies prompt-conditioned response distributions under a fixed elicitation design — what a model returns when asked, not how well it forecasts. The paper carries the full treatment; this page summarizes the estimand and discloses the per-model harness."
      />
      <div className="mx-auto max-w-[1100px] px-5 py-8">
        <section className="max-w-3xl space-y-4 text-sm leading-relaxed">
          <MethodBlock title="Elicitation">
            A memory-only prompt fixes each quantity&apos;s interpretation and
            requests structured JSON: a point estimate, five quantiles (p05,
            p25, p50, p75, p95), citations, and a reasoning summary. The
            prompt instructs the model to answer from background knowledge
            alone — no tools, no literature reconstruction. Two
            sign-ambiguous quantities carry direction-first clarifiers that
            define what the reported sign means without calling either
            direction correct. The exact prompt text appears verbatim on
            every quantity page and on the{" "}
            <Link
              href="/process"
              className="underline underline-offset-2"
              style={{ color: "var(--foreground)" }}
            >
              Process
            </Link>{" "}
            page, alongside costs and the full generation pipeline.
          </MethodBlock>
          <MethodBlock title="How the nine headline elasticities were chosen">
            Three criteria, fixed in the registry before the panel ran: each
            quantity is a standard named object with a published review anchor
            or an established calibration home; each has a direct policy
            consumer (six labor-and-tax parameters feed CBO-style and
            PolicyEngine-style tax-benefit microsimulation, three
            macro-and-trade parameters are core calibration inputs); and the
            two subpanels pair a domain where lower elasticities read as more
            room for redistribution with one where no such monotone mapping
            exists. Selection determines coverage, not model comparisons —
            every model answers the identical panel. Each quantity&apos;s page
            opens with the exact definition the models received.
          </MethodBlock>
          <MethodBlock title="Repeated runs and pooling">
            Each model-quantity cell runs 15 independent times. Each
            run&apos;s quantiles become a piecewise-uniform distribution;
            the pooled predictive distribution is the equal-weight mixture of
            the 15. The headline interval is the pooled 90 percent mixture
            interval — the predictive spread of repeated elicited answers,
            not a confidence interval for their mean. REML and Bayesian
            hierarchical estimators run alongside as robustness checks.
          </MethodBlock>
          <MethodBlock title="Panel">
            31 models from ten organizations — Anthropic, OpenAI, Google,
            xAI, DeepSeek, Alibaba, Moonshot AI, Zhipu AI, MiniMax, and
            Thinking Machines — elicited in six waves: 11 in April 2026,
            18 in July 2026 (six frontier updates, five Chinese-lab
            models, the GPT-5.6 family, and late Grok 4.5, Kimi K3,
            Gemini 3.6 Flash, and Claude Opus 5 additions), and 2 in
            August 2026 (Qwen 3.8 Max and Inkling), all under identical
            prompts. 26 U.S.-scoped quantities: 9 headline elasticities in
            two subpanels, 4 calibration parameters, a capital-gains
            convention sibling, and 12 simulation-facing coefficients.
            12,090 successful runs at a 100 percent parse rate, with every
            cell verified against the exact 15-run grid; failed slots re-ran as fresh independent draws,
            each traced in the{" "}
            <a
              className="underline underline-offset-2"
              href="https://github.com/PolicyEngine/llm-econ-beliefs/blob/main/results/failure-manifest.csv"
            >
              failure manifest
            </a>{" "}
            (58 July frontier slots) or in per-directory recovery archives
            (4,168 records across the Chinese-lab wave).
          </MethodBlock>
          <MethodBlock title="Reproduce it">
            Every table rebuilds from the committed artifacts without API
            access, and a{" "}
            <a
              className="underline underline-offset-2"
              href="https://github.com/PolicyEngine/llm-econ-beliefs/blob/main/scripts/verify_paper_prose.py"
            >
              verification script
            </a>{" "}
            pins the paper&apos;s prose to the generated tables. The{" "}
            <a
              className="underline underline-offset-2"
              href="https://github.com/PolicyEngine/llm-econ-beliefs#reproducing-the-v4-panel"
            >
              README
            </a>{" "}
            documents both reproduction paths and costs; the{" "}
            <a className="underline underline-offset-2" href={`${basePath}/paper.pdf`}>
              paper
            </a>{" "}
            carries the full estimand discussion, robustness appendices, and
            limitations.
          </MethodBlock>
        </section>

        {harness.length > 0 ? (
          <section className="mt-10">
            <h2
              className="text-lg font-semibold"
              style={{ color: "var(--foreground)" }}
            >
              Per-model harness disclosure
            </h2>
            <p
              className="mt-1 max-w-2xl text-sm"
              style={{ color: "var(--muted-foreground)" }}
            >
              The structured-output mechanism, completion budget, sampling
              regime, and reasoning configuration follow each provider&apos;s
              API surface and are therefore confounded with model identity —
              disclosed in full here and bounded empirically by the
              paper&apos;s cross-mechanism ablation (max center movement
              0.03).
            </p>
            <div
              className="mt-3 overflow-x-auto rounded-lg border"
              style={{
                borderColor: "var(--border)",
                background: "var(--card)",
              }}
            >
              <table className="w-full text-left text-xs">
                <thead>
                  <tr
                    className="border-b"
                    style={{
                      borderColor: "var(--border)",
                      color: "var(--muted-foreground)",
                    }}
                  >
                    <th className="px-3 py-2 font-medium">Model</th>
                    <th className="px-3 py-2 font-medium">Path</th>
                    <th className="px-3 py-2 font-medium">Mechanism</th>
                    <th className="px-3 py-2 font-medium">Budget</th>
                    <th className="px-3 py-2 font-medium">Sampling</th>
                    <th className="px-3 py-2 font-medium">Reasoning</th>
                  </tr>
                </thead>
                <tbody>
                  {harness.map((row) => (
                    <tr
                      key={row.model}
                      className="border-b last:border-b-0"
                      style={{ borderColor: "var(--border)" }}
                    >
                      <td
                        className="px-3 py-2"
                        style={{ color: "var(--foreground)" }}
                      >
                        {row.model}
                      </td>
                      <td
                        className="px-3 py-2"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        {row.providerPath}
                      </td>
                      <td
                        className="px-3 py-2"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        {row.mechanism}
                      </td>
                      <td
                        className="px-3 py-2 font-mono"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        {row.budget}
                      </td>
                      <td
                        className="px-3 py-2"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        {row.sampling}
                      </td>
                      <td
                        className="px-3 py-2"
                        style={{ color: "var(--muted-foreground)" }}
                      >
                        {row.reasoning}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}

        <ProvenanceFooter runCount={totalRunCount()} />
      </div>
    </div>
  );
}

function MethodBlock({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h2
        className="text-base font-semibold"
        style={{ color: "var(--foreground)" }}
      >
        {title}
      </h2>
      <p className="mt-1" style={{ color: "var(--muted-foreground)" }}>
        {children}
      </p>
    </div>
  );
}
