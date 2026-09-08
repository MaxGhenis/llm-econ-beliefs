import type { ReactNode } from "react";

import { ProvenanceFooter } from "@/components/site-chrome";
import { totalRunCount } from "@/lib/site-data";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

// Bump on every manuscript revision, in lockstep everywhere this page
// links the render (tests/test_paper_embed.py locks the lockstep).
const PAPER_VERSION = "n31-eti-claim-20260907";

const STANDALONE_HREF = `${basePath}/paper/web/index.html?v=${PAPER_VERSION}`;
const PDF_HREF = `${basePath}/paper/web/index.pdf?v=${PAPER_VERSION}`;

export const metadata = {
  title: "Working paper — AI beliefs",
  description:
    "How Large Language Models Answer Questions About Economic Elasticities: a repeated-elicitation study of 31 models, 12,090 runs, with every number verified against the committed tables.",
  openGraph: {
    type: "article",
    title: "How Large Language Models Answer Questions About Economic Elasticities",
    url: "https://policyengine.org/ai-beliefs/paper",
  },
};

function ActionLink({
  href,
  children,
  primary = false,
  external = false,
}: {
  href: string;
  children: ReactNode;
  primary?: boolean;
  external?: boolean;
}): ReactNode {
  return (
    <a
      href={href}
      {...(external ? { target: "_blank", rel: "noopener" } : {})}
      className="rounded-md border px-3 py-1.5 text-sm transition hover:bg-[color:var(--muted)]"
      style={
        primary
          ? {
              background: "var(--foreground)",
              color: "var(--background)",
              borderColor: "var(--foreground)",
            }
          : { borderColor: "var(--border)", color: "var(--foreground)" }
      }
    >
      {children}
    </a>
  );
}

export default function PaperPage(): ReactNode {
  const runCount = totalRunCount();
  return (
    <main>
      <div
        className="border-b px-5 py-6"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="mx-auto max-w-[1100px]">
          <p
            className="text-xs font-semibold uppercase tracking-wider"
            style={{ color: "var(--muted-foreground)" }}
          >
            Working paper
          </p>
          <h1
            className="mt-1 font-sans text-2xl font-semibold tracking-tight"
            style={{ color: "var(--foreground)" }}
          >
            How large language models answer questions about economic
            elasticities
          </h1>
          <p
            className="mt-1.5 max-w-3xl text-sm leading-relaxed"
            style={{ color: "var(--muted-foreground)" }}
          >
            A repeated-elicitation study of prompt-conditioned response
            distributions: 31 models from ten organizations, 15 runs per
            model-quantity cell over 26 US-scoped quantities, pooled into
            predictive distributions and mapped through a fixed optimal-tax
            calibration. This page embeds the manuscript snapshot that
            matches the live site — the 31-model panel with capability
            correlates pinned to PolicyBench release dashboard-data-20260805
            — and every number in it is verified against the committed
            tables by a 131-check prose gate.
          </p>
          <p className="mt-3">
            <span
              className="rounded-full border px-3 py-1 text-xs"
              style={{
                borderColor: "var(--border)",
                color: "var(--muted-foreground)",
              }}
            >
              31-model panel · 2026-08-08 · Max Ghenis, PolicyEngine
            </span>
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <ActionLink href={STANDALONE_HREF} primary external>
              Open standalone HTML
            </ActionLink>
            <ActionLink href={PDF_HREF} external>
              Download PDF
            </ActionLink>
            <ActionLink href={`${basePath}/`}>Live AI beliefs</ActionLink>
            <ActionLink
              href="https://github.com/PolicyEngine/llm-econ-beliefs"
              external
            >
              Code and artifacts
            </ActionLink>
          </div>
        </div>
      </div>

      <div className="px-5 py-6">
        <div className="mx-auto max-w-[1100px]">
          <div
            className="overflow-hidden rounded-lg border"
            style={{ borderColor: "var(--border)" }}
          >
            <iframe
              src={STANDALONE_HREF}
              title="How large language models answer questions about economic elasticities — manuscript"
              loading="lazy"
              sandbox="allow-same-origin allow-popups allow-popups-to-escape-sandbox"
              referrerPolicy="same-origin"
              className="w-full"
              style={{
                height: "calc(100vh - 16rem)",
                minHeight: 720,
                background: "#fff",
                border: 0,
              }}
            />
          </div>
          <p
            className="mt-3 text-sm"
            style={{ color: "var(--muted-foreground)" }}
          >
            <a className="underline underline-offset-2" href="#top">
              ↑ Back to top
            </a>{" "}
            ·{" "}
            <a
              className="underline underline-offset-2"
              href={STANDALONE_HREF}
              target="_blank"
              rel="noopener"
            >
              Open manuscript in a new page
            </a>
          </p>
        </div>
      </div>

      <ProvenanceFooter runCount={runCount} />
    </main>
  );
}
