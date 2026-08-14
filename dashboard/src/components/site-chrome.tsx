import type { ReactNode } from "react";

import { NavLinks } from "@/components/nav-links";

const NAV_ITEMS = [
  { href: "/", label: "Overview" },
  { href: "/quantities", label: "Quantities" },
  { href: "/models", label: "Models" },
  { href: "/top-rates", label: "Top rates" },
  { href: "/generations", label: "Generations" },
  { href: "/process", label: "Process" },
  { href: "/methods", label: "Methods" },
  { href: "/paper", label: "Paper" },
] as const;

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export function SubNav(): ReactNode {
  return (
    <nav
      className="border-b px-5"
      style={{ borderColor: "var(--border)", background: "var(--card)" }}
      aria-label="AI beliefs sections"
    >
      <div className="mx-auto flex max-w-[1100px] flex-wrap items-center gap-1 py-2">
        <span
          className="mr-3 text-sm font-semibold"
          style={{ color: "var(--foreground)" }}
        >
          AI beliefs
        </span>
        <NavLinks items={NAV_ITEMS} />
      </div>
    </nav>
  );
}

interface ProvenanceFooterProps {
  runCount: number;
  extra?: ReactNode;
}

export function ProvenanceFooter({
  runCount,
  extra,
}: ProvenanceFooterProps): ReactNode {
  return (
    <div
      className="mt-10 border-t pt-4 text-xs leading-relaxed"
      style={{ borderColor: "var(--border)", color: "var(--muted-foreground)" }}
    >
      <p>
        {runCount.toLocaleString()} successful runs · elicited April through August
        2026 · v4 prompts · 15 runs per model-quantity cell.{" "}
        <a
          className="underline underline-offset-2"
          href="https://github.com/PolicyEngine/llm-econ-beliefs"
        >
          Code
        </a>
        {" · "}
        <a
          className="underline underline-offset-2"
          href="https://github.com/PolicyEngine/llm-econ-beliefs/tree/main/results"
        >
          Raw responses
        </a>
        {" · "}
        <a
          className="underline underline-offset-2"
          href={`${basePath}/paper.pdf`}
        >
          Paper (PDF)
        </a>
      </p>
      {extra}
    </div>
  );
}

interface PageBandProps {
  title: string;
  lede: ReactNode;
  aside?: ReactNode;
}

export function PageBand({ title, lede, aside }: PageBandProps): ReactNode {
  return (
    <div
      className="border-b px-5 py-6"
      style={{ borderColor: "var(--border)" }}
    >
      <div className="mx-auto flex max-w-[1100px] flex-wrap items-end justify-between gap-4">
        <div>
          <h1
            className="font-sans text-2xl font-semibold tracking-tight"
            style={{ color: "var(--foreground)" }}
          >
            {title}
          </h1>
          <p
            className="mt-1.5 max-w-2xl text-sm leading-relaxed"
            style={{ color: "var(--muted-foreground)" }}
          >
            {lede}
          </p>
        </div>
        {aside}
      </div>
    </div>
  );
}
