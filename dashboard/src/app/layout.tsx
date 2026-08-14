import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Script from "next/script";
import { PolicyEngineShell } from "@policyengine/ui-kit";
import { SubNav } from "@/components/site-chrome";
import { getSummaryData } from "@/lib/site-data";
import "./globals.css";

// Matches the GA4 property used by policyengine-app-v2/website so
// page_view events from this multizone land in the same account as the
// rest of policyengine.org traffic.
const GA_MEASUREMENT_ID = "G-2YHG89FY0N";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export function generateMetadata(): Metadata {
  const modelCount = getSummaryData().stats.modelCount;
  return {
    metadataBase: new URL("https://policyengine.org"),
    title: "AI beliefs · PolicyEngine",
    description: `How ${modelCount} frontier language models answer when asked for their beliefs about economic elasticities: point estimates, uncertainty bands, and run-level responses.`,
    alternates: { canonical: basePath || "/" },
    openGraph: {
      title: "AI beliefs about economic parameters",
      description: `Elicited beliefs from ${modelCount} frontier models on 26 economic quantities, with pooled, REML, and Bayesian uncertainty bands.`,
      url: basePath || "/",
      siteName: "PolicyEngine",
    },
    icons: { icon: `${basePath}/favicon.svg` },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const modelCount = getSummaryData().stats.modelCount;
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${jetbrains.variable} h-full antialiased`}
    >
      <head>
        <Script
          src={`https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`}
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            window.gtag = gtag;
            gtag('js', new Date());
            gtag('config', '${GA_MEASUREMENT_ID}');
          `}
        </Script>
      </head>
      <body className="min-h-full flex flex-col">
        <PolicyEngineShell country="us" showFooter mainClassName="flex-1">
          <SubNav />
          <div className="flex-1">{children}</div>
          <div
            className="border-t px-5 py-3 text-xs"
            style={{
              borderColor: "var(--border)",
              color: "var(--muted-foreground)",
            }}
          >
            <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-2">
              <a
                href="https://github.com/PolicyEngine/llm-econ-beliefs"
                className="underline-offset-2 hover:underline"
              >
                Code and data
              </a>
              <span>
                Elicited April through August 2026 · {modelCount} models · v4 prompts
              </span>
            </div>
          </div>
        </PolicyEngineShell>
      </body>
    </html>
  );
}
