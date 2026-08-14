import fs from "node:fs";
import path from "node:path";
import type { NextConfig } from "next";
import { parse as parseCsv } from "csv-parse/sync";

// Served as a policyengine.org zone: the host app rewrites
// /ai-beliefs/:path* to this deployment, so every route and asset lives
// under the base path.
const basePath = "/ai-beliefs";

function loadClientModelRegistry(): string {
  const localPath = path.resolve(__dirname, "..", "results", "model-registry.csv");
  const bundledPath = path.resolve(__dirname, "results", "model-registry.csv");
  const registryPath = fs.existsSync(localPath) ? localPath : bundledPath;
  if (!fs.existsSync(registryPath)) {
    throw new Error(`Missing required model registry: ${registryPath}`);
  }
  const rows = parseCsv(fs.readFileSync(registryPath, "utf-8"), {
    columns: true,
    skip_empty_lines: true,
  }) as Record<string, string>[];
  return JSON.stringify(
    rows.map((row) => ({
      modelId: row.model_id,
      displayLabel: row.display_label,
      organization: row.organization,
    })),
  );
}

const nextConfig: NextConfig = {
  basePath,
  // /paper.pdf predates the /paper wrapper and has been shared; keep it
  // serving by pointing it at the synced render.
  async redirects() {
    return [
      {
        source: "/paper.pdf",
        destination: "/paper/web/index.pdf",
        permanent: false,
      },
    ];
  },
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
    NEXT_PUBLIC_MODEL_REGISTRY_JSON: loadClientModelRegistry(),
  },
  // Every page is statically generated at build time; results/ and
  // tables/ are read only during the build, so nothing needs runtime
  // file tracing.
  ...(process.env.VERCEL
    ? {}
    : { outputFileTracingRoot: path.join(__dirname, "..") }),
};

export default nextConfig;
