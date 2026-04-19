import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  ...(process.env.VERCEL
    ? {}
    : { outputFileTracingRoot: path.join(__dirname, "..") }),
};

export default nextConfig;
