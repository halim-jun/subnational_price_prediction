import type { NextConfig } from "next";

// Optional: deploy under a sub-path (e.g. "/dashboard") by setting
// NEXT_PUBLIC_BASE_PATH at build time. Empty → root.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
  basePath: basePath || undefined,
  assetPrefix: basePath || undefined,
};

export default nextConfig;
