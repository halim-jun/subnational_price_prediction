import type { NextConfig } from "next";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const STATIC_MODE = process.env.NEXT_PUBLIC_STATIC_MODE === "true";

const nextConfig: NextConfig = STATIC_MODE
  ? {
      output: "export",
      images: { unoptimized: true },
      trailingSlash: true,
    }
  : {
      async rewrites() {
        return [
          {
            source: "/api/:path*",
            destination: `${API_URL}/api/:path*`,
          },
        ];
      },
    };

export default nextConfig;
