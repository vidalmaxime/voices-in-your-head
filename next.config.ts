import type { NextConfig } from "next";

const basePath = "/voices";

const nextConfig: NextConfig = {
  // Use Turbopack (Next.js 16 default)
  turbopack: {},
  basePath,
  // Exposed so client code can build URLs for files served from public/.
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "cross-origin-embedder-policy", value: "require-corp" },
          { key: "cross-origin-opener-policy", value: "same-origin" },
          { key: "cross-origin-resource-policy", value: "cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
