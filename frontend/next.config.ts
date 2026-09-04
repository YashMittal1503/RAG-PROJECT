import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow images from Supabase Storage if needed in the future
  images: {
    remotePatterns: [],
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
