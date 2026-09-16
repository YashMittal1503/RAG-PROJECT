import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow images from Supabase Storage if needed in the future
  images: {
    remotePatterns: [],
  },
};

export default nextConfig;
