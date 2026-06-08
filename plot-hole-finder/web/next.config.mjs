/** @type {import('next').NextConfig} */
const nextConfig = {
  // The Claude Agent SDK runs in a Node subprocess; keep it external to the bundle.
  serverExternalPackages: ["@anthropic-ai/claude-agent-sdk"],
};

export default nextConfig;
