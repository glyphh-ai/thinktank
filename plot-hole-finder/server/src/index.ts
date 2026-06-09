// API server entrypoint: `pnpm start` / `pnpm dev`. Boots Fastify (REST + agent
// SSE + the TRUE MCP server). Background analyze/fix jobs run in the separate
// worker process (`pnpm worker`).

import { buildApp } from "./http/app";

const PORT = Number(process.env.PORT ?? "3000");
const HOST = process.env.HOST ?? "0.0.0.0";

async function main() {
  const app = await buildApp();

  const shutdown = async (signal: string) => {
    app.log.info(`received ${signal}, shutting down`);
    await app.close();
    process.exit(0);
  };
  process.on("SIGTERM", () => void shutdown("SIGTERM"));
  process.on("SIGINT", () => void shutdown("SIGINT"));

  await app.listen({ port: PORT, host: HOST });
}

main().catch((err) => {
  console.error("failed to start server:", err);
  process.exit(1);
});
