# TS app image (Fastify API + MCP server + BullMQ worker). LibreOffice is the
# universal fallback converter for office documents (.doc/.odt/.rtf/.ppt(x)/.xls(x)).
FROM node:22-bookworm-slim

ENV NODE_ENV=production
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer libreoffice-calc libreoffice-impress \
    && rm -rf /var/lib/apt/lists/* \
    && corepack enable

# Install deps first for layer caching (keep dev deps: drizzle-kit/tsx run at runtime).
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .

EXPOSE 3000
# Run migrations, then start the API server (tsx, no build step). The worker
# service overrides this command with `pnpm worker`.
CMD ["sh", "-c", "pnpm db:migrate && pnpm start"]
