# Build context is the plot-hole-finder/ directory (so the sibling engine/ is
# available). Build with: docker build -f server/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first for layer caching.
COPY server/requirements.txt server/requirements-prod.txt /app/server/
RUN pip install --no-cache-dir -r /app/server/requirements.txt -r /app/server/requirements-prod.txt

# App code: the engine is a sibling package imported via glyphh_bridge.
COPY engine /app/engine
COPY server /app/server

WORKDIR /app/server
EXPOSE 8000

# API by default; the worker service overrides this with `python -m app.worker`.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
