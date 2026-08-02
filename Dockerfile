# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build the React/Vite reviewer UI into ui/dist
# ---------------------------------------------------------------------------
FROM node:20-slim AS ui-build
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — Python runtime that serves the API + the built UI
# ---------------------------------------------------------------------------
# Pinned to 3.12: torch / sentence-transformers ship CPU wheels for it, so the
# build stays fast and avoids compiling from source.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the sentence-transformers / HuggingFace model cache in a stable,
    # bakeable location so the model is embedded in the image (no cold-start
    # download on Render).
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Python dependencies first, for layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download the embedding model at build time so the first request is fast
# and the container needs no network to warm up.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Application source.
COPY . .

# Overwrite any stale local ui/dist with the freshly built assets from stage 1.
COPY --from=ui-build /ui/dist ./ui/dist

# Audit log directory (audit_logger also mkdir's it, but be explicit).
RUN mkdir -p logs

EXPOSE 8000

# Render injects $PORT; bind to it, falling back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
