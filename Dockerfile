# ─── Stage 1: Build React Frontend ───────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/front_end
COPY front_end/package*.json ./
RUN npm ci
COPY front_end/ ./
RUN npm run build

# ─── Stage 2: Python Backend ──────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies for llama-cpp-python (CPU build inside Docker)
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python (CPU-only for Docker; GPU requires special base image)
RUN pip install --no-cache-dir llama-cpp-python

# Copy source code
COPY api/        ./api/
COPY database/   ./database/
COPY monitor/    ./monitor/
COPY src/        ./src/
COPY system_prompt.txt .
COPY main.py     .

# Copy built React frontend into FastAPI static dir
COPY --from=frontend-builder /app/front_end/dist ./front_end/dist

# Serve static frontend via FastAPI (mount at /static)
# Runtime: mount LLM/ and data/ as Docker volumes

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["python", "main.py", "--api"]
