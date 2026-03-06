# Frontend: AI Shorts Creator (NiceGUI)
FROM python:3.12-slim

# System deps: ffmpeg for video processing, yt-dlp needs it too
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
# Pin uv to Python 3.13 — litellm is broken on 3.14
ENV UV_PYTHON=3.13

# Copy root project files (deps are in root pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync --frozen --no-cache --python 3.13

# Copy the full project (src/api/main.py for MCP server + src/app/ for frontend)
COPY . .

EXPOSE 3000

# Run the frontend — NiceGUI binds to 0.0.0.0:3000
CMD ["uv", "run", "src/app/main.py"]
