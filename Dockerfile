FROM python:3.11-slim

# Install dependances for audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependance files
COPY pyproject.toml uv.lock ./

# Install python dependance with uv
RUN uv sync --frozen --no-dev

# Copy the rest of the code
COPY . .

# Execute the API using the venv made by uv
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
