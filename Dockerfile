FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for tensorflow / h5py / pyarrow wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer).
# pytest + httpx are needed so `docker compose exec api pytest` works out of the box.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install . pytest pytest-asyncio httpx

# App code + trained model + tests (tests are mounted at runtime via compose,
# but COPY here keeps the image self-contained).
COPY src ./src
COPY models ./models
COPY tests ./tests

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
