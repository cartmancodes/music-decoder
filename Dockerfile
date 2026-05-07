FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libfluidsynth3 \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN pip install --upgrade pip && pip install -e ".[dev]"

COPY . /app/
RUN python -m music_decoder.cli.main doctor || true

EXPOSE 8501
CMD ["python", "-m", "music_decoder.cli.main", "ui"]
