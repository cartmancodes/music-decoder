# Dockerfile — secondary install path; matches the local install but containerized.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libfluidsynth3 libchromaprint1 \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --upgrade pip && pip install -e ".[dev]"

COPY . /app/
RUN python -c "from music_decoder.cli.models_download import warm_models; warm_models()" || \
    echo "warmup failed; models will fetch on first run"

EXPOSE 8501
CMD ["python", "-m", "music_decoder.cli.main"]
