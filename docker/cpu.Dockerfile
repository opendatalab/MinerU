FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir "mineru[core]>=3.0.0"

ENV MINERU_MODEL_SOURCE=local

EXPOSE 8000

ENTRYPOINT ["mineru-api"]
