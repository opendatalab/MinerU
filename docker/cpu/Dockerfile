# MinerU API Server - CPU-only deployment
# For environments without GPU (cloud VMs, ARM servers, etc.)
# Uses the pipeline backend instead of vllm-engine
#
# Build:   docker compose -f cpu-compose.yaml build
# Run:     docker compose -f cpu-compose.yaml up -d

FROM python:3.12-slim

RUN pip install --no-cache-dir uv && \
    uv pip install --system -U "mineru[all]"

EXPOSE 8888
CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8888"]
