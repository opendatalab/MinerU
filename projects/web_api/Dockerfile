FROM python:3.10-slim-bookworm AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1


FROM base AS build

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Build Python dependencies
COPY requirements.txt .
RUN python -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    find /app/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Download models
RUN . /app/venv/bin/activate && \
    export HF_HOME=/opt/models && \
    python -m mineru.cli.models_download

FROM base AS prod

# Copy Python dependencies and models from the build stage
COPY --from=build /app/venv /app/venv
COPY --from=build /opt/models /opt/models
COPY --from=build /opt/layoutreader /opt/layoutreader

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy the app and its configuration file
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh
COPY magic-pdf.json /root/magic-pdf.json
COPY app.py /app/app.py
COPY rabbitmq_consumer.py /app/rabbitmq_consumer.py

# Expose the port that FastAPI will run on (keeping for potential future use)
EXPOSE 8000

# Command to run RabbitMQ consumer instead of FastAPI
ENTRYPOINT [ "/app/entrypoint.sh" ]
CMD ["python", "rabbitmq_consumer.py"]
