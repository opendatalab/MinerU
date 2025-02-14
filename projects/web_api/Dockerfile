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
    rm -rf /var/lib/apt/lists/*

# Build Python dependencies
COPY requirements.txt .
RUN python -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install -r requirements.txt && \
    pip uninstall -y paddlepaddle && \
    pip install -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
        paddlepaddle-gpu==3.0.0rc1

# Download models
COPY download_models.py .
RUN . /app/venv/bin/activate && \
    ./download_models.py


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
    rm -rf /var/lib/apt/lists/*

# Create volume for paddleocr models
RUN mkdir -p /root/.paddleocr
VOLUME [ "/root/.paddleocr" ]

# Copy the app and its configuration file
COPY entrypoint.sh /app/entrypoint.sh
COPY magic-pdf.json /root/magic-pdf.json
COPY app.py /app/app.py

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run FastAPI using Uvicorn, pointing to app.py and binding to 0.0.0.0:8000
ENTRYPOINT [ "/app/entrypoint.sh" ]
CMD ["--host", "0.0.0.0", "--port", "8000"]
