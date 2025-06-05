FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Configure pip to use Alibaba Cloud mirror
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Install dependencies
RUN pip install --no-cache-dir poetry

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install the package
RUN poetry config virtualenvs.create false && \
    poetry install

# Create downloads directory
RUN mkdir -p /app/downloads

# Set environment variables
ENV OUTPUT_DIR=/app/downloads
# MINERU_API_KEY should be provided at runtime
ENV MINERU_API_BASE=https://mineru.net
ENV USE_LOCAL_API=false
ENV LOCAL_MINERU_API_BASE=""

# Expose the port that SSE will run on
EXPOSE 8001

# Set command to start the service with SSE transport
CMD ["mineru-mcp", "--transport", "sse", "--output-dir", "/app/downloads"] 