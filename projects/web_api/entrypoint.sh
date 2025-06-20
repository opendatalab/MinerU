#!/usr/bin/env bash
set -euo pipefail

. /app/venv/bin/activate
export HF_HOME=/opt/models
exec uvicorn app:app "$@"
