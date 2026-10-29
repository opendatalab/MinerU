#!/usr/bin/env bash
# 统一 HTTP 示例复用 Tasks 客户端；参数、恢复和退出码见 http_task_example.py。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/http_task_example.sh" "$@"
