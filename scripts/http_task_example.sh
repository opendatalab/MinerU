#!/usr/bin/env bash
# tasks HTTP 示例：默认预检后逐文件上传，轮询状态并下载服务返回的 ZIP 链接。
# SEND_HASH=0 可省略 hash；TASK_ID=... 恢复原任务；UPLOAD_MODE=direct 仅允许单文件。
# 成功退出 0，部分成功退出 2，失败退出 1，轮询预算耗尽退出 124。
set -euo pipefail
TASK_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${MINERU_PYTHON:-python3}" "$TASK_SCRIPT_DIR/http_task_example.py" "$@"
