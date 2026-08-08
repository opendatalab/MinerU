#!/usr/bin/env bash
# MinerU WebUI 管理脚本（macOS Apple Silicon）
# 用法: ./mineru-webui.sh {start|stop|restart|status|logs}

set -euo pipefail

APP_NAME="mineru-webui"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${MINERU_WEBUI_PORT:-7866}"
HOST="${MINERU_WEBUI_HOST:-127.0.0.1}"
DEVICE="${MINERU_DEVICE_MODE:-mps}"
PID_FILE="$REPO_ROOT/.$APP_NAME.pid"
LOG_FILE="$REPO_ROOT/$APP_NAME.log"
LAUNCHER="$REPO_ROOT/.venv/bin/mineru-gradio"

read_pid() {
    if [[ -f "$PID_FILE" ]]; then
        cat "$PID_FILE" 2>/dev/null || true
    fi
}

is_alive() {
    local pid
    pid="$(read_pid)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

port_pids() {
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | sort -u || true
}

start() {
    if is_alive; then
        echo "[$APP_NAME] 已在运行 (pid $(read_pid))，访问 http://$HOST:$PORT"
        return 0
    fi
    local existing
    existing="$(port_pids | tr '\n' ' ')"
    if [[ -n "$existing" ]]; then
        echo "[$APP_NAME] 端口 $PORT 已被其他实例占用 (pid: $existing)，请先执行 stop 或 restart"
        return 1
    fi
    cd "$REPO_ROOT"
    # 长文档（如数百页 PDF）转换可能超过默认 1 小时，这里放宽任务等待/下载超时，
    # 可通过同名环境变量覆盖。
    MINERU_DEVICE_MODE="$DEVICE" \
    MINERU_TASK_RESULT_TIMEOUT_SECONDS="${MINERU_TASK_RESULT_TIMEOUT_SECONDS:-14400}" \
    MINERU_TASK_RESULT_DOWNLOAD_TIMEOUT_SECONDS="${MINERU_TASK_RESULT_DOWNLOAD_TIMEOUT_SECONDS:-1800}" \
    nohup "$LAUNCHER" --server-port "$PORT" --server-name "$HOST" >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "[$APP_NAME] 已启动 (pid $(read_pid))，日志: $LOG_FILE"
    echo "[$APP_NAME] 访问 http://$HOST:$PORT"
}

stop() {
    local stopped=0 pid pids p
    pid="$(read_pid)"
    if is_alive; then
        kill "$pid" 2>/dev/null || true
        stopped=1
    fi
    pids="$(port_pids)"
    for p in $pids; do
        if [[ "$p" != "$pid" ]]; then
            kill "$p" 2>/dev/null || true
            stopped=1
        fi
    done
    # 等待进程退出（最多 10 秒），必要时强制结束
    for p in $pid $pids; do
        [[ -n "$p" ]] || continue
        for _ in $(seq 1 20); do
            kill -0 "$p" 2>/dev/null || break
            sleep 0.5
        done
        kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null || true
    done
    rm -f "$PID_FILE"
    if [[ "$stopped" -eq 1 ]]; then
        echo "[$APP_NAME] 已停止"
    else
        echo "[$APP_NAME] 没有运行中的实例"
    fi
}

restart() {
    stop
    sleep 1
    start
}

status() {
    local pid pids
    pid="$(read_pid)"
    pids="$(port_pids | tr '\n' ' ')"
    if is_alive; then
        echo "[$APP_NAME] 运行中: pid $pid，端口 $PORT"
    elif [[ -n "$pids" ]]; then
        echo "[$APP_NAME] 端口 $PORT 有实例在运行 (pid: $pids)，但 pidfile 缺失或过期"
    else
        echo "[$APP_NAME] 未运行"
    fi
    echo "[$APP_NAME] 日志: $LOG_FILE"
}

logs() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "[$APP_NAME] 日志文件不存在: $LOG_FILE"
        return 1
    fi
    tail -n 100 "$LOG_FILE"
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    restart) restart ;;
    status) status ;;
    logs) logs ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs}"
        echo "环境变量: MINERU_WEBUI_PORT(默认7866) MINERU_WEBUI_HOST(默认127.0.0.1) MINERU_DEVICE_MODE(默认mps)"
        exit 1
        ;;
esac
