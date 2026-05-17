# Copyright (c) Opendatalab. All rights reserved.

import base64
import asyncio
import html as html_lib
import httpx
import os
import re
import sys
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import click
import gradio as gr
from gradio_pdf import PDF
from loguru import logger

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
# 检测 Gradio 版本，用于兼容 Gradio 5 和 Gradio 6
_gradio_major_version = int(gr.__version__.split('.')[0])
IS_GRADIO_6 = _gradio_major_version >= 6

log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()  # 移除默认handler
logger.add(sys.stderr, level=log_level)  # 添加新handler

from mineru.cli.common import (
    docx_suffixes,
    image_suffixes,
    normalize_task_stem,
    office_suffixes,
    pdf_suffixes,
    read_fn,
)
from mineru.cli import api_client as _api_client
from mineru.cli.output_paths import resolve_parse_dir
from mineru.cli.vlm_preload import resolve_gradio_local_api_cli_args
from mineru.cli.visualization import VisualizationJob, run_visualization_job

_gradio_local_api_server = _api_client.ReusableLocalAPIServer()


@dataclass(frozen=True)
class GradioConcurrencyWaitSnapshot:
    limit: int
    active: int
    waiting: int
    ahead: int


@dataclass
class _LimiterState:
    semaphore: asyncio.Semaphore
    active: int = 0
    waiters: list[object] = field(default_factory=list)


class GradioRequestConcurrencyLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._states: dict[int, _LimiterState] = {}

    def _get_state(self, limit: int):
        if limit <= 0:
            return None
        with self._lock:
            state = self._states.get(limit)
            if state is None:
                state = _LimiterState(semaphore=asyncio.Semaphore(limit))
                self._states[limit] = state
            return state

    def _build_wait_snapshot(
        self,
        state: _LimiterState,
        limit: int,
        wait_token: object,
    ) -> GradioConcurrencyWaitSnapshot | None:
        if wait_token not in state.waiters:
            return None

        return GradioConcurrencyWaitSnapshot(
            limit=limit,
            active=state.active,
            waiting=len(state.waiters),
            ahead=state.waiters.index(wait_token),
        )

    def _remove_waiter(self, state: _LimiterState, wait_token: object) -> None:
        if wait_token in state.waiters:
            state.waiters.remove(wait_token)

    async def _cleanup_acquire_interruption(
        self,
        state: _LimiterState,
        acquire_task: asyncio.Task[bool],
        wait_token: object,
        should_wait: bool,
    ) -> None:
        if not acquire_task.done():
            acquire_task.cancel()
            await asyncio.gather(acquire_task, return_exceptions=True)
        elif not acquire_task.cancelled():
            try:
                acquired = acquire_task.result()
            except Exception:
                acquired = False
            if acquired:
                state.semaphore.release()

        if should_wait:
            with self._lock:
                self._remove_waiter(state, wait_token)

    @asynccontextmanager
    async def acquire(
        self,
        limit: int,
        on_wait: Callable[[GradioConcurrencyWaitSnapshot], None] | None = None,
    ):
        state = self._get_state(limit)
        if state is None:
            yield
            return

        wait_token = object()
        should_wait = False
        snapshot = None
        with self._lock:
            if state.active >= limit or state.waiters:
                state.waiters.append(wait_token)
                should_wait = True
                snapshot = self._build_wait_snapshot(state, limit, wait_token)

        acquire_task: asyncio.Task[bool] = asyncio.create_task(state.semaphore.acquire())
        last_wait_ahead = None
        if should_wait and on_wait is not None and snapshot is not None:
            on_wait(snapshot)
            last_wait_ahead = snapshot.ahead

        try:
            if should_wait:
                while True:
                    done, _ = await asyncio.wait(
                        {acquire_task},
                        timeout=STATUS_TIMER_INTERVAL_SECONDS,
                    )
                    if acquire_task in done:
                        acquire_task.result()
                        break

                    if on_wait is None:
                        continue

                    with self._lock:
                        snapshot = self._build_wait_snapshot(state, limit, wait_token)

                    if snapshot is None or snapshot.ahead == last_wait_ahead:
                        continue

                    on_wait(snapshot)
                    last_wait_ahead = snapshot.ahead
            else:
                await acquire_task
        except BaseException:
            await self._cleanup_acquire_interruption(
                state=state,
                acquire_task=acquire_task,
                wait_token=wait_token,
                should_wait=should_wait,
            )
            raise

        with self._lock:
            if should_wait:
                self._remove_waiter(state, wait_token)
            state.active += 1
        try:
            yield
        finally:
            with self._lock:
                state.active = max(0, state.active - 1)
            state.semaphore.release()


_gradio_request_concurrency_limiter = GradioRequestConcurrencyLimiter()

STATUS_BOX_AUTOSCROLL_JS = """
(value) => {
    const scrollToBottom = () => {
        const textarea = document.querySelector(".convert-status-box textarea");
        if (!textarea) {
            return;
        }
        textarea.scrollTop = textarea.scrollHeight;
    };

    requestAnimationFrame(() => {
        scrollToBottom();
        requestAnimationFrame(scrollToBottom);
    });

    return [];
}
"""

STATUS_TIMER_INTERVAL_SECONDS = 0.1
STATUS_QUEUE_ANIMATION_INTERVAL_SECONDS = 1.0
STATUS_QUEUE_ANIMATION_MAX_DOTS = 10

STATUS_PREPARING_REQUEST = "Preparing request..."
STATUS_CHECKING_SERVER = "Checking server status..."
STATUS_SUBMITTING_TASK = "Submitting task..."
STATUS_DOWNLOADING_RESULT = "Task completed, downloading result..."
STATUS_PROCESSING_OUTPUT = "Preparing outputs..."
STATUS_COMPLETED = "Completed"
STATUS_QUEUED_ON_SERVER = "Queued on server"
STATUS_PROCESSING_ON_SERVER = "Processing on server"
STATUS_QUEUED_LOCALLY_PREFIX = "Queued locally:"

BACKEND_CHOICE_DEFINITIONS = [
    "pipeline",
    "vlm-auto-engine",
    "hybrid-auto-engine",
]
HTTP_CLIENT_BACKEND_CHOICE_DEFINITIONS = [
    "vlm-http-client",
    "hybrid-http-client",
]
STATUS_STEP_DEFINITIONS = [
    ("status_step_prepare", STATUS_PREPARING_REQUEST),
    ("status_step_check", STATUS_CHECKING_SERVER),
    ("status_step_submit", STATUS_SUBMITTING_TASK),
    ("status_step_queue", STATUS_QUEUED_ON_SERVER),
    ("status_step_process", STATUS_PROCESSING_ON_SERVER),
    ("status_step_download", STATUS_DOWNLOADING_RESULT),
    ("status_step_outputs", STATUS_PROCESSING_OUTPUT),
    ("status_step_done", STATUS_COMPLETED),
]


def translate_ui(i18n, key):
    """读取自定义 HTML/选项需要的纯文本文案，避免直接渲染 Gradio I18nData 元数据。"""
    if i18n is None:
        return key
    translations = getattr(i18n, "translations", None)
    if translations:
        preferred_locale = os.getenv("MINERU_GRADIO_DEFAULT_LOCALE", "zh")
        preferred_text = translations.get(preferred_locale, {}).get(key)
        if preferred_text is not None:
            return preferred_text
        fallback_text = translations.get("en", {}).get(key)
        if fallback_text is not None:
            return fallback_text
        return key
    return i18n(key)


def build_backend_choices(http_client_enable, i18n):
    """构建后端选项列表，展示文案与提交给后端的 backend 值保持完全一致。"""
    choices = list(BACKEND_CHOICE_DEFINITIONS)
    if http_client_enable:
        choices.extend(HTTP_CLIENT_BACKEND_CHOICE_DEFINITIONS)
    return choices


def resolve_status_step_index(status_lines):
    """根据现有状态日志推断步骤面板中当前应高亮的步骤索引。"""
    if not status_lines:
        return -1, False
    if status_lines[-1].startswith("Failed:"):
        return len(STATUS_STEP_DEFINITIONS) - 1, True
    if any(line.startswith(STATUS_COMPLETED) for line in status_lines):
        return len(STATUS_STEP_DEFINITIONS) - 1, False

    for index in range(len(STATUS_STEP_DEFINITIONS) - 1, -1, -1):
        _, marker = STATUS_STEP_DEFINITIONS[index]
        if marker == STATUS_QUEUED_ON_SERVER:
            if any(StatusPanelState.is_queue_message(line) for line in status_lines):
                return index, False
            continue
        if any(line.startswith(marker) for line in status_lines):
            return index, False
    return 0, False


def render_status_steps_html(status_text, i18n):
    """把流式状态日志渲染为步骤式状态面板，底层日志格式保持不变。"""
    status_lines = [line for line in str(status_text or "").splitlines() if line]
    current_index, is_failed = resolve_status_step_index(status_lines)
    latest_status = status_lines[-1] if status_lines else translate_ui(i18n, "status_idle_hint")

    step_items = []
    for index, (label_key, _) in enumerate(STATUS_STEP_DEFINITIONS):
        classes = ["status-step"]
        if is_failed and index == current_index:
            classes.extend(["is-active", "is-error"])
            label = translate_ui(i18n, "status_step_failed")
        elif index < current_index or (current_index == len(STATUS_STEP_DEFINITIONS) - 1 and not is_failed):
            classes.append("is-done")
            label = translate_ui(i18n, label_key)
        elif index == current_index:
            classes.append("is-active")
            label = translate_ui(i18n, label_key)
        else:
            classes.append("is-pending")
            label = translate_ui(i18n, label_key)
        step_items.append(
            f'<div class="{" ".join(classes)}">'
            f'<span class="status-dot"></span>'
            f'<span class="status-label">{html_lib.escape(label)}</span>'
            "</div>"
        )

    title = translate_ui(i18n, "status_idle_title") if not status_lines else translate_ui(i18n, "status_latest")
    return (
        '<div class="status-steps-panel">'
        f'<div class="status-panel-title">{html_lib.escape(title)}</div>'
        f'<div class="status-steps-list">{"".join(step_items)}</div>'
        f'<div class="status-latest">{html_lib.escape(latest_status)}</div>'
        "</div>"
    )


APP_CSS = """
.gradio-container {
    --mineru-accent: #f97316;
    --mineru-panel: rgba(255, 255, 255, 0.62);
    --mineru-panel-border: rgba(17, 24, 39, 0.10);
    --mineru-status-panel-bg: rgba(255, 255, 255, 0.78);
    --mineru-status-latest-border: rgba(17, 24, 39, 0.13);
    --mineru-status-latest-text: #374151;
    --mineru-popover-border: rgba(17, 24, 39, 0.12);
    --mineru-popover-shell-bg: rgba(255, 255, 255, 0.88);
    --mineru-popover-card-bg: rgba(255, 255, 255, 0.96);
    --mineru-popover-shadow: 0 18px 42px rgba(15, 23, 42, 0.18), 0 0 0 1px rgba(17, 24, 39, 0.04) inset;
    --mineru-popover-dropdown-shadow: 0 14px 30px rgba(15, 23, 42, 0.16);
}
@media (prefers-color-scheme: light) {
    .gradio-container {
        --mineru-panel: rgba(255, 255, 255, 0.62);
        --mineru-panel-border: rgba(17, 24, 39, 0.10);
        --mineru-status-panel-bg: rgba(255, 255, 255, 0.78);
        --mineru-status-latest-border: rgba(17, 24, 39, 0.13);
        --mineru-status-latest-text: #374151;
        --mineru-popover-border: rgba(17, 24, 39, 0.12);
        --mineru-popover-shell-bg: rgba(255, 255, 255, 0.88);
        --mineru-popover-card-bg: rgba(255, 255, 255, 0.96);
        --mineru-popover-shadow: 0 18px 42px rgba(15, 23, 42, 0.18), 0 0 0 1px rgba(17, 24, 39, 0.04) inset;
        --mineru-popover-dropdown-shadow: 0 14px 30px rgba(15, 23, 42, 0.16);
    }
}
body.dark .gradio-container {
    --mineru-panel: rgba(255, 255, 255, 0.045);
    --mineru-panel-border: rgba(255, 255, 255, 0.1);
    --mineru-status-panel-bg: rgba(0, 0, 0, 0.18);
    --mineru-status-latest-border: var(--mineru-panel-border);
    --mineru-status-latest-text: var(--body-text-color-subdued);
    --mineru-popover-border: rgba(255, 255, 255, 0.08);
    --mineru-popover-shell-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.20), rgba(255, 255, 255, 0.05) 44%, rgba(249, 115, 22, 0.16));
    --mineru-popover-card-bg: linear-gradient(180deg, rgba(42, 43, 48, 0.97), rgba(32, 33, 36, 0.94));
    --mineru-popover-shadow: 0 18px 42px rgba(0, 0, 0, 0.34), 0 0 0 1px rgba(255, 255, 255, 0.04) inset;
    --mineru-popover-dropdown-shadow: 0 14px 30px rgba(0, 0, 0, 0.36);
}
.mineru-shell {
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.mineru-workspace-row {
    align-items: stretch;
}
.mineru-control-column,
.mineru-work-column {
    border: 1px solid var(--mineru-panel-border);
    border-radius: 8px;
    background: var(--mineru-panel);
    padding: 12px;
}
.mineru-control-column {
    position: sticky;
    top: 12px;
    align-self: flex-start;
}
.mineru-upload-file label[data-testid="block-label"] {
    align-items: flex-start !important;
    line-height: 1.35 !important;
    white-space: pre-line !important;
}
.mineru-upload-file label[data-testid="block-label"] > span {
    margin-top: 2px;
}
#mineru-example-files label[data-testid="block-label"] {
    display: none !important;
}
.mineru-actions {
    flex-wrap: nowrap !important;
    gap: 8px !important;
}
.mineru-actions > * {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
.mineru-actions button {
    min-width: 0 !important;
    width: 100% !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
}
.mineru-advanced-open,
.mineru-advanced-open button {
    min-width: 0 !important;
    width: 100% !important;
}
.mineru-advanced-popover {
    position: fixed !important;
    left: var(--mineru-popover-left, 316px) !important;
    top: var(--mineru-popover-top, 360px) !important;
    z-index: 1000 !important;
    width: min(420px, calc(100vw - var(--mineru-popover-left, 316px) - 18px)) !important;
    max-width: calc(100vw - 36px) !important;
    padding: 1px !important;
    border: 1px solid var(--mineru-popover-border) !important;
    border-radius: 14px !important;
    background: var(--mineru-popover-shell-bg) !important;
    box-shadow: var(--mineru-popover-shadow) !important;
    backdrop-filter: blur(12px);
    overflow: hidden !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    transform: translateY(-4px) scale(0.985) !important;
    transform-origin: left center !important;
    transition:
        opacity 140ms ease,
        transform 140ms ease,
        visibility 0s linear 140ms !important;
}
body.mineru-advanced-popover-open .gradio-container .contain .mineru-advanced-popover,
body.mineru-advanced-popover-open .mineru-advanced-popover {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    transform: translateY(0) scale(1) !important;
    transition:
        opacity 140ms ease,
        transform 140ms ease,
        visibility 0s linear 0s !important;
}
.mineru-advanced-popover::before {
    display: none !important;
}
.mineru-advanced-card {
    width: 100% !important;
    max-height: min(70vh, 620px) !important;
    margin: 0 !important;
    padding: 14px !important;
    overflow: auto !important;
    border: 0 !important;
    border-radius: 13px !important;
    background: var(--mineru-popover-card-bg) !important;
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.06) inset !important;
}
.mineru-advanced-popover-title .markdown {
    margin-bottom: 8px;
}
.mineru-advanced-popover .wrap,
.mineru-advanced-popover .block {
    min-width: 0 !important;
}
.mineru-advanced-popover .wrap,
.mineru-advanced-popover .block,
.mineru-advanced-popover .form,
.mineru-advanced-popover .styler,
.mineru-advanced-popover .gr-group {
    overflow: visible !important;
}
.mineru-advanced-popover ul.options {
    border-radius: 8px !important;
    box-shadow: var(--mineru-popover-dropdown-shadow) !important;
}
.mineru-result-tabs .tab-nav {
    border-bottom-color: var(--mineru-panel-border);
}
.mineru-compare-row {
    align-items: stretch;
}
.mineru-preview-pane,
.mineru-markdown-pane {
    min-width: 340px;
}
.mineru-preview-pane > .block,
.mineru-markdown-output {
    min-height: 684px;
}
.mineru-result-file {
    height: auto !important;
    min-height: 44px !important;
    margin-top: 2px;
}
.mineru-result-file .empty {
    display: none !important;
}
.block.mineru-status-panel .html-container.padding {
    padding-left: 0 !important;
    padding-right: 0 !important;
}
.block.mineru-status-panel .prose,
.block.mineru-status-panel .status-steps-panel {
    width: 100% !important;
    max-width: none !important;
}
.mineru-markdown-tabs .tab-nav {
    border-bottom-color: var(--mineru-panel-border);
}
.mineru-markdown-tabs textarea {
    min-height: 640px !important;
}
.status-steps-panel {
    border: 1px solid var(--mineru-panel-border);
    border-radius: 8px;
    padding: 12px;
    background: var(--mineru-status-panel-bg);
}
.status-panel-title {
    margin-bottom: 10px;
    font-size: 13px;
    font-weight: 700;
}
.status-steps-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}
.status-step {
    display: flex;
    align-items: center;
    gap: 7px;
    min-width: 0;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--border-color-primary);
    flex: none;
}
.status-step.is-done .status-dot {
    background: #22c55e;
}
.status-step.is-active {
    color: var(--body-text-color);
    font-weight: 700;
}
.status-step.is-active .status-dot {
    background: var(--mineru-accent);
    box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.22);
}
.status-step.is-error .status-dot {
    background: #ef4444;
    box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.2);
}
.status-latest {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--mineru-status-latest-border);
    color: var(--mineru-status-latest-text);
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1.45;
    word-break: break-word;
}
.office-preview-shell {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 720px;
}
.office-preview-notice {
    border: 1px solid rgba(249, 115, 22, 0.35);
    border-radius: 8px;
    padding: 10px 12px;
    background: rgba(249, 115, 22, 0.08);
}
.office-preview-notice strong,
.office-preview-notice span,
.office-preview-notice code {
    display: block;
}
.office-preview-notice span {
    margin-top: 4px;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
.office-preview-notice code {
    margin-top: 6px;
    white-space: normal;
    word-break: break-all;
}
.office-preview-frame {
    width: 100%;
    min-height: 720px;
    border: 1px solid var(--mineru-panel-border);
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.18);
}
@media (max-width: 900px) {
    .mineru-control-column {
        position: static;
    }
    .mineru-preview-pane,
    .mineru-markdown-pane {
        min-width: 100%;
    }
    .status-steps-list {
        grid-template-columns: 1fr;
    }
}
"""

APP_JS = """
() => {
    const POPOVER_SCRIPT_VERSION = "clipboard-upload-v1";
    if (window.__mineruAdvancedPopoverInstalled === POPOVER_SCRIPT_VERSION) {
        return;
    }
    window.__mineruAdvancedPopoverInstalled = POPOVER_SCRIPT_VERSION;

    const POPOVER_OPEN_CLASS = "mineru-advanced-popover-open";
    const OPEN_DELAY_MS = 120;
    const CLOSE_DELAY_MS = 280;
    const ANIMATION_DELAY_MS = 140;
    const CLIPBOARD_MIME_EXTENSIONS = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    };

    // 兼容 Gradio 将 elem_classes 挂到按钮自身或按钮外层容器的两种 DOM 结构。
    const findButton = () => document.querySelector(
        "button.mineru-advanced-open, .mineru-advanced-open button, .mineru-advanced-open"
    );
    const findPopover = () => document.querySelector(".mineru-advanced-popover");
    let openTimer = null;
    let closeTimer = null;
    let visibilityTimer = null;
    let hoverHandlersInstalled = false;
    const findUploadFileInput = () => {
        const uploadRoot = document.querySelector(".mineru-upload-file");
        if (!uploadRoot) {
            return null;
        }
        return uploadRoot.querySelector('input[type="file"]');
    };

    // 读取上传控件 accept 规则，后续粘贴文件仍复用 gr.File 的支持格式边界。
    const getUploadAcceptedTypes = (uploadInput) => {
        const accept = uploadInput?.getAttribute("accept") || "";
        return accept.split(",").map((item) => item.trim().toLowerCase()).filter(Boolean);
    };

    // 判断剪贴板文件是否匹配 gr.File 当前支持的扩展名或 MIME 类型。
    const fileMatchesAcceptedType = (file, acceptedTypes) => {
        if (!acceptedTypes.length) {
            return true;
        }
        const name = (file.name || "").toLowerCase();
        const type = (file.type || "").toLowerCase();
        return acceptedTypes.some((accepted) => {
            if (accepted.startsWith(".")) {
                return name.endsWith(accepted);
            }
            if (accepted.endsWith("/*")) {
                return type.startsWith(accepted.slice(0, -1));
            }
            return type === accepted;
        });
    };

    // 为截图等无文件名剪贴板图片补一个扩展名，确保后端按普通图片文件解析。
    const buildClipboardFileName = (file) => {
        const type = (file.type || "").toLowerCase();
        const extension = CLIPBOARD_MIME_EXTENSIONS[type];
        if (!extension) {
            return "";
        }
        const timestamp = new Date().toISOString()
            .replace(/[-:]/g, "")
            .replace(/[.].+/, "")
            .replace("T", "-");
        const prefix = type.startsWith("image/") ? "clipboard-image" : "clipboard-file";
        return `${prefix}-${timestamp}.${extension}`;
    };

    // 保留浏览器暴露的原始文件；仅在文件名缺少扩展名时复制一份并补齐名称。
    const normalizeClipboardFile = (file) => {
        if (/[.][^.]+$/.test(file.name || "")) {
            return file;
        }
        const fileName = buildClipboardFileName(file);
        if (!fileName || typeof File === "undefined") {
            return file;
        }
        return new File([file], fileName, {
            type: file.type,
            lastModified: file.lastModified || Date.now(),
        });
    };

    // 同时兼容剪贴板 files 与 items，两种入口在不同浏览器里暴露情况不一致。
    const collectClipboardFiles = (clipboardData) => {
        const files = Array.from(clipboardData.files || []);
        if (files.length) {
            return files;
        }
        return Array.from(clipboardData.items || [])
            .filter((item) => item.kind === "file")
            .map((item) => item.getAsFile())
            .filter(Boolean);
    };

    // 构造只包含目标文件的 FileList；部分浏览器不允许构造 DataTransfer，需要降级处理。
    const createUploadFileList = (file) => {
        try {
            const transfer = new DataTransfer();
            transfer.items.add(file);
            return transfer.files;
        } catch (error) {
            return null;
        }
    };

    // 把文件列表赋值给 gr.File 的原生 input，并触发 Gradio 监听的变更事件。
    const assignClipboardFileToUpload = (uploadInput, uploadFiles) => {
        if (!uploadFiles) {
            return false;
        }
        try {
            uploadInput.files = uploadFiles;
        } catch (error) {
            return false;
        }
        uploadInput.dispatchEvent(new Event("input", { bubbles: true }));
        uploadInput.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    };

    // 将剪贴板文件注入现有 gr.File input，避免为图片、PDF、Office 维护第二套上传链路。
    const uploadClipboardFile = (event) => {
        const clipboardData = event.clipboardData;
        const uploadInput = findUploadFileInput();
        if (!clipboardData || !uploadInput) {
            return false;
        }

        const acceptedTypes = getUploadAcceptedTypes(uploadInput);
        const rawClipboardFiles = clipboardData.files || null;
        const clipboardFiles = collectClipboardFiles(clipboardData)
            .map((rawFile) => ({ rawFile, uploadFile: normalizeClipboardFile(rawFile) }))
            .filter(({ uploadFile }) => fileMatchesAcceptedType(uploadFile, acceptedTypes));
        if (!clipboardFiles.length) {
            return false;
        }

        const { rawFile, uploadFile } = clipboardFiles[0];
        const uploadFiles = createUploadFileList(uploadFile)
            || (
                rawClipboardFiles?.length === 1
                && rawClipboardFiles[0] === rawFile
                && rawFile === uploadFile
                    ? rawClipboardFiles
                    : null
            );
        return assignClipboardFileToUpload(uploadInput, uploadFiles);
    };

    // 修正 Gradio Dropdown 在 fixed 浮层里按视口定位导致的下拉列表漂移。
    const positionAdvancedDropdowns = () => {
        const popover = findPopover();
        if (!popover || !document.body.classList.contains(POPOVER_OPEN_CLASS)) {
            return;
        }

        popover.querySelectorAll("ul.options").forEach((options) => {
            const wrap = options.closest(".wrap");
            if (!wrap) {
                return;
            }

            popover.querySelectorAll(".wrap").forEach((item) => {
                item.style.removeProperty("z-index");
            });

            const wrapRect = wrap.getBoundingClientRect();
            const popoverRect = popover.getBoundingClientRect();
            const viewportPadding = 12;
            const gap = 6;
            const belowSpace = Math.max(0, popoverRect.bottom - wrapRect.bottom - viewportPadding);
            const aboveSpace = Math.max(0, wrapRect.top - popoverRect.top - viewportPadding);
            const naturalHeight = Math.max(36, Math.min(options.scrollHeight || 220, 240));
            const openBelow = belowSpace >= Math.min(180, naturalHeight) || belowSpace >= aboveSpace;
            const availableHeight = Math.max(84, openBelow ? belowSpace : aboveSpace);
            const height = Math.min(naturalHeight, availableHeight);
            const top = openBelow ? wrap.offsetHeight + gap : -height - gap;

            wrap.style.setProperty("z-index", "1003", "important");
            options.style.setProperty("position", "absolute", "important");
            options.style.setProperty("left", "0", "important");
            options.style.setProperty("top", `${top}px`, "important");
            options.style.setProperty("bottom", "auto", "important");
            options.style.setProperty("width", `${wrapRect.width}px`, "important");
            options.style.setProperty("max-height", `${height}px`, "important");
            options.style.setProperty("z-index", "1004", "important");
        });
    };

    // 只在真正支持鼠标悬浮的桌面环境启用 hover 浮窗，触屏设备继续使用点击兜底。
    const supportsHoverPopover = () => (
        typeof window.matchMedia === "function"
        && window.matchMedia("(hover: hover) and (pointer: fine)").matches
    );

    // 取消尚未执行的打开/关闭计时，避免鼠标在按钮和气泡之间移动时闪烁。
    const cancelPopoverTimers = () => {
        if (openTimer !== null) {
            clearTimeout(openTimer);
            openTimer = null;
        }
        if (closeTimer !== null) {
            clearTimeout(closeTimer);
            closeTimer = null;
        }
        if (visibilityTimer !== null) {
            clearTimeout(visibilityTimer);
            visibilityTimer = null;
        }
    };

    // 清理旧版 display 开关留下的内联样式，后续统一交给 CSS 的可见性和动画状态控制。
    const clearLegacyPopoverDisplay = (popover) => {
        if (popover) {
            popover.style.removeProperty("display");
        }
    };

    // 用内联 important 同步动画属性，避免 Gradio 自动 scoped CSS 抬高隐藏规则优先级。
    const applyOpenPopoverStyle = (popover) => {
        if (!popover) {
            return;
        }
        popover.style.setProperty("visibility", "visible", "important");
        popover.style.setProperty("opacity", "1", "important");
        popover.style.setProperty("pointer-events", "auto", "important");
        popover.style.setProperty("transform", "translateY(0) scale(1)", "important");
    };

    // 关闭时先取消交互并播放淡出，动画结束后再隐藏可见性。
    const applyClosedPopoverStyle = (popover) => {
        if (!popover) {
            return;
        }
        popover.style.setProperty("opacity", "0", "important");
        popover.style.setProperty("pointer-events", "none", "important");
        popover.style.setProperty("transform", "translateY(-4px) scale(0.985)", "important");
        visibilityTimer = window.setTimeout(() => {
            if (!document.body.classList.contains(POPOVER_OPEN_CLASS)) {
                popover.style.setProperty("visibility", "hidden", "important");
            }
            visibilityTimer = null;
        }, ANIMATION_DELAY_MS);
    };

    // 等待 Gradio 完成下拉列表挂载后，再按当前输入框位置校正。
    const queueDropdownPosition = () => {
        requestAnimationFrame(() => {
            requestAnimationFrame(positionAdvancedDropdowns);
        });
    };

    // 根据高级选项按钮的位置，把气泡贴在左侧控制栏右侧并限制在视口内。
    const positionPopover = () => {
        const button = findButton();
        const popover = findPopover();
        if (!button || !popover) {
            return;
        }

        const buttonRect = button.getBoundingClientRect();
        const preferredWidth = Math.min(420, window.innerWidth - 36);
        const left = Math.min(
            Math.max(18, buttonRect.right + 12),
            Math.max(18, window.innerWidth - preferredWidth - 18)
        );
        const availableHeight = Math.max(260, window.innerHeight - 36);
        const measuredHeight = Math.min(
            popover.scrollHeight || 520,
            availableHeight,
            Math.round(window.innerHeight * 0.7)
        );
        const centeredTop = buttonRect.top + buttonRect.height / 2 - measuredHeight / 2;
        const top = Math.min(
            Math.max(18, centeredTop),
            Math.max(18, window.innerHeight - measuredHeight - 18)
        );

        popover.style.setProperty("--mineru-popover-left", `${left}px`);
        popover.style.setProperty("--mineru-popover-top", `${top}px`);
    };

    // 打开气泡时保持组件 DOM 挂载，只切换 body 状态类并重新计算位置。
    const openPopover = () => {
        const popover = findPopover();
        cancelPopoverTimers();
        clearLegacyPopoverDisplay(popover);
        document.body.classList.add(POPOVER_OPEN_CLASS);
        applyOpenPopoverStyle(popover);
        requestAnimationFrame(() => {
            positionPopover();
            queueDropdownPosition();
        });
    };

    // 收起气泡时不卸载 Gradio 控件，用户已经修改的高级配置会保留在原组件上。
    const closePopover = () => {
        const popover = findPopover();
        cancelPopoverTimers();
        clearLegacyPopoverDisplay(popover);
        document.body.classList.remove(POPOVER_OPEN_CLASS);
        applyClosedPopoverStyle(popover);
    };

    // 鼠标进入按钮后延迟打开，防止只是路过按钮时频繁弹出。
    const scheduleHoverOpen = () => {
        if (!supportsHoverPopover()) {
            return;
        }
        cancelPopoverTimers();
        openTimer = window.setTimeout(() => {
            openTimer = null;
            openPopover();
        }, OPEN_DELAY_MS);
    };

    // 鼠标离开按钮或气泡后延迟关闭，给用户从按钮移动到气泡留出缓冲时间。
    const scheduleHoverClose = () => {
        if (!supportsHoverPopover()) {
            return;
        }
        cancelPopoverTimers();
        closeTimer = window.setTimeout(() => {
            closeTimer = null;
            closePopover();
        }, CLOSE_DELAY_MS);
    };

    // 给真实桌面指针安装 hover 事件；如果 Gradio 稍后才挂载 DOM，就通过观察器重试。
    const installHoverPopoverHandlers = () => {
        if (hoverHandlersInstalled || !supportsHoverPopover()) {
            return;
        }
        const button = findButton();
        const popover = findPopover();
        if (!button || !popover) {
            return;
        }
        button.addEventListener("pointerenter", scheduleHoverOpen);
        button.addEventListener("pointerleave", scheduleHoverClose);
        button.addEventListener("mouseenter", scheduleHoverOpen);
        button.addEventListener("mouseleave", scheduleHoverClose);
        popover.addEventListener("pointerenter", cancelPopoverTimers);
        popover.addEventListener("pointerleave", scheduleHoverClose);
        popover.addEventListener("mouseenter", cancelPopoverTimers);
        popover.addEventListener("mouseleave", scheduleHoverClose);
        hoverHandlersInstalled = true;
    };

    installHoverPopoverHandlers();
    requestAnimationFrame(installHoverPopoverHandlers);
    if (typeof MutationObserver !== "undefined") {
        const hoverObserver = new MutationObserver(installHoverPopoverHandlers);
        hoverObserver.observe(document.body, { childList: true, subtree: true });
    }

    document.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        if (target.closest(".mineru-advanced-open")) {
            if (document.body.classList.contains(POPOVER_OPEN_CLASS)) {
                closePopover();
            } else {
                openPopover();
            }
            return;
        }
        if (target.closest(".mineru-advanced-popover")) {
            queueDropdownPosition();
        }
        if (!target.closest(".mineru-advanced-popover")) {
            closePopover();
        }
    });

    document.addEventListener("focusin", (event) => {
        const target = event.target;
        if (target instanceof Element && target.closest(".mineru-advanced-popover")) {
            queueDropdownPosition();
        }
    });

    document.addEventListener("input", (event) => {
        const target = event.target;
        if (target instanceof Element && target.closest(".mineru-advanced-popover")) {
            queueDropdownPosition();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closePopover();
            return;
        }
        const target = event.target;
        if (target instanceof Element && target.closest(".mineru-advanced-popover")) {
            queueDropdownPosition();
        }
    });

    document.addEventListener("paste", (event) => {
        if (uploadClipboardFile(event)) {
            event.preventDefault();
        }
    });

    window.addEventListener("resize", () => {
        if (document.body.classList.contains(POPOVER_OPEN_CLASS)) {
            positionPopover();
            positionAdvancedDropdowns();
        }
    });
}
"""

# Gradio 6 的 js 参数在部分托管环境里只注入函数文本，使用 head 包装确保页面加载后主动执行。
APP_HEAD = f"""
<script>
(() => {{
    const installMineruAdvancedPopover = {APP_JS};
    if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", installMineruAdvancedPopover, {{ once: true }});
    }} else {{
        installMineruAdvancedPopover();
    }}
}})();
</script>
"""


@dataclass
class StatusPanelState:
    lines: list[str] = field(default_factory=list)
    processing_index: int | None = None
    processing_started_at: float | None = None
    last_processing_elapsed_seconds: float | None = None
    queue_index: int | None = None
    queue_started_at: float | None = None
    queue_base_message: str | None = None

    def append(self, message: str) -> bool:
        if not message:
            return False

        if self.is_queue_message(message):
            self.finalize_processing()
            return self.update_queue(message)

        if message == STATUS_PROCESSING_ON_SERVER:
            self.finalize_queue()
            return self.start_processing()

        self.finalize_processing()
        self.finalize_queue()
        if message == STATUS_COMPLETED:
            message = format_completed_status(self.last_processing_elapsed_seconds)
        if not self.lines or self.lines[-1] != message:
            self.lines.append(message)
            return True
        return False

    def start_processing(self) -> bool:
        if self.processing_started_at is not None:
            return self.tick_processing()

        self.processing_started_at = time.monotonic()
        self.last_processing_elapsed_seconds = 0.0
        self.processing_index = len(self.lines)
        self.lines.append(format_processing_status(0.0))
        return True

    def tick_processing(self) -> bool:
        if self.processing_started_at is None or self.processing_index is None:
            return False

        elapsed_seconds = max(0.0, time.monotonic() - self.processing_started_at)
        self.last_processing_elapsed_seconds = elapsed_seconds
        updated = format_processing_status(elapsed_seconds)
        if self.lines[self.processing_index] != updated:
            self.lines[self.processing_index] = updated
            return True
        return False

    def finalize_processing(self) -> bool:
        if self.processing_started_at is None or self.processing_index is None:
            return False

        self.tick_processing()
        self.processing_started_at = None
        self.processing_index = None
        return True

    def update_queue(self, message: str) -> bool:
        if (
            self.queue_index is None
            or self.queue_started_at is None
            or self.queue_base_message is None
        ):
            self.queue_started_at = time.monotonic()
            self.queue_index = len(self.lines)
            self.queue_base_message = message
            self.lines.append(format_queue_status(message, 0.0))
            return True

        self.queue_base_message = message
        updated = format_queue_status(
            message,
            max(0.0, time.monotonic() - self.queue_started_at),
        )
        if self.lines[self.queue_index] != updated:
            self.lines[self.queue_index] = updated
            return True
        return False

    def tick_queue(self) -> bool:
        if (
            self.queue_index is None
            or self.queue_started_at is None
            or self.queue_base_message is None
        ):
            return False

        updated = format_queue_status(
            self.queue_base_message,
            max(0.0, time.monotonic() - self.queue_started_at),
        )
        if self.lines[self.queue_index] != updated:
            self.lines[self.queue_index] = updated
            return True
        return False

    def finalize_queue(self) -> bool:
        if (
            self.queue_index is None
            or self.queue_started_at is None
            or self.queue_base_message is None
        ):
            return False

        self.tick_queue()
        self.queue_index = None
        self.queue_started_at = None
        self.queue_base_message = None
        return True

    def tick(self) -> bool:
        if self.is_processing:
            return self.tick_processing()
        if self.is_queueing:
            return self.tick_queue()
        return False

    @property
    def is_processing(self) -> bool:
        return self.processing_started_at is not None

    @property
    def is_queueing(self) -> bool:
        return self.queue_started_at is not None

    @property
    def animation_interval_seconds(self) -> float | None:
        if self.is_processing:
            return STATUS_TIMER_INTERVAL_SECONDS
        if self.is_queueing:
            return STATUS_QUEUE_ANIMATION_INTERVAL_SECONDS
        return None

    @staticmethod
    def is_queue_message(message: str) -> bool:
        return (
            message.startswith(STATUS_QUEUED_LOCALLY_PREFIX)
            or message.startswith(STATUS_QUEUED_ON_SERVER)
        )

    def render(self) -> str:
        return "\n".join(self.lines)


def format_failed_status(error: Exception | str) -> str:
    return f"Failed: {error}"


def format_processing_status(elapsed_seconds: float) -> str:
    return f"{STATUS_PROCESSING_ON_SERVER} ({elapsed_seconds:.1f}s)"


def format_completed_status(elapsed_seconds: float | None) -> str:
    """生成完成状态文案，保留服务端解析阶段最终耗时。"""
    if elapsed_seconds is None:
        return STATUS_COMPLETED
    return f"{STATUS_COMPLETED} ({elapsed_seconds:.1f}s)"


def format_queue_status(base_message: str, elapsed_seconds: float) -> str:
    dots = "." * (
        (int(max(0.0, elapsed_seconds)) % STATUS_QUEUE_ANIMATION_MAX_DOTS) + 1
    )
    return f"{base_message}{dots}"


def format_concurrency_wait_message(snapshot: GradioConcurrencyWaitSnapshot) -> str:
    return f"{STATUS_QUEUED_LOCALLY_PREFIX} {snapshot.ahead} request(s) ahead"


def format_remote_status_message(
    status_snapshot: _api_client.TaskStatusSnapshot | str,
) -> str:
    if isinstance(status_snapshot, _api_client.TaskStatusSnapshot):
        status = status_snapshot.status
        queued_ahead = status_snapshot.queued_ahead
    else:
        status = status_snapshot
        queued_ahead = None

    if status == "pending":
        if queued_ahead is not None:
            return f"{STATUS_QUEUED_ON_SERVER}: {queued_ahead} request(s) ahead"
        return STATUS_QUEUED_ON_SERVER
    if status == "processing":
        return STATUS_PROCESSING_ON_SERVER
    if status == "completed":
        return STATUS_COMPLETED
    if status == "failed":
        return format_failed_status("server task failed")
    return f"Task status: {status}"


def compress_directory_to_zip(directory_path, output_zip_path):
    """压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # MIME类型映射
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
    }

    def _path_to_data_uri(relative_path):
        file_ext = os.path.splitext(relative_path)[1].lower()
        if file_ext not in mime_types:
            return None
        try:
            full_path = os.path.join(image_dir_path, relative_path)
            base64_image = image_to_base64(full_path)
            return f'data:{mime_types[file_ext]};base64,{base64_image}'
        except Exception as e:
            logger.warning(f"Failed to convert image {relative_path} to base64: {e}")
            return None

    # 匹配Markdown中的图片标签 ![...](path)
    def replace_md(match):
        relative_path = match.group(1)
        data_uri = _path_to_data_uri(relative_path)
        if data_uri:
            return f'![{relative_path}]({data_uri})'
        return match.group(0)

    result = re.sub(r'\!\[(?:[^\]]*)\]\(([^)]+)\)', replace_md, markdown_text)

    # 匹配HTML表格中的 <img src="path"> (跳过已有的data: URI)
    def replace_html_src(match):
        relative_path = match.group(1)
        data_uri = _path_to_data_uri(relative_path)
        if data_uri:
            return f'src="{data_uri}"'
        return match.group(0)

    result = re.sub(r'src="(?!data:)([^"]+)"', replace_html_src, result)

    return result


def _escape_latex_html_chars_for_gradio(content):
    """转义公式内部会被 Gradio HTML 解析链路误判的尖括号，保留 LaTeX 对齐用 &。"""
    return content.replace("<", "&lt;").replace(">", "&gt;")


def escape_latex_blocks_for_gradio_preview(markdown_text, latex_delimiters):
    """根据当前 LaTeX 分隔符，仅转义公式内容，避免影响公式外 Markdown/HTML。"""
    if not markdown_text or not latex_delimiters:
        return markdown_text

    delimiter_pairs = []
    for delimiter in latex_delimiters:
        left = delimiter.get("left")
        right = delimiter.get("right")
        if left and right:
            delimiter_pairs.append((left, right))
    delimiter_pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    if not delimiter_pairs:
        return markdown_text

    result = []
    position = 0
    text_length = len(markdown_text)
    while position < text_length:
        matched_pair = None
        for left, right in delimiter_pairs:
            if markdown_text.startswith(left, position):
                matched_pair = (left, right)
                break

        if matched_pair is None:
            result.append(markdown_text[position])
            position += 1
            continue

        left, right = matched_pair
        content_start = position + len(left)
        content_end = markdown_text.find(right, content_start)
        if content_end == -1:
            # 未闭合的分隔符保持原样，并继续扫描后续可能闭合的公式块。
            result.append(markdown_text[position])
            position += 1
            continue

        result.append(left)
        result.append(
            _escape_latex_html_chars_for_gradio(markdown_text[content_start:content_end])
        )
        result.append(right)
        position = content_end + len(right)

    return "".join(result)


def prepare_markdown_for_gradio_preview(markdown_text, latex_delimiters):
    """准备传给 gr.Markdown 的预览文本；原始 Markdown 文件内容不在这里改写。"""
    if not isinstance(markdown_text, str):
        return markdown_text
    return escape_latex_blocks_for_gradio_preview(markdown_text, latex_delimiters)


def normalize_language(language):
    if '(' in language and ')' in language:
        return language.split('(')[0].strip()
    return language


def resolve_parse_method(file_path, is_ocr, backend):
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    if file_suffix in office_suffixes:
        return "auto"
    if backend.startswith("vlm"):
        return "auto"
    return "ocr" if is_ocr else "auto"


def is_image_analysis_option_visible(backend):
    """判断 Gradio 图片分析开关是否应展示；pipeline 后端不消费该参数。"""
    return backend.startswith("vlm") or backend.startswith("hybrid")


def create_gradio_run_paths(file_path, output_root="./output"):
    run_id = f"{time.strftime('%y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe_stem(Path(file_path).stem)}"
    run_root = Path(output_root) / "gradio" / run_id
    extract_root = run_root / "result"
    archive_zip_path = run_root / f"{safe_stem(Path(file_path).stem)}.zip"
    return run_root, extract_root, archive_zip_path


def build_gradio_upload_name(file_path):
    path = Path(file_path)
    return f"{normalize_task_stem(path.stem)}{path.suffix}"


def resolve_result_file_name(submit_response, extract_root, file_path):
    if submit_response.file_names:
        return submit_response.file_names[0]

    candidate_dirs = sorted(path.name for path in Path(extract_root).iterdir() if path.is_dir())
    if len(candidate_dirs) == 1:
        return candidate_dirs[0]
    return normalize_task_stem(Path(file_path).stem)


async def resolve_server_health(http_client, api_url):
    if api_url:
        return await _api_client.fetch_server_health(
            http_client,
            _api_client.normalize_base_url(api_url),
        )

    local_server, started_now = _gradio_local_api_server.ensure_started()
    if started_now:
        logger.info(f"Started local mineru-api at {local_server.base_url}")
    return await _api_client.wait_for_local_api_ready(http_client, local_server)


async def ensure_local_api_ready_for_gradio_startup(
    timeout_seconds: float = _api_client.LOCAL_API_STARTUP_TIMEOUT_SECONDS,
):
    local_server, started_now = _gradio_local_api_server.ensure_started()
    if started_now:
        logger.info(f"Started local mineru-api at {local_server.base_url}")

    async with httpx.AsyncClient(
        timeout=_api_client.build_http_timeout(),
        follow_redirects=True,
    ) as http_client:
        return await _api_client.wait_for_local_api_ready(
            http_client,
            local_server,
            timeout_seconds=timeout_seconds,
        )


def maybe_prepare_local_api_for_gradio_startup(
    *,
    api_url: str | None,
    enable_vlm_preload: bool,
):
    if api_url is not None or not enable_vlm_preload:
        return None

    try:
        return asyncio.run(ensure_local_api_ready_for_gradio_startup())
    except Exception:
        _gradio_local_api_server.stop()
        raise


def resolve_gradio_max_concurrent_requests(api_url, server_health):
    if api_url is None:
        return server_health.max_concurrent_requests

    return _api_client.resolve_effective_max_concurrent_requests(
        local_max=_api_client.read_max_concurrent_requests(
            default=_api_client.DEFAULT_MAX_CONCURRENT_REQUESTS
        ),
        server_max=server_health.max_concurrent_requests,
    )


def maybe_generate_local_preview(extract_root, file_name, file_suffix, backend, parse_method):
    if file_suffix in office_suffixes:
        return None

    parse_dir = resolve_parse_dir(
        extract_root,
        file_name,
        backend,
        parse_method,
        allow_office_fallback=True,
    )
    visualization_job = VisualizationJob(
        document_stem=file_name,
        backend=backend,
        parse_method=parse_method,
        parse_dir=parse_dir,
        draw_span=backend.startswith("pipeline"),
    )
    result = run_visualization_job(visualization_job)
    if result.status != "finished":
        logger.warning(
            f"Skipping visualization for {visualization_job.document_stem}: {result.message}"
        )
    return resolve_preview_pdf_path(parse_dir, file_name)


async def _run_to_markdown_job(
    file_path,
    end_pages=10,
    is_ocr=False,
    formula_enable=True,
    table_enable=True,
    image_analysis=True,
    language="ch",
    backend="pipeline",
    url=None,
    api_url=None,
    status_callback: Callable[[str], None] | None = None,
):
    if file_path is None:
        return "", "", None, None

    def emit_status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    normalized_language = normalize_language(language)
    file_path = str(file_path)
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    parse_method = resolve_parse_method(file_path, is_ocr, backend)
    run_root, extract_root, archive_zip_path = create_gradio_run_paths(file_path)
    run_root.mkdir(parents=True, exist_ok=True)

    form_data = _api_client.build_parse_request_form_data(
        lang_list=[normalized_language],
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        image_analysis=image_analysis,
        server_url=url,
        start_page_id=0,
        end_page_id=end_pages - 1,
        return_md=True,
        return_middle_json=True,
        return_model_output=True,
        return_content_list=True,
        return_images=True,
        response_format_zip=True,
        return_original_file=True,
    )
    upload_assets = [
        _api_client.UploadAsset(
            path=Path(file_path),
            upload_name=build_gradio_upload_name(file_path),
        )
    ]

    async with httpx.AsyncClient(
        timeout=_api_client.build_http_timeout(),
        follow_redirects=True,
    ) as http_client:
        emit_status(STATUS_PREPARING_REQUEST)
        emit_status(STATUS_CHECKING_SERVER)
        server_health = await resolve_server_health(http_client, api_url)
        effective_max_concurrent_requests = resolve_gradio_max_concurrent_requests(
            api_url=api_url,
            server_health=server_health,
        )
        async with _gradio_request_concurrency_limiter.acquire(
            effective_max_concurrent_requests,
            on_wait=lambda snapshot: emit_status(
                format_concurrency_wait_message(snapshot)
            ),
        ):
            emit_status(STATUS_SUBMITTING_TASK)
            submit_response = await _api_client.submit_parse_task(
                base_url=server_health.base_url,
                upload_assets=upload_assets,
                form_data=form_data,
            )
            emit_status(f"Task submitted：task_id={submit_response.task_id}")

            last_task_snapshot = None

            def handle_task_status(
                status_snapshot: _api_client.TaskStatusSnapshot,
            ) -> None:
                nonlocal last_task_snapshot
                if status_snapshot == last_task_snapshot:
                    return
                last_task_snapshot = status_snapshot
                emit_status(format_remote_status_message(status_snapshot))

            await _api_client.wait_for_task_result(
                client=http_client,
                submit_response=submit_response,
                task_label=Path(file_path).name,
                status_snapshot_callback=handle_task_status,
            )
            emit_status(STATUS_DOWNLOADING_RESULT)
            result_zip_path = await _api_client.download_result_zip(
                client=http_client,
                submit_response=submit_response,
                task_label=Path(file_path).name,
            )

    try:
        _api_client.safe_extract_zip(result_zip_path, extract_root)
    finally:
        result_zip_path.unlink(missing_ok=True)

    file_name = resolve_result_file_name(submit_response, extract_root, file_path)
    local_md_dir = resolve_parse_dir(
        extract_root,
        file_name,
        backend,
        parse_method,
        allow_office_fallback=True,
    )
    preview_pdf_path = maybe_generate_local_preview(
        extract_root=extract_root,
        file_name=file_name,
        file_suffix=file_suffix,
        backend=backend,
        parse_method=parse_method,
    )

    emit_status(STATUS_PROCESSING_OUTPUT)
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info('Compression successful')
    else:
        logger.error('Compression failed')

    md_path = Path(local_md_dir) / f"{file_name}.md"
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)

    if file_suffix in office_suffixes:
        preview_pdf_path = None

    emit_status(STATUS_COMPLETED)
    return md_content, txt_content, str(archive_zip_path), preview_pdf_path


async def stream_to_markdown(
    file_path,
    end_pages=10,
    is_ocr=False,
    formula_enable=True,
    table_enable=True,
    image_analysis=True,
    language="ch",
    backend="pipeline",
    url=None,
    api_url=None,
):
    status_state = StatusPanelState()
    job_task: asyncio.Task | None = None
    queue_get_task: asyncio.Task | None = None
    timer_task: asyncio.Task | None = None
    yield status_state.render(), None, "", "", gr.skip()

    if file_path is None:
        return

    status_queue: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def enqueue_status(message: str) -> None:
        loop.call_soon_threadsafe(status_queue.put_nowait, message)

    try:
        job_task = asyncio.create_task(
            _run_to_markdown_job(
                file_path=file_path,
                end_pages=end_pages,
                is_ocr=is_ocr,
                formula_enable=formula_enable,
                table_enable=table_enable,
                image_analysis=image_analysis,
                language=language,
                backend=backend,
                url=url,
                api_url=api_url,
                status_callback=enqueue_status,
            )
        )

        while True:
            if job_task.done() and status_queue.empty():
                status_state.finalize_processing()
                status_state.finalize_queue()
                break

            queue_get_task = asyncio.create_task(status_queue.get())
            wait_tasks: set[asyncio.Task] = {job_task, queue_get_task}
            timer_task = None
            animation_interval = status_state.animation_interval_seconds
            if animation_interval is not None:
                timer_task = asyncio.create_task(
                    asyncio.sleep(animation_interval)
                )
                wait_tasks.add(timer_task)

            done, pending = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if queue_get_task in done:
                message = queue_get_task.result()
                if status_state.append(message):
                    yield status_state.render(), None, "", "", gr.skip()
            elif timer_task is not None and timer_task in done:
                if status_state.tick():
                    yield status_state.render(), None, "", "", gr.skip()
            else:
                queue_get_task.cancel()
                await asyncio.gather(queue_get_task, return_exceptions=True)

            for pending_task in pending:
                if pending_task is job_task:
                    continue
                pending_task.cancel()
                await asyncio.gather(pending_task, return_exceptions=True)
            queue_get_task = None
            timer_task = None

        while not status_queue.empty():
            status_state.append(status_queue.get_nowait())
    except Exception as exc:
        status_state.append(format_failed_status(exc))
        yield status_state.render(), None, "", "", gr.skip()
        raise
    finally:
        for task in (queue_get_task, timer_task, job_task):
            if task is None or task.done():
                continue
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        md_content, txt_content, archive_zip_path, preview_pdf_path = await job_task
    except Exception as exc:
        status_state.append(format_failed_status(exc))
        yield status_state.render(), None, "", "", gr.skip()
        raise

    status_state.append(STATUS_COMPLETED)
    yield (
        status_state.render(),
        archive_zip_path,
        md_content,
        txt_content,
        preview_pdf_path,
    )


def resolve_preview_pdf_path(local_md_dir, file_name):
    layout_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')
    if os.path.exists(layout_pdf_path):
        return layout_pdf_path

    origin_pdf_path = os.path.join(local_md_dir, file_name + '_origin.pdf')
    if os.path.exists(origin_pdf_path):
        logger.warning(
            f"Layout preview PDF not found for {file_name}, "
            f"falling back to origin PDF: {origin_pdf_path}"
        )
        return origin_pdf_path

    logger.warning(f"No preview PDF found for {file_name} under {local_md_dir}")
    return None


latex_delimiters_type_a = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False},
]
latex_delimiters_type_b = [
    {'left': '\\(', 'right': '\\)', 'display': False},
    {'left': '\\[', 'right': '\\]', 'display': True},
]
latex_delimiters_type_all = latex_delimiters_type_a + latex_delimiters_type_b

header_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'header.html')
with open(header_path, mode='r', encoding='utf-8') as header_file:
    header_template = header_file.read()

HEADER_I18N_PLACEHOLDERS = {
    "{{HEADER_TITLE}}": "header_title",
    "{{HEADER_SUBTITLE}}": "header_subtitle",
    "{{HEADER_SUPPORT_TEXT}}": "header_support_text",
    "{{HEADER_STARS_ALT}}": "header_stars_alt",
    "{{HEADER_CODE_LINK}}": "header_code_link",
    "{{HEADER_MODEL_LINK}}": "header_model_link",
    "{{HEADER_PAPER_LINK}}": "header_paper_link",
    "{{HEADER_HOMEPAGE_LINK}}": "header_homepage_link",
    "{{HEADER_DOWNLOAD_LINK}}": "header_download_link",
}
HEADER_GRADIO_VERSION_CLASS_PLACEHOLDER = "{{HEADER_GRADIO_VERSION_CLASS}}"


def render_header_html(i18n):
    """渲染支持 i18n 的顶部 Header，保留静态模板中的样式和链接。"""
    rendered_header = header_template
    for placeholder, translation_key in HEADER_I18N_PLACEHOLDERS.items():
        rendered_header = rendered_header.replace(
            placeholder,
            html_lib.escape(translate_ui(i18n, translation_key)),
        )
    rendered_header = rendered_header.replace(
        HEADER_GRADIO_VERSION_CLASS_PLACEHOLDER,
        "mineru-gradio6-header" if IS_GRADIO_6 else "",
    )
    return rendered_header

other_lang = [
    'ch (Chinese, English, Chinese Traditional)',
    'ch_lite (Chinese, English, Chinese Traditional, Japanese)',
    'ch_server (Chinese, English, Chinese Traditional, Japanese)',
    'en (English)',
    'korean (Korean, English)',
    'japan (Chinese, English, Chinese Traditional, Japanese)',
    'chinese_cht (Chinese, English, Chinese Traditional, Japanese)',
    'ta (Tamil, English)',
    'te (Telugu, English)',
    'ka (Kannada)',
    'el (Greek, English)',
    'th (Thai, English)'
]
add_lang = [
    'latin (French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua)',
    'arabic (Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English)',
    'east_slavic (Russian, Belarusian, Ukrainian, English)',
    'cyrillic (Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English)',
    'devanagari (Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi, English)'
]
all_lang = [*other_lang, *add_lang]


def safe_stem(file_path):
    stem = Path(file_path).stem
    # 只保留字母、数字、下划线和点，其他字符替换为下划线
    return re.sub(r'[^\w.]', '_', stem)


def to_pdf(file_path):

    if file_path is None:
        return None

    pdf_bytes = read_fn(file_path)

    # unique_filename = f'{uuid.uuid4()}.pdf'
    unique_filename = f'{safe_stem(file_path)}.pdf'

    # 构建完整的文件路径
    tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

    # 将字节数据写入文件
    with open(tmp_file_path, 'wb') as tmp_pdf_file:
        tmp_pdf_file.write(pdf_bytes)

    return tmp_file_path


def to_pdf_preview(file_path):
    """用于 PDF 预览的转换函数，office 文件不支持预览，返回 None。"""
    if file_path is None:
        return None
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    if file_suffix in office_suffixes:
        return None
    return to_pdf(file_path)


def build_office_preview_html(file_path, request: gr.Request, i18n=None):
    """生成 Office 在线预览 HTML，并提示该预览依赖外部 Microsoft 服务访问。"""
    host = (
        request.headers.get('x-forwarded-host')
        or request.headers.get('host', 'localhost:7860')
    )
    proto = request.headers.get('x-forwarded-proto', 'http')
    base_url = f"{proto}://{host}"
    public_url = f"{base_url}/gradio_api/file={file_path}"
    viewer_url = (
        "https://view.officeapps.live.com/op/embed.aspx?src="
        f"{quote(public_url, safe='')}"
    )
    file_name = html_lib.escape(Path(file_path).name)
    title = html_lib.escape(translate_ui(i18n, "office_preview_title"))
    notice = html_lib.escape(translate_ui(i18n, "office_preview_notice"))
    return (
        '<div class="office-preview-shell">'
        '<div class="office-preview-notice">'
        f"<strong>{title}</strong>"
        f"<span>{notice}</span>"
        f"<code>{file_name}</code>"
        "</div>"
        f'<iframe class="office-preview-frame" src="{html_lib.escape(viewer_url)}" '
        'frameborder="0"></iframe>'
        "</div>"
    )


def update_file_options_html(file_path, request: gr.Request, i18n=None):
    """处理文件上传第一阶段：根据文件类型更新 options_group 和 office_html。
    将 doc_show（gradio_pdf.PDF）的更新拆分到独立的 .then() 事件中，
    以规避 gradio_pdf 0.0.24 在 Gradio 6 中对 value=None 处理不当导致的
    整个事件 processing 状态卡死的兼容性问题。
    """
    if file_path is None:
        return (
            gr.update(visible=True),             # options_group - 恢复显示
            gr.update(value="", visible=False),  # office_html - 隐藏
        )

    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    is_office = file_suffix in office_suffixes

    if is_office:
        html_content = build_office_preview_html(file_path, request, i18n)
        return (
            gr.update(visible=False),                    # options_group - 隐藏
            gr.update(value=html_content, visible=True), # office_html - 显示
        )
    else:
        return (
            gr.update(visible=True),             # options_group - 显示
            gr.update(value="", visible=False),  # office_html - 隐藏
        )


def update_doc_show(file_path):
    """处理文件上传第二阶段：单独更新 doc_show（gradio_pdf.PDF）组件。
    对 office 文件仅改变 visible，避免传递 value=None 触发
    gradio_pdf 0.0.24 在 Gradio 6 中无法完成的加载周期。
    """
    if file_path is None:
        # 无文件时恢复显示并清空（clear 按钮路径）
        return gr.update(value=None, visible=True)

    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    is_office = file_suffix in office_suffixes

    if is_office:
        # 仅隐藏，不改变 value，避免触发 gradio_pdf 加载周期导致事件 pending 卡死
        return gr.update(visible=False)
    else:
        pdf_path = to_pdf_preview(file_path)
        return gr.update(value=pdf_path, visible=True)


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option(
    '--enable-example',
    'example_enable',
    type=bool,
    help="Enable example files for input."
         "The example files to be input need to be placed in the `example` folder within the directory where the command is currently executed.",
    default=True,
)
@click.option(
    '--enable-http-client',
    'http_client_enable',
    type=bool,
    help="Enable http-client backend to link openai-compatible servers.",
    default=False,
)
@click.option(
    '--enable-api',
    'api_enable',
    type=bool,
    help="Enable gradio API for serving the application.",
    default=True,
)
@click.option(
    '--max-convert-pages',
    'max_convert_pages',
    type=int,
    help="Set the maximum number of pages to convert from PDF to Markdown.",
    default=1000,
)
@click.option(
    '--server-name',
    'server_name',
    type=str,
    help="Set the server name for the Gradio app.",
    default=None,
)
@click.option(
    '--server-port',
    'server_port',
    type=int,
    help="Set the server port for the Gradio app.",
    default=None,
)
@click.option(
    '--api-url',
    'api_url',
    type=str,
    help="MinerU FastAPI base URL. If omitted, gradio starts a reusable local mineru-api service.",
    default=None,
)
@click.option(
    '--enable-vlm-preload',
    'enable_vlm_preload',
    type=bool,
    help="Preload the local VLM model when gradio starts a local mineru-api service.",
    default=False,
)
@click.option(
    '--latex-delimiters-type',
    'latex_delimiters_type',
    type=click.Choice(['a', 'b', 'all']),
    help="Set the type of LaTeX delimiters to use in Markdown rendering:"
         "'a' for type '$', 'b' for type '()[]', 'all' for both types.",
    default='all',
)
def main(ctx,
        example_enable,
        http_client_enable,
        api_enable, max_convert_pages,
        server_name, server_port, api_url, enable_vlm_preload, latex_delimiters_type, **kwargs
):

    # 创建 i18n 实例，支持中英文
    i18n = gr.I18n(
        en={
            "upload_file": "Select or paste a file to upload\nPDF, image, DOCX, PPTX, or XLSX",
            "header_title": "MinerU 3: Document Extraction Demo",
            "header_subtitle": "Open-source document extraction for PDF, DOCX, PPTX, XLSX, and images to Markdown and JSON.",
            "header_support_text": "If you found our project helpful, please give us a ⭐️ to support us!",
            "header_stars_alt": "stars",
            "header_code_link": "Code",
            "header_model_link": "Model",
            "header_paper_link": "Paper",
            "header_homepage_link": "Homepage",
            "header_download_link": "Download",
            "max_pages": "Max convert pages",
            "backend": "Backend",
            "backend_label_hybrid": "Hybrid (Recommended)",
            "backend_label_pipeline": "Pipeline (Stable multilingual)",
            "backend_label_vlm": "VLM (High-precision Chinese/English)",
            "backend_label_remote_vlm": "Remote VLM",
            "backend_label_remote_hybrid": "Remote Hybrid",
            "server_url": "Server URL",
            "server_url_info": "OpenAI-compatible server URL for http-client backend.",
            "recognition_options": "**Recognition Options:**",
            "advanced_options": "Advanced options",
            "table_enable": "Enable table recognition",
            "table_info": "If disabled, tables will be shown as images.",
            "image_analysis_enable": "Enable image analysis",
            "image_analysis_info": "If disabled, image/chart blocks will keep layout positions but skip VLM image/chart analysis.",
            "formula_label_vlm": "Enable display formula recognition",
            "formula_label_pipeline": "Enable formula recognition",
            "formula_label_hybrid": "Enable inline formula recognition",
            "formula_info_vlm": "If disabled, display formulas will be shown as images.",
            "formula_info_pipeline": "If disabled, display formulas will be shown as images, and inline formulas will not be detected or parsed.",
            "formula_info_hybrid": "If disabled, inline formulas will not be detected or parsed.",
            "ocr_language": "OCR Language",
            "ocr_language_info": "Select the OCR language for image-based PDFs and images.",
            "force_ocr": "Force enable OCR",
            "force_ocr_info": "Enable only if the result is extremely poor. Requires correct OCR language.",
            "convert": "Convert",
            "clear": "Clear",
            "doc_preview": "Document preview",
            "examples": "Examples:",
            "convert_status": "Conversion Status",
            "convert_result": "Convert result",
            "result_file": "Result file",
            "md_rendering": "Markdown rendering",
            "md_text": "Markdown text",
            "status_idle_title": "Waiting",
            "status_idle_hint": "Upload a file and start conversion.",
            "status_latest": "Latest status",
            "status_step_prepare": "Prepare",
            "status_step_check": "Check service",
            "status_step_submit": "Submit",
            "status_step_queue": "Queue",
            "status_step_process": "Parse",
            "status_step_download": "Download",
            "status_step_outputs": "Build outputs",
            "status_step_done": "Done",
            "status_step_failed": "Failed",
            "office_preview_title": "Office online preview",
            "office_preview_notice": "This preview requires the current file to be reachable by Microsoft Office Online. Conversion does not depend on this preview.",
            "backend_info_vlm": "High-precision parsing via VLM, supports Chinese and English documents only.",
            "backend_info_pipeline": "Traditional Multi-model pipeline parsing, supports multiple languages, hallucination-free.",
            "backend_info_hybrid": "High-precision hybrid parsing, supports multiple languages.",
            "backend_info_default": "Select the backend engine for document parsing.",
        },
        zh={
            "upload_file": "请选择或粘贴要上传的文件\nPDF、图片、DOCX、PPTX 或 XLSX",
            "header_title": "MinerU 3：文档提取演示",
            "header_subtitle": "开源文档提取工具，支持将 PDF、DOCX、PPTX、XLSX 和图片转换为 Markdown 与 JSON。",
            "header_support_text": "如果我们的项目对你有帮助，请点亮 ⭐️ 支持我们！",
            "header_stars_alt": "GitHub 星标",
            "header_code_link": "代码",
            "header_model_link": "模型",
            "header_paper_link": "论文",
            "header_homepage_link": "主页",
            "header_download_link": "下载",
            "max_pages": "最大转换页数",
            "backend": "解析后端",
            "backend_label_hybrid": "Hybrid 推荐",
            "backend_label_pipeline": "Pipeline 稳定多语言",
            "backend_label_vlm": "VLM 高精度中英文",
            "backend_label_remote_vlm": "Remote VLM",
            "backend_label_remote_hybrid": "Remote Hybrid",
            "server_url": "服务器地址",
            "server_url_info": "http-client 后端的 OpenAI 兼容服务器地址。",
            "recognition_options": "**识别选项：**",
            "advanced_options": "高级选项",
            "table_enable": "启用表格识别",
            "table_info": "禁用后，表格将显示为图片。",
            "image_analysis_enable": "启用图片分析",
            "image_analysis_info": "禁用后，图片/图表块仍保留版面位置，但跳过 VLM 图片/图表分析。",
            "formula_label_vlm": "启用行间公式识别",
            "formula_label_pipeline": "启用公式识别",
            "formula_label_hybrid": "启用行内公式识别",
            "formula_info_vlm": "禁用后，行间公式将显示为图片。",
            "formula_info_pipeline": "禁用后，行间公式将显示为图片，行内公式将不会被检测或解析。",
            "formula_info_hybrid": "禁用后，行内公式将不会被检测或解析。",
            "ocr_language": "OCR 语言",
            "ocr_language_info": "为扫描版 PDF 和图片选择 OCR 语言。",
            "force_ocr": "强制启用 OCR",
            "force_ocr_info": "仅在识别效果极差时启用，需选择正确的 OCR 语言。",
            "convert": "转换",
            "clear": "清除",
            "doc_preview": "文档预览",
            "examples": "示例：",
            "convert_status": "转换状态",
            "convert_result": "转换结果",
            "result_file": "结果文件",
            "md_rendering": "Markdown 渲染",
            "md_text": "Markdown 文本",
            "status_idle_title": "等待任务",
            "status_idle_hint": "上传文件后开始转换。",
            "status_latest": "最新状态",
            "status_step_prepare": "准备请求",
            "status_step_check": "检查服务",
            "status_step_submit": "提交任务",
            "status_step_queue": "排队",
            "status_step_process": "解析中",
            "status_step_download": "下载结果",
            "status_step_outputs": "整理输出",
            "status_step_done": "完成",
            "status_step_failed": "失败",
            "office_preview_title": "Office 在线预览",
            "office_preview_notice": "该预览需要当前文件可被 Microsoft 在线预览服务访问，转换不依赖该预览。",
            "backend_info_vlm": "多模态大模型高精度解析，仅支持中英文文档。",
            "backend_info_pipeline": "传统多模型管道解析，支持多语言，无幻觉。",
            "backend_info_hybrid": "高精度混合解析，支持多语言。",
            "backend_info_default": "选择文档解析的后端引擎。",
        },
    )

    # 根据后端类型获取公式识别标签（闭包函数以支持 i18n）
    def get_formula_label(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_label_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_label_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_label_hybrid")
        else:
            return i18n("formula_label_pipeline")

    def get_formula_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_info_hybrid")
        else:
            return ""

    def get_backend_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("backend_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("backend_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("backend_info_hybrid")
        else:
            return i18n("backend_info_default")

    # 更新界面函数
    def update_interface(backend_choice):
        formula_label_update = gr.update(label=get_formula_label(backend_choice), info=get_formula_info(backend_choice))
        backend_info_update = gr.update(info=get_backend_info(backend_choice))
        image_analysis_update = gr.update(visible=is_image_analysis_option_visible(backend_choice))
        if "http-client" in backend_choice:
            client_options_update = gr.update(visible=True)
        else:
            client_options_update = gr.update(visible=False)
        if "vlm" in backend_choice:
            ocr_options_update = gr.update(visible=False)
        else:
            ocr_options_update = gr.update(visible=True)

        return client_options_update, ocr_options_update, formula_label_update, backend_info_update, image_analysis_update

    del kwargs
    _gradio_local_api_server.configure(
        resolve_gradio_local_api_cli_args(
            ctx.args,
            api_url=api_url,
            enable_vlm_preload=enable_vlm_preload,
        )
    )

    if latex_delimiters_type == 'a':
        latex_delimiters = latex_delimiters_type_a
    elif latex_delimiters_type == 'b':
        latex_delimiters = latex_delimiters_type_b
    elif latex_delimiters_type == 'all':
        latex_delimiters = latex_delimiters_type_all
    else:
        raise ValueError(f"Invalid latex delimiters type: {latex_delimiters_type}.")


    async def convert_to_markdown_stream(
        file_path,
        end_pages=10,
        is_ocr=False,
        formula_enable=True,
        table_enable=True,
        image_analysis=True,
        language="ch",
        backend="pipeline",
        url=None,
    ):
        async for update in stream_to_markdown(
            file_path=file_path,
            end_pages=end_pages,
            is_ocr=is_ocr,
            formula_enable=formula_enable,
            table_enable=table_enable,
            image_analysis=image_analysis,
            language=language,
            backend=backend,
            url=url,
            api_url=api_url,
        ):
            update = (
                render_status_steps_html(update[0], i18n),
                update[1],
                prepare_markdown_for_gradio_preview(update[2], latex_delimiters),
                *update[3:],
            )
            yield update

    suffixes = [f".{suffix}" for suffix in pdf_suffixes + image_suffixes + office_suffixes]
    _blocks_kwargs = {} if IS_GRADIO_6 else {"css": APP_CSS, "js": APP_JS}
    with gr.Blocks(**_blocks_kwargs) as demo:
        gr.HTML(render_header_html(i18n))
        with gr.Row(elem_classes=["mineru-workspace-row"]):
            with gr.Column(variant='panel', scale=2, min_width=280, elem_classes=["mineru-control-column"]):
                input_file = gr.File(
                    label=i18n("upload_file"),
                    file_types=suffixes,
                    elem_classes=["mineru-upload-file"],
                )
                preferred_option = "hybrid-auto-engine"
                backend = gr.Dropdown(
                    build_backend_choices(http_client_enable, i18n),
                    label=i18n("backend"),
                    value=preferred_option,
                    info=get_backend_info(preferred_option),
                )
                # 下面这些选项在上传 office 文件时会被自动隐藏
                with gr.Group() as options_group:
                    max_pages = gr.Slider(1, max_convert_pages, max_convert_pages, step=1, label=i18n("max_pages"))
                    advanced_bu = gr.Button(
                        i18n("advanced_options"),
                        size="sm",
                        elem_classes=["mineru-advanced-open"],
                    )
                with gr.Row(elem_classes=["mineru-actions"]):
                    change_bu = gr.Button(i18n("convert"), variant="primary", scale=1, min_width=0)
                    clear_bu = gr.ClearButton(value=i18n("clear"), scale=1, min_width=0)
                output_file = gr.File(
                    label=i18n("convert_result"),
                    interactive=False,
                    elem_classes=["mineru-result-file"],
                )
                status_panel = gr.HTML(
                    value=render_status_steps_html("", i18n),
                    label=i18n("convert_status"),
                    elem_classes=["mineru-status-panel"],
                )
                if example_enable:
                    example_root = os.path.join(os.getcwd(), 'examples')
                    if os.path.exists(example_root):
                        example_files = [
                            os.path.join(example_root, _) for _ in os.listdir(example_root)
                            if _.endswith(tuple(suffixes))
                        ]
                        if example_files:
                            with gr.Accordion(i18n("examples"), open=True):
                                gr.Examples(
                                    examples=example_files,
                                    inputs=input_file,
                                    elem_id="mineru-example-files",
                                    label=None,
                                )

            with gr.Column(variant='panel', scale=8, min_width=560, elem_classes=["mineru-work-column"]):
                with gr.Row(equal_height=True, elem_classes=["mineru-compare-row"]):
                    with gr.Column(scale=1, min_width=340, elem_classes=["mineru-preview-pane"]):
                        _doc_preview_label = "doc preview" if IS_GRADIO_6 else i18n("doc_preview")
                        doc_show = PDF(label=_doc_preview_label, interactive=False, visible=True, height=720)
                        office_html = gr.HTML(value="", visible=False, min_height=760)
                    with gr.Column(scale=1, min_width=340, elem_classes=["mineru-markdown-pane"]):
                        _md_copy_kwargs = {"buttons": ["copy"]} if IS_GRADIO_6 else {"show_copy_button": True}
                        _textarea_copy_kwargs = {"buttons": ["copy"]} if IS_GRADIO_6 else {"show_copy_button": True}
                        with gr.Tabs(elem_classes=["mineru-markdown-tabs"]):
                            with gr.Tab(i18n("md_rendering")):
                                md = gr.Markdown(
                                    label=i18n("md_rendering"),
                                    height=684,
                                    elem_classes=["mineru-markdown-output"],
                                    latex_delimiters=latex_delimiters,
                                    line_breaks=True,
                                    **_md_copy_kwargs
                                )
                            with gr.Tab(i18n("md_text")):
                                md_text = gr.TextArea(
                                    lines=28,
                                    label=i18n("md_text"),
                                    elem_classes=["mineru-markdown-text"],
                                    **_textarea_copy_kwargs
                                )

        with gr.Column(elem_classes=["mineru-advanced-popover"]):
            with gr.Column(elem_classes=["mineru-advanced-card"]):
                gr.Markdown(
                    f"### {translate_ui(i18n, 'advanced_options')}",
                    elem_classes=["mineru-advanced-popover-title"],
                )
                with gr.Row(visible=False) as client_options:
                    url = gr.Textbox(
                        label=i18n("server_url"),
                        value='http://localhost:30000',
                        placeholder='http://localhost:30000',
                        info=i18n("server_url_info"),
                    )
                with gr.Group():
                    table_enable = gr.Checkbox(label=i18n("table_enable"), value=True, info=i18n("table_info"))
                    formula_enable = gr.Checkbox(label=get_formula_label(preferred_option), value=True, info=get_formula_info(preferred_option))
                    image_analysis = gr.Checkbox(
                        label=i18n("image_analysis_enable"),
                        value=True,
                        visible=is_image_analysis_option_visible(preferred_option),
                        info=i18n("image_analysis_info"),
                    )
                with gr.Group() as ocr_options:
                    language = gr.Dropdown(
                        all_lang,
                        label=i18n("ocr_language"),
                        value='ch (Chinese, English, Chinese Traditional)',
                        info=i18n("ocr_language_info"),
                    )
                    is_ocr = gr.Checkbox(label=i18n("force_ocr"), value=False, info=i18n("force_ocr_info"))

        # 添加事件处理
        _private_api_kwargs = (
            {"api_visibility": "private", "queue": False}
            if IS_GRADIO_6
            else {"api_name": False, "queue": False}
        )
        backend.change(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend, image_analysis],
            **_private_api_kwargs
        )
        # 添加demo.load事件，在页面加载时触发一次界面更新
        demo.load(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend, image_analysis],
            **_private_api_kwargs
        )
        clear_bu.add([input_file, md, doc_show, md_text, output_file, is_ocr, office_html, status_panel])

        def reset_primary_ui():
            """清除主界面状态。高级气泡由前端点击外部逻辑自动收起。"""
            return (
                gr.update(visible=True),
                gr.update(value=None, visible=True),
                gr.update(value="", visible=False),
                gr.update(value=render_status_steps_html("", i18n)),
            )

        # 清除按钮额外重置 UI 可见性（ClearButton 不一定触发 input_file.change）
        clear_bu.click(
            fn=reset_primary_ui,
            inputs=[],
            outputs=[options_group, doc_show, office_html, status_panel],
            **_private_api_kwargs
        )

        def update_file_options_html_for_ui(file_path, request: gr.Request):
            """绑定当前 i18n 的文件上传 UI 更新函数，避免事件签名暴露额外参数。"""
            return update_file_options_html(file_path, request, i18n)

        # 第一阶段：快速更新 options_group 和 office_html，不涉及 gradio_pdf 组件
        # 第二阶段（.then）：单独更新 doc_show，使 office_html 的 processing 遮罩
        # 在第一阶段完成后立即消失，规避 gradio_pdf 0.0.24 与 Gradio 6 的兼容性问题。
        input_file.change(
            fn=update_file_options_html_for_ui,
            inputs=input_file,
            outputs=[options_group, office_html],
            **_private_api_kwargs
        ).then(
            fn=update_doc_show,
            inputs=input_file,
            outputs=[doc_show],
            **_private_api_kwargs
        )
        _to_md_api_kwargs = (
            {
                "api_visibility": "public" if api_enable else "private",
                "queue": True,
                "show_progress": "hidden",
            }
            if IS_GRADIO_6
            else {
                "api_name": "to_markdown" if api_enable else False,
                "queue": True,
                "show_progress": "hidden",
            }
        )
        change_bu.click(
            fn=convert_to_markdown_stream,
            inputs=[input_file, max_pages, is_ocr, formula_enable, table_enable, image_analysis, language, backend, url],
            outputs=[status_panel, output_file, md, md_text, doc_show],
            **_to_md_api_kwargs
        )

    demo.queue(default_concurrency_limit=None)

    if IS_GRADIO_6:
        footer_links = ["gradio", "settings"]
        if api_enable:
            footer_links.append("api")
        _launch_kwargs = {"footer_links": footer_links, "css": APP_CSS, "head": APP_HEAD}
    else:
        _launch_kwargs = {"show_api": api_enable}
    maybe_prepare_local_api_for_gradio_startup(
        api_url=api_url,
        enable_vlm_preload=enable_vlm_preload,
    )
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        i18n=i18n,
        **_launch_kwargs,
    )


if __name__ == '__main__':
    main()
