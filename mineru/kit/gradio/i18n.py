"""Gradio 界面的唯一中英文词典，不翻译文档正文或外部错误详情。"""

from __future__ import annotations

import html
import re

# 每项依次为英文和简体中文；同时供 Gradio 属性、Python HTML 和前端事件使用。
MESSAGES: dict[str, tuple[str, str]] = {
    "header_title": ("MinerU 4: Document Extraction", "MinerU 4：文档提取"),
    "header_subtitle": (
        "Open-source document extraction for PDF, Office, EPUB, OFD, HTML, CSV and images.",
        "开源文档提取工具，支持 PDF、Office、EPUB、OFD、HTML、CSV 与图片。",
    ),
    "stars": ("GitHub stars", "GitHub 星标"),
    "code": ("Code", "代码"),
    "model": ("Models", "模型"),
    "paper": ("Papers", "论文"),
    "homepage": ("Homepage", "主页"),
    "download": ("Download", "下载"),
    "upload": ("Select a document to parse", "请选择要解析的文件"),
    "tier": ("Parsing tier", "解析 tier"),
    "tier_value": ("Parsing tier: {tier}{notice}", "解析 tier：{tier}{notice}"),
    "tier_unavailable_suffix": (" (unavailable on this server)", "（当前服务不可用）"),
    "force_ocr": ("Force OCR", "强制 OCR"),
    "force_ocr_info": ("Ignore the PDF text layer and perform OCR", "忽略 PDF 文本层并进行 OCR"),
    "page_range": ("Page range", "页码范围"),
    "start_page": ("Start page", "起始页"),
    "end_page": ("End page", "结束页"),
    "page_count": ("{count} pages", "{count} 页"),
    "page_limit": ("Up to {count} pages", "最多 {count} 页"),
    "page_unlimited": ("No page limit", "不限页数"),
    "reading_pages": ("Reading PDF page count…", "正在读取 PDF 页数…"),
    "page_read_failed": (
        "Cannot read the PDF page count. Check whether the file is damaged or password-protected.",
        "无法读取 PDF 页数，请检查文件是否损坏或需要密码。",
    ),
    "page_limit_exceeded": (
        "At most {limit} pages can be parsed per conversion; {count} pages are selected.",
        "单次最多解析 {limit} 页，当前选择了 {count} 页。",
    ),
    "flash_unavailable": (
        "This format requires Flash, which is unavailable on this server",
        "该格式仅支持 Flash，当前服务不可用",
    ),
    "convert": ("Convert", "转换"),
    "clear": ("Clear", "清除"),
    "preview": ("Document preview", "文档预览"),
    "empty_preview": ("No source document preview", "暂无源文档预览"),
    "source_preview": ("Source document preview", "源文档预览"),
    "unsupported_preview": ("No source preview for this format", "该格式暂无源文档预览"),
    "result_ready": ("Results are ready", "结果已生成"),
    "markdown_rendered": ("Markdown preview", "Markdown 渲染"),
    "markdown_source": ("Markdown source", "Markdown 源码"),
    "structured_source": ("Structured Content source", "Structured Content 源码"),
    "download_results": ("Download results", "下载结果"),
    "latex_bundle": ("LaTeX bundle", "LaTeX 压缩包"),
    "examples": ("Examples", "示例"),
    "preparing_download": ("Preparing…", "准备中…"),
    "download_failed": ("{format} download failed: {error}", "{format} 下载失败：{error}"),
    "missing_download": ("No download file was received. Please try again.", "未获得下载文件，请重试。"),
    "stale_download": ("The parsing result has changed. Please download again.", "解析结果已变更，请重新下载。"),
    "office_preview_title": ("Office online preview", "Office 在线预览"),
    "office_preview_source_link": ("File url", "文件链接"),
    "office_notice": (
        "This preview requires the current file to be reachable by Microsoft Office Online. "
        "Conversion does not depend on this preview.",
        "该预览需要当前文件可被 Microsoft 在线预览服务访问，转换不依赖该预览。",
    ),
    "ignore_once": ("Dismiss", "忽略"),
    "ignore_forever": ("Always dismiss", "不再提示"),
    "status_idle_title": ("Waiting", "等待任务"),
    "status_idle_hint": ("Upload a file and start conversion.", "上传文件后开始转换。"),
    "status_latest": ("Latest status", "最新状态"),
    "status_step_prepare": ("Prepare", "准备请求"),
    "status_step_check": ("Check service", "检查服务"),
    "status_step_submit": ("Submit", "提交任务"),
    "status_step_queue": ("Queue", "排队"),
    "status_step_process": ("Parse", "解析中"),
    "status_step_download": ("Download", "下载结果"),
    "status_step_outputs": ("Build outputs", "整理输出"),
    "status_step_done": ("Done", "完成"),
    "status_step_failed": ("Failed", "失败"),
    "preparing_request": ("Preparing request...", "正在准备请求…"),
    "checking_server": ("Checking server status...", "正在检查服务状态…"),
    "submitting_task": ("Submitting task...", "正在提交任务…"),
    "queued_locally": ("Queued locally", "本地排队中"),
    "queued_on_server": ("Queued on server", "服务端排队中"),
    "processing_on_server": ("Processing on server...", "服务端解析中…"),
    "processing_elapsed": ("Processing on server ({elapsed}s)", "服务端解析中（{elapsed} 秒）"),
    "downloading_result": ("Task completed, downloading result...", "任务已完成，正在下载结果…"),
    "processing_output": ("Preparing outputs...", "正在整理输出…"),
    "completed": ("Completed", "已完成"),
    "completed_elapsed": ("Completed ({elapsed}s)", "已完成（{elapsed} 秒）"),
    "failed": ("Failed: {error}", "失败：{error}"),
    "missing_input": ("input file does not exist", "输入文件不存在"),
    "unsupported_input": ("unsupported file type '.{suffix}'", "不支持的文件类型 '.{suffix}'"),
    "invalid_tier": ("Invalid tier slider position", "解析档位无效"),
}


def translations() -> dict[str, dict[str, str]]:
    """为原生组件提供命名空间词典，中文地区统一使用简体中文。"""
    english = {f"mineru.{key}": pair[0] for key, pair in MESSAGES.items()}
    chinese = {f"mineru.{key}": pair[1] for key, pair in MESSAGES.items()}
    return {"en": english, "zh": chinese, "zh-CN": chinese, "zh-TW": chinese}


def localized_text(key: str, **values: object) -> str:
    """为文本叶节点生成双语标记；参数只作为转义后的普通文本插入。"""
    english, chinese = (template.format(**values) for template in MESSAGES[key])
    return _localized_span(key, english, chinese)


def _localized_span(key: str, english: str, chinese: str) -> str:
    """统一转义标记属性与可见文字，避免错误详情成为活动 HTML。"""
    return (
        f'<span data-mineru-i18n-key="{html.escape(key, quote=True)}"'
        f' data-mineru-i18n-en="{html.escape(english, quote=True)}"'
        f' data-mineru-i18n-zh="{html.escape(chinese, quote=True)}">{html.escape(chinese)}</span>'
    )


def _message_pair(message: str) -> tuple[str, str]:
    """按固定消息及已知参数模板翻译应用提示，未知详情保留原文。"""
    for pair in MESSAGES.values():
        if message in pair:
            return pair
    if message.startswith("Failed: "):
        return tuple(template.format(error=detail) for template, detail in zip(MESSAGES["failed"], _message_pair(message[8:])))
    for prefix in ("tier_unavailable: ", "page_range_invalid: "):
        if message.startswith(prefix):
            return tuple(prefix + detail for detail in _message_pair(message[len(prefix) :]))
    patterns = (
        (r"Processing on server \((?P<elapsed>[\d.]+)s\)", "processing_elapsed"),
        (r"Completed \((?P<elapsed>[\d.]+)s\)", "completed_elapsed"),
        (r"unsupported file type '\.(?P<suffix>[^']*)'", "unsupported_input"),
        (r"单次最多解析 (?P<limit>\d+) 页，当前选择了 (?P<count>\d+) 页。", "page_limit_exceeded"),
    )
    for pattern, key in patterns:
        if match := re.fullmatch(pattern, message):
            return tuple(template.format(**match.groupdict()) for template in MESSAGES[key])
    for key in ("queued_locally", "queued_on_server"):
        english, chinese = MESSAGES[key]
        if message.startswith(english) and set(message[len(english) :]) <= {"."}:
            return message, chinese + message[len(english) :]
    return message, message


def localized_message(message: str) -> str:
    """保持后端状态协议不变，只在生成界面 HTML 时附加双语显示。"""
    return _localized_span("status_message", *_message_pair(message))


def preview_placeholder(key: str) -> str:
    """保留预览占位容器，只对内部文字应用本地化。"""
    return f'<div class="mineru-kit-empty-preview">{localized_text(key)}</div>'


__all__ = ["MESSAGES", "localized_message", "localized_text", "preview_placeholder", "translations"]
