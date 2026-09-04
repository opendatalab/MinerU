"""MinerU Kit 的 V1 Gradio 应用组装与启动入口。"""

from __future__ import annotations

import asyncio
import html
import json
import os
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import quote

from ...errors import MineruError
from ...filetypes import IMAGE_EXTENSIONS, OFFICE_EXTENSIONS, PARSEABLE_EXTENSIONS, PDF_EXTENSIONS
from ...parser.page_range import normalize_page_range_input
from ...types import TIERS, Tier
from ...utils.stdio import configure_standard_streams
from .artifacts import RunArtifacts, markdown_for_gradio, persist_parse_result, render_download
from .client import (
    ManagedLocalApiServer,
    STATUS_COMPLETED,
    STATUS_PREPARING_REQUEST,
    STATUS_PROCESSING_OUTPUT,
    V1ArtifactClient,
    V1ServerCapabilities,
)

_DOWNLOAD_FORMATS: tuple[tuple[str, str], ...] = (
    ("zip", "ZIP"),
    ("html", "HTML"),
    ("docx", "DOCX"),
    ("latex", "LaTeX bundle"),
    ("epub", "EPUB"),
    ("pdf", "PDF"),
)
_DEFAULT_TIER = "standard"
_DEFAULT_STATUS = "Upload a file and start conversion."
_STATUS_STEPS = (
    ("prepare", "准备"),
    ("check", "检查服务"),
    ("submit", "提交任务"),
    ("process", "服务端解析"),
    ("download", "下载结果"),
    ("output", "整理输出"),
    ("done", "完成"),
)
_LATEX_DELIMITERS_A = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]
_LATEX_DELIMITERS_B = [
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]
_DOWNLOAD_ICON_HTML = """
<button type="button" class="mineru-kit-download-icon" title="下载结果" aria-label="下载结果"
        aria-controls="mineru-kit-download-options">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
        <path d="M12 3v12m-5-5 5 5 5-5M5 16v4h14v-4" />
    </svg>
</button>
"""
_KIT_MENU_CSS = """
.mineru-kit-results { position: relative; padding: 12px !important; overflow: visible; }
/* Gradio 5/6 的导航容器不同；只缩窄标题行，让正文继续占满结果栏。 */
.mineru-kit-results > .mineru-markdown-tabs > .tab-nav,
.mineru-kit-results > .mineru-markdown-tabs > .tab-wrapper {
    width: calc(100% - 40px);
}
.mineru-kit-results > .mineru-kit-download-menu {
    position: absolute; top: 12px; right: 12px; z-index: 40;
    width: 32px !important; min-width: 0 !important; height: 32px; gap: 0;
    overflow: visible;
}
.mineru-kit-download-trigger,
.mineru-kit-download-trigger .html-container,
.mineru-kit-download-trigger .prose {
    min-width: 0 !important; padding: 0 !important; margin: 0; line-height: 0; overflow: visible;
}
.mineru-kit-download-trigger { min-height: 32px; border: 0; background: transparent; }
.mineru-kit-download-trigger .mineru-kit-download-icon {
    display: flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; margin: 0; padding: 6px; border: 0; border-radius: 6px;
    color: var(--body-text-color, #1f2937); background: transparent; cursor: pointer;
}
.mineru-kit-download-icon svg { width: 20px; height: 20px; margin: 0; }
.mineru-kit-download-menu:hover .mineru-kit-download-icon,
.mineru-kit-download-icon:focus-visible { background: var(--background-fill-secondary, #f3f4f6); }
.mineru-kit-download-icon:focus-visible { outline: 2px solid var(--mineru-accent, #f97316); outline-offset: 2px; }
.mineru-kit-download-options {
    position: absolute; right: 0; top: calc(100% + 6px); z-index: 40;
    width: 192px !important; min-width: 0 !important;
    display: flex !important; flex-direction: column; gap: 4px; padding: 6px;
    border: 1px solid var(--mineru-panel-border, rgba(17,24,39,.12)); border-radius: 8px;
    background: var(--background-fill-primary, #fff); box-shadow: 0 12px 28px rgba(15,23,42,.18);
    opacity: 0; pointer-events: none; transform: translateY(-4px); visibility: hidden;
    transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
}
.mineru-kit-download-menu:hover .mineru-kit-download-options,
.mineru-kit-download-menu:focus-within .mineru-kit-download-options {
    opacity: 1; pointer-events: auto; transform: translateY(0); visibility: visible;
}
/* 填满图标与浮层之间的间隙，避免鼠标移向下载项时菜单提前关闭。 */
.mineru-kit-download-options::before { content: ""; position: absolute; left: 0; right: 0; top: -7px; height: 7px; }
.mineru-kit-download-options :is(button, a) {
    justify-content: flex-start; width: 100%; min-height: 34px; padding: 6px 10px;
    border: 0; border-radius: 6px; background: transparent; box-shadow: none; text-align: left;
}
.mineru-kit-download-options :is(button, a):hover { background: var(--background-fill-secondary, #f3f4f6); }
.mineru-kit-status { min-height: 84px; }
.mineru-kit-status-steps { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.mineru-kit-status-steps .status-step {
    display: inline-flex; align-items: center; gap: 4px;
    color: var(--body-text-color-subdued, #6b7280); font-size: 12px;
}
.mineru-kit-status-steps .status-step i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; opacity: .35; }
.mineru-kit-status-steps .status-step.is-active { color: var(--mineru-accent, #f97316); font-weight: 600; }
.mineru-kit-status-steps .status-step.is-active i, .mineru-kit-status-steps .status-step.is-done i { opacity: 1; }
.mineru-kit-status-steps .status-step.is-error { color: #dc2626; font-weight: 600; }
.mineru-kit-empty-preview { min-height: 160px; display: grid; place-items: center; opacity: .65; }
.mineru-kit-image-preview img { max-height: var(--mineru-pdf-page-height, 720px); object-fit: contain; }
@media (max-width: 900px) {
  .mineru-kit-workspace { flex-direction: column !important; }
  .mineru-kit-control, .mineru-kit-preview, .mineru-kit-results { min-width: 0 !important; width: 100% !important; }
}
"""


def _resource_text(resource_name: str) -> str:
    """从已安装 MinerU 包资源中读取 Gradio 静态文本。"""
    resource_path = Path(__file__).resolve().parents[2] / "resources" / resource_name
    return resource_path.read_text(encoding="utf-8")


def _render_header(*, gradio_major_version: int = 5) -> str:
    """渲染复用旧版视觉风格的静态 Header。"""
    template = _resource_text("gradio_header.html")
    values = {
        "{{HEADER_TITLE}}": "MinerU 4：文档提取",
        "{{HEADER_SUBTITLE}}": "开源文档提取工具，支持 PDF、Office、EPUB、OFD、HTML、CSV 与图片。",
        "{{HEADER_SUPPORT_TEXT}}": "",
        "{{HEADER_STARS_ALT}}": "GitHub 星标",
        "{{HEADER_CODE_LINK}}": "代码",
        "{{HEADER_MODEL_LINK}}": "模型",
        "{{HEADER_MODEL_HUGGINGFACE_LINK}}": "Hugging Face",
        "{{HEADER_MODEL_MODELSCOPE_LINK}}": "ModelScope",
        "{{HEADER_PAPER_LINK}}": "论文",
        "{{HEADER_PAPER_MINERU_REPORT}}": "MinerU · arXiv",
        "{{HEADER_PAPER_MINERU25_REPORT}}": "MinerU 2.5 · arXiv",
        "{{HEADER_PAPER_MINERU25PRO_REPORT}}": "MinerU 2.5 Pro · arXiv",
        "{{HEADER_HOMEPAGE_LINK}}": "主页",
        "{{HEADER_DOWNLOAD_LINK}}": "下载",
        "{{HEADER_GRADIO_VERSION_CLASS}}": " mineru-gradio6-header" if gradio_major_version >= 6 else "",
    }
    rendered = template
    for placeholder, value in values.items():
        rendered = rendered.replace(placeholder, html.escape(value, quote=True))
    return rendered


def _supported_file_types() -> list[str]:
    """把统一 filetypes 集合转换为 Gradio 文件选择器后缀。"""
    return [f".{extension}" for extension in sorted(PARSEABLE_EXTENSIONS)]


def _default_tier(capabilities: V1ServerCapabilities) -> str:
    """按服务能力选择默认 tier，优先保持 Standard 语义。"""
    if _DEFAULT_TIER in capabilities.tiers:
        return _DEFAULT_TIER
    for tier in ("advanced", "standard", "basic", "flash"):
        if tier in capabilities.tiers:
            return tier
    return capabilities.tiers[0]


def _tier_for_position(position: int | float, tier_choices: list[Tier]) -> Tier:
    """把离散滑块位置映射为服务支持的 tier，并拒绝越界或非整数位置。"""
    if isinstance(position, bool) or position not in range(len(tier_choices)):
        raise ValueError("Invalid tier slider position")
    return tier_choices[int(position)]


def _status_html(message: str) -> str:
    """生成不解释协议细节、只显示当前阶段的安全状态片段。"""
    raw_message = str(message or _DEFAULT_STATUS)
    safe_message = html.escape(raw_message)
    lowered = raw_message.lower()
    if raw_message.startswith("Failed:"):
        current_index = len(_STATUS_STEPS) - 1
        state_for_index = "error"
    elif "downloading" in lowered or "downloaded" in lowered:
        current_index, state_for_index = 4, "active"
    elif "completed" in lowered:
        current_index = len(_STATUS_STEPS) - 1
        state_for_index = "done"
    elif "preparing outputs" in lowered:
        current_index, state_for_index = 5, "active"
    elif "preparing" in lowered:
        current_index, state_for_index = 0, "active"
    elif "checking" in lowered:
        current_index, state_for_index = 1, "active"
    elif "submitting" in lowered:
        current_index, state_for_index = 2, "active"
    elif "processing" in lowered:
        current_index, state_for_index = 3, "active"
    else:
        current_index, state_for_index = -1, "pending"
    items: list[str] = []
    for index, (_key, label) in enumerate(_STATUS_STEPS):
        if state_for_index == "error" and index == current_index:
            state = "is-error"
        elif state_for_index == "done" or index < current_index:
            state = "is-done"
        elif index == current_index:
            state = "is-active"
        else:
            state = "is-pending"
        items.append(f'<span class="status-step {state}"><i></i>{label}</span>')
    return (
        '<div class="mineru-kit-status">'
        f'<div class="mineru-kit-status-steps">{"".join(items)}</div>'
        f"<strong>{safe_message}</strong>"
        "</div>"
    )


def _file_suffix(path_value: str | Path | None) -> str:
    """读取上传文件的小写后缀，供 UI 预览和页码控件判断。"""
    if not path_value:
        return ""
    return Path(path_value).suffix.lower().lstrip(".")


def _effective_page_range(path_value: str | Path | None, raw_page_range: str | None) -> str:
    """仅为 PDF 保留用户输入的 V1 page_range，其它格式统一返回空字符串。"""
    if _file_suffix(path_value) not in PDF_EXTENSIONS:
        return ""
    return normalize_page_range_input(raw_page_range)


def _is_pdf_or_image(path_value: str | Path | None) -> bool:
    """判断上传文件是否可以在 PDF 预览组件中展示。"""
    suffix = _file_suffix(path_value)
    return suffix in PDF_EXTENSIONS or suffix in IMAGE_EXTENSIONS


def _is_office(path_value: str | Path | None) -> bool:
    """判断上传文件是否适合展示 Office 在线预览提示。"""
    return _file_suffix(path_value) in OFFICE_EXTENSIONS


def _preview_update(gr: Any, value: object, *, visible: bool) -> Any:
    """构造跨 Gradio 5/6 兼容的组件更新对象。"""
    return gr.update(value=value, visible=visible)


def _download_updates(gr: Any, *, interactive: bool) -> tuple[Any, ...]:
    """清空全部下载按钮的旧文件，并统一切换交互状态。"""
    return tuple(gr.update(value=None, interactive=interactive) for _format_name, _label in _DOWNLOAD_FORMATS)


def _latex_delimiters(delimiters_type: Literal["a", "b", "all"]) -> list[dict[str, Any]]:
    """按 CLI 选择返回 Gradio Markdown 组件使用的公式分隔符。"""
    if delimiters_type == "a":
        return list(_LATEX_DELIMITERS_A)
    if delimiters_type == "b":
        return list(_LATEX_DELIMITERS_B)
    return [*(_LATEX_DELIMITERS_A), *(_LATEX_DELIMITERS_B)]


def build_gradio_app(
    client: V1ArtifactClient,
    capabilities: V1ServerCapabilities,
    *,
    output_root: Path,
    enable_example: bool = True,
    enable_api: bool = True,
    latex_delimiters_type: Literal["a", "b", "all"] = "all",
) -> Any:
    """构建不启动监听端口的 Gradio Blocks 应用，便于单元测试和外部托管。"""
    import gradio as gr
    from gradio_pdf import PDF

    tier_choices = [tier for tier in TIERS if tier in capabilities.tiers]
    if not tier_choices:
        raise ValueError("V1 API server did not advertise any parsing tier")
    preferred_tier = _default_tier(capabilities)
    file_types = _supported_file_types()
    markdown_copy_kwargs = {"buttons": ["copy"]} if _gradio_major_version(gr) >= 6 else {"show_copy_button": True}

    with gr.Blocks() as demo:
        gr.HTML(
            _render_header(gradio_major_version=_gradio_major_version(gr)),
            elem_classes=["mineru-header-html"],
        )
        with gr.Row(elem_classes=["mineru-kit-workspace"]):
            with gr.Column(scale=2, min_width=280, elem_classes=["mineru-kit-control", "mineru-control-column"]):
                input_file = gr.File(
                    label="请选择要解析的文件",
                    file_types=file_types,
                    file_count="single",
                    type="filepath",
                    elem_classes=["mineru-upload-file"],
                )
                with gr.Group():
                    tier_label = gr.Markdown(
                        value=f"解析 tier：{preferred_tier}",
                        padding=True,
                        elem_classes=["mineru-tier-label"],
                    )
                    tier = gr.Slider(
                        minimum=0,
                        # 单档位时保留非零跨度，避免原生滑块计算进度时除零。
                        maximum=max(1, len(tier_choices) - 1),
                        value=tier_choices.index(preferred_tier),
                        step=1,
                        precision=0,
                        interactive=len(tier_choices) > 1,
                        label="解析 tier",
                        show_label=False,
                        elem_classes=["mineru-tier-slider"],
                    )
                page_range = gr.Textbox(
                    value="",
                    label="页码范围",
                    placeholder="例如 1-5,8,r1 或 all；留空表示全部",
                    interactive=False,
                    elem_classes=["mineru-kit-page-range"],
                )
                with gr.Row(elem_classes=["mineru-actions"]):
                    convert_button = gr.Button("转换", variant="primary", scale=1, min_width=0)
                    clear_button = gr.ClearButton(value="清除", scale=1, min_width=1)
                status_panel = gr.HTML(_status_html(_DEFAULT_STATUS), elem_classes=["mineru-kit-status"])

            with gr.Column(scale=4, min_width=340, elem_classes=["mineru-kit-preview", "mineru-preview-pane"]):
                pdf_preview = PDF(
                    label="文档预览",
                    interactive=False,
                    visible=False,
                    height=720,
                    elem_classes=["mineru-kit-pdf-preview"],
                )
                image_preview = gr.Image(
                    label="文档预览",
                    type="filepath",
                    interactive=False,
                    visible=False,
                    height=720,
                    elem_classes=["mineru-kit-image-preview"],
                )
                office_preview = gr.HTML(
                    value="",
                    visible=False,
                    min_height=320,
                    elem_classes=["mineru-kit-office-preview", "mineru-office-preview-html"],
                )
                generic_preview = gr.HTML(
                    value='<div class="mineru-kit-empty-preview">暂无源文档预览</div>',
                    visible=True,
                    elem_classes=["mineru-kit-generic-preview"],
                )

            with gr.Column(scale=4, min_width=340, elem_classes=["mineru-kit-results", "mineru-markdown-pane"]):
                with gr.Tabs(elem_classes=["mineru-markdown-tabs"]):
                    with gr.Tab("Markdown 渲染"):
                        markdown_output = gr.Markdown(
                            value="",
                            height=775,
                            line_breaks=True,
                            latex_delimiters=_latex_delimiters(latex_delimiters_type),
                            **markdown_copy_kwargs,
                            elem_classes=["mineru-markdown-output"],
                        )
                    with gr.Tab("Markdown 源码"):
                        markdown_source = gr.Code(
                            value="",
                            language="markdown",
                            lines=28,
                            interactive=False,
                            elem_classes=["mineru-markdown-text"],
                        )
                    with gr.Tab("Structured Content 源码"):
                        structured_source = gr.Code(
                            value="",
                            language="json",
                            lines=28,
                            interactive=False,
                            elem_classes=["mineru-structured-content"],
                        )
                with gr.Column(scale=0, min_width=0, elem_classes=["mineru-kit-download-menu"]):
                    gr.HTML(_DOWNLOAD_ICON_HTML, elem_classes=["mineru-kit-download-trigger"])
                    with gr.Column(
                        min_width=0,
                        elem_id="mineru-kit-download-options",
                        elem_classes=["mineru-kit-download-options"],
                    ):
                        download_buttons: dict[str, Any] = {}
                        for format_name, label in _DOWNLOAD_FORMATS:
                            download_buttons[format_name] = gr.DownloadButton(
                                label,
                                visible=True,
                                interactive=False,
                                size="sm",
                                elem_classes=[f"mineru-kit-download-{format_name}"],
                            )

        if enable_example:
            examples = _example_files(file_types)
            if examples:
                gr.Examples(examples=examples, inputs=input_file, label="示例", elem_id="mineru-kit-examples")

        artifact_state = gr.State(value=None)
        clear_button.add(
            [
                input_file,
                page_range,
                markdown_output,
                markdown_source,
                structured_source,
                pdf_preview,
                image_preview,
                office_preview,
                generic_preview,
                status_panel,
            ]
        )

        def update_file_preview(file_path: str | None) -> tuple[Any, ...]:
            """切换源文件预览，并清除上一份文档的结果与下载状态。"""
            reset_result = (_status_html(_DEFAULT_STATUS), "", "", "", None, *_download_updates(gr, interactive=False))
            if not file_path:
                return (
                    gr.update(value="", interactive=False),
                    _preview_update(gr, None, visible=False),
                    _preview_update(gr, None, visible=False),
                    _preview_update(gr, "", visible=False),
                    _preview_update(gr, '<div class="mineru-kit-empty-preview">暂无源文档预览</div>', visible=True),
                    *reset_result,
                )
            suffix = _file_suffix(file_path)
            page_update = gr.update(interactive=suffix in PDF_EXTENSIONS, value="")
            if suffix in PDF_EXTENSIONS:
                return (
                    page_update,
                    _preview_update(gr, file_path, visible=True),
                    _preview_update(gr, None, visible=False),
                    _preview_update(gr, "", visible=False),
                    _preview_update(gr, '<div class="mineru-kit-empty-preview">源文档预览</div>', visible=False),
                    *reset_result,
                )
            if suffix in IMAGE_EXTENSIONS:
                return (
                    page_update,
                    _preview_update(gr, None, visible=False),
                    _preview_update(gr, file_path, visible=True),
                    _preview_update(gr, "", visible=False),
                    _preview_update(gr, '<div class="mineru-kit-empty-preview">源文档预览</div>', visible=False),
                    *reset_result,
                )
            if _is_office(file_path):
                return (
                    page_update,
                    _preview_update(gr, None, visible=False),
                    _preview_update(gr, None, visible=False),
                    _preview_update(
                        gr,
                        '<div class="mineru-kit-empty-preview">Office 文件将在转换后提供结果</div>',
                        visible=True,
                    ),
                    _preview_update(gr, "", visible=False),
                    *reset_result,
                )
            return (
                page_update,
                _preview_update(gr, None, visible=False),
                _preview_update(gr, None, visible=False),
                _preview_update(gr, "", visible=False),
                _preview_update(gr, '<div class="mineru-kit-empty-preview">该格式暂无源文档预览</div>', visible=True),
                *reset_result,
            )

        private_event_kwargs = _private_event_kwargs(gr)
        tier.input(
            fn=None,
            inputs=tier,
            outputs=tier_label,
            js="""(position) => {
                // 在浏览器端即时显示当前档位；实际解析独立读取滑块位置。
                const tiers = %s;
                return ["解析 tier：" + tiers[position]];
            }"""
            % json.dumps(tier_choices),
            **private_event_kwargs,
        )
        input_file.change(
            fn=update_file_preview,
            inputs=input_file,
            outputs=[
                page_range,
                pdf_preview,
                image_preview,
                office_preview,
                generic_preview,
                status_panel,
                markdown_output,
                markdown_source,
                structured_source,
                artifact_state,
                *download_buttons.values(),
            ],
            **private_event_kwargs,
        )

        async def convert_handler(
            file_path: str | None,
            tier_position: int | float,
            raw_page_range: str,
            request: object | None = None,
        ) -> Any:
            """执行单文件 V1 解析并流式更新状态与三个结果标签。"""
            empty_result = (
                _status_html(_DEFAULT_STATUS),
                "",
                "",
                "",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                gr.update(value='<div class="mineru-kit-empty-preview">暂无源文档预览</div>', visible=True),
                None,
                *_download_updates(gr, interactive=False),
            )
            if not file_path:
                yield empty_result
                return
            source_path = Path(file_path).resolve()
            if not source_path.is_file():
                yield (_status_html("Failed: input file does not exist"), "", "", "", *empty_result[4:])
                return
            suffix = _file_suffix(source_path)
            if suffix not in PARSEABLE_EXTENSIONS:
                yield (_status_html(f"Failed: unsupported file type '.{suffix}'"), "", "", "", *empty_result[4:])
                return
            try:
                selected_tier = _tier_for_position(tier_position, tier_choices)
            except ValueError as exc:
                yield (_status_html(f"Failed: {exc}"), "", "", "", *empty_result[4:])
                return
            try:
                page_text = _effective_page_range(source_path, raw_page_range)
            except MineruError as exc:
                yield (_status_html(f"Failed: {exc.code}: {exc}"), "", "", "", *empty_result[4:])
                return
            yield (_status_html(STATUS_PREPARING_REQUEST), "", "", "", *empty_result[4:])
            status_queue: asyncio.Queue[str] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def emit(message: str) -> None:
                """把同步 client 状态安全投递到当前 Gradio 事件循环。"""
                loop.call_soon_threadsafe(status_queue.put_nowait, message)

            task = asyncio.create_task(
                client.parse_file(
                    source_path,
                    tier=selected_tier,
                    page_range=page_text,
                    status_callback=emit,
                )
            )
            try:
                while not task.done():
                    try:
                        message = await asyncio.wait_for(status_queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                    yield (_status_html(message), "", "", "", *empty_result[4:])
                while not status_queue.empty():
                    yield (_status_html(status_queue.get_nowait()), "", "", "", *empty_result[4:])
                result = await task
                emit(STATUS_PROCESSING_OUTPUT)
                yield (_status_html(STATUS_PROCESSING_OUTPUT), "", "", "", *empty_result[4:])
                artifacts = await asyncio.to_thread(
                    persist_parse_result,
                    result,
                    source_path,
                    output_root=output_root,
                    page_range=page_text,
                )
                markdown_text = artifacts.markdown_path.read_text(encoding="utf-8")
                structured_text = artifacts.structured_content_path.read_text(encoding="utf-8")
                preview_path = artifacts.layout_pdf_path or artifacts.origin_pdf_path
                office_html = _build_office_result_html(artifacts, request) if _is_office(source_path) else ""
                generic_html = "" if preview_path or office_html else '<div class="mineru-kit-empty-preview">结果已生成</div>'
                show_image_preview = suffix in IMAGE_EXTENSIONS and preview_path is None
                yield (
                    _status_html(STATUS_COMPLETED),
                    markdown_for_gradio(markdown_text, artifacts),
                    markdown_text,
                    structured_text,
                    gr.update(value=str(preview_path) if preview_path else None, visible=preview_path is not None),
                    gr.update(value=str(source_path) if show_image_preview else None, visible=show_image_preview),
                    gr.update(value=office_html, visible=bool(office_html)),
                    gr.update(value=generic_html, visible=bool(generic_html)),
                    artifacts.as_state(),
                    *_download_updates(gr, interactive=True),
                )
            except Exception as exc:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                yield (_status_html(f"Failed: {exc}"), "", "", "", *empty_result[4:])

        convert_outputs = [
            status_panel,
            markdown_output,
            markdown_source,
            structured_source,
            pdf_preview,
            image_preview,
            office_preview,
            generic_preview,
            artifact_state,
            *download_buttons.values(),
        ]
        # Gradio 在读取函数签名时需要真实的 Request 类型对象；注解在运行时补回以保持延迟导入。
        convert_handler.__annotations__["request"] = gr.Request
        event_kwargs: dict[str, Any] = {"queue": True, "show_progress": "hidden"}
        if _gradio_major_version(gr) >= 6:
            event_kwargs["api_visibility"] = "public" if enable_api else "private"
        else:
            event_kwargs["api_name"] = "to_markdown" if enable_api else False
        convert_button.click(
            fn=convert_handler,
            inputs=[input_file, tier, page_range],
            outputs=convert_outputs,
            **event_kwargs,
        )

        reset_outputs = [
            status_panel,
            markdown_output,
            markdown_source,
            structured_source,
            pdf_preview,
            image_preview,
            office_preview,
            generic_preview,
            artifact_state,
            *download_buttons.values(),
        ]

        def reset_ui() -> tuple[Any, ...]:
            """清除当前任务结果并恢复空预览状态。"""
            return (
                _status_html(_DEFAULT_STATUS),
                "",
                "",
                "",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                gr.update(value='<div class="mineru-kit-empty-preview">暂无源文档预览</div>', visible=True),
                None,
                *_download_updates(gr, interactive=False),
            )

        clear_button.click(fn=reset_ui, inputs=[], outputs=reset_outputs, queue=False)

        for format_name, _label in _DOWNLOAD_FORMATS:
            download_buttons[format_name].click(
                fn=_download_handler(format_name, output_root),
                inputs=artifact_state,
                outputs=download_buttons[format_name],
                queue=True,
                api_name=False,
            )

    demo._mineru_kit_css = _resource_text("gradio_app.css") + _KIT_MENU_CSS
    demo._mineru_kit_js = _resource_text("gradio_app.js")
    demo.queue(default_concurrency_limit=1)
    return demo


def launch_gradio(
    *,
    api_url: str | None,
    api_key: str | None,
    server_name: str,
    server_port: int | None,
    output_dir: str,
    enable_example: bool,
    enable_api: bool,
    latex_delimiters_type: Literal["a", "b", "all"],
    api_server_tier: str,
    api_server_no_flash: bool,
    api_server_concurrency: int,
    api_server_language: str,
    api_server_ocr_mode: Literal["auto", "txt", "ocr"],
    api_server_disable_image_analysis: bool,
    api_server_preload_models: bool,
) -> None:
    """启动 Gradio；未指定端口时自动选择，未指定外部 URL 时托管本地 V1 API server。"""
    configure_standard_streams()
    resolved_api_key = api_key if api_key is not None else os.environ.get("MINERU_API_KEY")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    managed_server: ManagedLocalApiServer | None = None
    resolved_api_url = api_url
    try:
        if resolved_api_url is None:
            managed_server = ManagedLocalApiServer(
                tier=api_server_tier,  # type: ignore[arg-type]
                no_flash=api_server_no_flash,
                concurrency=api_server_concurrency,
                language=api_server_language,
                ocr_mode=api_server_ocr_mode,
                disable_image_analysis=api_server_disable_image_analysis,
                preload_models=api_server_preload_models,
                api_key=resolved_api_key,
            )
            resolved_api_url = managed_server.start()
        client = V1ArtifactClient(api_url=resolved_api_url, api_key=resolved_api_key)
        capabilities = asyncio.run(client.discover())
        demo = build_gradio_app(
            client,
            capabilities,
            output_root=output_root,
            enable_example=enable_example,
            enable_api=enable_api,
            latex_delimiters_type=latex_delimiters_type,
        )
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            allowed_paths=[str(output_root)],
            css=getattr(demo, "_mineru_kit_css", None),
            js=getattr(demo, "_mineru_kit_js", None),
        )
    finally:
        if managed_server is not None:
            managed_server.stop()


def _example_files(file_types: list[str]) -> list[str]:
    """读取当前工作目录 examples 下的受支持示例文件。"""
    example_root = Path.cwd() / "examples"
    if not example_root.is_dir():
        return []
    suffixes = set(file_types)
    return [str(path) for path in sorted(example_root.iterdir()) if path.is_file() and path.suffix.lower() in suffixes]


def _build_office_result_html(artifacts: RunArtifacts, request: Any = None) -> str:
    """为 Office 结果生成带可选在线预览的安全提示，转换不依赖该 iframe。"""
    source_name = html.escape(artifacts.source_path.name, quote=True)
    headers = getattr(request, "headers", None) or {}
    host = headers.get("x-forwarded-host") or headers.get("host") or "localhost:7860"
    protocol = headers.get("x-forwarded-proto") or "http"
    public_url = f"{protocol}://{host}/gradio_api/file={quote(str(artifacts.source_path), safe='/:')}"
    viewer_url = "https://view.officeapps.live.com/op/embed.aspx?src=" + quote(public_url, safe="")
    return (
        '<div class="office-preview-shell">'
        '<div class="office-preview-notice">'
        '<div class="office-preview-copy">'
        f"<strong>{source_name} 已完成解析</strong>"
        "<span>Office 在线预览依赖外部服务，转换结果不依赖该预览。</span>"
        f'<div class="office-preview-source-link">{html.escape(public_url, quote=True)}</div>'
        "</div>"
        '<div class="office-preview-actions">'
        '<button type="button" class="office-preview-ignore-once">忽略</button>'
        '<button type="button" class="office-preview-ignore-forever">不再提示</button>'
        "</div>"
        "</div>"
        f'<iframe class="office-preview-frame" src="{html.escape(viewer_url, quote=True)}" frameborder="0"></iframe>'
        "</div>"
    )


def _gradio_major_version(gr: Any) -> int:
    """读取 Gradio 主版本，用于兼容 Gradio 5/6 事件参数。"""
    raw_version = str(getattr(gr, "__version__", "5"))
    try:
        return int(raw_version.split(".", 1)[0])
    except (TypeError, ValueError):
        return 5


def _private_event_kwargs(gr: Any) -> dict[str, Any]:
    """生成不把内部文件事件公开为 Gradio API 的兼容参数。"""
    if _gradio_major_version(gr) >= 6:
        return {"queue": False, "api_visibility": "private"}
    return {"queue": False, "api_name": False}


def _download_handler(format_name: str, output_root: Path) -> Callable[[object], str]:
    """创建一个绑定格式和 output root 的 Gradio 下载回调。"""

    def handler(state: object) -> str:
        """从当前任务 State 生成或读取指定格式的下载文件。"""
        return render_download(state, format_name, allowed_root=output_root)

    return handler


__all__ = ["build_gradio_app", "launch_gradio"]
