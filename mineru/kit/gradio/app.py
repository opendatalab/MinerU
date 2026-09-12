"""MinerU Kit 的 V1 Gradio 应用组装与启动入口。"""

from __future__ import annotations

import asyncio
import html
import json
import os
import time
from contextlib import aclosing
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import quote

from ...errors import MineruError
from ...filetypes import FLASH_ONLY_PARSE_EXTENSIONS, IMAGE_EXTENSIONS, OFFICE_EXTENSIONS, PARSEABLE_EXTENSIONS, PDF_EXTENSIONS
from ...types import TIERS, Tier
from ...utils.stdio import configure_standard_streams
from .artifacts import RunArtifacts, persist_parse_result, render_download, render_html_preview
from .client import (
    GradioArtifactClient,
    ManagedLocalApiServer,
    V1ArtifactClient,
    V1ServerCapabilities,
)
from .i18n import MESSAGES, localized_text, preview_placeholder, translations
from .ofd_preview import prepare_ofd_preview
from .page_range import effective_page_range as _effective_page_range
from .page_range import pdf_page_metadata, validate_max_pages
from .pdf_preview import pdf_preview_js, register_pdf_preview_resources
from .status import (
    DEFAULT_STATUS as _DEFAULT_STATUS,
    STATUS_COMPLETED,
    STATUS_PREPARING_REQUEST,
    STATUS_PROCESSING_OUTPUT,
    STATUS_QUEUED_LOCALLY,
    StatusPanelState,
    status_html as _status_html,
    stream_status_updates,
)

_DOWNLOAD_FORMATS: tuple[tuple[str, str], ...] = (
    ("markdown", "Markdown"),
    ("json", "JSON"),
    ("html", "HTML"),
    ("docx", "DOCX"),
    ("latex", "LaTeX"),
    ("epub", "EPUB"),
    ("pdf", "PDF"),
)
# 下载图标采用统一线宽：Markdown 标记、JSON 花括号、HTML 标签、文档、公式、书籍和 PDF 文件。
_DOWNLOAD_ICON_PATHS: dict[str, str] = {
    "markdown": "M3 17V7l4 5 4-5v10M15 13l3 4 3-4M18 7v10",
    "json": "M9 4H7a2 2 0 0 0-2 2v3a3 3 0 0 1-2 3 3 3 0 0 1 2 3v3a2 2 0 0 0 2 2h2"
    "M15 4h2a2 2 0 0 1 2 2v3a3 3 0 0 0 2 3 3 3 0 0 0-2 3v3a2 2 0 0 1-2 2h-2",
    "html": "M7 7l-5 5 5 5M17 7l5 5-5 5M14 4l-4 16",
    "docx": "M14 3H5v18h14V8l-5-5v5h5M8 12h8M8 16h8",
    "latex": "M19 5H5l8 7-8 7h14M19 5v3M19 16v3",
    "epub": "M12 6c-3-2-6-2-10-2v15c4 0 7 0 10 2 3-2 6-2 10-2V4c-4 0-7 0-10 2v15",
    "pdf": "M14 3H5v18h14V8l-5-5v5h5M8 17c3-4 5-8 4-8-2 0-1 7 4 7 3 0-5-3-8 1-1 2 1 1 2 0",
}
_DEFAULT_TIER = "standard"
_LATEX_DELIMITERS_A = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]
_LATEX_DELIMITERS_B = [
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]
_DOWNLOAD_ICON_HTML = f"""
<button type="button" class="mineru-kit-download-icon" title="Download results" aria-label="Download results"
        data-mineru-i18n-key="download_results" data-mineru-i18n-attr="title aria-label"
        aria-controls="mineru-kit-download-options">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
        <path d="M12 3v12m-5-5 5 5 5-5M5 16v4h14v-4" />
    </svg>
    {localized_text("download")}
</button>
"""
_KIT_MENU_CSS = """
.mineru-kit-results {
    --mineru-download-trigger-width: 108px;
    position: relative; padding: 12px !important; overflow: visible;
    /* 保留原输出栏总高度，将空提示区释放的空间交给正文。 */
    min-height: calc(var(--mineru-preview-content-height) + 110px);
}
.mineru-kit-results > .mineru-markdown-tabs { display: flex !important; flex-direction: column; gap: 16px; }
.mineru-kit-results [role="tabpanel"] { margin-top: 0 !important; }
.mineru-kit-results > .mineru-markdown-tabs,
.mineru-kit-results [role="tabpanel"],
.mineru-kit-results [role="tabpanel"] > .column { flex: 1 1 0; min-height: 0; }
.mineru-kit-results .mineru-markdown-output { flex: 1 1 0; height: 100%; min-height: 0 !important; }
.mineru-kit-results .mineru-markdown-output .html-container,
.mineru-kit-results .mineru-markdown-output .prose { height: 100%; }
.mineru-kit-results .mineru-rendered-html-frame { height: 100% !important; }
.mineru-kit-results .mineru-structured-json { flex: 1 1 0; height: 100%; min-height: 0 !important; }
.mineru-structured-json .code-container,
.mineru-structured-json .cm-editor { height: 100% !important; min-height: 0 !important; max-height: none !important; }
.mineru-structured-json .cm-scroller { overflow: auto; }
.mineru-kit-download-notice:not(:has([role="alert"])) { display: none !important; }
/* 只缩窄 Gradio 6 的标题行，让正文继续占满结果栏。 */
.mineru-kit-results > .mineru-markdown-tabs > .tab-wrapper {
    width: calc(100% - var(--mineru-download-trigger-width) - 8px);
    margin-bottom: 0 !important;
}
.mineru-kit-results > .mineru-kit-download-menu {
    position: absolute; top: 12px; right: 12px; z-index: 40;
    width: var(--mineru-download-trigger-width) !important; min-width: 0 !important; height: 32px; gap: 0;
    overflow: visible;
}
.mineru-kit-download-trigger,
.mineru-kit-download-trigger .html-container,
.mineru-kit-download-trigger .prose {
    min-width: 0 !important; padding: 0 !important; margin: 0; line-height: 0; overflow: visible;
}
.mineru-kit-download-trigger { min-height: 32px; border: 0; background: transparent; }
.mineru-kit-download-trigger .mineru-kit-download-icon {
    display: flex; align-items: center; justify-content: center; gap: 6px;
    width: 100%; height: 32px; margin: 0; padding: 6px 8px; border: 0; border-radius: 6px;
    font-size: 14px; line-height: 20px; white-space: nowrap;
    color: var(--body-text-color, #1f2937); background: transparent; cursor: pointer;
}
.mineru-kit-download-icon svg { width: 20px; height: 20px; margin: 0; flex: 0 0 20px; }
.mineru-kit-download-menu:hover .mineru-kit-download-icon,
.mineru-kit-download-icon:focus-visible { background: var(--background-fill-secondary, #f3f4f6); }
.mineru-kit-download-icon:focus-visible { outline: 2px solid var(--mineru-accent, #f97316); outline-offset: 2px; }
.mineru-kit-download-options {
    position: absolute; right: 0; top: calc(100% + 6px); z-index: 40;
    width: max-content !important; min-width: 0 !important;
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
    justify-content: flex-start; width: 100%; min-height: 34px; padding: 6px 10px; white-space: nowrap;
    border: 0; border-radius: 6px; background: transparent; box-shadow: none; text-align: left; gap: 8px;
}
.mineru-kit-download-options button::before {
    content: ""; display: block; width: 16px; height: 16px; flex: 0 0 16px;
    background-color: currentColor;
    -webkit-mask: var(--mineru-download-format-icon) center / contain no-repeat;
    mask: var(--mineru-download-format-icon) center / contain no-repeat;
}
.mineru-kit-download-options :is(button, a):hover { background: var(--background-fill-secondary, #f3f4f6); }
.mineru-kit-empty-preview { min-height: 160px; display: grid; place-items: center; opacity: .65; }
/* Gradio 6.8 会按逗号拆分并重写选择器，PDF/OFD 使用独立选择器避免破坏 :has。 */
/* PDF/OFD 直接贴合面板边框，独立预览不再沿用旧组件的标签留白与额外高度。 */
.mineru-kit-preview:has(.mineru-pdf-frame), .mineru-kit-preview:has(.mineru-ofd-frame) { padding: 0; gap: 0; overflow: hidden; }
.mineru-kit-preview > .block.mineru-kit-pdf-preview,
.mineru-kit-preview > .block.mineru-kit-ofd-preview {
    height: var(--mineru-preview-content-height, 775px) !important;
    min-height: 0 !important; max-height: none !important;
}
.mineru-kit-pdf-preview, .mineru-kit-ofd-preview { height: 100%; padding: 0 !important; }
.mineru-kit-pdf-preview .html-container, .mineru-kit-pdf-preview .prose,
.mineru-kit-ofd-preview .html-container, .mineru-kit-ofd-preview .prose { height: 100%; padding: 0 !important; }
.mineru-kit-pdf-preview:not(:has(.mineru-pdf-frame, [role="alert"])) { display: none !important; }
.mineru-pdf-frame, .mineru-ofd-frame { display: block; width: 100%; height: 100%; border: 0; }
.mineru-kit-ofd-preview:not(:has(iframe)):not(:has([data-mineru-i18n-key])) { display: none !important; }
.mineru-kit-image-preview img { max-height: var(--mineru-pdf-page-height, 720px); object-fit: contain; }
/* 桌面两栏共用行高，PDF/OFD 填满伸展后的面板；窄屏仍采用独立预览高度。 */
@media (min-width: 901px) {
  .mineru-kit-results, .mineru-kit-preview:has(.mineru-pdf-frame),
  .mineru-kit-preview:has(.mineru-ofd-frame) { align-self: stretch !important; height: auto; }
  .mineru-kit-preview > .block.mineru-kit-pdf-preview,
  .mineru-kit-preview > .block.mineru-kit-ofd-preview { flex: 1 1 0; height: auto !important; }
}
@media (max-width: 900px) {
  .mineru-kit-workspace { flex-direction: column !important; }
  .mineru-kit-control, .mineru-kit-preview, .mineru-kit-results { min-width: 0 !important; width: 100% !important; }
}
"""


def _resource_text(resource_name: str) -> str:
    """从已安装 MinerU 包资源中读取 Gradio 静态文本。"""
    resource_path = Path(__file__).resolve().parents[2] / "resources" / resource_name
    return resource_path.read_text(encoding="utf-8")


def _download_icon_css() -> str:
    """为各下载格式生成本地 SVG 遮罩，随按钮文字适配主题且不改变无障碍名称。"""
    rules = []
    for format_name, path in _DOWNLOAD_ICON_PATHS.items():
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="{path}"/></svg>'
        )
        rules.append(
            f'.mineru-kit-download-{format_name} {{ --mineru-download-format-icon: url("data:image/svg+xml,{quote(svg)}"); }}'
        )
    return "\n".join(rules)


def _render_header() -> str:
    """渲染复用旧版视觉风格的静态 Header。"""
    template = _resource_text("gradio_header.html")
    values = {
        "{{HEADER_MODEL_HUGGINGFACE_LINK}}": "Hugging Face",
        "{{HEADER_MODEL_MODELSCOPE_LINK}}": "ModelScope",
        "{{HEADER_PAPER_MINERU_REPORT}}": "MinerU · arXiv",
        "{{HEADER_PAPER_MINERU25_REPORT}}": "MinerU 2.5 · arXiv",
        "{{HEADER_PAPER_MINERU25PRO_REPORT}}": "MinerU 2.5 Pro · arXiv",
    }
    for placeholder, key in {
        "HEADER_TITLE": "header_title",
        "HEADER_SUBTITLE": "header_subtitle",
        "HEADER_SUPPORT_TEXT": "header_support_text",
        "HEADER_CODE_LINK": "code",
        "HEADER_MODEL_LINK": "model",
        "HEADER_PAPER_LINK": "paper",
        "HEADER_HOMEPAGE_LINK": "homepage",
    }.items():
        template = template.replace("{{" + placeholder + "}}", localized_text(key))
    template = template.replace(
        'alt="{{HEADER_STARS_ALT}}"', 'alt="GitHub stars" data-mineru-i18n-attr="alt" data-mineru-i18n-key="stars"'
    )
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


def _file_suffix(path_value: str | Path | None) -> str:
    """读取上传文件的小写后缀，供预览、选页与 OCR 控件判断。"""
    if not path_value:
        return ""
    return Path(path_value).suffix.lower().lstrip(".")


def _is_pdf_or_image(path_value: str | Path | None) -> bool:
    """判断上传文件是否可以在 PDF 预览组件中展示。"""
    suffix = _file_suffix(path_value)
    return suffix in PDF_EXTENSIONS or suffix in IMAGE_EXTENSIONS


def _is_office(path_value: str | Path | None) -> bool:
    """判断上传文件是否适合展示 Office 在线预览提示。"""
    return _file_suffix(path_value) in OFFICE_EXTENSIONS


def _preview_update(gr: Any, value: object, *, visible: bool) -> Any:
    """构造原生组件的内容与显隐更新。"""
    return gr.update(value=value, visible=visible)


def _pdf_preview_update(gr: Any, value: str | None) -> Any:
    """保持文件输出的 FileData 契约，实际预览由独立 HTML 组件承载。"""
    return gr.update(value=value, visible=False)


def _download_updates(gr: Any, *, interactive: bool, run_id: str = "") -> tuple[Any, ...]:
    """同步当前结果标识，并恢复全部下载按钮的标签与交互状态。"""
    return (run_id, *(gr.update(value=label, interactive=interactive) for _format_name, label in _DOWNLOAD_FORMATS))


def _latex_delimiters(delimiters_type: Literal["a", "b", "all"]) -> list[dict[str, Any]]:
    """按 CLI 选择返回 Gradio Markdown 组件使用的公式分隔符。"""
    if delimiters_type == "a":
        return list(_LATEX_DELIMITERS_A)
    if delimiters_type == "b":
        return list(_LATEX_DELIMITERS_B)
    return [*(_LATEX_DELIMITERS_A), *(_LATEX_DELIMITERS_B)]


def build_gradio_app(
    client: V1ArtifactClient | GradioArtifactClient,
    capabilities: V1ServerCapabilities,
    *,
    output_root: Path,
    enable_example: bool = True,
    enable_api: bool = True,
    latex_delimiters_type: Literal["a", "b", "all"] = "all",
    max_pages: int | None = None,
) -> Any:
    """构建不启动监听端口的 Gradio Blocks 应用，便于单元测试和外部托管。"""
    import gradio as gr

    viewer_entry = register_pdf_preview_resources()
    validate_max_pages(max_pages)
    tier_choices = [tier for tier in TIERS if tier in capabilities.tiers]
    if not tier_choices:
        raise ValueError("V1 API server did not advertise any parsing tier")
    preferred_tier = _default_tier(capabilities)
    file_types = _supported_file_types()
    app_css = _resource_text("gradio_app.css") + _KIT_MENU_CSS + _download_icon_css()
    i18n = gr.I18n(**translations())
    app_js = _resource_text("gradio_app.js").replace(
        "__MINERU_I18N__",
        f"({_resource_text('gradio_i18n.js')})({json.dumps(MESSAGES, ensure_ascii=False)})",
    )
    # 等待限制放在生成器内部，使其他会话也能立即显示本地排队状态。
    conversion_slot = asyncio.Semaphore(1)
    session_tasks: dict[str, set[asyncio.Task[tuple[Any, ...]]]] = {}

    with gr.Blocks() as demo:
        gr.HTML(
            _render_header(),
            elem_classes=["mineru-header-html"],
        )
        with gr.Row(elem_classes=["mineru-kit-workspace"]):
            with gr.Column(scale=2, min_width=280, elem_classes=["mineru-kit-control", "mineru-control-column"]):
                input_file = gr.File(
                    label=i18n("mineru.upload"),
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
                        label=i18n("mineru.tier"),
                        show_label=False,
                        elem_classes=["mineru-tier-slider"],
                    )
                    # 独立保存会话的未锁定档位，不随文件和页码选区一起重置。
                    tier_selection = gr.Textbox(value=json.dumps({"tier": preferred_tier, "locked": False}), visible=False)
                force_ocr = gr.Checkbox(
                    value=False,
                    label=i18n("mineru.force_ocr"),
                    info=i18n("mineru.force_ocr_info"),
                    visible=False,
                    interactive=True,
                    elem_classes=["mineru-force-ocr"],
                )
                page_range = gr.Textbox(
                    value="",
                    label=i18n("mineru.page_range"),
                    visible=False,
                )
                # 用 JSON 文本承载前端状态，保持现有事件输入契约。
                page_metadata = gr.Textbox(value="{}", visible=False)
                page_selection = gr.Textbox(value="{}", visible=False)
                # 通过子 HTML 的明确状态控制显隐，保留原生滑块的交互状态。
                with gr.Column(min_width=0, elem_classes=["mineru-kit-page-range"]):
                    page_summary = gr.HTML(value="", elem_classes=["mineru-page-summary"])
                    with gr.Row(elem_classes=["mineru-page-sliders"]):
                        page_handle_a = gr.Slider(
                            minimum=1,
                            # 新版 Gradio 拒绝零跨度，单页与空状态仍保持禁用和值为 1。
                            maximum=2,
                            value=1,
                            step=1,
                            precision=0,
                            label=i18n("mineru.start_page"),
                            interactive=False,
                            container=False,
                            elem_classes=["mineru-page-handle-a"],
                        )
                        page_handle_b = gr.Slider(
                            minimum=1,
                            maximum=2,
                            value=1,
                            step=1,
                            precision=0,
                            label=i18n("mineru.end_page"),
                            interactive=False,
                            container=False,
                            elem_classes=["mineru-page-handle-b"],
                        )
                page_notice = gr.HTML(value="", visible=False, elem_classes=["mineru-page-notice"])
                with gr.Row(elem_classes=["mineru-actions"]):
                    convert_button = gr.Button(
                        i18n("mineru.convert"), variant="primary", scale=1, min_width=0, interactive=False
                    )
                    clear_button = gr.ClearButton(value=i18n("mineru.clear"), scale=1, min_width=1)
                status_panel = gr.HTML(_status_html(), elem_classes=["mineru-status-panel"])

            with gr.Column(scale=4, min_width=340, elem_classes=["mineru-kit-preview", "mineru-preview-pane"]):
                pdf_preview = gr.File(visible=False, interactive=False, label="PDF", type="filepath")
                pdf_viewer_entry = gr.File(value=str(viewer_entry), visible=False, interactive=False)
                pdf_viewer = gr.HTML(
                    value="",
                    apply_default_css=False,
                    # 保持原生 HTML 挂载，避免较早的 Gradio 6 忽略纯前端的 visible 更新。
                    visible=True,
                    elem_classes=["mineru-kit-pdf-preview"],
                )
                image_preview = gr.Image(
                    label=i18n("mineru.preview"),
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
                ofd_preview = gr.HTML(
                    value="", apply_default_css=False,
                    elem_classes=["mineru-kit-ofd-preview"],
                )
                ofd_ticket = gr.Textbox(value="", visible=False)
                ofd_receipt = gr.Textbox(value="", visible=False)
                generic_preview = gr.HTML(
                    value=preview_placeholder("empty_preview"),
                    visible=True,
                    elem_classes=["mineru-kit-generic-preview"],
                )

            with gr.Column(scale=4, min_width=340, elem_classes=["mineru-kit-results", "mineru-markdown-pane"]):
                with gr.Tabs(elem_classes=["mineru-markdown-tabs"]):
                    with gr.Tab(i18n("mineru.markdown")):
                        html_output = gr.HTML(
                            value="",
                            elem_classes=["mineru-markdown-output"],
                        )
                    with gr.Tab(i18n("mineru.json_view")):
                        json_output = gr.Code(
                            value="",
                            language="json",
                            interactive=False,
                            show_label=False,
                            container=False,
                            lines=1,
                            wrap_lines=True,
                            buttons=["copy"],
                            elem_classes=["mineru-structured-json"],
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
                            download_buttons[format_name] = gr.Button(
                                label,
                                visible=True,
                                interactive=False,
                                size="sm",
                                min_width=0,
                                elem_classes=[f"mineru-kit-download-{format_name}"],
                            )
                download_notice = gr.HTML(value="", elem_classes=["mineru-kit-download-notice"])

        if enable_example:
            examples = _example_files(file_types)
            if examples:
                gr.Examples(examples=examples, inputs=input_file, label=i18n("mineru.examples"), elem_id="mineru-kit-examples")

        artifact_state = gr.State(value=None)
        active_run_id = gr.Textbox(value="", visible=False)
        download_files = {name: gr.File(visible=False, interactive=False) for name, _label in _DOWNLOAD_FORMATS}
        download_requests = {name: gr.Textbox(value="", visible=False) for name, _label in _DOWNLOAD_FORMATS}
        download_receipts = {name: gr.Textbox(value="", visible=False) for name, _label in _DOWNLOAD_FORMATS}
        clear_button.add(
            [
                input_file,
                page_range,
                force_ocr,
                html_output,
                json_output,
                pdf_preview,
                image_preview,
                office_preview,
                generic_preview,
                status_panel,
            ]
        )

        def update_file_preview(file_path: str | None, request: object | None = None) -> tuple[Any, ...]:
            """切换源文件预览，并清除上一份文档的结果与下载状态。"""
            reset_result = (_status_html(_DEFAULT_STATUS), "", None, *_download_updates(gr, interactive=False), "")
            if not file_path:
                return (
                    _pdf_preview_update(gr, None),
                    _preview_update(gr, None, visible=False),
                    _preview_update(gr, "", visible=False),
                    _preview_update(gr, preview_placeholder("empty_preview"), visible=True),
                    *reset_result,
                )
            suffix = _file_suffix(file_path)
            if suffix in PDF_EXTENSIONS:
                return (
                    _pdf_preview_update(gr, file_path),
                    _preview_update(gr, None, visible=False),
                    _preview_update(gr, "", visible=False),
                    _preview_update(gr, preview_placeholder("source_preview"), visible=False),
                    *reset_result,
                )
            if suffix in IMAGE_EXTENSIONS:
                return (
                    _pdf_preview_update(gr, None),
                    _preview_update(gr, file_path, visible=True),
                    _preview_update(gr, "", visible=False),
                    _preview_update(gr, preview_placeholder("source_preview"), visible=False),
                    *reset_result,
                )
            if suffix == "ofd":
                return (
                    _pdf_preview_update(gr, None),
                    _preview_update(gr, None, visible=False),
                    _preview_update(gr, "", visible=False),
                    _preview_update(gr, "", visible=False),
                    *reset_result,
                )
            if _is_office(file_path):
                return (
                    _pdf_preview_update(gr, None),
                    _preview_update(gr, None, visible=False),
                    _preview_update(
                        gr,
                        _build_office_preview_html(file_path, request),
                        visible=True,
                    ),
                    _preview_update(gr, "", visible=False),
                    *reset_result,
                )
            return (
                _pdf_preview_update(gr, None),
                _preview_update(gr, None, visible=False),
                _preview_update(gr, "", visible=False),
                _preview_update(gr, preview_placeholder("unsupported_preview"), visible=True),
                *reset_result,
            )

        # 保持公开输入只有文件，访问地址由 Gradio 的请求上下文注入。
        update_file_preview.__annotations__["request"] = gr.Request
        private_event_kwargs = {"queue": False, "api_visibility": "private"}

        # 换文件和清除先在浏览器中撤销旧预览，文件输出仍由原有 Python 事件管理。
        input_file.change(fn=None, inputs=input_file, outputs=pdf_viewer, js=pdf_preview_js("reset"), **private_event_kwargs)
        clear_button.click(fn=None, inputs=[], outputs=pdf_viewer, js=pdf_preview_js("clear"), **private_event_kwargs)

        ofd_script = _resource_text("gradio_ofd_preview.js")
        input_file.change(
            fn=None, inputs=input_file, outputs=[ofd_ticket, ofd_preview],
            js=f"(...args) => ({ofd_script})('begin', ...args)",
            **private_event_kwargs,
        )
        # Gradio 6.8 的纯前端事件不可靠地触发 then；通过请求值变化启动后台转换。
        render_ofd = ofd_ticket.change(
            fn=prepare_ofd_preview, inputs=[input_file, ofd_ticket], outputs=ofd_receipt,
            concurrency_limit=None, trigger_mode="multiple", **private_event_kwargs,
        )
        render_ofd.then(
            fn=None, inputs=ofd_receipt, outputs=ofd_preview,
            js=f"(...args) => ({ofd_script})('apply', ...args)", **private_event_kwargs,
        )
        clear_button.click(
            fn=None, inputs=[], outputs=[ofd_ticket, ofd_preview],
            js=f"(...args) => ({ofd_script})('clear', ...args)", **private_event_kwargs,
        )

        download_script = _resource_text("gradio_download.js")

        def download_js(action: str, format_name: str = "", label: str = "") -> str:
            """为下载事件绑定明确的动作与格式，复用同一份前端状态处理脚本。"""
            arguments = ", ".join(json.dumps(value) for value in (action, _DOWNLOAD_FORMATS, format_name, label))
            return f"(...args) => ({download_script})({arguments}, ...args)"

        def reset_download_ui() -> tuple[Any, ...]:
            """返回下载组件的初始状态，为转换提供可可靠串联的完成事件。"""
            count = len(_DOWNLOAD_FORMATS)
            return ("", *((None,) * count), *(("",) * count * 2), *_download_updates(gr, interactive=False)[1:], "")

        download_reset_outputs = [
            active_run_id,
            *download_files.values(),
            *download_requests.values(),
            *download_receipts.values(),
            *download_buttons.values(),
            download_notice,
        ]

        # 先在前端失效旧请求，再等待上传/清除/转换的 Python 回调，防止迟到的下载被触发。
        gr.on(
            triggers=[input_file.change, clear_button.click],
            fn=None,
            inputs=[],
            outputs=download_reset_outputs,
            js=download_js("reset"),
            **private_event_kwargs,
        )
        active_run_id.change(fn=None, inputs=active_run_id, outputs=[], js=download_js("activate"), **private_event_kwargs)

        def update_ocr_control(file_path: str | None) -> Any:
            """仅为原始 PDF 显示开关，并在更换或清除文件时重置为自动判断。"""
            return gr.update(value=False, visible=_file_suffix(file_path) in PDF_EXTENSIONS)

        input_file.change(
            fn=update_ocr_control,
            inputs=input_file,
            outputs=force_ocr,
            trigger_mode="always_last",
            **private_event_kwargs,
        )

        async def cancel_session_conversion(request: object | None = None) -> None:
            """主动回收当前会话任务，确保关闭异步生成器时也能完成资源清理。"""
            session_hash = getattr(request, "session_hash", None)
            tasks = session_tasks.pop(session_hash, set())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        cancel_session_conversion.__annotations__["request"] = gr.Request
        # 显式注册幂等加载事件，初始化前端状态和自定义文案。
        demo.load(fn=None, js=app_js, **private_event_kwargs)
        preview_outputs = [
            pdf_preview,
            image_preview,
            office_preview,
            generic_preview,
            status_panel,
            html_output,
            artifact_state,
            active_run_id,
            *download_buttons.values(),
            json_output,
        ]

        # 文件页数只在上传后读取；前端缓存元数据，tier 切换和拖动不发起 Python 请求。
        range_inputs = [input_file, tier, page_metadata, page_selection, page_handle_a, page_handle_b, tier_selection]
        range_outputs = [
            page_handle_a,
            page_handle_b,
            page_summary,
            page_range,
            page_selection,
            convert_button,
            page_notice,
            tier,
            tier_label,
            tier_selection,
        ]
        range_script = _resource_text("gradio_page_range.js")
        # 共用一个 always_last 事件流，避免文件、元数据、清除与拖动的并行回调互相覆盖。
        gr.on(
            triggers=[input_file.change, tier.input, page_metadata.change, page_handle_a.input, page_handle_b.input],
            fn=None,
            inputs=range_inputs,
            outputs=range_outputs,
            js=(
                f"(...args) => ({range_script})({json.dumps(tier_choices)}, "
                f"{json.dumps(sorted(FLASH_ONLY_PARSE_EXTENSIONS))}, {json.dumps(max_pages)}, ...args)"
            ),
            trigger_mode="always_last",
            **private_event_kwargs,
        )

        def read_page_metadata(file_path: str | None) -> str:
            """把页数元数据编码为稳定的 JSON 文本，供两个 Gradio 主版本共用。"""
            return json.dumps(pdf_page_metadata(file_path), ensure_ascii=False)

        input_file.change(
            fn=read_page_metadata,
            inputs=input_file,
            outputs=page_metadata,
            trigger_mode="always_last",
            show_progress="hidden",
            **private_event_kwargs,
        )

        async def convert_handler(
            file_path: str | None,
            tier_position: int | float,
            raw_page_range: str,
            force_ocr: bool = False,
            request: object | None = None,
        ) -> Any:
            """执行单文件 V1 解析并流式更新状态、HTML 和 Structured Content JSON。"""
            # 开始或失败只重置结果与下载；预览内容、显隐和浏览位置保持不变。
            reset_result = (
                _status_html(_DEFAULT_STATUS),
                "",
                *(gr.skip() for _ in range(4)),
                None,
                *_download_updates(gr, interactive=False),
                "",
            )
            if not file_path:
                yield reset_result
                return
            source_path = Path(file_path).resolve()
            if not source_path.is_file():
                yield (_status_html("Failed: input file does not exist"), *reset_result[1:])
                return
            suffix = _file_suffix(source_path)
            if suffix not in PARSEABLE_EXTENSIONS:
                yield (_status_html(f"Failed: unsupported file type '.{suffix}'"), *reset_result[1:])
                return
            try:
                selected_tier = _tier_for_position(tier_position, tier_choices)
            except ValueError as exc:
                yield (_status_html(f"Failed: {exc}"), *reset_result[1:])
                return
            if suffix in FLASH_ONLY_PARSE_EXTENSIONS:
                # 提交端独立约束有效档位，避免事件 API 或前端残留值绕过 Flash 锁定。
                if "flash" not in tier_choices:
                    message = "Failed: tier_unavailable: 该格式仅支持 Flash，当前服务不可用"
                    yield (_status_html(message), *reset_result[1:])
                    return
                selected_tier = "flash"
            try:
                page_text = await asyncio.to_thread(
                    _effective_page_range, source_path, raw_page_range, tier=selected_tier, max_pages=max_pages
                )
            except MineruError as exc:
                yield (_status_html(f"Failed: {exc.code}: {exc}"), *reset_result[1:])
                return
            state = StatusPanelState()
            state.append(STATUS_PREPARING_REQUEST)
            yield (state.render(), *reset_result[1:])
            status_queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def emit(message: str) -> None:
                """记录通知时刻，避免队列消费延迟被误算为服务端解析耗时。"""
                loop.call_soon_threadsafe(status_queue.put_nowait, (message, time.monotonic()))

            async def run_conversion() -> tuple[Any, ...]:
                """在单任务槽内解析和整理结果；取消或失败均自动释放等待位置。"""
                if conversion_slot.locked():
                    emit(STATUS_QUEUED_LOCALLY)
                async with conversion_slot:
                    result = await client.parse_file(
                        source_path,
                        tier=selected_tier,
                        page_range=page_text,
                        # 再次检查源文件类型，避免事件 API 或隐藏控件残留值强制处理非 PDF。
                        ocr_mode="ocr" if suffix in PDF_EXTENSIONS and force_ocr else "auto",
                        status_callback=emit,
                    )
                    emit(STATUS_PROCESSING_OUTPUT)
                    artifacts = await asyncio.to_thread(
                        persist_parse_result,
                        result,
                        source_path,
                        output_root=output_root,
                        page_range=page_text,
                    )
                    rendered_html = await asyncio.to_thread(
                        render_html_preview,
                        artifacts,
                        public_base_url=_gradio_public_base_url(request),
                    )
                    # 直接复用下载所用的 Structured Content 文件，避免展示成 Middle JSON 或重复渲染。
                    structured_json = await asyncio.to_thread(artifacts.structured_content_path.read_text, encoding="utf-8")
                    preview_path = artifacts.layout_pdf_path or artifacts.origin_pdf_path
                    generic_html = "" if preview_path else preview_placeholder("result_ready")
                    show_image_preview = suffix in IMAGE_EXTENSIONS and preview_path is None
                    result_preview_updates = (
                        _pdf_preview_update(gr, str(preview_path) if preview_path else None),
                        gr.update(value=str(source_path) if show_image_preview else None, visible=show_image_preview),
                        gr.update(value="", visible=False),
                        gr.update(value=generic_html, visible=bool(generic_html)),
                    )
                    if _is_office(source_path) or suffix == "ofd":
                        # Office/OFD 源预览已在上传时挂载，成功后保留原内容和浏览位置。
                        result_preview_updates = tuple(gr.skip() for _ in range(4))
                    return (
                        rendered_html,
                        *result_preview_updates,
                        artifacts.as_state(),
                        *_download_updates(gr, interactive=True, run_id=artifacts.root.name),
                        structured_json,
                    )

            task = asyncio.create_task(run_conversion())
            session_hash = getattr(request, "session_hash", None)
            if session_hash:
                session_tasks.setdefault(session_hash, set()).add(task)

                def forget_task(done_task: asyncio.Task[tuple[Any, ...]]) -> None:
                    """任务结束即释放会话索引，即使前端已丢弃生成器也不保留任务引用。"""
                    tasks = session_tasks.get(session_hash)
                    if tasks is not None:
                        tasks.discard(done_task)
                        if not tasks:
                            session_tasks.pop(session_hash, None)

                task.add_done_callback(forget_task)
            try:
                async with aclosing(stream_status_updates(task, status_queue, state)) as updates:
                    async for status in updates:
                        # 动画只更新状态卡片，避免反复重建预览和清空结果组件。
                        yield (status, *(gr.skip() for _ in reset_result[1:]))
                result_outputs = await task
                state.append(STATUS_COMPLETED)
                yield (state.render(), *result_outputs)
            except asyncio.CancelledError:
                # 会话重置后静默结束旧流，避免把取消异常或旧状态写回新界面。
                return
            except Exception as exc:
                state.append(f"Failed: {exc}")
                yield (state.render(), *reset_result[1:])
            finally:
                # 清除、换文件或断开流时仅取消本地等待，不发送远端取消请求。
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        convert_outputs = [
            status_panel,
            html_output,
            pdf_preview,
            image_preview,
            office_preview,
            generic_preview,
            artifact_state,
            active_run_id,
            *download_buttons.values(),
            json_output,
        ]
        # Gradio 在读取函数签名时需要真实的 Request 类型对象；注解在运行时补回以保持延迟导入。
        convert_handler.__annotations__["request"] = gr.Request
        event_kwargs: dict[str, Any] = {
            "queue": True,
            "show_progress": "hidden",
            "concurrency_limit": None,
            "api_visibility": "public" if enable_api else "private",
        }
        # 重置与转换必须显式串联；独立 click 监听可能在转换完成后才执行重置。
        begin_conversion = convert_button.click(
            fn=reset_download_ui,
            inputs=[],
            outputs=download_reset_outputs,
            js=f"() => {{ ({pdf_preview_js('begin')})(); ({download_js('reset')})(); return []; }}",
            **private_event_kwargs,
        )
        convert_event = begin_conversion.then(
            fn=convert_handler,
            inputs=[input_file, tier, page_range, force_ocr],
            outputs=convert_outputs,
            **event_kwargs,
        )
        source_preview_event = input_file.change(
            fn=update_file_preview,
            inputs=input_file,
            outputs=preview_outputs,
            cancels=[convert_event],
            **private_event_kwargs,
        )
        # 成功事件接收已经序列化的原生 FileData，避免首次挂载时丢失 URL。
        source_preview_event.success(
            fn=None,
            inputs=[pdf_preview, input_file, pdf_viewer_entry],
            outputs=pdf_viewer,
            js=pdf_preview_js("source"),
            **private_event_kwargs,
        )
        convert_event.success(
            fn=None,
            inputs=[pdf_preview, input_file, pdf_viewer_entry, active_run_id],
            outputs=pdf_viewer,
            js=pdf_preview_js("result"),
            **private_event_kwargs,
        )
        input_file.change(fn=cancel_session_conversion, inputs=[], outputs=[], **private_event_kwargs)

        reset_outputs = [
            status_panel,
            html_output,
            pdf_preview,
            image_preview,
            office_preview,
            generic_preview,
            artifact_state,
            active_run_id,
            *download_buttons.values(),
            json_output,
        ]

        def reset_ui() -> tuple[Any, ...]:
            """清除当前任务结果并恢复空预览状态。"""
            return (
                _status_html(_DEFAULT_STATUS),
                "",
                _pdf_preview_update(gr, None),
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                gr.update(value=preview_placeholder("empty_preview"), visible=True),
                None,
                *_download_updates(gr, interactive=False),
                "",
            )

        clear_button.click(fn=reset_ui, inputs=[], outputs=reset_outputs, cancels=[convert_event], **private_event_kwargs)
        clear_button.click(fn=cancel_session_conversion, inputs=[], outputs=[], **private_event_kwargs)

        for format_name, label in _DOWNLOAD_FORMATS:
            begin_download = download_buttons[format_name].click(
                fn=_download_request_handler,
                inputs=active_run_id,
                outputs=download_requests[format_name],
                js=f"(...args) => [({download_js('begin', format_name, label)})(...args)[0]]",
                **private_event_kwargs,
            )
            # 开始回执也必须经过前端令牌校验，避免慢请求在清除后重新显示旧的“准备中”。
            begin_download.success(
                fn=None,
                inputs=[download_requests[format_name], active_run_id],
                outputs=[download_buttons[format_name], download_notice],
                js=download_js("busy", format_name, label),
                **private_event_kwargs,
            )
            download_handler = _download_handler(format_name, output_root)
            download_handler.__annotations__["request"] = gr.Request
            prepare_download = begin_download.then(
                fn=download_handler,
                inputs=[artifact_state, download_requests[format_name]],
                outputs=[download_files[format_name], download_receipts[format_name]],
                queue=True,
                show_progress="hidden",
                api_visibility="private",
            )
            prepare_download.success(
                fn=None,
                inputs=[download_files[format_name], download_receipts[format_name], active_run_id],
                outputs=[download_buttons[format_name], download_notice],
                js=download_js("complete", format_name, label),
                **private_event_kwargs,
            )

    demo._mineru_kit_css = app_css
    demo._mineru_kit_js = app_js
    demo._mineru_kit_launch_kwargs = {"i18n": i18n, "css": app_css, "js": app_js}
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
    api_server_concurrency: int,
    api_server_language: str,
    api_server_disable_image_analysis: bool,
    api_server_preload_models: bool,
    max_pages: int | None = None,
) -> None:
    """启动 Gradio，并在外部服务缺少 Flash 时托管本地 Flash V1 服务。"""
    configure_standard_streams()
    validate_max_pages(max_pages)
    resolved_api_key = api_key if api_key is not None else os.environ.get("MINERU_API_KEY")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    managed_server: ManagedLocalApiServer | None = None
    resolved_api_url = api_url
    try:
        if resolved_api_url is None:
            managed_server = ManagedLocalApiServer(
                tier=api_server_tier,  # type: ignore[arg-type]
                concurrency=api_server_concurrency,
                language=api_server_language,
                disable_image_analysis=api_server_disable_image_analysis,
                preload_models=api_server_preload_models,
                api_key=resolved_api_key,
            )
            resolved_api_url = managed_server.start()
        client = V1ArtifactClient(api_url=resolved_api_url, api_key=resolved_api_key)
        capabilities = asyncio.run(client.discover())
        local_flash: V1ArtifactClient | None = None
        if api_url is not None and "flash" not in capabilities.tiers:
            managed_server = ManagedLocalApiServer(
                tier="flash",
                concurrency=api_server_concurrency,
                language=api_server_language,
                disable_image_analysis=api_server_disable_image_analysis,
                preload_models=api_server_preload_models,
                api_key="",
            )
            local_flash = V1ArtifactClient(api_url=managed_server.start(), api_key="")
            asyncio.run(local_flash.discover())
        routed_client = GradioArtifactClient(client, local_flash=local_flash)
        demo = build_gradio_app(
            routed_client,
            routed_client.capabilities,
            output_root=output_root,
            enable_example=enable_example,
            enable_api=enable_api,
            latex_delimiters_type=latex_delimiters_type,
            max_pages=max_pages,
        )
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            allowed_paths=[str(output_root)],
            **demo._mineru_kit_launch_kwargs,
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


def _gradio_public_base_url(request: object | None = None) -> str:
    """从 Gradio 请求获取当前外部站点根地址，保留反向代理协议、域名和挂载路径。"""
    headers = getattr(request, "headers", None) or {}
    raw_request = getattr(request, "request", None)
    request_url = getattr(raw_request, "url", None)
    host = headers.get("x-forwarded-host") or headers.get("host") or getattr(request_url, "netloc", "localhost:7860")
    protocol = headers.get("x-forwarded-proto") or getattr(request_url, "scheme", "http")
    host = host.split(",", 1)[0].strip()
    protocol = protocol.split(",", 1)[0].strip()
    scope = getattr(raw_request, "scope", None) or {}
    root_path = scope.get("root_path", "")
    if root_path.startswith(("http://", "https://")):
        return root_path.rstrip("/")
    root_path = "/" + root_path.strip("/") if root_path else ""
    return f"{protocol}://{host}{root_path}"


def _build_office_preview_html(file_path: str | Path, request: object | None = None) -> str:
    """复用 3.4.5 的上传即预览结构；短地址只供展示，iframe 始终使用完整地址。"""
    source_path = Path(file_path)
    headers = getattr(request, "headers", None) or {}
    host = headers.get("x-forwarded-host") or headers.get("host") or "localhost:7860"
    protocol = headers.get("x-forwarded-proto") or "http"
    public_url = f"{protocol}://{host}/gradio_api/file={quote(source_path.as_posix(), safe='/:')}"
    short_name = f"{source_path.stem[-12:]}{source_path.suffix}" if source_path.stem else source_path.name
    short_public_url = f"{protocol}://{host}/....{short_name}"
    viewer_url = "https://view.officeapps.live.com/op/embed.aspx?src=" + quote(public_url, safe="")
    return (
        '<div class="office-preview-shell">'
        '<div class="office-preview-notice">'
        '<div class="office-preview-copy">'
        f"<strong>{localized_text('office_preview_title')}</strong>"
        f"<span>{localized_text('office_notice')}</span>"
        '<div class="office-preview-source-link">'
        f"{localized_text('office_preview_source_link')}: {html.escape(short_public_url, quote=True)}</div>"
        "</div>"
        '<div class="office-preview-actions">'
        f'<button type="button" class="office-preview-ignore-once">{localized_text("ignore_once")}</button>'
        f'<button type="button" class="office-preview-ignore-forever">{localized_text("ignore_forever")}</button>'
        "</div>"
        "</div>"
        f'<iframe class="office-preview-frame" src="{html.escape(viewer_url, quote=True)}" frameborder="0"></iframe>'
        "</div>"
    )


def _download_request_handler(request_token: str) -> str:
    """传回前端下载令牌以可靠触发后续事件，不直接修改可能已属于新文档的按钮。"""
    return request_token if isinstance(request_token, str) else ""


def _download_handler(format_name: str, output_root: Path) -> Callable[..., tuple[str | None, str]]:
    """创建一个绑定格式和 output root 的 Gradio 下载回调。"""

    def handler(state: object, request_token: str, request: object | None = None) -> tuple[str | None, str]:
        """校验请求所属结果，返回文件与回执；生成失败也交给前端按请求标识恢复按钮。"""
        receipt = {"request": request_token, "error": ""}
        try:
            token = json.loads(request_token)
            artifacts = RunArtifacts.from_state(state)
            if token["run_id"] != artifacts.root.name:
                raise ValueError("解析结果已变更，请重新下载。")
            path = render_download(
                state,
                format_name,
                allowed_root=output_root,
                public_base_url=_gradio_public_base_url(request),
            )
            return path, json.dumps(receipt, ensure_ascii=False)
        except Exception as exc:
            receipt["error"] = str(exc)
            return None, json.dumps(receipt, ensure_ascii=False)

    return handler


__all__ = ["build_gradio_app", "launch_gradio"]
