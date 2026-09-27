"""MinerU Kit 的 V1 Gradio 应用组装与启动入口。"""

from __future__ import annotations

import asyncio
import html
import json
import os
import time
import uuid
from contextlib import suppress
from pathlib import Path
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from loguru import logger

from ...errors import MineruError
from ...filetypes import (
    FLASH_ONLY_PARSE_EXTENSIONS,
    HTML_EXTENSIONS,
    MHTML_EXTENSIONS,
    IMAGE_EXTENSIONS,
    OFFICE_EXTENSIONS,
    PARSEABLE_EXTENSIONS,
    PDF_EXTENSIONS,
)
from ...types import TIERS, Tier
from ...utils.logger import configure_global_log_level
from ...utils.stdio import configure_standard_streams
from .artifacts import RunArtifacts, persist_parse_result, render_download, render_html_preview
from .client import (
    GradioArtifactClient,
    ManagedLocalApiServer,
    V1ArtifactClient,
    V1ServerCapabilities,
)
from .epub_preview import register_epub_preview_resources
from .conversion import ConversionRun, SessionConversions, await_task_completion, run_sync_output
from .i18n import MESSAGES, localized_text, preview_placeholder, translations
from .page_range import effective_page_range as _effective_page_range
from .page_range import pdf_page_metadata, validate_max_pages
from .pdf_preview import pdf_preview_js, register_pdf_preview_resources
from .source_preview import prepare_source_preview
from .status import (
    DEFAULT_STATUS as _DEFAULT_STATUS,
    STATUS_COMPLETED,
    STATUS_PREPARING_REQUEST,
    STATUS_PROCESSING_OUTPUT,
    STATUS_QUEUED_LOCALLY,
    ParseStatusUpdate,
    status_html as _status_html,
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
    min-width: 0 !important; padding: 0 !important; margin: 0; line-height: 0 !important; overflow: visible;
}
.mineru-kit-download-trigger { min-height: 32px; border: 0; background: transparent; }
/* Gradio 主题对 .gradio-style button 的原生按钮样式优先级更高（HF Space 上
   无前缀改写时），此处声明须带 !important 才能稳定生效；hover 变色同理。 */
.mineru-kit-download-trigger .mineru-kit-download-icon {
    display: flex; align-items: center; justify-content: center; gap: 6px;
    width: 100% !important; height: 32px !important; margin: 0; padding: 6px 8px !important;
    border: 0; border-radius: 6px;
    font-size: 14px !important; line-height: 20px; white-space: nowrap;
    color: var(--body-text-color, #1f2937) !important; background: transparent !important; cursor: pointer;
}
.mineru-kit-download-icon svg { width: 20px; height: 20px; margin: 0; flex: 0 0 20px; }
.mineru-kit-download-menu:hover .mineru-kit-download-icon,
.mineru-kit-download-icon:focus-visible { background: var(--background-fill-secondary, #f3f4f6) !important; }
.mineru-kit-download-icon:focus-visible { outline: 2px solid var(--mineru-accent, #f97316); outline-offset: 2px; }
/* Gradio 会在页面端给自定义 CSS 加 .gradio-container-<ver> .contain 前缀抬升
   优先级，但该改写在部分环境（如 HF Space 实测）不会执行，此时类选择器
   (0,1,0) 会输给 Column 自带的 div.svelte-siXXXX { position: relative }
   scoped 规则 (0,1,1)，浮窗被顶开 38px 并使 hover 桥失效。
   组件已带 elem_id，用 ID 选择器不依赖该改写，在所有环境稳赢；
   hover/focus 显示规则须同步用 ID，否则压不过基础规则 (1,0,0)。 */
#mineru-kit-download-options {
    position: absolute; right: 0; top: calc(100% + 6px); z-index: 40;
    width: max-content !important; min-width: 0 !important;
    display: flex !important; flex-direction: column; gap: 4px; padding: 6px;
    border: 1px solid var(--mineru-panel-border, rgba(17,24,39,.12)); border-radius: 8px;
    background: var(--background-fill-primary, #fff); box-shadow: 0 12px 28px rgba(15,23,42,.18);
    opacity: 0; pointer-events: none; transform: translateY(-4px); visibility: hidden;
    transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
}
.mineru-kit-download-menu:hover #mineru-kit-download-options,
.mineru-kit-download-menu:focus-within #mineru-kit-download-options {
    opacity: 1; pointer-events: auto; transform: translateY(0); visibility: visible;
}
/* 填满图标与浮层之间的间隙，避免鼠标移向下载项时菜单提前关闭。 */
.mineru-kit-download-options::before { content: ""; position: absolute; left: 0; right: 0; top: -7px; height: 7px; }
#mineru-kit-download-options :is(button, a) {
    justify-content: flex-start; width: 100%; min-height: 34px; padding: 6px 10px; white-space: nowrap;
    border: 0; border-radius: 6px; background: transparent; box-shadow: none; text-align: left; gap: 8px;
}
.mineru-kit-download-options button::before {
    content: ""; display: block; width: 16px; height: 16px; flex: 0 0 16px;
    background-color: currentColor;
    -webkit-mask: var(--mineru-download-format-icon) center / contain no-repeat;
    mask: var(--mineru-download-format-icon) center / contain no-repeat;
}
#mineru-kit-download-options :is(button, a):hover { background: var(--background-fill-secondary, #f3f4f6); }
.mineru-kit-empty-preview { min-height: 160px; display: grid; place-items: center; opacity: .65; }
/* Gradio 6.8 会按逗号拆分并重写选择器，PDF/源文档预览使用独立选择器避免破坏 :has。 */
/* PDF/源文档预览直接贴合面板边框，独立预览不再沿用旧组件的标签留白与额外高度。 */
.mineru-kit-preview:has(.mineru-pdf-frame),
.mineru-kit-preview:has(.mineru-source-frame),
.mineru-kit-preview:has(.mineru-epub-frame) {
    padding: 0; gap: 0; overflow: hidden;
    min-height: var(--mineru-preview-content-height, 775px) !important;
}
.mineru-kit-preview > .block.mineru-kit-pdf-preview,
.mineru-kit-preview > .block.mineru-kit-source-preview {
    height: var(--mineru-preview-content-height, 775px) !important;
    min-height: 0 !important; max-height: none !important;
}
.mineru-kit-pdf-preview, .mineru-kit-source-preview { height: 100%; padding: 0 !important; }
.mineru-kit-source-preview:has(.mineru-epub-frame) {
    height: var(--mineru-preview-content-height, 775px) !important;
    min-height: var(--mineru-preview-content-height, 775px) !important;
}
.mineru-kit-pdf-preview .html-container, .mineru-kit-pdf-preview .prose,
.mineru-kit-source-preview .html-container, .mineru-kit-source-preview .prose { height: 100%; padding: 0 !important; }
.mineru-kit-pdf-preview:not(:has(.mineru-pdf-frame, [role="alert"])) { display: none !important; }
/* HTML 源预览用固定 viewport 裁剪逻辑舞台。缩放舞台而不是 iframe 本体，
   避免 Safari/WebKit 在 transform iframe 时把子文档绘制层裁成局部区域。 */
.mineru-source-viewport {
    position: relative; display: block; width: 100%; height: 100%; min-height: 0; overflow: hidden;
}
.mineru-source-stage {
    display: block; width: 100%; height: 100%; min-height: 0; transform-origin: 0 0;
}
.mineru-pdf-frame, .mineru-source-frame, .mineru-epub-frame {
    display: block; width: 100%; height: 100%; min-height: 0; border: 0;
}
.mineru-kit-source-preview:not(:has(iframe)):not(:has([data-mineru-i18n-key])) { display: none !important; }
.mineru-kit-image-preview img { max-height: var(--mineru-pdf-page-height, 720px); object-fit: contain; }
/* 桌面两栏共用行高，PDF/源文档预览填满伸展后的面板；窄屏仍采用独立预览高度。 */
@media (min-width: 901px) {
  .mineru-kit-results, .mineru-kit-preview:has(.mineru-pdf-frame),
  .mineru-kit-preview:has(.mineru-source-frame),
  .mineru-kit-preview:has(.mineru-epub-frame) {
    align-self: stretch !important;
    height: auto;
  }
  .mineru-kit-preview > .block.mineru-kit-pdf-preview,
  .mineru-kit-preview > .block.mineru-kit-source-preview {
    flex: 1 1 0;
    height: auto !important;
  }
  /* 结果栏变高时，EPUB iframe 也必须沿着完整的 Gradio 高度链拉伸，
     否则固定的默认预览高度下方会露出父容器背景。 */
  .mineru-kit-workspace:has(.mineru-epub-frame) {
    align-items: stretch !important;
  }
  .mineru-kit-preview:has(.mineru-epub-frame) {
    height: auto !important;
    min-height: var(--mineru-preview-content-height, 775px) !important;
    background: #fff !important;
  }
  .mineru-kit-preview:has(.mineru-epub-frame) > .block.mineru-kit-source-preview,
  .mineru-kit-preview:has(.mineru-epub-frame) .mineru-kit-source-preview,
  .mineru-kit-preview:has(.mineru-epub-frame) .html-container,
  .mineru-kit-preview:has(.mineru-epub-frame) .prose,
  .mineru-kit-preview:has(.mineru-epub-frame) .mineru-epub-frame {
    height: 100% !important;
    min-height: 0 !important;
  }
}
@media (max-width: 900px) {
  .mineru-kit-workspace { flex-direction: column !important; }
  .mineru-kit-control, .mineru-kit-preview, .mineru-kit-results { min-width: 0 !important; width: 100% !important; }
  /* 窄屏改用固定高度、整栏按内容收口：不依赖上面的 flex 拉伸链（移动端引擎
     可能不把拉伸所得高度视为定值，height:100% 断链时 iframe 会塌到默认
     ~150px），也避免拉伸模式下 885px min-height 与 775px 正文之间在卡片
     底部留下死区。与 PDF/源文档预览的窄屏策略一致。 */
  .mineru-kit-results { min-height: 0; flex: 0 0 auto !important; }
  .mineru-kit-results > .mineru-markdown-tabs,
  .mineru-kit-results [role="tabpanel"],
  .mineru-kit-results [role="tabpanel"] > .column { flex: 0 0 auto; }
  .mineru-kit-results .mineru-markdown-output,
  .mineru-kit-results .mineru-structured-json {
    flex: 0 0 auto !important;
    height: var(--mineru-preview-content-height, 775px) !important;
    min-height: var(--mineru-preview-content-height, 775px) !important;
  }
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


def build_gradio_app(
    client: V1ArtifactClient | GradioArtifactClient,
    capabilities: V1ServerCapabilities,
    *,
    output_root: Path,
    enable_example: bool = True,
    enable_api: bool = True,
    max_pages: int | None = None,
) -> Any:
    """构建不启动监听端口的 Gradio Blocks 应用，便于单元测试和外部托管。"""
    import gradio as gr

    viewer_entry = register_pdf_preview_resources()
    epub_viewer_entry = register_epub_preview_resources()
    validate_max_pages(max_pages)
    tier_choices = [tier for tier in TIERS if tier in capabilities.tiers]
    if not tier_choices:
        raise ValueError("V1 API server did not advertise any parsing tier")
    preferred_tier = _default_tier(capabilities)
    file_types = _supported_file_types()
    examples = _example_files(file_types) if enable_example else []
    app_css = _resource_text("gradio_app.css") + _KIT_MENU_CSS + _download_icon_css()
    i18n = gr.I18n(**translations())
    app_js = (
        _resource_text("gradio_app.js")
        .replace(
            "__MINERU_I18N__",
            f"({_resource_text('gradio_i18n.js')})({json.dumps(MESSAGES, ensure_ascii=False)})",
        )
        .replace(
            "__MINERU_STATUS_TIMER__",
            _resource_text("gradio_status_timer.js"),
        )
    )
    # 等待限制放在生成器内部，使其他会话也能立即显示本地排队状态。
    conversions = SessionConversions()

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
                        value=MESSAGES["tier_value"][1].format(tier=MESSAGES[f"tier_{preferred_tier}"][1], notice=""),
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
                        i18n("mineru.convert"),
                        variant="primary",
                        scale=1,
                        min_width=0,
                        interactive=False,
                        elem_classes=["mineru-convert-button"],
                    )
                    clear_button = gr.ClearButton(value=i18n("mineru.clear"), scale=1, min_width=1)
                status_panel = gr.HTML(_status_html(), elem_classes=["mineru-status-panel"])
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=input_file,
                        label=i18n("mineru.examples"),
                        elem_id="mineru-kit-examples",
                        # 示例全部落在第一页，超出由卡片内部滚动；避免出现翻页控件。
                        examples_per_page=_EXAMPLES_PER_PAGE,
                    )

            with gr.Column(scale=4, min_width=340, elem_classes=["mineru-kit-preview", "mineru-preview-pane"]):
                pdf_preview = gr.File(visible=False, interactive=False, label="PDF", type="filepath")
                pdf_viewer_entry = gr.File(value=str(viewer_entry), visible=False, interactive=False)
                epub_viewer_entry_component = gr.File(value=str(epub_viewer_entry), visible=False, interactive=False)
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
                source_preview = gr.HTML(
                    value="",
                    apply_default_css=False,
                    elem_classes=["mineru-kit-source-preview"],
                )
                source_ticket = gr.Textbox(value="", visible=False)
                source_receipt = gr.Textbox(value="", visible=False)
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

        artifact_state = gr.State(value=None)
        active_run_id = gr.Textbox(value="", visible=False)
        conversion_ticket = gr.Textbox(value="", visible=False)
        conversion_receipt = gr.Textbox(value="", visible=False)
        conversion_cancel = gr.Textbox(value="", visible=False)
        status_snapshot = gr.Textbox(value="", visible=False)
        status_poll = gr.Timer(value=1.0, active=False)
        api_conversion_button = gr.Button(visible=False)
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
            if suffix in {"ofd", "epub"} or suffix in HTML_EXTENSIONS | MHTML_EXTENSIONS:
                # OFD/EPUB/HTML 源预览由独立异步事件挂载，此处只隐藏占位组件。
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

        source_script = _resource_text("gradio_source_preview.js")
        input_file.change(
            fn=None,
            inputs=input_file,
            outputs=[source_ticket, source_preview],
            js=f"(...args) => ({source_script})('begin', ...args)",
            **private_event_kwargs,
        )
        # Gradio 6.8 的纯前端事件不可靠地触发 then；通过请求值变化启动后台转换。
        render_source_preview = source_ticket.change(
            fn=prepare_source_preview,
            inputs=[input_file, source_ticket],
            outputs=source_receipt,
            concurrency_limit=None,
            trigger_mode="multiple",
            **private_event_kwargs,
        )
        render_source_preview.then(
            fn=None,
            inputs=[source_receipt, epub_viewer_entry_component],
            outputs=source_preview,
            js=f"(...args) => ({source_script})('apply', ...args)",
            **private_event_kwargs,
        )
        clear_button.click(
            fn=None,
            inputs=[],
            outputs=[source_ticket, source_preview],
            js=f"(...args) => ({source_script})('clear', ...args)",
            **private_event_kwargs,
        )

        download_script = _resource_text("gradio_download.js")

        def download_js(action: str, format_name: str = "", label: str = "") -> str:
            """为下载事件绑定明确的动作与格式，复用同一份前端状态处理脚本。"""
            arguments = ", ".join(json.dumps(value) for value in (action, _DOWNLOAD_FORMATS, format_name, label))
            return f"(...args) => ({download_script})({arguments}, ...args)"

        download_reset_outputs = [
            active_run_id,
            *download_files.values(),
            *download_requests.values(),
            *download_receipts.values(),
            *download_buttons.values(),
            download_notice,
        ]
        # 文件切换只重置下载控件；转换点击另加状态卡片，避免重置结果错位清空卡片。
        begin_conversion_outputs = [status_panel, *download_reset_outputs]

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
            """取消会话任务，并等待同步输出退出后回收后台工作。"""
            task = conversions.cancel(getattr(request, "session_hash", "") or "")
            if task is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await await_task_completion(task)

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

        async def execute_conversion(
            run: ConversionRun,
            file_path: str | None,
            tier_position: int | float,
            raw_page_range: str,
            force_ocr: bool,
            request: object | None,
        ) -> tuple[Any, ...]:
            """执行一次转换，只返回完整终态；进度写入独立的会话快照。"""
            reset_result = (
                _status_html(_DEFAULT_STATUS),
                "",
                *(gr.skip() for _ in range(4)),
                None,
                *_download_updates(gr, interactive=False),
                "",
            )

            def failure(message: str) -> tuple[Any, ...]:
                """把输入或输出错误收敛成可轮询的失败终态。"""
                run.publish(f"Failed: {message}")
                return (run.state.render(), *reset_result[1:])

            if not file_path:
                return failure("No input file")
            source_path = Path(file_path).resolve()
            if not source_path.is_file():
                return failure("input file does not exist")
            suffix = _file_suffix(source_path)
            if suffix not in PARSEABLE_EXTENSIONS:
                return failure(f"unsupported file type '.{suffix}'")
            try:
                selected_tier = _tier_for_position(tier_position, tier_choices)
            except ValueError as exc:
                return failure(str(exc))
            if suffix in FLASH_ONLY_PARSE_EXTENSIONS:
                if "flash" not in tier_choices:
                    return failure("tier_unavailable: 该格式仅支持 Flash，当前服务不可用")
                selected_tier = "flash"
            loop = asyncio.get_running_loop()

            def emit(message: str | ParseStatusUpdate) -> None:
                """在通知发生时记录时间，再由事件循环串行更新会话状态。"""
                at = time.monotonic()
                loop.call_soon_threadsafe(lambda: run.publish(message, at=at))

            async def run_conversion() -> tuple[Any, ...]:
                """实际同步工作退出前一直持有执行槽，避免取消后与新任务重叠。"""
                if conversions.slot.locked():
                    run.publish(STATUS_QUEUED_LOCALLY)
                async with conversions.slot:
                    run.publish(STATUS_PREPARING_REQUEST)
                    page_text = await run_sync_output(_effective_page_range, source_path, raw_page_range, max_pages=max_pages)
                    result = await client.parse_file(
                        source_path,
                        tier=selected_tier,
                        page_range=page_text,
                        ocr_mode="ocr" if suffix in PDF_EXTENSIONS and force_ocr else "auto",
                        status_callback=emit,
                    )
                    # 先消费同一轮已发出的解析通知，保持阶段和解析耗时的顺序。
                    await asyncio.sleep(0)
                    run.publish(STATUS_PROCESSING_OUTPUT)
                    output_started = time.monotonic()
                    artifacts = await run_sync_output(
                        persist_parse_result, result, source_path, output_root=output_root, page_range=page_text
                    )
                    rendered_html = await run_sync_output(
                        render_html_preview, artifacts, public_base_url=_gradio_public_base_url(request)
                    )
                    structured_json = await run_sync_output(artifacts.structured_content_path.read_text, encoding="utf-8")
                    preview_path = artifacts.layout_pdf_path or artifacts.origin_pdf_path
                    generic_html = "" if preview_path else preview_placeholder("result_ready")
                    show_image_preview = suffix in IMAGE_EXTENSIONS and preview_path is None
                    result_preview_updates = (
                        _pdf_preview_update(gr, str(preview_path) if preview_path else None),
                        gr.update(value=str(artifacts.source_path) if show_image_preview else None, visible=show_image_preview),
                        gr.update(value="", visible=False),
                        gr.update(value=generic_html, visible=bool(generic_html)),
                    )
                    if _is_office(source_path) or suffix in {"ofd", "epub"} or suffix in HTML_EXTENSIONS | MHTML_EXTENSIONS:
                        result_preview_updates = tuple(gr.skip() for _ in range(4))
                    run.artifacts = artifacts.as_state()
                    logger.debug(
                        "WebUI output ready run_id={} artifacts={} elapsed={:.3f}s html_bytes={} json_bytes={} ready_at={:.3f}",
                        run.run_id,
                        artifacts.root.name,
                        time.monotonic() - output_started,
                        len(rendered_html.encode("utf-8")),
                        len(structured_json.encode("utf-8")),
                        time.time(),
                    )
                    return (
                        rendered_html,
                        *result_preview_updates,
                        run.artifacts,
                        *_download_updates(gr, interactive=True, run_id=artifacts.root.name),
                        structured_json,
                    )

            task = asyncio.create_task(run_conversion())
            run.task = task
            try:
                result_outputs = await asyncio.shield(task)
                if run.cancelled:
                    return tuple(gr.skip() for _ in reset_result)
                run.publish(STATUS_COMPLETED)
                return (run.state.render(), *result_outputs)
            except asyncio.CancelledError:
                run.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await await_task_completion(task)
                return tuple(gr.skip() for _ in reset_result)
            except Exception as exc:
                logger.exception("WebUI conversion failed run_id={}", run.run_id)
                message = f"{exc.code}: {exc}" if isinstance(exc, MineruError) else str(exc)
                return failure(message)
            finally:
                run.task = None

        async def convert_handler(
            file_path: str | None,
            tier_position: int | float,
            raw_page_range: str,
            force_ocr: bool = False,
            request: object | None = None,
        ) -> tuple[Any, ...]:
            """保留公开转换 API 的四个输入和原生组件输出，普通响应一次返回结果。"""
            session = getattr(request, "session_hash", None) or uuid.uuid4().hex
            run = conversions.start(session, uuid.uuid4().hex)
            assert run is not None
            try:
                return await execute_conversion(run, file_path, tier_position, raw_page_range, force_ocr, request)
            finally:
                if not getattr(request, "session_hash", None):
                    conversions.cancel(session, run.run_id)

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
        # 公开 API 仍由 Gradio 原生组件序列化，浏览器通过私有回执校验后才应用结果。
        convert_event = api_conversion_button.click(
            fn=convert_handler,
            inputs=[input_file, tier, page_range, force_ocr],
            outputs=convert_outputs,
            **event_kwargs,
        )
        conversion_script = _resource_text("gradio_conversion.js")

        def conversion_js(action: str) -> str:
            """绑定自有任务协调脚本，不访问 Gradio 的内部状态或差分缓存。"""
            return f"(...args) => ({conversion_script})({json.dumps(action)}, ...args)"

        def read_conversion_status(ticket: str, request: object | None = None) -> str:
            """普通请求返回完整快照；没有有效任务时不创建新的轮询状态。"""
            try:
                run_id = json.loads(ticket)["run_id"]
            except (ValueError, TypeError, KeyError):
                return ""
            run = conversions.current(getattr(request, "session_hash", "") or "", run_id)
            return run.snapshot if run is not None else ""

        async def cancel_ui_conversion(ticket: str, request: object | None = None) -> None:
            """只取消浏览器刚刚撤销的那次任务，避免晚到请求误伤新任务。"""
            try:
                identity = json.loads(ticket)
                run_id = uuid.UUID(identity["run_id"]).hex
                revision = identity["revision"]
                if type(revision) is not int or revision <= 0:
                    return
            except (ValueError, TypeError, KeyError, AttributeError):
                return
            session = getattr(request, "session_hash", "") or ""
            # 取消可能先于队列中的提交到达，记住修订号以拒绝尚未开始的旧请求。
            conversions.revisions[session] = max(conversions.revisions.get(session, 0), revision)
            task = conversions.cancel(session, run_id)
            if task is not None:
                with suppress(asyncio.CancelledError, Exception):
                    await await_task_completion(task)

        async def unload_session(request: object | None = None) -> None:
            """标签页关闭后释放会话快照，并等待剩余同步任务退出。"""
            await cancel_session_conversion(request)
            conversions.revisions.pop(getattr(request, "session_hash", "") or "", None)

        async def convert_ui(
            file_path: str | None,
            tier_position: int | float,
            raw_page_range: str,
            force_ocr: bool,
            ticket: str,
            request: object | None = None,
        ) -> str:
            """用完整回执传递浏览器结果，使迟到响应无法直接覆盖可见组件。"""
            from gradio.data_classes import FileData

            try:
                identity = json.loads(ticket)
                run_id = uuid.UUID(identity["run_id"]).hex
                revision = identity["revision"]
                if type(revision) is not int or revision <= 0:
                    return ""
            except (ValueError, TypeError, KeyError, AttributeError):
                return ""
            session = getattr(request, "session_hash", "") or ""
            run = conversions.start(session, run_id, revision=revision)
            if run is None:
                return ""
            result = await execute_conversion(run, file_path, tier_position, raw_page_range, force_ocr, request)
            if conversions.current(session, run_id) is not run:
                return ""
            values = list(result)
            # 文件来自输出根目录的不可变任务目录，使用现有 allowed_paths 文件路由。
            for index in (2, 3):
                update = values[index]
                if isinstance(update, dict) and update.get("value"):
                    path = Path(update["value"]).resolve()
                    path.relative_to(output_root.resolve())
                    file_url = f"{_gradio_public_base_url(request)}/gradio_api/file={quote(path.as_posix(), safe='/')}"
                    values[index] = {
                        **update,
                        "value": FileData(path=str(path), url=file_url, orig_name=path.name).model_dump(mode="json"),
                    }
            return json.dumps(
                {
                    "run_id": run_id,
                    "sequence": run.state.sequence,
                    "ready_at": time.time(),
                    "outputs": values[:6] + values[7:],
                },
                ensure_ascii=False,
            )

        for callback in (read_conversion_status, cancel_ui_conversion, unload_session, convert_ui):
            callback.__annotations__["request"] = gr.Request
        # 纯前端开始事件通过票据 change 发起转换，兼容旧版 Gradio 的 then 限制。
        convert_button.click(
            fn=None,
            inputs=[],
            outputs=[*begin_conversion_outputs, conversion_ticket, status_poll, html_output, json_output],
            js=(
                f"() => {{ ({pdf_preview_js('begin')})(); "
                f"const ticket = ({conversion_js('begin')})(); "
                f"return [{json.dumps(_status_html(STATUS_PREPARING_REQUEST))}, "
                f"...({download_js('reset')})(), ...ticket, '', '']; }}"
            ),
            **private_event_kwargs,
        )
        ui_convert_event = conversion_ticket.change(
            fn=convert_ui,
            inputs=[input_file, tier, page_range, force_ocr, conversion_ticket],
            outputs=conversion_receipt,
            queue=True,
            concurrency_limit=None,
            trigger_mode="multiple",
            show_progress="hidden",
            api_visibility="private",
        )
        status_poll.tick(
            fn=read_conversion_status,
            inputs=conversion_ticket,
            outputs=status_snapshot,
            trigger_mode="always_last",
            show_progress="hidden",
            **private_event_kwargs,
        )
        status_snapshot.change(
            fn=None,
            inputs=status_snapshot,
            outputs=[status_panel, status_poll],
            js=conversion_js("status"),
            **private_event_kwargs,
        )
        ui_result_outputs = [*convert_outputs[:6], *convert_outputs[7:], status_poll]
        conversion_receipt.change(
            fn=None,
            inputs=[conversion_receipt, input_file, pdf_viewer_entry],
            outputs=[*ui_result_outputs, pdf_viewer],
            js=(
                "(receipt, source, viewer) => { "
                f"const values = ({conversion_js('result')})(receipt); "
                "const pdf = values[2]; "
                f"const preview = pdf?.__type__ === 'update' && 'value' in pdf ? ({pdf_preview_js('result')})("
                "pdf.value, source, viewer, values[6]) : {__type__: 'update'}; "
                "return [...values, preview]; }"
            ),
            **private_event_kwargs,
        )
        gr.on(
            triggers=[input_file.change, clear_button.click],
            fn=None,
            inputs=[],
            outputs=[conversion_ticket, conversion_cancel, status_poll],
            js=conversion_js("cancel"),
            **private_event_kwargs,
        )
        conversion_cancel.change(
            fn=cancel_ui_conversion,
            inputs=conversion_cancel,
            outputs=[],
            trigger_mode="multiple",
            **private_event_kwargs,
        )
        demo.unload(unload_session)
        source_preview_event = input_file.change(
            fn=update_file_preview,
            inputs=input_file,
            outputs=preview_outputs,
            cancels=[convert_event, ui_convert_event],
            # 示例切换也会触发文件变更，预览准备期间保持状态卡片可见。
            show_progress="hidden",
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

        clear_button.click(
            fn=reset_ui, inputs=[], outputs=reset_outputs, cancels=[convert_event, ui_convert_event], **private_event_kwargs
        )

        def session_download_handler(format_name: str) -> Callable[..., tuple[Any, str]]:
            """下载时读取当前会话的素材路径，公开 API 的 State 仍可单独使用。"""
            renderer = _download_handler(format_name, output_root)

            def handler(state: dict[str, Any] | None, token: str, request: object | None = None) -> tuple[Any, str]:
                """使用服务端当前任务核对下载标识，拒绝过期下载请求。"""
                run = conversions.runs.get(getattr(request, "session_hash", "") or "")
                if run is not None:
                    state = run.artifacts if not run.cancelled else None
                return renderer(state, token, request)

            return handler

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
            download_handler = session_download_handler(format_name)
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
    api_server_tier: str,
    api_server_concurrency: int,
    api_server_disable_image_analysis: bool,
    api_server_preload_models: bool,
    max_pages: int | None = None,
) -> None:
    """启动 Gradio，并在外部服务缺少 Flash 时托管本地 Flash V1 服务。"""
    configure_standard_streams()
    configure_global_log_level()
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


# 示例数量预期很小；单页放全部文件，翻页交给卡片内部滚动。
_EXAMPLES_PER_PAGE = 1000


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
