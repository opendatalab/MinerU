# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 ReportLab PDF bytes 的公共渲染实现。"""

from __future__ import annotations

from functools import partial
import html
from io import BytesIO
import re
from typing import Any, Iterable

from bs4 import BeautifulSoup
from loguru import logger
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    Image as ReportLabImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ....backend.postprocess.inline import inline_plain_text, join_inline_spans
from ..common.index import strip_index_page_tail
from ..common.list_items import parse_list_item_marker, reference_list_needs_bullets
from ..common.planner import PlannedBlock, build_render_plan
from ...contracts import AssetResolver
from ....types import (
    PAGE_AUXILIARY_BLOCK_TYPES,
    RAW_ALGORITHM,
    AlgorithmBodyBlock,
    BlockBase,
    BlockType,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeAnnotationBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
    HyperlinkSpan,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    ImagePayloadBlock,
    IndexBlock,
    InlineSpan,
    ListBlock,
    MiddleJson,
    NonLinkInlineSpan,
    PageFootnoteBlock,
    ParagraphTitleBlock,
    RefTextBlock,
    TableAnnotationBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
    TextSpan,
    TitleBlockBase,
)
from .assets import PdfAssetError, PreparedImage, prepare_block_image, prepare_html_image
from .formula import (
    DisplayFormulaFlowable,
    FormulaRenderer,
    InlineFormulaImage,
    PdfFormulaError,
    draw_inline_formula,
    split_formula_tag,
)
from .inline import PdfAnchorRegistry, PdfInlineContext, build_pdf_paragraph, render_plain_text_markup
from .styles import (
    BORDER_COLOR,
    PAGE_MARGIN,
    SURFACE_COLOR,
    build_pdf_styles,
)
from .table import PdfTableError, build_pdf_tables

_HTML_TABLE_RE = re.compile(r"<table\b", re.IGNORECASE)
_INVALID_METADATA_TEXT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff\ufffe\uffff]")
_VISIBLE_HTML_TAGS = (
    "a",
    "b",
    "blockquote",
    "br",
    "code",
    "div",
    "em",
    "eq",
    "i",
    "li",
    "ol",
    "p",
    "pre",
    "span",
    "strong",
    "sub",
    "sup",
    "table",
    "u",
    "ul",
)
_MAX_PLACEHOLDER_TEXT = 320
_MIN_IMAGE_WIDTH = 5 * mm
_PLACEHOLDER_HEIGHT = 18 * mm


class _PdfCanvas(Canvas):
    """提供确定性 metadata 与行内 ZiaMath 矢量绘制的 ReportLab Canvas。"""

    def __init__(self, filename: Any, *args: Any, document_title: str, **kwargs: Any) -> None:
        """创建启用压缩和 invariant 的 PDF canvas，并写入稳定 metadata。"""
        kwargs.setdefault("pageCompression", 1)
        kwargs["invariant"] = 1
        super().__init__(filename, *args, **kwargs)
        self.setTitle(document_title)
        self.setAuthor("MinerU")
        self.setCreator("MinerU PDF Renderer")
        self.setSubject("Semantic document rendering from MiddleJson")
        self.setKeywords("MinerU, MiddleJson, PDF")

    def drawImage(
        self,
        image: Any,
        x: float,
        y: float,
        width: float | None = None,
        height: float | None = None,
        mask: Any = None,
        preserveAspectRatio: bool = False,
        anchor: str = "c",
        anchorAtXY: bool = False,
        showBoundary: bool = False,
    ) -> Any:
        """识别 Paragraph 传入的公式代理，否则沿用标准 raster 图片行为。"""
        if isinstance(image, InlineFormulaImage):
            resolved_width = image.vector.width if width is None else float(width)
            resolved_height = image.vector.height if height is None else float(height)
            return draw_inline_formula(self, image, x, y, resolved_width, resolved_height)
        return super().drawImage(
            image,
            x,
            y,
            width,
            height,
            mask=mask,
            preserveAspectRatio=preserveAspectRatio,
            anchor=anchor,
            anchorAtXY=anchorAtXY,
            showBoundary=showBoundary,
        )


class _PdfRenderer:
    """维护一次 MiddleJson 到 PDF 渲染所需的样式、公式与素材状态。"""

    def __init__(
        self,
        middle_json: MiddleJson,
        *,
        asset_resolver: AssetResolver | None,
        document_title: str | None,
    ) -> None:
        """保存严格输入，并预注册标题及页面脚注 anchor。"""
        self.middle_json = middle_json
        self.asset_resolver = asset_resolver
        self.document_title = _resolve_document_title(middle_json, document_title)
        self.styles = build_pdf_styles()
        self.available_width = A4[0] - 2 * PAGE_MARGIN
        self.available_height = A4[1] - 2 * PAGE_MARGIN
        self.inline_context = PdfInlineContext(
            formulas=FormulaRenderer(),
            anchors=PdfAnchorRegistry(_iter_document_anchors(middle_json)),
        )

    def render(self) -> bytes:
        """构造逐页 story，并将确定性 ReportLab 文档序列化为 bytes。"""
        story: list[Flowable] = []
        planned_pages = build_render_plan(self.middle_json)
        for planned_blocks in planned_pages:
            for planned in planned_blocks:
                if planned.removed:
                    continue
                if planned.block.type in PAGE_AUXILIARY_BLOCK_TYPES:
                    continue
                rendered = self._render_planned_block(planned)
                story.extend(rendered)
        if not story:
            story.append(Spacer(1, 1))

        output = BytesIO()
        document = SimpleDocTemplate(
            output,
            pagesize=A4,
            leftMargin=PAGE_MARGIN,
            rightMargin=PAGE_MARGIN,
            topMargin=PAGE_MARGIN,
            bottomMargin=PAGE_MARGIN,
            title=self.document_title,
            author="MinerU",
            creator="MinerU PDF Renderer",
            subject="Semantic document rendering from MiddleJson",
            keywords="MinerU, MiddleJson, PDF",
            invariant=1,
            pageCompression=1,
        )
        document.build(
            story,
            canvasmaker=partial(_PdfCanvas, document_title=self.document_title),
        )
        return output.getvalue()

    def _render_planned_block(self, planned: PlannedBlock) -> list[Flowable]:
        """按严格 PageBlock 具体类型分派 PDF Flowable visitor。"""
        block = planned.block
        if isinstance(block, (TextBlock, RefTextBlock)):
            spans = join_inline_spans(planned.text_contents or [block.content])
            return [self._paragraph(spans, self.styles.body, planned.page_idx, block)]
        if isinstance(block, (DocTitleBlock, ParagraphTitleBlock)):
            return [
                self._paragraph(
                    block.content,
                    self.styles.heading(block.level),
                    planned.page_idx,
                    block,
                    anchor=block.anchor,
                )
            ]
        if isinstance(block, PageFootnoteBlock):
            return [
                self._paragraph(
                    block.content,
                    self.styles.footnote,
                    planned.page_idx,
                    block,
                    anchor=block.anchor,
                )
            ]
        if isinstance(block, EquationBlock):
            return self._render_equation(block, planned.page_idx)
        if isinstance(block, ListBlock):
            return self._render_list(block, planned.page_idx, depth=0)
        if isinstance(block, IndexBlock):
            return self._render_index(block, planned.page_idx, depth=0)
        if isinstance(block, ImageBlock):
            return self._render_image_block(block, planned.page_idx)
        if isinstance(block, TableBlock):
            return self._render_table_block(block, planned.page_idx)
        if isinstance(block, ChartBlock):
            return self._render_chart_block(block, planned.page_idx)
        if isinstance(block, CodeBlock):
            return self._render_code_block(block, planned.page_idx)
        raise TypeError(f"Unsupported PageBlock type: {type(block).__name__}")

    def _paragraph(
        self,
        spans: list[InlineSpan],
        style: ParagraphStyle,
        page_idx: int,
        block: BlockBase,
        *,
        anchor: str | None = None,
        preserve_newlines: bool = False,
        max_width: float | None = None,
    ) -> Paragraph:
        """为当前 block 构造带定位、anchor 与矢量公式的 Paragraph。"""
        return build_pdf_paragraph(
            spans,
            style,
            context=self.inline_context,
            page_idx=page_idx,
            block_index=block.index,
            block_type=str(block.type),
            max_width=self.available_width if max_width is None else max_width,
            anchor=anchor,
            preserve_newlines=preserve_newlines,
        )

    def _render_equation(self, block: EquationBlock, page_idx: int) -> list[Flowable]:
        """优先输出 ZiaMath display 矢量，失败后使用图片、LaTeX 或占位。"""
        content = block.content.strip()
        if content:
            formula, tag = split_formula_tag(content)
            try:
                vector = self.inline_context.formulas.render(formula or content, inline=False, font_size=14)
                tag_vector = None
                if tag:
                    tag_vector = self.inline_context.formulas.render(f"({tag})", inline=True, font_size=9)
                flowable = DisplayFormulaFlowable(vector, tag_vector)
                flowable.spaceBefore = 5
                flowable.spaceAfter = 7
                return [flowable]
            except PdfFormulaError as exc:
                logger.warning("PDF display formula fallback: {} ({})", exc, self._location(page_idx, block))
        if _has_image_payload(block):
            image = self._try_prepared_block_image(block, page_idx)
            if image is not None:
                return [self._prepared_image_flowable(image, block)]
        if content:
            return [
                self._paragraph(
                    [TextSpan(type="text", content=content)],
                    self.styles.formula_fallback,
                    page_idx,
                    block,
                    preserve_newlines=True,
                )
            ]
        return [self._placeholder("formula unavailable", block=block, page_idx=page_idx)]

    def _render_list(self, block: ListBlock, page_idx: int, *, depth: int) -> list[Flowable]:
        """保留 producer marker，以缩进和悬挂缩进表达递归列表。"""
        rendered: list[Flowable] = []
        add_reference_bullets = reference_list_needs_bullets(block)
        for child in block.content:
            if isinstance(child, ListBlock):
                rendered.extend(self._render_list(child, page_idx, depth=depth + 1))
                continue
            spans = list(child.content)
            parsed = parse_list_item_marker(spans)
            if add_reference_bullets and parsed.marker is None and inline_plain_text(spans).strip():
                spans = [TextSpan(type="text", content="- "), *spans]
                parsed = parse_list_item_marker(spans)
            style = ParagraphStyle(
                f"MinerU PDF List {depth} {len(rendered)}",
                parent=self.styles.body,
                leftIndent=(depth + (1 if parsed.marker else 0)) * 14,
                firstLineIndent=-14 if parsed.marker else 0,
                spaceAfter=3,
            )
            rendered.append(self._paragraph(spans, style, page_idx, child))
        return rendered

    def _render_index(self, block: IndexBlock, page_idx: int, *, depth: int) -> list[Flowable]:
        """递归输出目录叶子，并把已注册标题 anchor 写成内部链接。"""
        rendered: list[Flowable] = []
        for child in block.content:
            if isinstance(child, IndexBlock):
                rendered.extend(self._render_index(child, page_idx, depth=depth + 1))
                continue
            content = strip_index_page_tail(child.content)
            if not content:
                continue
            spans: list[InlineSpan] = [TextSpan(type="text", content="• ")]
            if isinstance(child, TitleBlockBase) and child.anchor:
                link_content = _flatten_non_link_spans(content)
                if link_content:
                    spans.append(HyperlinkSpan(type="hyperlink", url=f"#{child.anchor}", content=link_content))
            else:
                spans.extend(content)
            style = ParagraphStyle(
                f"MinerU PDF Index {depth} {len(rendered)}",
                parent=self.styles.body,
                leftIndent=(depth + 1) * 14,
                firstLineIndent=-10,
                spaceAfter=3,
            )
            rendered.append(self._paragraph(spans, style, page_idx, child))
        return rendered

    def _render_image_block(self, block: ImageBlock, page_idx: int) -> list[Flowable]:
        """按原始子块顺序输出图片主体、宽松占位与说明。"""
        rendered: list[Flowable] = []
        for child in block.content:
            if isinstance(child, ImageBodyBlock):
                alt_text = _plain_html_text(child.content) or block.sub_type or "image"
                flowable, succeeded = self._image_or_placeholder(child, page_idx=page_idx, alt_text=alt_text)
                rendered.append(flowable)
                if not succeeded and child.content.strip():
                    rendered.append(
                        self._paragraph(
                            [TextSpan(type="text", content=_plain_html_text(child.content))],
                            self.styles.footnote,
                            page_idx,
                            child,
                        )
                    )
            elif isinstance(child, ImageAnnotationBlock):
                rendered.append(self._render_annotation(child, page_idx))
            else:
                raise TypeError(f"Unsupported image child: {type(child).__name__}")
        return rendered

    def _render_table_block(self, block: TableBlock, page_idx: int) -> list[Flowable]:
        """优先输出原生 HTML table，再回退空间文本、图片或占位。"""
        rendered: list[Flowable] = []
        for child in block.content:
            if isinstance(child, TableBodyBlock):
                content = child.content.strip()
                if content and _HTML_TABLE_RE.search(content):
                    try:
                        rendered.extend(self._html_tables(content, page_idx=page_idx, block=child))
                        continue
                    except PdfTableError as exc:
                        logger.warning("PDF HTML table fallback: {} ({})", exc, self._location(page_idx, child))
                    if _has_image_payload(child):
                        image = self._try_prepared_block_image(child, page_idx)
                        if image is not None:
                            rendered.append(self._prepared_image_flowable(image, child))
                            continue
                    plain = _plain_html_text(content)
                    if plain:
                        rendered.append(self._preformatted(plain, page_idx, child, self.styles.spatial_table))
                    rendered.append(self._placeholder("table unavailable", block=child, page_idx=page_idx))
                elif content:
                    rendered.append(self._preformatted(child.content, page_idx, child, self.styles.spatial_table))
                else:
                    flowable, _succeeded = self._image_or_placeholder(child, page_idx=page_idx, alt_text="table")
                    rendered.append(flowable)
            elif isinstance(child, TableAnnotationBlock):
                rendered.append(self._render_annotation(child, page_idx))
            else:
                raise TypeError(f"Unsupported table child: {type(child).__name__}")
        return rendered

    def _render_chart_block(self, block: ChartBlock, page_idx: int) -> list[Flowable]:
        """输出 chart 图片或占位，并继续保留可物化的结构化内容。"""
        rendered: list[Flowable] = []
        for child in block.content:
            if isinstance(child, ChartBodyBlock):
                has_image = _has_image_payload(child)
                image_succeeded = False
                if has_image:
                    flowable, image_succeeded = self._image_or_placeholder(
                        child,
                        page_idx=page_idx,
                        alt_text=block.sub_type or "chart",
                    )
                    rendered.append(flowable)
                content = child.content.strip()
                if content and _HTML_TABLE_RE.search(content):
                    try:
                        rendered.extend(self._html_tables(content, page_idx=page_idx, block=child))
                    except PdfTableError as exc:
                        logger.warning("PDF chart table omitted: {} ({})", exc, self._location(page_idx, child))
                        if not image_succeeded:
                            rendered.append(self._preformatted(_plain_html_text(content), page_idx, child, self.styles.body))
                elif content and not image_succeeded:
                    rendered.append(
                        self._paragraph(
                            [TextSpan(type="text", content=_plain_html_text(content))],
                            self.styles.body,
                            page_idx,
                            child,
                        )
                    )
                elif not has_image and not content:
                    rendered.append(self._placeholder("chart unavailable", block=child, page_idx=page_idx))
            elif isinstance(child, ChartAnnotationBlock):
                rendered.append(self._render_annotation(child, page_idx))
            else:
                raise TypeError(f"Unsupported chart child: {type(child).__name__}")
        return rendered

    def _render_code_block(self, block: CodeBlock, page_idx: int) -> list[Flowable]:
        """使用等宽浅色块输出代码或带矢量公式的算法正文。"""
        rendered: list[Flowable] = []
        for child in block.content:
            if isinstance(child, CodeBodyBlock):
                if block.sub_type != BlockType.CODE:
                    raise TypeError("code_body requires code subtype")
                rendered.append(
                    self._paragraph(
                        [TextSpan(type="text", content=child.content or " ")],
                        self.styles.code,
                        page_idx,
                        child,
                        preserve_newlines=True,
                    )
                )
            elif isinstance(child, AlgorithmBodyBlock):
                if block.sub_type != RAW_ALGORITHM:
                    raise TypeError("algorithm_body requires algorithm subtype")
                rendered.append(
                    self._paragraph(
                        child.content,
                        self.styles.code,
                        page_idx,
                        child,
                        preserve_newlines=True,
                    )
                )
            elif isinstance(child, CodeAnnotationBlock):
                rendered.append(self._render_annotation(child, page_idx))
            else:
                raise TypeError(f"Unsupported code child: {type(child).__name__}")
        return rendered

    def _render_annotation(
        self,
        block: ImageAnnotationBlock | TableAnnotationBlock | ChartAnnotationBlock | CodeAnnotationBlock,
        page_idx: int,
    ) -> Paragraph:
        """根据 caption/footnote discriminator 选择弱化说明样式。"""
        style = self.styles.caption if str(block.type).endswith("caption") else self.styles.footnote
        return self._paragraph(block.content, style, page_idx, block)

    def _preformatted(
        self,
        content: str,
        page_idx: int,
        block: BlockBase,
        style: ParagraphStyle,
    ) -> Paragraph:
        """把需要保留换行和空白的普通字符串写为 Paragraph。"""
        return self._paragraph(
            [TextSpan(type="text", content=content or " ")],
            style,
            page_idx,
            block,
            preserve_newlines=True,
        )

    def _html_tables(self, content: str, *, page_idx: int, block: BlockBase) -> list[Table]:
        """使用当前 block 上下文把 HTML table 物化为 ReportLab 表格。"""

        def build_paragraph(spans: list[InlineSpan], style: object, max_width: float) -> Paragraph:
            """为表格单元格构造支持公式和链接的 Paragraph。"""
            if not isinstance(style, ParagraphStyle):
                raise TypeError("PDF table paragraph style must be a ParagraphStyle")
            return self._paragraph(
                spans,
                style,
                page_idx,
                block,
                preserve_newlines=True,
                max_width=max_width,
            )

        def build_image(source: str, max_width: float, alt_text: str) -> Flowable:
            """为表格单元格构造离线图片或宽松占位。"""
            try:
                prepared = prepare_html_image(source, self.asset_resolver)
            except PdfAssetError as exc:
                logger.warning("PDF table image placeholder: {} ({})", exc, self._location(page_idx, block))
                return self._placeholder(
                    f"image unavailable: {alt_text}",
                    block=block,
                    page_idx=page_idx,
                    width=max_width,
                    url=source if _is_remote_url(source) else None,
                )
            return self._prepared_image_flowable(prepared, None, max_width=max_width)

        return list(
            build_pdf_tables(
                content,
                available_width=self.available_width,
                styles=self.styles,
                build_paragraph=build_paragraph,
                build_image=build_image,
            )
        )

    def _image_or_placeholder(
        self,
        block: ImagePayloadBlock,
        *,
        page_idx: int,
        alt_text: str,
    ) -> tuple[Flowable, bool]:
        """加载 block 图片；任何离线失败都转换为可见占位而不抛出。"""
        prepared = self._try_prepared_block_image(block, page_idx)
        if prepared is not None:
            return self._prepared_image_flowable(prepared, block), True
        remote_url = block.image_url if block.image_path is None and block.image_base64 is None else None
        return (
            self._placeholder(
                f"image unavailable: {alt_text}",
                block=block,
                page_idx=page_idx,
                url=remote_url,
            ),
            False,
        )

    def _try_prepared_block_image(self, block: ImagePayloadBlock, page_idx: int) -> PreparedImage | None:
        """尝试离线准备图片并把所有素材错误降级为 warning。"""
        try:
            return prepare_block_image(block, self.asset_resolver)
        except PdfAssetError as exc:
            logger.warning("PDF image placeholder: {} ({})", exc, self._location(page_idx, block))
            return None

    def _prepared_image_flowable(
        self,
        prepared: PreparedImage,
        block: ImagePayloadBlock | None,
        *,
        max_width: float | None = None,
    ) -> ReportLabImage:
        """按自然尺寸或 bbox 宽度限制图片，并保持宽高比和左对齐。"""
        available_width = self.available_width if max_width is None else max(1.0, max_width)
        natural_width = prepared.width_px / 96 * 72
        desired_width = natural_width
        if block is not None and block.bbox is not None:
            desired_width = available_width * (block.bbox[2] - block.bbox[0])
        desired_width = max(min(_MIN_IMAGE_WIDTH, available_width), min(desired_width, available_width))
        desired_height = desired_width * prepared.height_px / max(prepared.width_px, 1)
        max_height = self.available_height
        if desired_height > max_height:
            scale = max_height / desired_height
            desired_width *= scale
            desired_height *= scale
        image = ReportLabImage(BytesIO(prepared.data), width=desired_width, height=desired_height)
        image.hAlign = "LEFT"
        image.spaceBefore = 5
        image.spaceAfter = 5
        return image

    def _placeholder(
        self,
        label: str,
        *,
        block: BlockBase,
        page_idx: int,
        width: float | None = None,
        url: str | None = None,
    ) -> Table:
        """创建带浅色边框、可选远程链接和定位文本的稳定占位框。"""
        normalized = re.sub(r"\s+", " ", label).strip()[:_MAX_PLACEHOLDER_TEXT] or "content unavailable"
        markup = render_plain_text_markup(normalized)
        if url:
            markup += f'<br/><a href="{html.escape(url, quote=True)}" color="#0b6fc2">{html.escape(url)}</a>'
        location = self._location(page_idx, block)
        markup += f'<br/><font size="7" color="#6b7280">{html.escape(location)}</font>'
        paragraph = Paragraph(markup, self.styles.placeholder)
        target_width = self.available_width if width is None else max(1.0, min(width, self.available_width))
        table = Table([[paragraph]], colWidths=[target_width], rowHeights=[_PLACEHOLDER_HEIGHT], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), SURFACE_COLOR),
                    ("BOX", (0, 0), (-1, -1), 0.6, BORDER_COLOR),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        table.spaceBefore = 5
        table.spaceAfter = 5
        return table

    @staticmethod
    def _location(page_idx: int, block: BlockBase) -> str:
        """返回 PDF 告警与占位使用的稳定 page/block 定位。"""
        return f"page_idx={page_idx}, block_index={block.index}, block_type={block.type}"


def render_pdf(
    middle_json: MiddleJson,
    *,
    asset_resolver: AssetResolver | None = None,
    document_title: str | None = None,
) -> bytes:
    """把严格 MiddleJson 无副作用地渲染为完整 PDF bytes。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("render_pdf expects a MiddleJson instance")
    if asset_resolver is not None and not callable(asset_resolver):
        raise TypeError("asset_resolver must be callable or None")
    if document_title is not None and not isinstance(document_title, str):
        raise TypeError("document_title must be a string or None")
    return _PdfRenderer(
        middle_json,
        asset_resolver=asset_resolver,
        document_title=document_title,
    ).render()


def _resolve_document_title(middle_json: MiddleJson, explicit: str | None) -> str:
    """按显式标题、首个文档标题和固定回退的顺序生成 metadata title。"""
    if explicit is not None:
        normalized = re.sub(r"\s+", " ", _INVALID_METADATA_TEXT_RE.sub("\ufffd", explicit)).strip()
        return normalized or "MinerU Document"
    for page in middle_json.pages:
        for block in page.blocks:
            if isinstance(block, DocTitleBlock):
                normalized = re.sub(
                    r"\s+",
                    " ",
                    _INVALID_METADATA_TEXT_RE.sub("\ufffd", inline_plain_text(block.content)),
                ).strip()
                if normalized:
                    return normalized
    return "MinerU Document"


def _iter_document_anchors(middle_json: MiddleJson) -> Iterable[str]:
    """按文档顺序枚举标题和页面脚注的非空 anchor。"""
    for page in middle_json.pages:
        for block in page.blocks:
            if isinstance(block, TitleBlockBase) and block.anchor:
                yield block.anchor
            elif isinstance(block, PageFootnoteBlock) and block.anchor:
                yield block.anchor


def _plain_html_text(content: str) -> str:
    """把视觉 body 的 HTML 或普通字符串压缩为可见文本。"""
    if not content:
        return ""
    soup = BeautifulSoup(content, "html.parser")
    if soup.find(_VISIBLE_HTML_TAGS) is None:
        return re.sub(r"[ \t]+", " ", content).strip()
    return re.sub(r"[ \t]+", " ", soup.get_text("\n")).strip()


def _flatten_non_link_spans(spans: list[InlineSpan]) -> list[NonLinkInlineSpan]:
    """递归移除已有 hyperlink 包装，供目录目标建立单层内部链接。"""
    flattened: list[NonLinkInlineSpan] = []
    for span in spans:
        if isinstance(span, HyperlinkSpan):
            flattened.extend(span.content)
        else:
            flattened.append(span)
    return flattened


def _has_image_payload(block: ImagePayloadBlock) -> bool:
    """判断统一图片载荷是否声明 sidecar、data URI 或远程 URL。"""
    return block.image_path is not None or block.image_base64 is not None or block.image_url is not None


def _is_remote_url(source: str) -> bool:
    """判断图片 source 是否是不会被 PDF renderer 下载的 HTTP(S) URL。"""
    normalized = source.strip().casefold()
    return normalized.startswith("http://") or normalized.startswith("https://")


__all__ = ["render_pdf"]
