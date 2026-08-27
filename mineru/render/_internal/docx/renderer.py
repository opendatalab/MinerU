# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到可编辑 DOCX bytes 的公共渲染实现。"""

from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
import re

from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.document import Document as DocumentType
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.part import Part
from docx.shared import Emu, Mm, Pt, Twips
from docx.table import _Cell
from docx.text.paragraph import Paragraph
from lxml import etree
from loguru import logger

from ..common.index import strip_index_page_tail
from ....backend.postprocess.inline import (
    InlineEquation,
    InlineLink,
    InlineNode,
    InlineStyled,
    InlineText,
    parse_inline_content,
)
from ..common.list_items import parse_list_item_marker
from ..common.planner import PlannedBlock, build_render_plan
from .assets import (
    DocxAssetError,
    PreparedImage,
    prepare_block_image,
    prepare_html_image,
)
from .inline import (
    BookmarkRegistry,
    InlineRenderContext,
    append_inline_content,
    append_inline_nodes,
    append_internal_link,
    append_joined_inline_contents,
    sanitize_xml_text,
)
from .math import DocxFormulaError, latex_to_omml, split_formula_tag
from .styles import (
    AUXILIARY_STYLE,
    BODY_STYLE,
    CAPTION_STYLE,
    CODE_STYLE,
    FOOTNOTE_STYLE,
    FORMULA_FALLBACK_STYLE,
    SPATIAL_TABLE_STYLE,
    configure_document,
    usable_width_emu,
    usable_width_twips,
)
from .table import DocxTableError, NestedTableWriter, materialize_docx_tables
from ...contracts import AssetResolver, RenderMode
from ...docx import DocxRenderError
from ....types import (
    PAGE_AUXILIARY_BLOCK_TYPES,
    RAW_ALGORITHM,
    BlockBase,
    BlockType,
    BBox,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeAnnotationBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    ImagePayloadBlock,
    IndexBlock,
    ListBlock,
    MiddleJson,
    PageAuxTextBlock,
    PageFootnoteBlock,
    ParagraphTitleBlock,
    RefTextBlock,
    TableAnnotationBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
    TitleBlockBase,
)

_SVG_BLIP_NAMESPACE = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
_SVG_BLIP_EXTENSION_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"

_HTML_TABLE_RE = re.compile(r"<table\b", re.IGNORECASE)
_ANNOTATION_CAPTION_TYPES = {
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_CAPTION,
    BlockType.CHART_CAPTION,
    BlockType.CODE_CAPTION,
}
_ANNOTATION_FOOTNOTE_TYPES = {
    BlockType.IMAGE_FOOTNOTE,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CHART_FOOTNOTE,
    BlockType.CODE_FOOTNOTE,
}


class _DocxRenderer:
    """持有单次 DOCX 渲染所需的 document、素材解析器和书签状态。"""

    def __init__(
        self,
        middle_json: MiddleJson,
        *,
        mode: RenderMode,
        asset_resolver: AssetResolver | None,
    ) -> None:
        """初始化 renderer，并预注册标题与默认可见页面脚注 anchor。"""
        self.middle_json = middle_json
        self.mode = mode
        self.asset_resolver = asset_resolver
        self.document = Document()
        configure_document(self.document)
        self.bookmarks = BookmarkRegistry(_iter_document_anchors(middle_json))
        self.usable_width_emu = usable_width_emu(self.document)
        self.usable_width_twips = usable_width_twips(self.document)

    def render(self) -> bytes:
        """执行逐页 visitor，并把 python-docx document 序列化为 bytes。"""
        planned_pages = build_render_plan(self.middle_json, self.mode)
        for page_position, planned_blocks in enumerate(planned_pages):
            if self.mode is RenderMode.FULL and page_position > 0:
                self.document.add_page_break()
            for planned in planned_blocks:
                if planned.removed:
                    continue
                if self.mode is RenderMode.DEFAULT and planned.block.type in PAGE_AUXILIARY_BLOCK_TYPES:
                    continue
                self._render_planned_block(planned)

        output = BytesIO()
        self.document.save(output)
        return output.getvalue()

    def _render_planned_block(self, planned: PlannedBlock) -> None:
        """按严格 PageBlock 具体类型分派 Word 写入逻辑。"""
        block = planned.block
        context = self._context(planned.page_idx, block)
        if isinstance(block, (TextBlock, RefTextBlock)):
            paragraph = self.document.add_paragraph(style=BODY_STYLE)
            append_joined_inline_contents(
                paragraph,
                planned.text_contents or [block.content],
                context=context,
            )
            return
        if isinstance(block, (DocTitleBlock, ParagraphTitleBlock)):
            self._render_title(block, context)
            return
        if isinstance(block, PageFootnoteBlock):
            paragraph = self.document.add_paragraph(style=FOOTNOTE_STYLE)
            append_inline_content(paragraph, block.content, context=context)
            self.bookmarks.attach(paragraph, block.anchor)
            return
        if isinstance(block, PageAuxTextBlock):
            paragraph = self.document.add_paragraph(style=AUXILIARY_STYLE)
            append_inline_content(paragraph, block.content, context=context)
            return
        if isinstance(block, EquationBlock):
            self._render_equation(block, context)
            return
        if isinstance(block, ListBlock):
            self._render_list(block, context, depth=0)
            return
        if isinstance(block, IndexBlock):
            self._render_index(block, context, depth=0)
            return
        if isinstance(block, ImageBlock):
            self._render_image_block(block, context)
            return
        if isinstance(block, TableBlock):
            self._render_table_block(block, context)
            return
        if isinstance(block, ChartBlock):
            self._render_chart_block(block, context)
            return
        if isinstance(block, CodeBlock):
            self._render_code_block(block, context)
            return
        raise TypeError(f"Unsupported PageBlock type: {type(block).__name__}")

    def _context(self, page_idx: int, block: BlockBase) -> InlineRenderContext:
        """为当前 block 构造行内渲染上下文。"""
        return InlineRenderContext(
            bookmarks=self.bookmarks,
            page_idx=page_idx,
            block_index=block.index,
            block_type=block.type,
        )

    def _render_title(
        self,
        block: DocTitleBlock | ParagraphTitleBlock,
        context: InlineRenderContext,
    ) -> None:
        """用 Heading 1–6 写严格标题，并在段落内容外添加 bookmark。"""
        level = min(max(block.level, 1), 9)
        paragraph = self.document.add_paragraph(style=f"Heading {level}")
        append_inline_content(paragraph, block.content, context=context)
        self.bookmarks.attach(paragraph, block.anchor)

    def _render_equation(self, block: EquationBlock, context: InlineRenderContext) -> None:
        """优先写可编辑 OMML，失败时按图片、可见 LaTeX 的顺序回退。"""
        content = block.content.strip()
        if content:
            formula, tag = split_formula_tag(content)
            try:
                math_element = latex_to_omml(formula, display=True)
                self._append_display_formula(math_element, tag, context=context)
                return
            except DocxFormulaError as exc:
                logger.warning("DOCX display formula fallback: {} ({})", exc, context.location())

        if block.image_base64 is not None or block.image_path is not None:
            try:
                self._append_block_image(block, context=context, alt_text="formula")
                return
            except DocxRenderError:
                if not content:
                    raise
                logger.warning("DOCX formula image fallback failed; preserving LaTeX ({})", context.location())

        if content:
            paragraph = self.document.add_paragraph(style=FORMULA_FALLBACK_STYLE)
            paragraph.add_run(sanitize_xml_text(content, context=context))
            return
        raise self._render_error("Equation does not contain usable LaTeX or image", context)

    def _append_display_formula(
        self,
        math_element: etree._Element,
        tag: str | None,
        *,
        context: InlineRenderContext,
    ) -> None:
        """把块公式居中写入；带 tag 时使用中心与右对齐 tab。"""
        if tag is None:
            properties = OxmlElement("m:oMathParaPr")
            justification = OxmlElement("m:jc")
            justification.set(qn("m:val"), "center")
            properties.append(justification)
            math_element.insert(0, properties)
            paragraph = self.document.add_paragraph(style=BODY_STYLE)
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(5)
            paragraph._p.append(math_element)
            return

        equation = math_element.find(qn("m:oMath"))
        if equation is None:
            raise ValueError("Display OMML does not contain m:oMath")
        math_element.remove(equation)
        paragraph = self.document.add_paragraph(style=BODY_STYLE)
        paragraph.paragraph_format.space_before = Pt(4)
        paragraph.paragraph_format.space_after = Pt(5)
        paragraph.paragraph_format.tab_stops.add_tab_stop(
            Twips(self.usable_width_twips // 2),
            WD_TAB_ALIGNMENT.CENTER,
        )
        paragraph.paragraph_format.tab_stops.add_tab_stop(
            Twips(self.usable_width_twips),
            WD_TAB_ALIGNMENT.RIGHT,
        )
        paragraph.add_run().add_tab()
        paragraph._p.append(equation)
        paragraph.add_run().add_tab()
        rendered_tag = tag if tag.startswith("(") and tag.endswith(")") else f"({tag})"
        paragraph.add_run(sanitize_xml_text(rendered_tag, context=context))

    def _render_list(
        self,
        block: ListBlock,
        context: InlineRenderContext,
        *,
        depth: int,
    ) -> None:
        """保留叶子原 marker，并仅用缩进表达递归列表层级。"""
        for child in block.content:
            if isinstance(child, ListBlock):
                self._render_list(child, context, depth=depth + 1)
                continue
            paragraph = self.document.add_paragraph(style=BODY_STYLE)
            item = parse_list_item_marker(child.content)
            if item.marker is None:
                paragraph.paragraph_format.left_indent = Mm(depth * 6)
            else:
                paragraph.paragraph_format.left_indent = Mm((depth + 1) * 6)
                paragraph.paragraph_format.first_line_indent = Mm(-6)
            append_inline_content(paragraph, child.content, context=context)

    def _render_index(
        self,
        block: IndexBlock,
        context: InlineRenderContext,
        *,
        depth: int,
    ) -> None:
        """递归输出目录项，并把标题叶子链接到预注册 bookmark。"""
        for child in block.content:
            if isinstance(child, IndexBlock):
                self._render_index(child, context, depth=depth + 1)
                continue
            content = strip_index_page_tail(child.content)
            nodes = parse_inline_content(content)
            if not nodes:
                continue
            paragraph = self.document.add_paragraph(style=BODY_STYLE)
            paragraph.paragraph_format.left_indent = Mm((depth + 1) * 6)
            paragraph.paragraph_format.first_line_indent = Mm(-4)
            paragraph.add_run("• ")
            anchor = child.anchor if isinstance(child, TitleBlockBase) else None
            append_internal_link(paragraph, nodes, anchor=anchor, context=context)

    def _render_image_block(self, block: ImageBlock, context: InlineRenderContext) -> None:
        """按原始子块顺序写图片主体、caption 与 footnote。"""
        for child in block.content:
            if isinstance(child, ImageBodyBlock):
                alt_text = _plain_html_text(child.content) or block.sub_type or "image"
                self._append_block_image(child, context=context, alt_text=alt_text)
            elif isinstance(child, ImageAnnotationBlock):
                self._render_annotation(child, context)
            else:
                raise TypeError(f"Unsupported image child: {type(child).__name__}")

    def _render_table_block(self, block: TableBlock, context: InlineRenderContext) -> None:
        """按原始子块顺序写原生 HTML table 或预格式空间投影文本。"""
        for child in block.content:
            if isinstance(child, TableBodyBlock):
                if _HTML_TABLE_RE.search(child.content):
                    try:
                        self._append_html_tables(child.content, context=context, depth=0)
                    except DocxRenderError as exc:
                        logger.warning("DOCX HTML table fallback: {}", exc)
                        if child.image_base64 is None and child.image_path is None:
                            raise self._render_error(
                                "HTML table cannot be materialized and has no image fallback",
                                context,
                            ) from exc
                        self._append_block_image(child, context=context, alt_text="table")
                else:
                    if child.content and not child.content.isspace():
                        paragraph = self.document.add_paragraph(style=SPATIAL_TABLE_STYLE)
                        paragraph.add_run(sanitize_xml_text(child.content, context=context))
                    elif child.image_base64 is not None or child.image_path is not None:
                        self._append_block_image(child, context=context, alt_text="table")
                    else:
                        raise self._render_error(
                            "Spatial table does not contain text content or image",
                            context,
                        )
            elif isinstance(child, TableAnnotationBlock):
                self._render_annotation(child, context)
            else:
                raise TypeError(f"Unsupported table child: {type(child).__name__}")

    def _render_chart_block(self, block: ChartBlock, context: InlineRenderContext) -> None:
        """先写图表图片，再把 HTML 结构化数据追加为原生表格。"""
        for child in block.content:
            if isinstance(child, ChartBodyBlock):
                has_image = child.image_base64 is not None or child.image_path is not None
                if has_image:
                    self._append_block_image(
                        child,
                        context=context,
                        alt_text=_plain_html_text(child.content) or block.sub_type or "chart",
                    )
                if _HTML_TABLE_RE.search(child.content):
                    try:
                        self._append_html_tables(child.content, context=context, depth=0)
                    except DocxRenderError as exc:
                        if not has_image:
                            raise
                        logger.warning("DOCX chart data table omitted after image fallback: {}", exc)
                elif child.content.strip() and not has_image:
                    paragraph = self.document.add_paragraph(style=BODY_STYLE)
                    append_inline_content(paragraph, child.content, context=context)
                elif not has_image and not child.content.strip():
                    raise self._render_error("Chart does not contain image or structured content", context)
            elif isinstance(child, ChartAnnotationBlock):
                self._render_annotation(child, context)
            else:
                raise TypeError(f"Unsupported chart child: {type(child).__name__}")

    def _render_code_block(self, block: CodeBlock, context: InlineRenderContext) -> None:
        """按原始子块顺序写代码或算法主体及说明。"""
        for child in block.content:
            if isinstance(child, CodeBodyBlock):
                paragraph = self.document.add_paragraph(style=CODE_STYLE)
                if block.sub_type == BlockType.CODE:
                    paragraph.add_run(sanitize_xml_text(child.content, context=context))
                elif block.sub_type == RAW_ALGORITHM:
                    append_inline_content(paragraph, child.content, context=context)
                else:
                    raise ValueError(f"Unsupported code subtype: {block.sub_type}")
            elif isinstance(child, CodeAnnotationBlock):
                self._render_annotation(child, context)
            else:
                raise TypeError(f"Unsupported code child: {type(child).__name__}")

    def _render_annotation(
        self,
        block: ImageAnnotationBlock | TableAnnotationBlock | ChartAnnotationBlock | CodeAnnotationBlock,
        context: InlineRenderContext,
    ) -> None:
        """根据说明类型选择 Caption 或 Footnote 样式并写入行内内容。"""
        if block.type in _ANNOTATION_CAPTION_TYPES:
            style = CAPTION_STYLE
        elif block.type in _ANNOTATION_FOOTNOTE_TYPES:
            style = FOOTNOTE_STYLE
        else:
            raise TypeError(f"Unsupported annotation type: {block.type}")
        paragraph = self.document.add_paragraph(style=style)
        append_inline_content(paragraph, block.content, context=context)

    def _append_block_image(
        self,
        block: ImagePayloadBlock,
        *,
        context: InlineRenderContext,
        alt_text: str,
    ) -> None:
        """安全加载 block 图片，并按 bbox/自然尺寸限制到可用页面范围。"""
        if block.image_base64 is None and block.image_path is None and block.image_url:
            paragraph = self.document.add_paragraph(style=BODY_STYLE)
            append_inline_nodes(
                paragraph,
                [InlineLink([InlineText(alt_text or "remote image")], block.image_url)],
                context=context,
            )
            return
        try:
            prepared = prepare_block_image(block, self.asset_resolver)
        except DocxAssetError as exc:
            raise self._render_error(str(exc), context) from exc
        self._append_prepared_image(
            prepared,
            bbox=block.bbox,
            alt_text=alt_text,
            max_width_emu=self.usable_width_emu,
            context=context,
        )

    def _append_prepared_image(
        self,
        prepared: PreparedImage,
        *,
        bbox: BBox | None,
        alt_text: str,
        max_width_emu: int,
        context: InlineRenderContext,
        paragraph: Paragraph | None = None,
    ) -> None:
        """把准备好的图片按比例插入居中段落并写入 alt description。"""
        target_paragraph = paragraph or self.document.add_paragraph(style=BODY_STYLE)
        target_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        natural_width = round(prepared.width_px / 96 * 914400)
        desired_width = natural_width
        if bbox is not None:
            desired_width = round(max_width_emu * max(0.0, min(1.0, bbox[2] - bbox[0])))
        minimum_width = min(int(Mm(5)), max_width_emu)
        desired_width = max(minimum_width, min(desired_width, max_width_emu))

        usable_height = int(Mm(257))
        rendered_height = round(desired_width * prepared.height_px / max(prepared.width_px, 1))
        if rendered_height > usable_height:
            desired_width = round(usable_height * prepared.width_px / max(prepared.height_px, 1))

        shape = target_paragraph.add_run().add_picture(
            BytesIO(prepared.data),
            width=Emu(max(1, desired_width)),
        )
        if prepared.svg_data is not None:
            self._attach_native_svg(shape._inline, target_paragraph.part, prepared.svg_data)
        description = sanitize_xml_text(
            re.sub(r"\s+", " ", alt_text).strip()[:2048],
            context=context,
        )
        if description:
            shape._inline.docPr.set("descr", description)

    @staticmethod
    def _attach_native_svg(inline: etree._Element, target_part: Part, svg_data: bytes) -> None:
        """为 fallback PNG 的 a:blip 添加 Office 2016 原生 SVG relationship。"""
        package = target_part.package
        svg_part = Part(
            package.next_partname("/word/media/image%d.svg"),
            "image/svg+xml",
            svg_data,
            package,
        )
        relationship_id = target_part.relate_to(svg_part, RT.IMAGE)
        blip = inline.find(f".//{qn('a:blip')}")
        if blip is None:
            raise ValueError("DOCX picture does not contain an a:blip element")
        extension_list = blip.find(qn("a:extLst"))
        if extension_list is None:
            extension_list = OxmlElement("a:extLst")
            blip.append(extension_list)
        extension = OxmlElement("a:ext")
        extension.set("uri", _SVG_BLIP_EXTENSION_URI)
        svg_blip = etree.SubElement(
            extension,
            f"{{{_SVG_BLIP_NAMESPACE}}}svgBlip",
            nsmap={"asvg": _SVG_BLIP_NAMESPACE},
        )
        svg_blip.set(qn("r:embed"), relationship_id)
        extension_list.append(extension)

    def _append_html_tables(
        self,
        markup: str,
        *,
        context: InlineRenderContext,
        depth: int,
        container: DocumentType | _Cell | None = None,
    ) -> None:
        """物化 HTML 表格，并通过共享行内/素材逻辑填充 origin cell。"""
        if depth >= 4:
            raise self._render_error("Nested HTML table depth exceeds 4", context)
        target_container = container or self.document
        target_xml = target_container._element.body if isinstance(target_container, DocumentType) else target_container._tc
        existing_child_count = len(target_xml)
        insertion_index = existing_child_count - 1 if isinstance(target_container, DocumentType) else existing_child_count
        existing_relationship_ids = set(self.document.part.rels)

        def fill_cell(cell: _Cell, source: Tag, write_nested: NestedTableWriter) -> None:
            """填充一个 origin cell，并在遇到 nested table 时调用递归 writer。"""
            self._fill_html_cell(
                cell,
                source,
                write_nested=write_nested,
                context=context,
            )

        try:
            materialize_docx_tables(
                target_container,
                markup,
                width_twips=self.usable_width_twips,
                fill_cell=fill_cell,
            )
        except DocxRenderError:
            self._rollback_table_materialization(
                target_xml,
                existing_child_count=existing_child_count,
                insertion_index=insertion_index,
                existing_relationship_ids=existing_relationship_ids,
            )
            raise
        except DocxTableError as exc:
            self._rollback_table_materialization(
                target_xml,
                existing_child_count=existing_child_count,
                insertion_index=insertion_index,
                existing_relationship_ids=existing_relationship_ids,
            )
            raise self._render_error(str(exc), context) from exc

    def _rollback_table_materialization(
        self,
        target_xml: etree._Element,
        *,
        existing_child_count: int,
        insertion_index: int,
        existing_relationship_ids: set[str],
    ) -> None:
        """回滚本轮表格新增的 XML 根节点和 relationship，避免残留半成品。"""
        while len(target_xml) > existing_child_count:
            target_xml.remove(target_xml[insertion_index])
        for relationship_id in tuple(self.document.part.rels):
            if relationship_id not in existing_relationship_ids:
                self.document.part.drop_rel(relationship_id)

    def _fill_html_cell(
        self,
        cell: _Cell,
        source: Tag,
        *,
        write_nested: NestedTableWriter,
        context: InlineRenderContext,
    ) -> None:
        """按源顺序填充单元格文本、图片和任意包装层中的嵌套表格。"""
        cell.text = ""
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        _set_cell_margins(cell, top=100, start=120, bottom=100, end=120)
        max_image_width_emu = _cell_content_width_emu(cell, horizontal_margin_twips=240)
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0

        for child in source.children:
            if not isinstance(child, (NavigableString, Tag)):
                continue
            if isinstance(child, Tag) and child.name.lower() in {"p", "div"} and _paragraph_has_content(paragraph):
                paragraph = cell.add_paragraph()
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
            styles = ("bold",) if source.name == "th" else ()
            paragraph, _ = self._append_html_node(
                paragraph,
                child,
                cell=cell,
                write_nested=write_nested,
                context=context,
                inherited_styles=styles,
                max_image_width_emu=max_image_width_emu,
            )

    def _append_html_node(
        self,
        paragraph: Paragraph,
        node: NavigableString | Tag,
        *,
        cell: _Cell,
        write_nested: NestedTableWriter,
        context: InlineRenderContext,
        inherited_styles: tuple[str, ...],
        max_image_width_emu: int,
    ) -> tuple[Paragraph, bool]:
        """按 HTML 源顺序写入单元格行内文本、样式、公式和图片。"""
        if isinstance(node, NavigableString):
            text = str(node)
            if not text:
                return paragraph, False
            nodes: list[InlineNode] = [InlineText(text)]
            if inherited_styles:
                nodes = [InlineStyled(nodes, inherited_styles)]
            append_inline_nodes(paragraph, nodes, context=context)
            return paragraph, bool(text.strip())

        name = node.name.lower()
        if name == "table":
            write_nested(node)
            trailing_paragraph = cell.paragraphs[-1]
            trailing_paragraph.paragraph_format.space_after = Pt(0)
            trailing_paragraph.paragraph_format.line_spacing = 1.0
            return trailing_paragraph, True
        if name == "img":
            self._append_html_image(
                paragraph,
                str(node.get("src", "")),
                context=context,
                alt_text=str(node.get("alt", "")),
                max_width_emu=max_image_width_emu,
            )
            return paragraph, True
        if name == "br":
            paragraph.add_run().add_break()
            return paragraph, True
        if name == "eq":
            latex = node.get_text().strip()
            nodes = [InlineEquation(latex)] if latex else []
            if inherited_styles and nodes:
                nodes = [InlineStyled(nodes, inherited_styles)]
            append_inline_nodes(paragraph, nodes, context=context)
            return paragraph, bool(nodes)
        if name == "a":
            children = _html_inline_nodes(node)
            if inherited_styles and children:
                children = [InlineStyled(children, inherited_styles)]
            append_inline_nodes(
                paragraph,
                [InlineLink(children, str(node.get("href", "")).strip())],
                context=context,
            )
            return paragraph, bool(children)
        if name in {"ul", "ol"}:
            start_value = node.get("start", "1")
            try:
                ordered_number = int(str(start_value))
            except ValueError:
                ordered_number = 1
            rendered = False
            for item in node.find_all("li", recursive=False):
                if _paragraph_has_content(paragraph) or rendered:
                    paragraph.add_run().add_break()
                marker = f"{ordered_number}. " if name == "ol" else "- "
                paragraph.add_run(marker)
                if name == "ol":
                    ordered_number += 1
                for child in item.children:
                    if isinstance(child, (NavigableString, Tag)):
                        paragraph, child_rendered = self._append_html_node(
                            paragraph,
                            child,
                            cell=cell,
                            write_nested=write_nested,
                            context=context,
                            inherited_styles=inherited_styles,
                            max_image_width_emu=max_image_width_emu,
                        )
                        rendered = child_rendered or rendered
            return paragraph, rendered
        if name == "li":
            rendered = False
            for child in node.children:
                if isinstance(child, (NavigableString, Tag)):
                    paragraph, child_rendered = self._append_html_node(
                        paragraph,
                        child,
                        cell=cell,
                        write_nested=write_nested,
                        context=context,
                        inherited_styles=inherited_styles,
                        max_image_width_emu=max_image_width_emu,
                    )
                    rendered = child_rendered or rendered
            return paragraph, rendered
        if name == "text":
            nodes = parse_inline_content(str(node))
            if inherited_styles and nodes:
                nodes = [InlineStyled(nodes, inherited_styles)]
            append_inline_nodes(paragraph, nodes, context=context)
            return paragraph, bool(nodes)

        style = {
            "strong": "bold",
            "b": "bold",
            "em": "italic",
            "i": "italic",
            "u": "underline",
            "s": "strikethrough",
            "sup": "superscript",
            "sub": "subscript",
        }.get(name)
        styles = tuple(dict.fromkeys((*inherited_styles, style))) if style else inherited_styles
        rendered = False
        for child in node.children:
            if isinstance(child, (NavigableString, Tag)):
                paragraph, child_rendered = self._append_html_node(
                    paragraph,
                    child,
                    cell=cell,
                    write_nested=write_nested,
                    context=context,
                    inherited_styles=styles,
                    max_image_width_emu=max_image_width_emu,
                )
                rendered = child_rendered or rendered
        return paragraph, rendered

    def _append_html_image(
        self,
        paragraph: Paragraph,
        source: str,
        *,
        context: InlineRenderContext,
        alt_text: str,
        max_width_emu: int,
    ) -> None:
        """安全加载表格单元格 img，并限制到紧凑的单元格宽度。"""
        try:
            prepared = prepare_html_image(source, self.asset_resolver)
        except DocxAssetError as exc:
            raise self._render_error(str(exc), context) from exc
        self._append_prepared_image(
            prepared,
            bbox=None,
            alt_text=alt_text or "table image",
            max_width_emu=min(self.usable_width_emu, max_width_emu),
            context=context,
            paragraph=paragraph,
        )

    def _render_error(self, message: str, context: InlineRenderContext) -> DocxRenderError:
        """用当前 context 构造带 page/block 定位的公共异常。"""
        return DocxRenderError(
            message,
            page_idx=context.page_idx,
            block_index=context.block_index,
            block_type=context.block_type,
        )


def render_docx(
    middle_json: MiddleJson,
    *,
    mode: RenderMode = RenderMode.DEFAULT,
    asset_resolver: AssetResolver | None = None,
) -> bytes:
    """把严格 MiddleJson 无副作用地渲染为完整 DOCX bytes。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("render_docx expects a MiddleJson instance")
    if not isinstance(mode, RenderMode):
        raise TypeError("mode must be a RenderMode value")
    return _DocxRenderer(
        middle_json,
        mode=mode,
        asset_resolver=asset_resolver,
    ).render()


def _iter_document_anchors(middle_json: MiddleJson) -> Iterable[str]:
    """遍历标题和默认可见页面脚注实际会写入的 bookmark anchor。"""
    for page in middle_json.pages:
        for block in page.blocks:
            if isinstance(block, TitleBlockBase) and block.anchor:
                yield block.anchor
            elif isinstance(block, PageFootnoteBlock) and block.anchor:
                yield block.anchor


def _plain_html_text(content: str) -> str:
    """把图片或图表 body 内容收敛为适合 alt description 的纯文本。"""
    if not content.strip():
        return ""
    return BeautifulSoup(content, "html.parser").get_text(" ", strip=True)


def _html_inline_nodes(node: Tag) -> list[InlineNode]:
    """把表格单元格中的安全 HTML 行内标签转换为共享 InlineNode。"""
    nodes: list[InlineNode] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            if str(child):
                nodes.append(InlineText(str(child)))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        children = _html_inline_nodes(child)
        if name == "eq":
            latex = child.get_text().strip()
            if latex:
                nodes.append(InlineEquation(latex))
        elif name in {"strong", "b"}:
            nodes.append(InlineStyled(children, ("bold",)))
        elif name in {"em", "i"}:
            nodes.append(InlineStyled(children, ("italic",)))
        elif name == "u":
            nodes.append(InlineStyled(children, ("underline",)))
        elif name == "s":
            nodes.append(InlineStyled(children, ("strikethrough",)))
        elif name == "sup":
            nodes.append(InlineStyled(children, ("superscript",)))
        elif name == "sub":
            nodes.append(InlineStyled(children, ("subscript",)))
        elif name == "a":
            nodes.append(InlineLink(children, str(child.get("href", "")).strip()))
        elif name == "br":
            nodes.append(InlineText("\n"))
        elif name == "text":
            nodes.extend(parse_inline_content(str(child)))
        elif name not in {"img", "table"}:
            nodes.extend(children)
    return nodes


def _set_cell_margins(
    cell: _Cell,
    *,
    top: int,
    start: int,
    bottom: int,
    end: int,
) -> None:
    """用 DXA 为 Word 单元格写入四边内边距。"""
    cell_properties = cell._tc.get_or_add_tcPr()
    margins = cell_properties.find(qn("w:tcMar"))
    if margins is None:
        margins = OxmlElement("w:tcMar")
        cell_properties.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        element = margins.find(qn(f"w:{side}"))
        if element is None:
            element = OxmlElement(f"w:{side}")
            margins.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def _cell_content_width_emu(cell: _Cell, *, horizontal_margin_twips: int) -> int:
    """从当前物理或合并单元格 tcW 计算扣除左右内边距后的可用宽度。"""
    cell_width = cell._tc.get_or_add_tcPr().get_or_add_tcW().w
    width_twips = int(cell_width) if cell_width is not None else 1
    return int(Twips(max(1, width_twips - horizontal_margin_twips)))


def _paragraph_has_content(paragraph: Paragraph) -> bool:
    """判断段落是否包含 pPr 之外的 run、公式、链接或 drawing。"""
    return any(child.tag != qn("w:pPr") for child in paragraph._p)


__all__ = ["render_docx"]
