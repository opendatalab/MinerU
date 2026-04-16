# Copyright (c) Opendatalab. All rights reserved.
import base64
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Final, BinaryIO, Optional

from lxml import etree
from pptx import Presentation, presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.oxml.text import CT_TextLineBreak
from loguru import logger
from PIL import Image, UnidentifiedImageError

from mineru.utils.enum_class import BlockType
from mineru.backend.utils.office_image import (
    is_vector_image,
    serialize_vector_image_with_placeholder,
)
from mineru.model.docx.tools.math.omml import oMath2Latex
from mineru.backend.utils.office_chart import html_table_from_excel_bytes
from mineru.model.pptx.xycut_pp_sorter import sort_entries
from mineru.utils.pdf_reader import image_to_b64str

IGNORED_NOTES_PLACEHOLDER_TYPES: Final = {
    PP_PLACEHOLDER.SLIDE_IMAGE,
    PP_PLACEHOLDER.SLIDE_NUMBER,
    PP_PLACEHOLDER.DATE,
    PP_PLACEHOLDER.FOOTER,
}
MIN_PICTURE_DIMENSION_RATIO: Final = 0.1
MIN_PICTURE_AREA_RATIO: Final = 0.01
BACKGROUND_PICTURE_TEXT_COVERAGE_RATIO: Final = 0.1
# PPTX_XYCUT_BETA: Final = 0.7
PPTX_XYCUT_BETA: Final = 2.0
PPTX_XYCUT_DENSITY_THRESHOLD: Final = 0.9
DRAWINGML_NS: Final = "http://schemas.openxmlformats.org/drawingml/2006/main"
RELATIONSHIP_NS: Final = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
SVG_BLIP_NS: Final = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
A14_DRAWING_NS: Final = "http://schemas.microsoft.com/office/drawing/2010/main"
OMML_NS: Final = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_EFFECTIVE_FONT_SIZE_KEY: Final = "_effective_font_size_pt"
_EFFECTIVE_ALL_BOLD_KEY: Final = "_effective_all_bold"


@dataclass(frozen=True)
class _SlideTransform:
    scale_x: float = 1.0
    scale_y: float = 1.0
    translate_x: float = 0.0
    translate_y: float = 0.0

    def apply_bbox(
        self,
        bbox: Optional[tuple[float, float, float, float]],
    ) -> Optional[tuple[float, float, float, float]]:
        if bbox is None:
            return None

        left = self.scale_x * bbox[0] + self.translate_x
        top = self.scale_y * bbox[1] + self.translate_y
        right = self.scale_x * bbox[2] + self.translate_x
        bottom = self.scale_y * bbox[3] + self.translate_y
        return (left, top, right, bottom)

    def compose(self, inner: "_SlideTransform") -> "_SlideTransform":
        return _SlideTransform(
            scale_x=self.scale_x * inner.scale_x,
            scale_y=self.scale_y * inner.scale_y,
            translate_x=self.scale_x * inner.translate_x + self.translate_x,
            translate_y=self.scale_y * inner.translate_y + self.translate_y,
        )


@dataclass(frozen=True)
class _FlattenedShape:
    shape: Any
    bbox: Optional[tuple[float, float, float, float]]


class PptxConverter:

    def __init__(self):
        self.namespaces = {
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
            "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        }
        self.file_stream = None
        self.pptx_obj = None
        self.pages = []
        self.cur_page = []
        self.list_block_stack: list = []  # 列表块堆栈
        self.equation_bookends: str = "<eq>{EQ}</eq>"  # 公式标记格式

    def convert(
        self,
        file_stream: BinaryIO,
    ):
        self.pages = []
        self.cur_page = []
        self.list_block_stack = []
        self.file_stream = file_stream
        self.pptx_obj = Presentation(self.file_stream)
        self.pages.append(self.cur_page)
        if self.pptx_obj:
            self._walk_linear(self.pptx_obj)
        if self.pages and self.pages[-1] == []:
            self.pages.pop()

    def _walk_linear(self, pptx_obj: presentation.Presentation):
        slide_width = int(pptx_obj.slide_width)
        slide_height = int(pptx_obj.slide_height)

        # 遍历每一张幻灯片
        for _, slide in enumerate(pptx_obj.slides):
            linear_shapes = self._flatten_slide_shapes(slide.shapes)
            sortable_shape_entries = []
            tail_blocks = []

            # 遍历幻灯片中的每一个形状
            for shape_index, shape_entry in enumerate(linear_shapes):
                shape_blocks = self._collect_shape_blocks(
                    shape_entry,
                    linear_shapes,
                    shape_index,
                    slide_width,
                    slide_height,
                )
                if not shape_blocks:
                    continue

                if shape_entry.bbox is None:
                    tail_blocks.extend(shape_blocks)
                    continue

                sortable_shape_entries.append(
                    {
                        "bbox": shape_entry.bbox,
                        "blocks": shape_blocks,
                    }
                )

            sorted_shape_entries = sort_entries(
                sortable_shape_entries,
                beta=PPTX_XYCUT_BETA,
                density_threshold=PPTX_XYCUT_DENSITY_THRESHOLD,
            )
            for entry in sorted_shape_entries:
                self.cur_page.extend(entry["blocks"])
            self.cur_page.extend(tail_blocks)

            self._handle_slide_notes(slide)
            self._promote_slide_text_blocks_to_titles(self.cur_page)
            self._cleanup_slide_text_block_metadata(self.cur_page)
            self.cur_page = []
            self.pages.append(self.cur_page)

    def _flatten_slide_shapes(
        self,
        shapes,
        slide_transform: Optional[_SlideTransform] = None,
    ) -> list[_FlattenedShape]:
        if slide_transform is None:
            slide_transform = _SlideTransform()

        linear_shapes: list[_FlattenedShape] = []
        for shape in shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                group_transform = self._group_shape_transform(shape)
                linear_shapes.extend(
                    self._flatten_slide_shapes(
                        shape.shapes,
                        slide_transform.compose(group_transform),
                    )
                )
            else:
                linear_shapes.append(
                    _FlattenedShape(
                        shape=shape,
                        bbox=slide_transform.apply_bbox(self._shape_bbox(shape)),
                    )
                )
        return linear_shapes

    def _collect_shape_blocks(
        self,
        shape_entry: _FlattenedShape,
        linear_shapes: list[_FlattenedShape],
        shape_index: int,
        slide_width: int,
        slide_height: int,
    ) -> list:
        shape = shape_entry.shape
        shape_blocks = []
        previous_page = self.cur_page
        previous_list_block_stack = self.list_block_stack
        self.cur_page = shape_blocks
        self.list_block_stack = []

        try:
            if shape.has_table:
                self._handle_tables(shape)

            if getattr(shape, "has_chart", False):
                self._handle_chart(shape)

            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                later_shapes = linear_shapes[shape_index + 1 :]
                if not self._should_skip_picture(
                    shape_entry,
                    later_shapes,
                    slide_width,
                    slide_height,
                ):
                    self._handle_pictures(shape)

            if not hasattr(shape, "text"):
                return shape_blocks
            if shape.text is None:
                return shape_blocks
            if len(shape.text.strip()) == 0:
                return shape_blocks
            if not shape.has_text_frame:
                logger.warning("Warning: shape has text but not text_frame")
                return shape_blocks

            self._handle_text_elements(shape)
            return shape_blocks
        finally:
            self.cur_page = previous_page
            self.list_block_stack = previous_list_block_stack

    @staticmethod
    def _shape_bbox(shape) -> Optional[tuple[float, float, float, float]]:
        try:
            left = float(shape.left)
            top = float(shape.top)
            width = float(shape.width)
            height = float(shape.height)
        except Exception:
            return None

        if width <= 0 or height <= 0:
            return None

        return (left, top, left + width, top + height)

    @staticmethod
    def _group_shape_transform(shape) -> _SlideTransform:
        group_properties = getattr(shape._element, "grpSpPr", None)
        xfrm = (
            getattr(group_properties, "xfrm", None)
            if group_properties is not None
            else None
        )
        if xfrm is None:
            return _SlideTransform()

        child_offset = getattr(xfrm, "chOff", None)
        child_extent = getattr(xfrm, "chExt", None)
        if child_offset is None or child_extent is None:
            return _SlideTransform()

        try:
            offset_x = float(xfrm.x)
            offset_y = float(xfrm.y)
            extent_x = float(xfrm.cx)
            extent_y = float(xfrm.cy)
            child_offset_x = float(child_offset.x)
            child_offset_y = float(child_offset.y)
            child_extent_x = float(child_extent.cx)
            child_extent_y = float(child_extent.cy)
        except Exception:
            return _SlideTransform()

        if (
            extent_x <= 0
            or extent_y <= 0
            or child_extent_x <= 0
            or child_extent_y <= 0
        ):
            return _SlideTransform()

        scale_x = extent_x / child_extent_x
        scale_y = extent_y / child_extent_y
        return _SlideTransform(
            scale_x=scale_x,
            scale_y=scale_y,
            translate_x=offset_x - child_offset_x * scale_x,
            translate_y=offset_y - child_offset_y * scale_y,
        )

    @staticmethod
    def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    @staticmethod
    def _bbox_intersection(
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> Optional[tuple[float, float, float, float]]:
        x0 = max(bbox1[0], bbox2[0])
        y0 = max(bbox1[1], bbox2[1])
        x1 = min(bbox1[2], bbox2[2])
        y1 = min(bbox1[3], bbox2[3])

        if x1 <= x0 or y1 <= y0:
            return None

        return (x0, y0, x1, y1)

    @classmethod
    def _rectangles_union_area(cls, bboxes: list[tuple[float, float, float, float]]) -> float:
        if not bboxes:
            return 0.0

        xs = sorted({bbox[0] for bbox in bboxes} | {bbox[2] for bbox in bboxes})
        total_area = 0.0

        for idx in range(len(xs) - 1):
            x_left = xs[idx]
            x_right = xs[idx + 1]
            if x_right <= x_left:
                continue

            y_intervals = []
            for bbox in bboxes:
                if bbox[0] < x_right and bbox[2] > x_left:
                    y_intervals.append((bbox[1], bbox[3]))

            if not y_intervals:
                continue

            y_intervals.sort()
            merged_height = 0.0
            current_y0, current_y1 = y_intervals[0]

            for y0, y1 in y_intervals[1:]:
                if y0 <= current_y1:
                    current_y1 = max(current_y1, y1)
                    continue

                merged_height += max(0.0, current_y1 - current_y0)
                current_y0, current_y1 = y0, y1

            merged_height += max(0.0, current_y1 - current_y0)
            total_area += (x_right - x_left) * merged_height

        return total_area

    @staticmethod
    def _is_nonempty_text_shape(shape) -> bool:
        if not getattr(shape, "has_text_frame", False):
            return False

        text = getattr(shape, "text", None)
        if text is None:
            return False

        return len(text.strip()) > 0

    def _is_small_picture(
        self,
        picture_bbox: Optional[tuple[float, float, float, float]],
        slide_width: int,
        slide_height: int,
    ) -> bool:
        if picture_bbox is None:
            return False

        picture_width = picture_bbox[2] - picture_bbox[0]
        picture_height = picture_bbox[3] - picture_bbox[1]

        if picture_width <= 0 or picture_height <= 0:
            return False

        slide_area = float(slide_width) * float(slide_height)
        if slide_area <= 0:
            return False

        if picture_width < MIN_PICTURE_DIMENSION_RATIO * float(slide_width):
            return True
        if picture_height < MIN_PICTURE_DIMENSION_RATIO * float(slide_height):
            return True

        picture_area_ratio = (picture_width * picture_height) / slide_area
        return picture_area_ratio < MIN_PICTURE_AREA_RATIO

    def _is_background_picture(
        self,
        picture_entry: _FlattenedShape,
        later_shapes: list[_FlattenedShape],
    ) -> bool:
        picture_bbox = picture_entry.bbox
        if picture_bbox is None:
            return False

        picture_area = self._bbox_area(picture_bbox)
        if picture_area <= 0:
            return False

        overlap_bboxes = []
        for later_shape in later_shapes:
            if not self._is_nonempty_text_shape(later_shape.shape):
                continue

            later_bbox = later_shape.bbox
            if later_bbox is None:
                continue

            overlap_bbox = self._bbox_intersection(picture_bbox, later_bbox)
            if overlap_bbox is not None:
                overlap_bboxes.append(overlap_bbox)

        if not overlap_bboxes:
            return False

        covered_area = self._rectangles_union_area(overlap_bboxes)
        return (
            covered_area / picture_area
            >= BACKGROUND_PICTURE_TEXT_COVERAGE_RATIO
        )

    def _should_skip_picture(
        self,
        picture_entry: _FlattenedShape,
        later_shapes: list[_FlattenedShape],
        slide_width: int,
        slide_height: int,
    ) -> bool:
        return self._is_small_picture(
            picture_entry.bbox,
            slide_width,
            slide_height,
        ) or self._is_background_picture(
            picture_entry,
            later_shapes,
        )

    def _handle_slide_notes(self, slide) -> None:
        if not slide.has_notes_slide:
            return

        try:
            notes_slide = slide.notes_slide
        except Exception as e:
            logger.warning(f"Warning: notes slide cannot be loaded: {e}")
            return

        def handle_notes_shape(shape) -> None:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                for grouped_shape in shape.shapes:
                    handle_notes_shape(grouped_shape)
                return

            if self._should_skip_notes_shape(shape):
                return

            for paragraph in shape.text_frame.paragraphs:
                note_text = self._normalize_text_block_content(
                    self._build_paragraph_rich_text(paragraph, shape)
                )
                if not note_text:
                    continue
                self.cur_page.append(
                    {
                        "type": BlockType.PAGE_FOOTNOTE,
                        "content": note_text,
                    }
                )

        for shape in notes_slide.shapes:
            handle_notes_shape(shape)

    @staticmethod
    def _should_skip_notes_shape(shape) -> bool:
        if not getattr(shape, "has_text_frame", False):
            return True

        text = getattr(shape, "text", None)
        if text is None or len(text.strip()) == 0:
            return True

        if not getattr(shape, "is_placeholder", False):
            return False

        try:
            return shape.placeholder_format.type in IGNORED_NOTES_PLACEHOLDER_TYPES
        except Exception:
            return False

    def _handle_tables(self, shape):
        """将PowerPoint表格转换为HTML格式。

        Args:
            shape: 包含表格的形状对象。
            parent_slide: 父幻灯片组。
            slide_ind: 当前幻灯片索引。
            doc: 文档对象(此实现中未使用)。
            slide_size: 幻灯片尺寸。

        Returns:
            str: 表格的HTML字符串，如果没有表格则返回None。
        """
        if not shape.has_table:
            return None

        table = shape.table
        table_xml = shape._element

        # 开始构建HTML表格
        html_parts = ['<table border="1">']

        # 跟踪已被合并单元格占用的位置
        # 格式: {(row, col): True}
        occupied_cells = {}

        for row_idx, row in enumerate(table.rows):
            html_parts.append("  <tr>")

            for col_idx, cell in enumerate(row.cells):
                # 跳过被合并占用的单元格
                if (row_idx, col_idx) in occupied_cells:
                    continue
                # 获取单元格XML以读取跨度信息
                cell_xml = table_xml.xpath(
                    f".//a:tbl/a:tr[{row_idx + 1}]/a:tc[{col_idx + 1}]"
                )

                if not cell_xml:
                    continue

                cell_xml = cell_xml[0]

                # 解析行跨度和列跨度
                row_span = cell_xml.get("rowSpan")
                col_span = cell_xml.get("gridSpan")

                row_span = int(row_span) if row_span else 1
                col_span = int(col_span) if col_span else 1

                # 标记被此单元格占用的位置
                for r in range(row_idx, row_idx + row_span):
                    for c in range(col_idx, col_idx + col_span):
                        if (r, c) != (row_idx, col_idx):
                            occupied_cells[(r, c)] = True

                # 确定标签类型：第一行使用<th>，其他使用<td>
                tag = "th" if row_idx == 0 else "td"

                # 构建属性字符串
                attrs = []
                if row_span > 1:
                    attrs.append(f'rowspan="{row_span}"')
                if col_span > 1:
                    attrs.append(f'colspan="{col_span}"')

                attr_str = " " + " ".join(attrs) if attrs else ""

                # 获取单元格文本内容
                cell_text = cell.text.strip() if cell.text else ""
                # 转义HTML特殊字符，防止XSS
                cell_text = (
                    cell_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )

                html_parts.append(f"    <{tag}{attr_str}>{cell_text}</{tag}>")

            html_parts.append("  </tr>")

        html_parts.append("</table>")

        self.cur_page.append(
            {
                "type": BlockType.TABLE,
                "content": "\n".join(html_parts),
            }
        )

        return None

    def _handle_chart(self, shape) -> None:
        try:
            chart_workbook = shape.chart.part.chart_workbook
            xlsx_part = chart_workbook.xlsx_part
            if xlsx_part is None:
                logger.warning("Warning: chart workbook part is missing")
                return

            chart_html = html_table_from_excel_bytes(xlsx_part.blob)
        except Exception as e:
            logger.warning(f"Warning: chart workbook cannot be loaded: {e}")
            return

        if not chart_html:
            return

        self.cur_page.append(
            {
                "type": BlockType.CHART,
                "content": chart_html,
            }
        )

    def _handle_pictures(self, shape):
        image_data = self._get_shape_image_data(shape)
        if image_data is None:
            return

        image_bytes, content_type = image_data

        if content_type == "image/svg+xml":
            image_block = {
                "type": BlockType.IMAGE,
                "content": self._bytes_to_data_uri(image_bytes, content_type),
            }
            self.cur_page.append(image_block)
            return

        # 使用PIL打开图像
        try:
            pil_image = Image.open(BytesIO(image_bytes))

            if is_vector_image(pil_image):
                img_base64 = serialize_vector_image_with_placeholder(pil_image)
            else:
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                img_base64 = image_to_b64str(pil_image)
            image_block = {
                "type": BlockType.IMAGE,
                "content": img_base64,
            }
            self.cur_page.append(image_block)

        except (UnidentifiedImageError, OSError) as e:
            logger.warning(f"Warning: image cannot be loaded by Pillow: {e}")
        return

    @staticmethod
    def _bytes_to_data_uri(image_bytes: bytes, content_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{encoded}"

    @staticmethod
    def _find_first_embedded_image_rid(shape) -> Optional[str]:
        svg_blips = shape._element.findall(f".//{{{SVG_BLIP_NS}}}svgBlip")
        for svg_blip in svg_blips:
            relationship_id = svg_blip.get(f"{{{RELATIONSHIP_NS}}}embed")
            if relationship_id:
                return relationship_id

        blips = shape._element.findall(f".//{{{DRAWINGML_NS}}}blip")
        for blip in blips:
            relationship_id = blip.get(f"{{{RELATIONSHIP_NS}}}embed")
            if relationship_id:
                return relationship_id

        return None

    def _get_shape_image_data(self, shape) -> Optional[tuple[bytes, Optional[str]]]:
        relationship_id = None
        if hasattr(shape, "_element"):
            relationship_id = self._find_first_embedded_image_rid(shape)

        if relationship_id:
            try:
                image_part = shape.part.related_part(relationship_id)
                image_bytes = image_part.blob
            except Exception as e:
                logger.warning(
                    f"Warning: embedded image relation {relationship_id} cannot be loaded: {e}"
                )
            else:
                return image_bytes, getattr(image_part, "content_type", None)

        try:
            image = shape.image
        except ValueError as e:
            logger.warning(f"Warning: shape image cannot be loaded: {e}")
            return None
        except AttributeError:
            return None

        return image.blob, None

    @staticmethod
    def _normalize_xml_toggle_attr(value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None

        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "t", "on"}:
            return True
        if normalized in {"0", "false", "f", "off", "none"}:
            return False
        return None

    @classmethod
    def _parse_toggle_attr_from_rpr(
        cls,
        rpr: Optional[etree._Element],
        attr_name: str,
    ) -> Optional[bool]:
        if rpr is None:
            return None
        return cls._normalize_xml_toggle_attr(rpr.get(attr_name))

    @classmethod
    def _parse_underline_from_rpr(
        cls,
        rpr: Optional[etree._Element],
    ) -> Optional[bool]:
        if rpr is None:
            return None

        underline = rpr.get("u")
        if underline is None:
            return None

        normalized = str(underline).strip().lower()
        if normalized in {"0", "false", "f", "off", "none"}:
            return False
        return True

    @classmethod
    def _parse_strikethrough_from_rpr(
        cls,
        rpr: Optional[etree._Element],
    ) -> Optional[bool]:
        if rpr is None:
            return None

        strike = rpr.get("strike")
        if strike is None:
            return None

        normalized = str(strike).strip().lower()
        if normalized in {"0", "false", "f", "off", "none", "nostrike"}:
            return False
        return True

    def _get_run_rpr(
        self,
        run,
    ) -> Optional[etree._Element]:
        if run is None:
            return None

        run_xml = getattr(run, "_r", None)
        if run_xml is None:
            return None

        try:
            return run_xml.find("a:rPr", namespaces=self.namespaces)
        except Exception:
            return None

    def _resolve_effective_run_bool(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
        parser,
    ) -> bool:
        for source in [self._get_run_rpr(run), *paragraph_font_sources]:
            resolved = parser(source)
            if resolved is not None:
                return resolved
        return False

    def _resolve_effective_run_italic(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> bool:
        return self._resolve_effective_run_bool(
            run,
            paragraph_font_sources,
            self._parse_italic_from_rpr,
        )

    def _resolve_effective_run_underline(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> bool:
        return self._resolve_effective_run_bool(
            run,
            paragraph_font_sources,
            self._parse_underline_from_rpr,
        )

    def _resolve_effective_run_strikethrough(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> bool:
        return self._resolve_effective_run_bool(
            run,
            paragraph_font_sources,
            self._parse_strikethrough_from_rpr,
        )

    def _get_style_str_from_run(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> Optional[str]:
        """从PPTX run对象提取可序列化的生效字体样式字符串。"""
        if run is None:
            return None

        styles = []
        if self._resolve_effective_run_bold(run, paragraph_font_sources):
            styles.append("bold")
        if self._resolve_effective_run_italic(run, paragraph_font_sources):
            styles.append("italic")
        if self._resolve_effective_run_underline(run, paragraph_font_sources):
            styles.append("underline")
        if self._resolve_effective_run_strikethrough(run, paragraph_font_sources):
            styles.append("strikethrough")

        return ",".join(styles) if styles else None

    @staticmethod
    def _format_text_with_hyperlink(
        text: str,
        hyperlink: Optional[str],
        style_str: Optional[str] = None,
    ) -> str:
        """按Office约定格式输出带样式/超链接的文本片段。"""
        if not text:
            return ""

        if hyperlink is None or str(hyperlink).strip() in ("", "."):
            if style_str:
                return f'<text style="{style_str}">{text}</text>'
            return text

        if style_str:
            text_tag = f'<text style="{style_str}">{text}</text>'
        else:
            text_tag = f"<text>{text}</text>"

        return f"<hyperlink>{text_tag}<url>{hyperlink}</url></hyperlink>"

    def _resolve_hyperlink_from_run(self, run, shape) -> Optional[str]:
        """解析 run 对应的超链接，优先公开 API，回退到 XML + rels。"""
        try:
            if hasattr(run, "hyperlink") and run.hyperlink is not None:
                address = run.hyperlink.address
                if address and str(address).strip():
                    return str(address).strip()
        except Exception:
            pass

        try:
            rPr = run._r.find("a:rPr", namespaces=self.namespaces)
            if rPr is None:
                return None

            hlink_click = rPr.find("a:hlinkClick", namespaces=self.namespaces)
            if hlink_click is None:
                return None

            rid = hlink_click.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            if not rid:
                return None

            rels = shape.part.rels
            if rid not in rels:
                return None

            rel = rels[rid]
            target_ref = getattr(rel, "target_ref", None)
            if target_ref and str(target_ref).strip():
                return str(target_ref).strip()

            target_part = getattr(rel, "target_part", None)
            if target_part is not None:
                partname = getattr(target_part, "partname", None)
                if partname and str(partname).strip():
                    return str(partname).strip()
        except Exception:
            return None

        return None

    def _build_paragraph_plain_text(self, paragraph) -> str:
        """构建段落纯文本（保留软换行为空格）。"""
        p = paragraph._element
        text_parts = []
        for node in p.content_children:
            if isinstance(node, CT_TextLineBreak):
                text_parts.append(" ")
                continue

            node_text = getattr(node, "text", None)
            if node_text is not None:
                text_parts.append(node_text)

        return "".join(text_parts)

    @staticmethod
    def _is_math_content_node(node) -> bool:
        tag = getattr(node, "tag", None)
        return tag in {
            f"{{{A14_DRAWING_NS}}}m",
            f"{{{OMML_NS}}}oMath",
            f"{{{OMML_NS}}}oMathPara",
        }

    @staticmethod
    def _strip_math_delimiters(math_text: str) -> str:
        stripped = math_text.strip()
        if (
            stripped.startswith("$$")
            and stripped.endswith("$$")
            and len(stripped) >= 4
        ):
            return stripped[2:-2].strip()
        if (
            stripped.startswith("$")
            and stripped.endswith("$")
            and len(stripped) >= 2
        ):
            return stripped[1:-1].strip()
        return stripped

    def _convert_math_node_to_latex(self, node) -> Optional[str]:
        omath = None
        if getattr(node, "tag", None) == f"{{{OMML_NS}}}oMath":
            omath = node
        else:
            omath = node.find(".//m:oMath", namespaces=self.namespaces)

        if omath is not None:
            try:
                latex = str(oMath2Latex(omath)).strip()
            except Exception as exc:
                logger.debug(f"Failed to convert PPTX OMML equation to LaTeX: {exc}")
            else:
                if latex:
                    return latex

        fallback_text = getattr(node, "text", None)
        if isinstance(fallback_text, str):
            latex = self._strip_math_delimiters(fallback_text)
            if latex:
                return latex

        return None

    def _build_paragraph_rich_text(self, paragraph, shape) -> str:
        """按 run 维度构建段落富文本，支持样式与超链接标签。"""
        paragraph_font_sources = self._get_paragraph_font_sources(shape, paragraph)
        run_map = {}
        for run in paragraph.runs:
            try:
                run_map[id(run._r)] = run
            except Exception:
                continue

        segments = []

        for node in paragraph._element.content_children:
            if isinstance(node, CT_TextLineBreak):
                segments.append(
                    {
                        "text": " ",
                        "style_str": None,
                        "hyperlink": None,
                    }
                )
                continue

            if self._is_math_content_node(node):
                latex = self._convert_math_node_to_latex(node)
                if latex:
                    segments.append(
                        {
                            "text": self.equation_bookends.format(EQ=latex),
                            "style_str": None,
                            "hyperlink": None,
                        }
                    )
                    continue

            node_text = getattr(node, "text", None)
            if node_text is None:
                continue
            if node_text == "":
                continue

            run = run_map.get(id(node))
            if run is None:
                segments.append(
                    {
                        "text": node_text,
                        "style_str": None,
                        "hyperlink": None,
                    }
                )
                continue

            segments.append(
                {
                    "text": node_text,
                    "style_str": self._get_style_str_from_run(
                        run,
                        paragraph_font_sources,
                    ),
                    "hyperlink": self._resolve_hyperlink_from_run(run, shape),
                }
            )

        segments = self._trim_rich_text_segments(segments)
        if not segments:
            return ""

        merged_segments = []
        for segment in segments:
            if (
                merged_segments
                and merged_segments[-1]["hyperlink"] is None
                and segment["hyperlink"] is None
                and merged_segments[-1]["style_str"] == segment["style_str"]
            ):
                merged_segments[-1]["text"] += segment["text"]
            else:
                merged_segments.append(segment)

        return "".join(
            self._format_text_with_hyperlink(
                segment["text"],
                segment["hyperlink"],
                segment["style_str"],
            )
            for segment in merged_segments
        )

    @staticmethod
    def _trim_rich_text_segments(segments: list[dict]) -> list[dict]:
        trimmed_segments = [dict(segment) for segment in segments if segment.get("text") is not None]
        if not trimmed_segments:
            return []

        start_idx = 0
        while start_idx < len(trimmed_segments):
            normalized_text = trimmed_segments[start_idx]["text"].lstrip()
            if normalized_text:
                trimmed_segments[start_idx]["text"] = normalized_text
                break
            start_idx += 1

        if start_idx == len(trimmed_segments):
            return []

        trimmed_segments = trimmed_segments[start_idx:]
        end_idx = len(trimmed_segments) - 1
        while end_idx >= 0:
            normalized_text = trimmed_segments[end_idx]["text"].rstrip()
            if normalized_text:
                trimmed_segments[end_idx]["text"] = normalized_text
                break
            end_idx -= 1

        if end_idx < 0:
            return []

        return trimmed_segments[:end_idx + 1]

    @staticmethod
    def _normalize_text_block_content(content: str) -> str:
        """Normalize extracted text-block content without changing internal spacing."""
        if not content:
            return ""
        return content.strip()

    @staticmethod
    def _parse_font_size_pt_from_rpr(
        rpr: Optional[etree._Element],
    ) -> Optional[float]:
        if rpr is None:
            return None

        size = rpr.get("sz")
        if size is None:
            return None

        try:
            return round(int(size) / 100, 1)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_bold_from_rpr(
        rpr: Optional[etree._Element],
    ) -> Optional[bool]:
        return PptxConverter._parse_toggle_attr_from_rpr(rpr, "b")

    @staticmethod
    def _parse_italic_from_rpr(
        rpr: Optional[etree._Element],
    ) -> Optional[bool]:
        return PptxConverter._parse_toggle_attr_from_rpr(rpr, "i")

    def _find_def_rpr(
        self,
        paragraph_properties: Optional[etree._Element],
    ) -> Optional[etree._Element]:
        if paragraph_properties is None:
            return None
        return paragraph_properties.find("a:defRPr", namespaces=self.namespaces)

    def _find_end_para_rpr(
        self,
        paragraph: Optional[etree._Element],
    ) -> Optional[etree._Element]:
        if paragraph is None:
            return None
        return paragraph.find("a:endParaRPr", namespaces=self.namespaces)

    def _get_font_sources_from_paragraph(
        self,
        paragraph: Optional[etree._Element],
    ) -> list[etree._Element]:
        if paragraph is None:
            return []

        sources = []
        paragraph_properties = paragraph.find("a:pPr", namespaces=self.namespaces)
        paragraph_def_rpr = self._find_def_rpr(paragraph_properties)
        if paragraph_def_rpr is not None:
            sources.append(paragraph_def_rpr)

        end_para_rpr = self._find_end_para_rpr(paragraph)
        if end_para_rpr is not None:
            sources.append(end_para_rpr)

        return sources

    def _get_font_sources_from_text_body(
        self,
        tx_body: Optional[etree._Element],
        level: int,
    ) -> list[etree._Element]:
        if tx_body is None:
            return []

        lst_style = tx_body.find("a:lstStyle", namespaces=self.namespaces)
        if lst_style is None:
            return []

        sources = []
        level_properties = self._find_level_properties_in_list_style(
            lst_style,
            level,
        )
        level_def_rpr = self._find_def_rpr(level_properties)
        if level_def_rpr is not None:
            sources.append(level_def_rpr)

        default_properties = lst_style.find("a:defPPr", namespaces=self.namespaces)
        default_def_rpr = self._find_def_rpr(default_properties)
        if default_def_rpr is not None:
            sources.append(default_def_rpr)

        return sources

    def _get_font_sources_from_text_style_bucket(
        self,
        style_bucket: Optional[etree._Element],
        level: int,
    ) -> list[etree._Element]:
        if style_bucket is None:
            return []

        sources = []
        level_properties = style_bucket.find(
            f"a:lvl{level + 1}pPr",
            namespaces=self.namespaces,
        )
        level_def_rpr = self._find_def_rpr(level_properties)
        if level_def_rpr is not None:
            sources.append(level_def_rpr)

        default_properties = style_bucket.find(
            "a:defPPr",
            namespaces=self.namespaces,
        )
        default_def_rpr = self._find_def_rpr(default_properties)
        if default_def_rpr is not None:
            sources.append(default_def_rpr)

        return sources

    def _resolve_layout_placeholder(self, shape):
        if not getattr(shape, "is_placeholder", False):
            return None

        try:
            idx = shape.placeholder_format.idx
            layout = shape.part.slide.slide_layout
            layout_ph = layout.placeholders.get(idx)
        except Exception:
            layout_ph = None

        if layout_ph is not None:
            return layout_ph

        try:
            placeholder_type = shape.placeholder_format.type
            layout = shape.part.slide.slide_layout
            for candidate in layout.placeholders:
                if candidate.placeholder_format.type == placeholder_type:
                    return candidate
        except Exception:
            return None

        return None

    def _get_paragraph_font_sources(self, shape, paragraph) -> list[etree._Element]:
        level = self._get_paragraph_level(paragraph._element)
        sources = self._get_font_sources_from_paragraph(paragraph._element)

        tx_body = shape._element.find(".//p:txBody", namespaces=self.namespaces)
        sources.extend(self._get_font_sources_from_text_body(tx_body, level))

        if getattr(shape, "is_placeholder", False):
            layout_placeholder = self._resolve_layout_placeholder(shape)
            if layout_placeholder is not None:
                layout_tx_body = layout_placeholder._element.find(
                    ".//p:txBody",
                    namespaces=self.namespaces,
                )
                sources.extend(
                    self._get_font_sources_from_text_body(layout_tx_body, level)
                )

            try:
                placeholder_type = shape.placeholder_format.type
                slide_master = shape.part.slide.slide_layout.slide_master
            except Exception:
                return sources

            style_bucket = self._get_master_text_style_node(
                slide_master,
                placeholder_type,
            )
            sources.extend(
                self._get_font_sources_from_text_style_bucket(
                    style_bucket,
                    level,
                )
            )

        return sources

    def _resolve_effective_run_font_size_pt(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> Optional[float]:
        for source in [self._get_run_rpr(run), *paragraph_font_sources]:
            font_size_pt = self._parse_font_size_pt_from_rpr(source)
            if font_size_pt is not None:
                return font_size_pt
        return None

    def _resolve_effective_run_bold(
        self,
        run,
        paragraph_font_sources: list[etree._Element],
    ) -> bool:
        return self._resolve_effective_run_bool(
            run,
            paragraph_font_sources,
            self._parse_bold_from_rpr,
        )

    def _build_paragraph_style_profile(self, shape, paragraph) -> dict[str, Optional[float] | bool]:
        paragraph_font_sources = self._get_paragraph_font_sources(shape, paragraph)
        effective_font_size_pt = None
        all_bold = True
        has_non_whitespace_run = False

        for run in paragraph.runs:
            run_text = getattr(run, "text", None)
            if run_text is None or not run_text.strip():
                continue

            has_non_whitespace_run = True

            run_font_size_pt = self._resolve_effective_run_font_size_pt(
                run,
                paragraph_font_sources,
            )
            if run_font_size_pt is not None:
                if effective_font_size_pt is None:
                    effective_font_size_pt = run_font_size_pt
                else:
                    effective_font_size_pt = max(
                        effective_font_size_pt,
                        run_font_size_pt,
                    )

            if (
                self._resolve_effective_run_bold(run, paragraph_font_sources)
                is not True
            ):
                all_bold = False

        return {
            "font_size_pt": effective_font_size_pt,
            "all_bold": has_non_whitespace_run and all_bold,
        }

    def _get_paragraph_list_info(self, shape, paragraph) -> dict:
        """基于段落->文本框->布局->母版继承链解析段落列表属性。"""
        marker_info = self._get_effective_list_marker(shape, paragraph)
        p = paragraph._element
        level = marker_info.get("level", self._get_paragraph_level(p))
        kind = marker_info.get("kind")

        if marker_info.get("is_list") is False:
            return {
                "is_list": False,
                "attribute": "unordered",
                "level": level,
                "kind": kind,
            }

        if kind == "buAutoNum":
            return {
                "is_list": True,
                "attribute": "ordered",
                "level": level,
                "kind": kind,
            }

        if kind in ("buChar", "buBlip"):
            return {
                "is_list": True,
                "attribute": "unordered",
                "level": level,
                "kind": kind,
            }

        if marker_info.get("is_list") is True:
            return {
                "is_list": True,
                "attribute": "unordered",
                "level": level,
                "kind": kind,
            }

        # 兜底：段落级标记 + 缩进层级判断
        if p.find(".//a:buAutoNum", namespaces={"a": self.namespaces["a"]}) is not None:
            return {
                "is_list": True,
                "attribute": "ordered",
                "level": paragraph.level,
                "kind": "buAutoNum",
            }

        if p.find(".//a:buChar", namespaces={"a": self.namespaces["a"]}) is not None:
            return {
                "is_list": True,
                "attribute": "unordered",
                "level": paragraph.level,
                "kind": "buChar",
            }

        if paragraph.level > 0:
            return {
                "is_list": True,
                "attribute": "unordered",
                "level": paragraph.level,
                "kind": None,
            }

        return {
            "is_list": False,
            "attribute": "unordered",
            "level": 0,
            "kind": None,
        }

    def _ensure_list_level(self, list_stack: list, level: int, attribute: str):
        """将列表栈调整到目标层级，并在必要时创建同级/子级列表块。"""
        while len(list_stack) > level + 1:
            list_stack.pop()

        if len(list_stack) == level + 1 and list_stack[level].get("attribute") != attribute:
            list_stack.pop()

        while len(list_stack) < level + 1:
            ilevel = len(list_stack)
            new_list_block = {
                "type": BlockType.LIST,
                "attribute": attribute,
                "ilevel": ilevel,
                "content": [],
            }

            if list_stack:
                list_stack[-1]["content"].append(new_list_block)
            else:
                self.cur_page.append(new_list_block)

            list_stack.append(new_list_block)

    def _append_list_item(
        self,
        list_stack: list,
        level: int,
        attribute: str,
        content: str,
    ):
        """向目标层级列表追加文本项。"""
        self._ensure_list_level(list_stack, level, attribute)
        list_stack[-1]["content"].append(
            {
                "type": BlockType.TEXT,
                "content": content,
            }
        )

    @staticmethod
    def _most_common_size(font_sizes: list[float]) -> Optional[float]:
        if not font_sizes:
            return None

        counts = Counter(font_sizes)
        return min(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]

    def _promote_slide_text_blocks_to_titles(self, slide_blocks: list[dict]) -> None:
        body_font_size_pt = self._most_common_size(
            [
                block[_EFFECTIVE_FONT_SIZE_KEY]
                for block in slide_blocks
                if (
                    block.get("type") == BlockType.TEXT
                    and block.get(_EFFECTIVE_FONT_SIZE_KEY) is not None
                    and not block.get(_EFFECTIVE_ALL_BOLD_KEY, False)
                )
            ]
        )

        self._promote_level2_text_blocks(slide_blocks, body_font_size_pt)
        self._promote_level3_text_blocks(slide_blocks, body_font_size_pt)

    def _promote_level2_text_blocks(
        self,
        slide_blocks: list[dict],
        body_font_size_pt: Optional[float],
    ) -> None:
        bold_text_blocks = [
            block
            for block in slide_blocks
            if (
                block.get("type") == BlockType.TEXT
                and block.get(_EFFECTIVE_ALL_BOLD_KEY, False)
                and block.get(_EFFECTIVE_FONT_SIZE_KEY) is not None
            )
        ]
        if not bold_text_blocks:
            return

        bold_font_sizes = sorted(
            {
                block[_EFFECTIVE_FONT_SIZE_KEY]
                for block in bold_text_blocks
            },
            reverse=True,
        )
        level2_font_size_pt = bold_font_sizes[0]
        level2_candidates = [
            block
            for block in bold_text_blocks
            if block[_EFFECTIVE_FONT_SIZE_KEY] == level2_font_size_pt
        ]

        if len(level2_candidates) != 1:
            return

        if (
            body_font_size_pt is not None
            and level2_font_size_pt < body_font_size_pt + 4
        ):
            return

        if (
            len(bold_font_sizes) > 1
            and level2_font_size_pt < bold_font_sizes[1] + 2
        ):
            return

        level2_candidates[0]["type"] = BlockType.TITLE
        level2_candidates[0]["level"] = 2

    def _promote_level3_text_blocks(
        self,
        slide_blocks: list[dict],
        body_font_size_pt: Optional[float],
    ) -> None:
        if body_font_size_pt is None:
            return

        level2_font_sizes = sorted(
            {
                block[_EFFECTIVE_FONT_SIZE_KEY]
                for block in slide_blocks
                if (
                    block.get("type") == BlockType.TITLE
                    and block.get("level") == 2
                    and block.get(_EFFECTIVE_FONT_SIZE_KEY) is not None
                )
            },
            reverse=True,
        )
        if not level2_font_sizes:
            return

        level2_font_size_pt = level2_font_sizes[0]
        level3_font_sizes = sorted(
            {
                block[_EFFECTIVE_FONT_SIZE_KEY]
                for block in slide_blocks
                if (
                    block.get("type") == BlockType.TEXT
                    and block.get(_EFFECTIVE_ALL_BOLD_KEY, False)
                    and block.get(_EFFECTIVE_FONT_SIZE_KEY) is not None
                    and block[_EFFECTIVE_FONT_SIZE_KEY] < level2_font_size_pt
                )
            },
            reverse=True,
        )
        if not level3_font_sizes:
            return

        level3_font_size_pt = level3_font_sizes[0]
        if level3_font_size_pt < body_font_size_pt + 2:
            return
        if level2_font_size_pt < level3_font_size_pt + 2:
            return

        for block in slide_blocks:
            if (
                block.get("type") == BlockType.TEXT
                and block.get(_EFFECTIVE_ALL_BOLD_KEY, False)
                and block.get(_EFFECTIVE_FONT_SIZE_KEY) == level3_font_size_pt
            ):
                block["type"] = BlockType.TITLE
                block["level"] = 3

    @staticmethod
    def _cleanup_slide_text_block_metadata(slide_blocks: list[dict]) -> None:
        for block in slide_blocks:
            block.pop(_EFFECTIVE_FONT_SIZE_KEY, None)
            block.pop(_EFFECTIVE_ALL_BOLD_KEY, None)

    def _handle_text_elements(self, shape):
        self.list_block_stack = []

        # 遍历段落以构建文本
        for paragraph in shape.text_frame.paragraphs:
            list_info = self._get_paragraph_list_info(shape, paragraph)

            if list_info["is_list"]:
                rich_text = self._normalize_text_block_content(
                    self._build_paragraph_rich_text(paragraph, shape)
                )
                if rich_text:
                    self._append_list_item(
                        self.list_block_stack,
                        list_info["level"],
                        list_info["attribute"],
                        rich_text,
                    )
                continue

            # 段落不是列表项，关闭当前 shape 的列表上下文
            self.list_block_stack.clear()

            p_text = self._normalize_text_block_content(
                self._build_paragraph_rich_text(paragraph, shape)
            )
            if not p_text:
                continue

            style_profile = self._build_paragraph_style_profile(shape, paragraph)

            block = {
                "type": BlockType.TEXT,
                "content": p_text,
                _EFFECTIVE_FONT_SIZE_KEY: style_profile["font_size_pt"],
                _EFFECTIVE_ALL_BOLD_KEY: style_profile["all_bold"],
            }
            if shape.is_placeholder:
                placeholder_type = shape.placeholder_format.type
                if placeholder_type in [
                    PP_PLACEHOLDER.CENTER_TITLE,
                    PP_PLACEHOLDER.TITLE,
                    PP_PLACEHOLDER.SUBTITLE,
                ]:
                    block["type"] = BlockType.TITLE
                    block["level"] = 2

            self.cur_page.append(block)

        # shape 结束后清理列表上下文，避免跨 shape 污染
        self.list_block_stack.clear()
        return

    def _is_list_item(self, paragraph) -> tuple[bool, str]:
        """
        判断段落是否应被视为列表项。
        该方法首先尝试通过拥有该段落的形状来解析列表样式信息。
        如果无法做到，则回退到基于段落属性和级别的更简单检查。
        Args:
            paragraph: 需要检查的'python-pptx'段落对象。

        Returns:
            返回一个2元组(`is_list`, `bullet_type`)，其中：
            `is_list` - 若段落被视为列表项，为True，否则为False；
            `bullet_type` - 为以下之一：'Bullet'(项目符号)、'Numbered'(编号)或'None'，
            描述列表标记类型。
        """
        # 尝试从段落获取形状（包含该段落的对象），如果可能的话
        shape = None
        try:
            # 这个路径适用于python-pptx段落对象
            # 首先获取文本框架(段落的父对象)
            text_frame = paragraph._parent
            # 然后获取形状(文本框架的父对象)
            shape = text_frame._parent
        except AttributeError:
            pass

        if shape is not None:
            list_info = self._get_paragraph_list_info(shape, paragraph)
            if not list_info["is_list"]:
                return (False, "None")

            if list_info["attribute"] == "ordered":
                return (True, "Numbered")
            return (True, "Bullet")

        # 如果无法获取形状，使用更简单的检查方式
        p = paragraph._element
        if p.find(".//a:buChar", namespaces={"a": self.namespaces["a"]}) is not None:
            return (True, "Bullet")
        elif (
            p.find(".//a:buAutoNum", namespaces={"a": self.namespaces["a"]}) is not None
        ):
            return (True, "Numbered")
        elif paragraph.level > 0:
            # 很可能是子列表项(缩进表示嵌套)
            return (True, "None")
        else:
            return (False, "None")

    def _get_effective_list_marker(self, shape, paragraph) -> dict:
        """
        返回描述段落的有效列表标记的字典。
        列表标记信息可以来自多个来源：直接段落属性、形状级别的列表样式、
        布局占位符或主幻灯片文本样式。此辅助方法解析所有这些层，并返回
        有效标记的统一视图。

        Args:
            shape: 包含段落的形状对象。
            paragraph: 需要检查的'python-pptx'段落对象。

        Returns:
            返回列表标记信息的字典，其中：
            `is_list` - True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`、`buAutoNum`、`buBlip`、`buNone`或None，描述标记类型；
            `detail` - 项目符号字符或编号类型字符串，或如果不适用则为None；
            `level` - 段落级别，范围在(0, 8)内。
        """
        p = paragraph._element
        lvl = self._get_paragraph_level(p)

        # 1) 直接段落属性
        pPr = p.find("a:pPr", namespaces=self.namespaces)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(pPr)
        if is_list is not None:
            return {
                "is_list": is_list,
                "kind": kind,
                "detail": detail,
                "level": lvl,
            }

        # 2) 形状级别的列表样式(txBody/a:lstStyle)
        txBody = shape._element.find(".//p:txBody", namespaces=self.namespaces)
        is_list, kind, detail = self._parse_bullet_from_text_body_list_style(
            txBody, lvl
        )
        if is_list is not None:
            return {
                "is_list": is_list,
                "kind": kind,
                "detail": detail,
                "level": lvl,
            }

        # 3) 布局占位符列表样式(如果这是一个占位符)
        layout_result = None
        if shape.is_placeholder:
            layout_ph = self._resolve_layout_placeholder(shape)

            if layout_ph is not None:
                layout_tx = layout_ph._element.find(
                    ".//p:txBody", namespaces=self.namespaces
                )
                is_list, kind, detail = self._parse_bullet_from_text_body_list_style(
                    layout_tx, lvl
                )

                # 仅在is_list明确为True/False时使用布局结果
                if is_list is not None:
                    layout_result = {
                        "is_list": is_list,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                    }

                # 4) 解析主文本样式
                ph_type = shape.placeholder_format.type
                master = shape.part.slide.slide_layout.slide_master
                is_list, kind, detail = self._parse_bullet_from_master_text_styles(
                    master, ph_type, lvl
                )

                # 检查主样式是否有标记信息
                if kind in ("buChar", "buAutoNum", "buBlip"):
                    return {
                        "is_list": True,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                    }
                elif is_list is not None:
                    return {
                        "is_list": is_list,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                    }

            # If layout has explicit is_list value but master didn't override it, use layout
            # 如果布局有显式的is_list值但主样式没有覆盖它，则使用布局结果
            if layout_result is not None:
                return layout_result

        return {
            "is_list": None,
            "kind": None,
            "detail": None,
            "level": lvl,
        }

    def _get_paragraph_level(self, paragraph) -> int:
        """
        返回段落XML元素的缩进级别。
        段落可以有不同的缩进级别(0-8)。级别存储在段落属性XML元素的'lvl'属性中。

        Args:
            paragraph: 需要提取级别的段落XML元素。

        Returns:
            返回范围在(0, 8)内的段落级别。当找不到'a:pPr'元素、没有'lvl'属性
            或'lvl'属性值无效时，返回0。
        """
        pPr = paragraph.find("a:pPr", namespaces=self.namespaces)
        if pPr is not None and "lvl" in pPr.attrib:
            try:
                return int(pPr.get("lvl"))
            except ValueError:
                pass
        return 0

    def _parse_bullet_from_paragraph_properties(
        self, pPr
    ) -> tuple[Optional[bool], Optional[str], Optional[str]]:
        """
        从段落属性节点解析项目符号或编号信息。
        检查'a:pPr'或'a:lvlXpPr'元素，并提取关于项目符号字符、自动编号、
        图片项目符号或显式'buNone'标记的信息。

        Args:
            pPr: 段落属性XML元素('a:pPr'或'a:lvlXpPr')。

        Returns:
            返回一个3元组(`is_list`, `kind`, `detail`)，其中：
            `is_list` - 为True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`(项目符号字符)、`buAutoNum`(自动编号)、
            `buBlip`(图片项目符号)、`buNone`(无标记)或None，描述标记类型；
            `detail` - 项目符号字符、编号类型字符串，或如果不适用则为None。
        """
        if pPr is None:
            return (None, None, None)

        # 显式指定无项目符号
        if pPr.find("a:buNone", namespaces=self.namespaces) is not None:
            return (False, "buNone", None)

        # 项目符号字符
        buChar = pPr.find("a:buChar", namespaces=self.namespaces)
        if buChar is not None:
            return (True, "buChar", buChar.get("char"))

        # 自动编号
        buAuto = pPr.find("a:buAutoNum", namespaces=self.namespaces)
        if buAuto is not None:
            return (True, "buAutoNum", buAuto.get("type"))

        # 图片项目符号
        buBlip = pPr.find("a:buBlip", namespaces=self.namespaces)
        if buBlip is not None:
            return (True, "buBlip", "image")

        return (None, None, None)

    def _parse_bullet_from_text_body_list_style(
        self, txBody, lvl: int
    ) -> tuple[Optional[bool], Optional[str], Optional[str]]:
        """
        从文本体的列表样式中解析项目符号或编号信息。
        在'txBody'下搜索'a:lstStyle/a:lvl{lvl+1}pPr'，并使用级别特定的段落属性
        推断项目符号或编号信息。

        Args:
            txBody: 文本体XML元素'p:txBody'。
            lvl: 段落级别，范围在(0, 8)内。
        Returns:
            返回一个3元组(`is_list`, `kind`, `detail`)，其中：
            `is_list` - 为True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`、`buAutoNum`、`buBlip`、`buNone`或None；
            `detail` - 项目符号字符、编号类型字符串，或如果不适用则为None。
        """
        if txBody is None:
            return (None, None, None)
        lstStyle = txBody.find("a:lstStyle", namespaces=self.namespaces)
        lvl_pPr = self._find_level_properties_in_list_style(lstStyle, lvl)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(lvl_pPr)
        return (is_list, kind, detail)

    def _parse_bullet_from_master_text_styles(
        self, slide_master, placeholder_type, lvl: int
    ) -> tuple[Optional[bool], Optional[str], Optional[str]]:
        """
        从主幻灯片的文本样式中解析项目符号或编号信息。
        在主幻灯片的'p:txStyles'中查找相应的样式bucket('titleStyle'、'bodyStyle'或
        'otherStyle')，并为给定的级别提取项目符号或编号信息。

        Args:
            slide_master: 与当前幻灯片关联的主幻灯片对象。
            placeholder_type: 来自'PP_PLACEHOLDER'的占位符类型枚举。
            lvl: 段落级别，范围在(0, 8)内。

        Returns:
            返回一个3元组(`is_list`, `kind`, `detail`)，其中：
            `is_list` - 为True/False/None，表示这是否是列表项；
            `kind` - 为以下之一：`buChar`、`buAutoNum`、`buBlip`、`buNone`或None；
            `detail` - 项目符号字符、编号类型字符串，或如果不适用则为None。
        """
        style = self._get_master_text_style_node(slide_master, placeholder_type)
        if style is None:
            return (None, None, None)

        lvl_pPr = style.find(f".//a:lvl{lvl + 1}pPr", namespaces=self.namespaces)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(lvl_pPr)
        return (is_list, kind, detail)

    def _find_level_properties_in_list_style(self, lstStyle, lvl: int):
        """Find the level-specific paragraph properties node from a list style.
        从列表样式中查找指定级别的段落属性节点。

        This looks for an `a:lvl{lvl+1}pPr` node inside an `a:lstStyle` element, where
        在'a:lstStyle'元素内查找'a:lvl{lvl+1}pPr'节点，其中'a:lvl1pPr'对应级别0，
        `a:lvl1pPr` corresponds to level 0, `a:lvl2pPr` to level 1, and so on.
        'a:lvl2pPr'对应级别1，依此类推。

        Args:
            lstStyle: List style XML element `a:lstStyle`.
            lstStyle: 列表样式XML元素'a:lstStyle'。
            lvl: Paragraph level in the range (0, 8).
            lvl: 段落级别，范围在(0, 8)内。

        Returns:
            Matching `a:lvl{lvl+1}pPr` XML element, or None if no matching element is
            匹配的'a:lvl{lvl+1}pPr'XML元素，如果未找到匹配元素则返回None。
                found.
        """
        if lstStyle is None:
            return None
        tag = f"a:lvl{lvl + 1}pPr"
        return lstStyle.find(tag, namespaces=self.namespaces)

    def _get_master_text_style_node(
        self, slide_master, placeholder_type
    ) -> Optional[etree._Element]:
        """
        获取占位符的相应主文本样式节点。
        大多数内容占位符(BODY/OBJECT)使用'p:bodyStyle'，而标题使用'p:titleStyle'。
        所有其他占位符默认使用'p:otherStyle'。

        Args:
            slide_master: 与当前幻灯片关联的主幻灯片对象。
            placeholder_type: 来自'PP_PLACEHOLDER'的占位符类型枚举。

        Returns:
            从主幻灯片的'p:txStyles'中匹配的样式节点('p:bodyStyle'、'p:titleStyle'或'p:otherStyle')，或当未定义样式时返回None。
        """
        txStyles = slide_master._element.find(
            ".//p:txStyles", namespaces=self.namespaces
        )
        if txStyles is None:
            return None

        if placeholder_type in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT):
            return txStyles.find("p:bodyStyle", namespaces=self.namespaces)

        if placeholder_type in (
            PP_PLACEHOLDER.TITLE,
            PP_PLACEHOLDER.CENTER_TITLE,
            PP_PLACEHOLDER.SUBTITLE,
        ):
            return txStyles.find("p:titleStyle", namespaces=self.namespaces)

        return txStyles.find("p:otherStyle", namespaces=self.namespaces)
