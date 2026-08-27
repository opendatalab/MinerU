# Copyright (c) Opendatalab. All rights reserved.
import collections
import hashlib
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from typing import BinaryIO, cast

from loguru import logger
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XlsImage
from openpyxl.utils.cell import range_to_tuple
from openpyxl.worksheet.worksheet import Worksheet
from ..image import serialize_office_image
from ..equation.image import (
    OfficeImageEquationDecoder,
)
from ..equation.ooxml import OoxmlEquationDecoder
from ..errors import LegacyOfficeResourceLimitError
from ..limits import MAX_ENTRY_BYTES
from ..equation.omml import oMath2Latex
from ..streams import read_stream_bytes_from_start, rewind_stream
from ..spreadsheet.html import EQUATION_BOOKENDS, render_spreadsheet_table
from ..spreadsheet.models import AnchoredBlock, FormulaMap, SheetImage
from ..spreadsheet.projector import SpreadsheetProjector
from .package_normalizer import (
    normalize_xlsx_package,
    strip_xlsx_ole_objects_for_openpyxl,
)
from .ooxml_ole import (
    XlsxOleEquationArtifact,
    package_has_sheet_ole_objects,
    read_sheet_image_artifacts,
    read_sheet_equation_artifacts,
    workbook_sheet_parts,
)
from .....types import BlockType


class XlsxConverter(SpreadsheetProjector):
    def __init__(
        self,
        treat_singleton_as_text: bool = True,
        gap_tolerance: int | None = None,
        include_hidden_sheets: bool = False,
    ) -> None:
        super().__init__(
            treat_singleton_as_text=treat_singleton_as_text,
            gap_tolerance=gap_tolerance,
            include_hidden_sheets=include_hidden_sheets,
        )
        self.zf = None
        self.image_map = {}
        self.cell_image_map = {}
        self._sheet_part_by_title: dict[str, str] = {}
        self._ole_artifacts: list[XlsxOleEquationArtifact] = []
        self._omml_shape_ids: set[str] = set()
        self._omml_artifacts: list[tuple[int, int, str, int]] = []
        self._suppressed_ole_previews: set[
            tuple[tuple[int, int], str]
        ] = set()
        self._ooxml_equation_decoder = OoxmlEquationDecoder()
        self._image_equation_decoder = OfficeImageEquationDecoder()

    def convert(
        self,
        file_stream: BinaryIO,
    ) -> None:
        if rewind_stream(file_stream):
            try:
                self._convert_package_stream(file_stream)
                return
            except Exception as exc:
                file_bytes = read_stream_bytes_from_start(file_stream)
                self._retry_convert_package_bytes_after_normalization(file_bytes, exc)
                return

        file_bytes = file_stream.read()
        try:
            self._convert_package_bytes(file_bytes)
        except Exception as exc:
            self._retry_convert_package_bytes_after_normalization(file_bytes, exc)

    def _reset_state(self) -> None:
        """重置解析状态，确保失败重试时不会残留上一次半解析结果。"""
        if self.zf:
            self.zf.close()
        self._reset_projection_state()
        self.zf = None
        self.image_map = {}
        self.cell_image_map = {}
        self._sheet_part_by_title = {}
        self._ole_artifacts = []
        self._omml_shape_ids = set()
        self._omml_artifacts = []
        self._suppressed_ole_previews = set()
        self._ooxml_equation_decoder = OoxmlEquationDecoder()
        self._image_equation_decoder = OfficeImageEquationDecoder()

    def _convert_package_bytes(self, file_bytes: bytes) -> None:
        """用独立字节流解析 XLSX 包，便于原始包失败后用规范化包重试。"""
        self._convert_package_stream(BytesIO(file_bytes))

    def _convert_package_stream(self, file_stream: BinaryIO) -> None:
        """直接使用可复位的 XLSX 流解析正常路径，避免提前复制完整包字节。"""
        self._reset_state()
        try:
            self.zf = zipfile.ZipFile(file_stream)
            self._sheet_part_by_title = workbook_sheet_parts(self.zf)
        except Exception as e:
            logger.warning(f"Failed to open zip file: {e}")
            self.zf = None

        try:
            workbook_stream: BinaryIO = file_stream
            if self.zf is not None and package_has_sheet_ole_objects(
                self.zf,
                self._sheet_part_by_title,
            ):
                file_bytes = read_stream_bytes_from_start(file_stream)
                workbook_stream = BytesIO(
                    strip_xlsx_ole_objects_for_openpyxl(file_bytes)
                )
            else:
                rewind_stream(file_stream)
            self.workbook = load_workbook(
                filename=workbook_stream,
                data_only=True,
                rich_text=True,
            )
            if self.workbook is not None:
                # 遍历需要参与转换的工作表，避免为隐藏表或尾部空页生成无效页面。
                sheet_pages = []
                for idx, sheet in enumerate(self._iter_sheets_to_convert(), start=1):
                    logger.debug(f"正在处理第 {idx} 个工作表：{sheet.title}")
                    self.cur_page = []
                    self._convert_sheet(sheet)
                    sheet_pages.append((sheet.title, self.cur_page))
                if self._should_emit_sheet_titles([page for _, page in sheet_pages]):
                    self._prepend_sheet_titles(sheet_pages)
                self.pages.extend(page for _, page in sheet_pages)
            else:
                logger.error("工作簿未初始化。")
        finally:
            if self.zf:
                self.zf.close()
                self.zf = None

    def _retry_convert_package_bytes_after_normalization(
        self,
        file_bytes: bytes,
        exc: Exception,
    ) -> None:
        """首次解析失败后，仅在包规范化确实产生变化时使用规范化字节重试。"""
        normalized_bytes = normalize_xlsx_package(file_bytes)
        if normalized_bytes == file_bytes:
            raise exc
        logger.warning(f"Retrying XLSX parsing after package normalization: {exc}")
        self._convert_package_bytes(normalized_bytes)

    def _prepare_sheet_assets(self, sheet: Worksheet) -> None:
        """准备 XLSX 公式、图片与 OLE 素材，并保持既有预览抑制优先级。"""
        self.math_map = self._map_math_formulas_to_cells(sheet)
        self._ole_artifacts = self._read_ole_equation_artifacts(sheet)
        self._suppressed_ole_previews = {
            ((artifact.row, artifact.col), artifact.preview_base64)
            for artifact in self._ole_artifacts
            if artifact.shape_id in self._omml_shape_ids
            and artifact.row is not None
            and artifact.col is not None
            and artifact.preview_base64 is not None
        }
        self._ole_artifacts = [
            artifact
            for artifact in self._ole_artifacts
            if artifact.shape_id not in self._omml_shape_ids
        ]
        for artifact in self._ole_artifacts:
            if artifact.latex is None or artifact.row is None or artifact.col is None:
                continue
            self.math_map.setdefault((artifact.row, artifact.col), []).append(artifact.latex)

        self.sheet_images = self._collect_sheet_images(sheet)
        ole_previews = self._suppressed_ole_previews | {
            ((artifact.row, artifact.col), artifact.preview_base64)
            for artifact in self._ole_artifacts
            if artifact.row is not None
            and artifact.col is not None
            and artifact.preview_base64
        }
        self.sheet_images = [
            image
            for image in self.sheet_images
            if (image.anchor, image.image_base64) not in ole_previews
        ]
        self.table_image_map = collections.defaultdict(list)
        for image in self.sheet_images:
            row, col = image.anchor
            if row is None or col is None:
                continue
            if image.latex:
                self.table_image_map[(row, col)].append(EQUATION_BOOKENDS.format(EQ=image.latex))
            elif image.image_base64:
                self.table_image_map[(row, col)].append(f'<img src="{image.image_base64}" />')
        for artifact in self._ole_artifacts:
            if (
                artifact.latex is not None
                or artifact.preview_base64 is None
                or artifact.row is None
                or artifact.col is None
            ):
                continue
            self.table_image_map[(artifact.row, artifact.col)].append(
                f'<img src="{artifact.preview_base64}" />'
            )

    def _find_additional_visual_artifacts(
        self,
        used_cells: set[tuple[int, int]],
    ) -> list[AnchoredBlock]:
        """输出未被表格吸收的 XLSX 公式、OLE 预览和图片公式。"""
        return [
            *self._find_equation_artifacts_in_sheet(used_cells),
            *self._find_image_equation_artifacts_in_sheet(used_cells),
        ]

    def _read_ole_equation_artifacts(
        self,
        sheet: Worksheet,
    ) -> list[XlsxOleEquationArtifact]:
        """从当前 worksheet part 读取 MathType/Equation 公式和预览。"""

        if self.zf is None:
            return []
        worksheet_part = self._sheet_part_by_title.get(sheet.title)
        if worksheet_part is None:
            return []
        return read_sheet_equation_artifacts(
            self.zf,
            worksheet_part,
            self._ooxml_equation_decoder,
            self._image_equation_decoder,
        )

    def _find_equation_artifacts_in_sheet(
        self,
        used_cells: set[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], int, dict]]:
        """输出未被表格吸收的 OMML/MTEF 公式或缓存预览。"""

        artifacts: list[tuple[tuple[int, int], int, dict]] = []
        for row, col, latex, order in self._omml_artifacts:
            if (row, col) in used_cells:
                continue
            artifacts.append(
                (
                    (row, col),
                    15_000 + order,
                    {
                        "type": BlockType.EQUATION,
                        "content": latex,
                    },
                )
            )
        for artifact in self._ole_artifacts:
            coordinate = (
                artifact.row if artifact.row is not None else 10**9,
                artifact.col if artifact.col is not None else 10**9,
            )
            if (
                artifact.row is not None
                and artifact.col is not None
                and (artifact.row, artifact.col) in used_cells
            ):
                continue
            block = None
            if artifact.latex is not None:
                block = {
                    "type": BlockType.EQUATION,
                    "content": artifact.latex,
                }
            elif artifact.preview_base64 is not None:
                block = {
                    "type": BlockType.IMAGE,
                    "image_base64": artifact.preview_base64,
                }
            if block is not None:
                artifacts.append((coordinate, 20_000 + artifact.order, block))
        return artifacts

    def _find_image_equation_artifacts_in_sheet(
        self,
        used_cells: set[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], int, dict]]:
        """按原始 drawing anchor 输出未被表格吸收的图片 comment 公式。"""

        artifacts: list[tuple[tuple[int, int], int, dict]] = []
        for image in self.sheet_images:
            if not image.latex:
                continue
            row, col = image.anchor
            if (
                row is not None
                and col is not None
                and (row, col) in used_cells
            ):
                continue
            coordinate = (
                row if row is not None else 10**9,
                col if col is not None else 10**9,
            )
            artifacts.append(
                (
                    coordinate,
                    25_000 + image.order,
                    {"type": BlockType.EQUATION, "content": image.latex},
                )
            )
        return artifacts

    def _read_xlsx_image_member(self, part_name: str) -> bytes | None:
        """从原始 XLSX ZIP 有界读取 media member。"""

        if self.zf is None:
            return None
        normalized = posixpath.normpath(part_name.lstrip("/"))
        if normalized.startswith("../") or normalized not in self.zf.namelist():
            return None
        info = self.zf.getinfo(normalized)
        if info.file_size > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"XLSX image exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        with self.zf.open(info) as stream:
            payload = stream.read(MAX_ENTRY_BYTES + 1)
        if len(payload) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"XLSX image exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        return payload

    def _raw_sheet_image(
        self,
        image: XlsImage,
    ) -> tuple[bytes, str | None, str | None] | None:
        """优先从原 ZIP 读取 openpyxl 图片的未转码原始字节。"""

        part_name = str(getattr(image, "path", "") or "") or None
        payload = (
            self._read_xlsx_image_member(part_name)
            if part_name is not None
            else None
        )
        if payload is None:
            try:
                payload = image._data()  # type: ignore[attr-defined]
            except Exception:
                return None
            if len(payload) > MAX_ENTRY_BYTES:
                raise LegacyOfficeResourceLimitError(
                    f"XLSX image exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
                )
        image_format = str(getattr(image, "format", "") or "").casefold()
        content_type = f"image/{image_format}" if image_format else None
        return payload, part_name, content_type

    def _collect_sheet_images(self, sheet: Worksheet) -> list[SheetImage]:
        """读取当前工作表的原始图片并识别图片公式。"""
        images: list[SheetImage] = []
        if self.workbook is None:
            return images

        seen: set[tuple[tuple[int | None, int | None], bytes]] = set()
        worksheet_part = self._sheet_part_by_title.get(sheet.title)
        if self.zf is not None and worksheet_part is not None:
            for artifact in read_sheet_image_artifacts(
                self.zf,
                worksheet_part,
            ):
                anchor = (artifact.row, artifact.col)
                digest = hashlib.sha256(artifact.payload).digest()
                key = (anchor, digest)
                if key in seen:
                    continue
                seen.add(key)
                latex = self._image_equation_decoder.decode(
                    artifact.payload,
                    part_name=artifact.part_name,
                )
                image_base64 = serialize_office_image(
                    artifact.payload,
                    part_name=artifact.part_name,
                    content_type=None,
                )
                if latex is None and image_base64 is None:
                    continue
                images.append(
                    SheetImage(
                        anchor=anchor,
                        image_base64=image_base64,
                        latex=latex,
                        order=artifact.order,
                    )
                )

        for image_order, item in enumerate(
            getattr(sheet, "_images", []),  # type: ignore[attr-defined]
            start=10_000,
        ):
            try:
                image: XlsImage = cast(XlsImage, item)
                raw_image = self._raw_sheet_image(image)
                if raw_image is None:
                    continue
                payload, part_name, content_type = raw_image
                anchor = self._get_anchor_pos(item.anchor)
                key = (anchor, hashlib.sha256(payload).digest())
                if key in seen:
                    continue
                seen.add(key)
                latex = self._image_equation_decoder.decode(
                    payload,
                    part_name=part_name,
                    content_type=content_type,
                )
                image_base64 = serialize_office_image(
                    payload,
                    part_name=part_name,
                    content_type=content_type,
                )
                if latex is None and image_base64 is None:
                    continue
                images.append(
                    SheetImage(
                        anchor=anchor,
                        image_base64=image_base64,
                        latex=latex,
                        order=image_order,
                    )
                )
            except Exception as e:
                logger.error(f"无法从 Excel 工作表中提取图片，错误信息：{e}")

        return images

    def _map_math_formulas_to_cells(self, sheet: Worksheet) -> FormulaMap:
        """从 worksheet drawing 恢复按 cell anchor 分组的 OMML 公式。"""
        math_map = collections.defaultdict(list)
        self._omml_shape_ids = set()
        self._omml_artifacts = []
        if not self.zf:
            return math_map

        # Find drawing relation
        drawing_rel = None
        if hasattr(sheet, "_rels"):
            for rel in sheet._rels:
                if rel.Type.endswith("/relationships/drawing"):
                    drawing_rel = rel
                    break

        if not drawing_rel:
            return math_map

        # Resolve path
        # Assuming relative path from worksheets/sheetX.xml to drawings/drawingY.xml
        # Usually target is like "../drawings/drawing1.xml"
        target = drawing_rel.Target
        if target.startswith("../"):
            path = target.replace("../", "xl/")  # simplistic resolution
        elif target.startswith("/"):
            path = target[1:]
        else:
            path = f"xl/worksheets/{target}"  # unlikely but default relative

        # Check if file exists in zip
        if path not in self.zf.namelist():
            # Try generic match if simplistic resolution failed
            # drawing1.xml -> xl/drawings/drawing1.xml
            basename = target.split("/")[-1]
            path = f"xl/drawings/{basename}"
            if path not in self.zf.namelist():
                return math_map

        try:
            with self.zf.open(path) as f:
                tree = ET.parse(f)
                root = tree.getroot()

            # Namespaces
            ns = {
                "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
                "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
                "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
            }

            # Iterate TwoCellAnchor and OneCellAnchor
            for anchor_tag in ["twoCellAnchor", "oneCellAnchor"]:
                for anchor in root.findall(f".//xdr:{anchor_tag}", ns):
                    # Get position
                    from_node = anchor.find("xdr:from", ns)
                    if from_node is None:
                        continue
                    col_node = from_node.find("xdr:col", ns)
                    row_node = from_node.find("xdr:row", ns)
                    if col_node is None or row_node is None:
                        continue

                    r = int(row_node.text)
                    c = int(col_node.text)

                    # Look for math content
                    # Usually in graphicalFrame -> graphic -> graphicData -> oMathPara
                    # But simpler to search descendant m:oMath
                    maths = anchor.findall(".//m:oMath", ns)
                    anchor_latex: list[str] = []
                    for math in maths:
                        # # Simple text extraction
                        # text = "".join(math.itertext())
                        # if text.strip():
                        #     # Wrap in latex block indicator if needed, or just plain text
                        #     # User asked for formula, assuming latex-like visual or text is acceptable
                        #     # Adding simple latex-like wrapper
                        #     math_map[(r, c)].append(f"${text}$")
                        latex = str(oMath2Latex(math)).strip()
                        if latex:
                            math_map[(r, c)].append(latex)
                            anchor_latex.append(latex)
                            self._omml_artifacts.append(
                                (r, c, latex, len(self._omml_artifacts))
                            )
                    if anchor_latex:
                        for node in anchor.findall(".//xdr:cNvPr", ns):
                            shape_id = node.get("id")
                            if shape_id:
                                self._omml_shape_ids.add(shape_id)

        except Exception as e:
            logger.warning(f"Error parsing math formulas: {e}")

        return math_map

    def _get_anchor_pos(self, anchor):
        """Helper to get (row, col) from anchor."""
        if hasattr(anchor, "_from"):
            return anchor._from.row, anchor._from.col
        return None, None




    def _extract_chart_range_formula(self, value_source) -> str | None:
        if value_source is None:
            return None

        for attr_name in ("numRef", "strRef", "multiLvlStrRef"):
            ref = getattr(value_source, attr_name, None)
            formula = getattr(ref, "f", None)
            if formula:
                return formula

        return None

    def _iter_chart_reference_formulas(self, chart):
        for series in getattr(chart, "ser", []):
            for attr_name in ("cat", "val", "xVal", "yVal", "bubbleSize"):
                formula = self._extract_chart_range_formula(getattr(series, attr_name, None))
                if formula:
                    yield formula

            tx = getattr(series, "tx", None)
            tx_formula = getattr(getattr(tx, "strRef", None), "f", None)
            if tx_formula:
                yield tx_formula

    def _parse_chart_reference_formula(self, formula: str, sheet_title: str) -> tuple[list[int], list[int]] | None:
        try:
            (
                formula_sheet_name,
                (
                    min_col,
                    min_row,
                    max_col,
                    max_row,
                ),
            ) = range_to_tuple(formula)
        except ValueError:
            logger.debug("Skip unsupported chart reference formula: {}", formula)
            return None

        if formula_sheet_name != sheet_title:
            logger.debug(
                "Skip chart reference formula from different sheet: {} != {}",
                formula_sheet_name,
                sheet_title,
            )
            return None

        if not all(isinstance(bound, int) for bound in (min_col, min_row, max_col, max_row)):
            logger.debug(
                "Skip chart reference formula with open-ended bounds: {}",
                formula,
            )
            return None

        rows = list(range(min_row - 1, max_row))
        cols = list(range(min_col - 1, max_col))
        return rows, cols

    def _collect_chart_source_axes(self, sheet: Worksheet, chart) -> tuple[list[int], list[int]] | None:
        referenced_rows = set()
        referenced_cols = set()
        formulas_found = False

        for formula in self._iter_chart_reference_formulas(chart):
            formulas_found = True
            parsed_axes = self._parse_chart_reference_formula(formula, sheet.title)
            if parsed_axes is None:
                return None

            rows, cols = parsed_axes
            referenced_rows.update(rows)
            referenced_cols.update(cols)

        if not formulas_found or not referenced_rows or not referenced_cols:
            return None

        return sorted(referenced_rows), sorted(referenced_cols)



    def _find_charts_in_sheet(self, sheet: Worksheet) -> list[AnchoredBlock]:
        chart_artifacts = []
        for order, chart in enumerate(getattr(sheet, "_charts", [])):
            axes = self._collect_chart_source_axes(sheet, chart)
            if axes is None:
                logger.debug(
                    "Skip chart on sheet '{}' because chart source ranges are unsupported",
                    sheet.title,
                )
                continue

            rows, cols = axes
            chart_table = self._build_synthetic_table_from_sheet_selection(
                sheet,
                rows,
                cols,
            )
            anchor_row, anchor_col = self._get_anchor_pos(getattr(chart, "anchor", None))
            chart_artifacts.append(
                (
                    self._get_block_sort_anchor(anchor_row, anchor_col),
                    10_000 + order,
                    {
                        "type": BlockType.CHART,
                        "content": render_spreadsheet_table(chart_table),
                    },
                )
            )

        return chart_artifacts





















    def _resolve_cell_image(self, text: str) -> str:
        """解析 WPS DISPIMG 单元格函数并返回图片或公式 HTML。"""
        match = re.search(r'"([^"]+)"', text)
        if match:
            image_id = match.group(1)

        else:
            logger.error(f"无法从单元格文本中提取图片 ID，文本内容：{text}")
            return ""

        cell_image_map = self._load_cell_image_mappings()

        zip_target_path = posixpath.normpath(posixpath.join("xl", cell_image_map.get(image_id, "")))
        if self.zf is None or zip_target_path not in self.zf.namelist():
            logger.warning(f"图片目标文件不存在，image_id={image_id}, target={zip_target_path}")
            return ""

        try:
            image_payload = self._read_xlsx_image_member(zip_target_path)
            if image_payload is None:
                return ""
            latex = self._image_equation_decoder.decode(
                image_payload,
                part_name=zip_target_path,
            )
            if latex:
                return EQUATION_BOOKENDS.format(EQ=latex)
            img_base64 = serialize_office_image(
                image_payload,
                part_name=zip_target_path,
                content_type=None,
            )
            return rf'<img src="{img_base64}" />' if img_base64 is not None else ""
        except Exception as e:
            logger.warning(f"读取单元格图片失败，image_id={image_id}, target={zip_target_path}, error={e}")
            return ""

    def _load_cell_image_mappings(self):
        if self.cell_image_map:
            return self.cell_image_map

        if self.zf is None:
            return {}
        cell_image_embed_to_name = {}
        cellimages_path = "xl/cellimages.xml"
        rels_path = "xl/_rels/cellimages.xml.rels"
        if cellimages_path not in self.zf.namelist() or rels_path not in self.zf.namelist():
            return {}

        try:
            with self.zf.open(cellimages_path) as f:
                root = ET.parse(f).getroot()

            ns = {
                "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
                "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
                "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                "etc": "http://www.wps.cn/officeDocument/2017/etCustomData",
            }

            for cell_image in root.findall(".//etc:cellImage", ns):
                c_nv_pr = cell_image.find(".//xdr:cNvPr", ns)
                blip = cell_image.find(".//a:blip", ns)
                if c_nv_pr is None or blip is None:
                    continue

                image_name = c_nv_pr.attrib.get("name")
                embed_id = blip.attrib.get(f"{{{ns['r']}}}embed")
                if image_name and embed_id:
                    cell_image_embed_to_name[embed_id] = image_name

            with self.zf.open(rels_path) as f:
                rel_root = ET.parse(f).getroot()

            rel_ns = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
            for rel in rel_root.findall("pr:Relationship", rel_ns):
                rel_id = rel.attrib.get("Id")
                target = rel.attrib.get("Target")
                if rel_id and target:
                    image_name = cell_image_embed_to_name.get(rel_id)
                    if not image_name:
                        logger.warning(f"跳过缺少 cellImage 名称映射的关系: {rel_id}")
                        continue
                    self.cell_image_map[image_name] = target

        except Exception as e:
            logger.warning(f"解析 cellimages 映射失败: {e}")
            return {}

        return self.cell_image_map

    @staticmethod
    def _get_sheet_content_layer(sheet: Worksheet):
        """根据工作表的可见性返回对应的内容层。

        若工作表可见，返回 None（默认层）；否则返回 INVISIBLE 层。

        参数：
            sheet: 待检查的工作表。

        返回：
            ContentLayer.INVISIBLE 或 None。
        """
        return None if sheet.sheet_state == Worksheet.SHEETSTATE_VISIBLE else "INVISIBLE"
