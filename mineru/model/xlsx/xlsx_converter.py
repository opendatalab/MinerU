import collections
import html
import posixpath
import zipfile
import re
import xml.etree.ElementTree as ET
from typing import BinaryIO, Annotated, cast


from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.drawing.image import Image as XlsImage
from PIL import Image
from loguru import logger
from pydantic import PositiveInt, Field, BaseModel, NonNegativeInt
from pydantic.dataclasses import dataclass

from mineru.utils.check_sys_env import is_windows_environment
from mineru.utils.enum_class import BlockType
from mineru.utils.pdf_reader import image_to_b64str
from mineru.model.docx.tools.math.omml import oMath2Latex


@dataclass
class DataRegion:
    """表示工作表中非空单元格的边界矩形区域。"""

    min_row: Annotated[
        PositiveInt, Field(description="Smallest row index (1-based index).")
    ]
    max_row: Annotated[
        PositiveInt, Field(description="Largest row index (1-based index).")
    ]
    min_col: Annotated[
        PositiveInt, Field(description="Smallest column index (1-based index).")
    ]
    max_col: Annotated[
        PositiveInt, Field(description="Largest column index (1-based index).")
    ]

    def width(self) -> PositiveInt:
        """返回数据区域的列数。"""
        return self.max_col - self.min_col + 1

    def height(self) -> PositiveInt:
        """返回数据区域的行数。"""
        return self.max_row - self.min_row + 1


class ExcelCell(BaseModel):
    """表示一个 Excel 单元格。

    属性：
        row: 单元格的行号。
        col: 单元格的列号。
        text: 单元格的文本内容。
        row_span: 单元格跨越的行数。
        col_span: 单元格跨越的列数。
    """

    row: int
    col: int
    text: str
    row_span: int
    col_span: int
    styles: dict = Field(default_factory=dict)
    media: list[str] = Field(default_factory=list)


class ExcelTable(BaseModel):
    """表示工作表上的一个 Excel 表格。

    属性：
        anchor: 表格左上角单元格的列和行索引（从0开始）。
        num_rows: 表格的行数。
        num_cols: 表格的列数。
        data: 表格数据，以 ExcelCell 对象列表的形式表示。
    """

    anchor: tuple[NonNegativeInt, NonNegativeInt]
    num_rows: int
    num_cols: int
    data: list[ExcelCell]


class XlsxConverter:
    def __init__(
        self,
        treat_singleton_as_text: bool = True,
        gap_tolerance: int = 0,
    ):
        self.workbook = None
        self.zf = None
        self.treat_singleton_as_text = treat_singleton_as_text
        self.gap_tolerance = gap_tolerance
        self.pages = []
        self.cur_page = []
        self.pages.append(self.cur_page)
        self.image_map = {}
        self.cell_image_map: dict[tuple[int, int], list[str]] = collections.defaultdict(
            list
        )
        self._cell_image_id_to_embed: dict[str, str] = {}
        self._cell_image_embed_to_target: dict[str, str] = {}

    def convert(
        self,
        file_stream: BinaryIO,
    ):
        if hasattr(file_stream, "seek"):
            file_stream.seek(0)

        try:
            self.zf = zipfile.ZipFile(file_stream)
        except Exception as e:
            logger.warning(f"Failed to open zip file: {e}")
            self.zf = None

        if hasattr(file_stream, "seek"):
            file_stream.seek(0)

        self.workbook = load_workbook(filename=file_stream, data_only=True)
        if self.workbook is not None:
            # 遍历工作簿中的所有工作表
            for idx, name in enumerate(self.workbook.sheetnames):
                logger.info(f"正在处理第 {idx + 1} 个工作表：{name}")
                sheet = self.workbook[name]
                self._convert_sheet(sheet)
                self.cur_page = []
                self.pages.append(self.cur_page)
        else:
            logger.error("工作簿未初始化。")

        if self.zf:
            self.zf.close()
            self.zf = None

    def _convert_sheet(self, sheet):
        if isinstance(sheet, Worksheet):
            # Pre-calc maps
            self.math_map = self._map_math_formulas_to_cells(sheet)

            used_cells = self._find_tables_in_sheet(sheet)  # 提取表格
            self._find_images_in_sheet(sheet, used_cells)  # 提取图片

    def _map_math_formulas_to_cells(self, sheet: Worksheet) -> dict:
        """Parse drawings to find math formulas and map them to cells."""
        math_map = collections.defaultdict(list)
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

        except Exception as e:
            logger.warning(f"Error parsing math formulas: {e}")

        return math_map

    def _get_anchor_pos(self, anchor):
        """Helper to get (row, col) from anchor."""
        if hasattr(anchor, "_from"):
            return anchor._from.row, anchor._from.col
        return None, None

    def _find_tables_in_sheet(self, sheet: Worksheet) -> set[tuple[int, int]]:
        used_cells = set()
        if self.workbook is not None:
            content_layer = self._get_sheet_content_layer(sheet)  # 检测工作表的可见性
            tables = self._find_data_tables(sheet)  # 检测工作表中的所有数据表格

            for excel_table in tables:
                # Record used cells
                anchor_c, anchor_r = excel_table.anchor
                for cell in excel_table.data:
                    used_cells.add((anchor_r + cell.row, anchor_c + cell.col))

                # 若只有一个单元格且启用了单元格文本选项，则作为文本添加
                if self.treat_singleton_as_text and len(excel_table.data) == 1:
                    self.cur_page.append(
                        {
                            "type": BlockType.TEXT,
                            "content": excel_table.data[0].text,
                        }
                    )

                else:
                    table_html_str = self.excel_table_to_html(excel_table)
                    self.cur_page.append(
                        {
                            "type": BlockType.TABLE,
                            "content": table_html_str,
                        }
                    )

        return used_cells

    def excel_table_to_html(self, excel_table) -> str:
        """
        将 ExcelTable 转换为 HTML 表格字符串，保留合并单元格结构。
        """
        # 1. 创建坐标到单元格的映射，方便快速查找
        cell_map = {(c.row, c.col): c for c in excel_table.data}

        # 2. 用于记录已被合并单元格占据的位置，避免重复生成 td
        covered_cells = set()

        # 开始构建 HTML
        lines = ["<table>"]  # 可以根据需要添加样式类或属性

        for r in range(excel_table.num_rows):
            lines.append("  <tr>")
            for c in range(excel_table.num_cols):
                # 如果当前位置已被之前的合并单元格占据，则跳过
                if (r, c) in covered_cells:
                    continue

                # 获取当前位置的单元格
                cell = cell_map.get((r, c))

                if cell:
                    # 确定标签类型：第一行通常作为表头
                    tag = "th" if cell.row == 0 else "td"

                    # 构建属性列表 (rowspan, colspan)
                    attrs = []
                    if cell.row_span > 1:
                        attrs.append(f'rowspan="{cell.row_span}"')
                    if cell.col_span > 1:
                        attrs.append(f'colspan="{cell.col_span}"')

                    # 标记该单元格覆盖的所有位置为已占用
                    for ir in range(cell.row_span):
                        for ic in range(cell.col_span):
                            covered_cells.add((r + ir, c + ic))

                    # 拼接属性字符串
                    attr_str = " " + " ".join(attrs) if attrs else ""

                    # 生成样式字符串
                    style_parts = []
                    for k, v in cell.styles.items():
                        style_parts.append(f"{k}: {v}")
                    if style_parts:
                        attr_str += f' style="{"; ".join(style_parts)}"'

                    # 生成 HTML 单元格，注意对文本进行 HTML 转义
                    text_content = html.escape(cell.text) if cell.text else ""

                    # 添加媒体内容 (Images, Math)
                    media_content = ""
                    if cell.media:
                        media_content = "<br>".join(cell.media)
                        if text_content:
                            text_content += "<br>" + media_content
                        else:
                            text_content = media_content

                    lines.append(f"    <{tag}{attr_str}>{text_content}</{tag}>")
                else:
                    # 如果既没被覆盖，又没有数据对象（理论上 _find_table_bounds 逻辑应避免此情况），生成空单元格
                    lines.append("    <td></td>")

            lines.append("  </tr>")

        lines.append("</table>")
        return "\n".join(lines)

    def _find_images_in_sheet(
        self, sheet: Worksheet, used_cells: set[tuple[int, int]] = None
    ):
        if self.workbook is not None:
            content_layer = self._get_sheet_content_layer(sheet)

            # 遍历工作表中的所有嵌入图片
            for item in sheet._images:  # type: ignore[attr-defined]

                # Check if image is already used in a table
                if used_cells:
                    r, c = self._get_anchor_pos(item.anchor)
                    if r is not None and c is not None and (r, c) in used_cells:
                        continue

                try:
                    image: XlsImage = cast(XlsImage, item)
                    pil_image = Image.open(image.ref)  # type: ignore[arg-type]
                    if pil_image.format in ("WMF", "EMF"):
                        if is_windows_environment():
                            # 在 Windows 上，Pillow 依赖底层的 Image.core.drawwmf 渲染
                            # 有时需要显式调用 .load() 确保矢量图被光栅化到内存中
                            try:
                                pil_image.load()
                                img_base64 = image_to_b64str(
                                    pil_image, image_format="PNG"
                                )
                            except OSError as e:
                                logger.warning(
                                    f"Failed to render {pil_image.format} image: {e}, size: {pil_image.size}. Using placeholder instead."
                                )
                                # 如果渲染失败，创建与原图同样大小的浅灰色占位图
                                placeholder = Image.new(
                                    "RGB", pil_image.size, (240, 240, 240)
                                )
                                img_base64 = image_to_b64str(
                                    placeholder, image_format="JPEG"
                                )
                        else:
                            logger.warning(
                                f"Skipping {pil_image.format} image on non-Windows environment, size: {pil_image.size}"
                            )
                            # 创建与原图同样大小的浅灰色占位图
                            placeholder = Image.new(
                                "RGB", pil_image.size, (240, 240, 240)
                            )
                            img_base64 = image_to_b64str(
                                placeholder, image_format="JPEG"
                            )
                    else:
                        # 处理常规图片
                        if pil_image.mode != "RGB":
                            # RGBA, P, L 等模式保留原貌并存为 PNG (PNG支持透明度)
                            img_base64 = image_to_b64str(pil_image, image_format="PNG")
                        else:
                            # 纯 RGB 图片存为 JPEG 以减小体积
                            img_base64 = image_to_b64str(pil_image, image_format="JPEG")
                    image_block = {
                        "type": BlockType.IMAGE,
                        "content": img_base64,
                    }
                    self.cur_page.append(image_block)

                except Exception as e:
                    logger.error(f"无法从 Excel 工作表中提取图片，错误信息：{e}")

        return

    def _find_data_tables(self, sheet: Worksheet) -> list[ExcelTable]:
        """在 Excel 工作表中查找所有紧凑的矩形数据表格。

        参数：
            sheet: 待解析的 Excel 工作表。

        返回：
            表示所有数据表格的 ExcelTable 对象列表。
        """
        bounds: DataRegion = self._find_true_data_bounds(sheet)  # 获取真实数据边界
        tables: list[ExcelTable] = []  # 存储已发现的表格
        visited: set[tuple[int, int]] = set()  # 记录已访问的单元格

        # 仅在真实数据边界范围内进行扫描
        for ri, row in enumerate(
            sheet.iter_rows(
                min_row=bounds.min_row,
                max_row=bounds.max_row,
                min_col=bounds.min_col,
                max_col=bounds.max_col,
                values_only=False,
            ),
            start=bounds.min_row - 1,  # 转换为0-based索引
        ):
            for rj, cell in enumerate(row, start=bounds.min_col - 1):
                # 跳过空单元格或已访问的单元格
                if cell.value is None or (ri, rj) in visited:
                    continue

                # 从当前单元格出发，通过洪水填充算法确定所属表格的边界
                table_bounds, visited_cells = self._find_table_bounds(
                    sheet, ri, rj, bounds.max_row, bounds.max_col
                )
                visited.update(visited_cells)  # 将已访问单元格加入全局记录
                tables.append(table_bounds)

        return tables

    def _find_true_data_bounds(self, sheet: Worksheet) -> DataRegion:
        """查找工作表中真实的数据边界（最小/最大行列）。

        该函数扫描所有单元格，找到包含所有非空单元格或合并单元格区域的
        最小矩形范围，返回边界的行列索引。

        参数：
            sheet: 待分析的工作表。

        返回：
            覆盖所有数据和合并单元格的最小矩形区域 DataRegion。
            若工作表为空，则默认返回 (1, 1, 1, 1)。
        """
        min_row, min_col = None, None
        max_row, max_col = 0, 0

        # 遍历所有有值的单元格，动态更新边界
        for cell in sheet._cells.values():
            if cell.value is not None:
                r, c = cell.row, cell.column
                min_row = r if min_row is None else min(min_row, r)
                min_col = c if min_col is None else min(min_col, c)
                max_row = max(max_row, r)
                max_col = max(max_col, c)

        # 将合并单元格的范围也纳入边界计算
        for merged in sheet.merged_cells.ranges:
            min_row = (
                merged.min_row if min_row is None else min(min_row, merged.min_row)
            )
            min_col = (
                merged.min_col if min_col is None else min(min_col, merged.min_col)
            )
            max_row = max(max_row, merged.max_row)
            max_col = max(max_col, merged.max_col)

        # 若工作表中没有任何数据，默认返回 (1, 1, 1, 1)
        if min_row is None or min_col is None:
            min_row = min_col = max_row = max_col = 1

        return DataRegion(min_row, max_row, min_col, max_col)

    def _find_table_bounds(
        self,
        sheet: Worksheet,
        start_row: int,
        start_col: int,
        max_row: int,
        max_col: int,
    ) -> tuple[ExcelTable, set[tuple[int, int]]]:
        """使用洪水填充（BFS）策略确定表格边界。

        该方法通过广度优先搜索（BFS）算法识别 Excel 工作表中连续的非空单元格区域，
        能够准确检测非矩形表格（如 L 形、错位列等），并支持通过间隔容忍度
        连接相邻但不直接相连的单元格。

        算法分两个阶段执行：
        1. 洪水填充阶段：使用 BFS 从给定位置出发，找出所有相连的单元格。
        2. 数据提取阶段：构建矩形边界框并提取单元格数据，正确处理合并单元格。

        参数：
            sheet: 待分析的 Excel 工作表。
            start_row: 洪水填充起始行索引（从0开始）。
            start_col: 洪水填充起始列索引（从0开始）。
            max_row: 工作表中可考虑的最大行索引（从0开始）。
            max_col: 工作表中可考虑的最大列索引（从0开始）。

        返回：
            一个元组，包含：
                - ExcelTable：表示检测到的表格对象，含锚点位置、尺寸和单元格数据。
                - set[tuple[int, int]]：洪水填充期间访问的所有 (行, 列) 元组集合，
                  用于防止重复扫描。

        说明：
            该方法遵循 GAP_TOLERANCE 选项，允许在容忍距离内将被空单元格隔开的
            单元格视为同一表格的一部分。
        """

        # BFS 队列，存储待处理的 (行, 列) 坐标
        queue = collections.deque([(start_row, start_col)])

        # 记录当前表格内已访问的单元格（避免重复加入队列）
        # 调用方维护全局 visited 集合，防止重复启动新表格
        table_cells: set[tuple[int, int]] = set()
        table_cells.add((start_row, start_col))

        # 动态记录当前表格的行列边界
        min_r, max_r = start_row, start_row
        min_c, max_c = start_col, start_col

        def has_content(r, c):
            """检查指定单元格（0-based索引）是否有内容（有值或属于合并区域）。"""
            if r < 0 or c < 0 or r > max_row or c > max_col:
                return False

            # 1. 检查单元格直接值
            cell = sheet.cell(row=r + 1, column=c + 1)
            if cell.value is not None:
                return True

            # 2. 检查是否属于某个合并单元格区域
            for mr in sheet.merged_cells.ranges:
                if cell.coordinate in mr:
                    return True
            return False

        # --- 第一阶段：洪水填充（连通性检测）---
        while queue:
            curr_r, curr_c = queue.popleft()

            # 动态更新表格边界
            min_r = min(min_r, curr_r)
            max_r = max(max_r, curr_r)
            min_c = min(min_c, curr_c)
            max_c = max(max_c, curr_c)

            # 四个方向（上、下、左、右）的邻居检测
            directions = [
                (0, 1),  # 右
                (0, -1),  # 左
                (1, 0),  # 下
                (-1, 0),  # 上
            ]

            for dr, dc in directions:
                # 在容忍距离范围内逐步检查邻居（优先检查最近的）
                for step in range(1, self.gap_tolerance + 2):
                    nr, nc = curr_r + (dr * step), curr_c + (dc * step)

                    if (nr, nc) in table_cells:
                        break  # 已属于当前表格，不跨越继续查找

                    if has_content(nr, nc):
                        table_cells.add((nr, nc))
                        queue.append((nr, nc))
                        # 在该方向找到连接点，停止扩展间隔
                        break

        # --- 第二阶段：数据提取（语义网格构建）---
        data = []

        # 识别被合并单元格"遮蔽"的单元格（即非合并区域左上角的单元格）
        hidden_merge_cells = set()
        for mr in sheet.merged_cells.ranges:
            mr_min_r, mr_min_c = mr.min_row - 1, mr.min_col - 1
            mr_max_r, mr_max_c = mr.max_row - 1, mr.max_col - 1
            for r in range(mr_min_r, mr_max_r + 1):
                for c in range(mr_min_c, mr_max_c + 1):
                    if r == mr_min_r and c == mr_min_c:
                        continue  # 左上角单元格保留，其余标记为隐藏
                    hidden_merge_cells.add((r, c))

        # 遍历发现区域的边界框（bbox内部的空格作为空单元格保留，维持矩形布局）
        for ri in range(min_r, max_r + 1):
            for rj in range(min_c, max_c + 1):
                # 跳过被合并单元格遮蔽的单元格（非左上角）
                if (ri, rj) in hidden_merge_cells:
                    continue

                cell = sheet.cell(row=ri + 1, column=rj + 1)
                cell_text = str(cell.value) if cell.value is not None else ""

                if "DISPIMG" in cell_text:
                    self._get_cell_image(ri, rj, cell_text)

                # 计算合并跨度（默认为 1x1）
                row_span = 1
                col_span = 1
                for mr in sheet.merged_cells.ranges:
                    if (ri + 1) == mr.min_row and (rj + 1) == mr.min_col:
                        row_span = (mr.max_row - mr.min_row) + 1
                        col_span = (mr.max_col - mr.min_col) + 1
                        break

                data.append(
                    ExcelCell(
                        row=ri - min_r,  # 相对于表格起始行的偏移
                        col=rj - min_c,  # 相对于表格起始列的偏移
                        text=cell_text,
                        row_span=row_span,
                        col_span=col_span,
                        styles=self._extract_cell_style(cell),
                        media=self._get_cell_media(ri, rj),
                    )
                )

        # 返回给调用方的 visited_cells 严格为包含数据/合并的单元格，
        # 使主循环不会重复扫描已处理的单元格。
        return (
            ExcelTable(
                anchor=(min_c, min_r),
                num_rows=max_r + 1 - min_r,
                num_cols=max_c + 1 - min_c,
                data=data,
            ),
            table_cells,
        )

    def _get_cell_image(self, r, c, text):
        match = re.search(r'"([^"]+)"', text)
        if match:
            image_id = match.group(1)

        else:
            logger.error(f"无法从单元格文本中提取图片 ID，文本内容：{text}")
            return None

        id_to_embed, embed_to_target = self._load_cell_image_mappings()
        if not id_to_embed:
            return None

        embed_id = id_to_embed.get(image_id)
        if not embed_id:
            logger.warning(f"在 cellimages.xml 中未找到 name={image_id} 的 cellImage")
            return None

        target = embed_to_target.get(embed_id)
        if not target:
            logger.warning(
                f"在 cellimages.xml.rels 中未找到 Id={embed_id} 的 Target 映射"
            )
            return None

        zip_target_path = posixpath.normpath(posixpath.join("xl", target))
        if self.zf is None or zip_target_path not in self.zf.namelist():
            logger.warning(
                f"图片目标文件不存在，image_id={image_id}, embed_id={embed_id}, target={zip_target_path}"
            )
            return None

        try:
            with self.zf.open(zip_target_path) as image_file:
                pil_image = Image.open(image_file)
                pil_image.load()

                if pil_image.mode != "RGB":
                    img_base64 = image_to_b64str(pil_image, image_format="PNG")
                else:
                    img_base64 = image_to_b64str(pil_image, image_format="JPEG")
                self.cell_image_map[(r, c)].append(f'<img src="{img_base64}" />')
        except Exception as e:
            logger.warning(
                f"读取单元格图片失败，image_id={image_id}, target={zip_target_path}, error={e}"
            )
            return None

        return zip_target_path

    def _load_cell_image_mappings(self) -> tuple[dict[str, str], dict[str, str]]:
        if self._cell_image_id_to_embed and self._cell_image_embed_to_target:
            return self._cell_image_id_to_embed, self._cell_image_embed_to_target

        if self.zf is None:
            return {}, {}

        cellimages_path = "xl/cellimages.xml"
        rels_path = "xl/_rels/cellimages.xml.rels"
        if (
            cellimages_path not in self.zf.namelist()
            or rels_path not in self.zf.namelist()
        ):
            return {}, {}

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
                embed_id = blip.attrib.get(f'{{{ns["r"]}}}embed')
                if image_name and embed_id:
                    self._cell_image_id_to_embed[image_name] = embed_id

            with self.zf.open(rels_path) as f:
                rel_root = ET.parse(f).getroot()

            rel_ns = {
                "pr": "http://schemas.openxmlformats.org/package/2006/relationships"
            }
            for rel in rel_root.findall("pr:Relationship", rel_ns):
                rel_id = rel.attrib.get("Id")
                target = rel.attrib.get("Target")
                if rel_id and target:
                    self._cell_image_embed_to_target[rel_id] = target

        except Exception as e:
            logger.warning(f"解析 cellimages 映射失败: {e}")
            return {}, {}

        return self._cell_image_id_to_embed, self._cell_image_embed_to_target



    def _extract_cell_style(self, cell):
        """Extract styles from an openpyxl cell."""
        style = {}
        if cell.font:
            if cell.font.b:
                style["font-weight"] = "bold"
            if cell.font.i:
                style["font-style"] = "italic"
            if cell.font.u:
                style["text-decoration"] = "underline"
            if cell.font.strike:
                style["text-decoration"] = "line-through"
            if (
                cell.font.color
                and hasattr(cell.font.color, "rgb")
                and cell.font.color.rgb
            ):
                # Color might be ARGB "FF000000"
                color = cell.font.color.rgb
                if isinstance(color, str) and len(color) == 8:
                    style["color"] = "#" + color[2:]
                elif isinstance(color, str):
                    style["color"] = "#" + color

        if cell.alignment:
            if cell.alignment.horizontal:
                style["text-align"] = cell.alignment.horizontal
            if cell.alignment.vertical:
                style["vertical-align"] = cell.alignment.vertical

        if cell.fill and cell.fill.patternType == "solid" and cell.fill.fgColor:
            # handle bg color
            color = cell.fill.fgColor.rgb
            if (
                hasattr(cell.fill.fgColor, "type")
                and cell.fill.fgColor.type == "rgb"
                and color
            ):
                if isinstance(color, str) and len(color) == 8:
                    style["background-color"] = "#" + color[2:]
        return style

    def _get_cell_media(self, r, c) -> list[str]:
        """Get media content (images, math) for a given cell (0-based row/col)."""
        media = []
        if (r, c) in self.cell_image_map:
            media.extend(self.cell_image_map[(r, c)])

        # Images
        if hasattr(self, "image_map") and (r, c) in self.image_map:
            for img in self.image_map[(r, c)]:
                # Convert image to base64 html tag
                try:
                    pil_image = Image.open(img.ref)
                    if pil_image.mode != "RGB":
                        img_base64 = image_to_b64str(pil_image, image_format="PNG")
                    else:
                        img_base64 = image_to_b64str(pil_image, image_format="JPEG")
                    media.append(f'<img src="{img_base64}" />')
                except Exception as e:
                    logger.warning(f"Failed to render cell image: {e}")

        # Math
        if hasattr(self, "math_map") and (r, c) in self.math_map:
            media.extend(self.math_map[(r, c)])

        return media

    @staticmethod
    def _get_sheet_content_layer(sheet: Worksheet):
        """根据工作表的可见性返回对应的内容层。

        若工作表可见，返回 None（默认层）；否则返回 INVISIBLE 层。

        参数：
            sheet: 待检查的工作表。

        返回：
            ContentLayer.INVISIBLE 或 None。
        """
        return (
            None if sheet.sheet_state == Worksheet.SHEETSTATE_VISIBLE else "INVISIBLE"
        )
