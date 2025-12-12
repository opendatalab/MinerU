# Copyright (c) Opendatalab. All rights reserved.

from loguru import logger
from bs4 import BeautifulSoup

from mineru.backend.vlm.vlm_middle_json_mkcontent import merge_para_with_text
from mineru.utils.enum_class import BlockType, SplitFlag


CONTINUATION_MARKERS = ["(续)", "(续表)", "(continued)", "(cont.)"]


def full_to_half(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters, numbers and punctuation (FF01-FF5E)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)


def calculate_table_total_columns(soup):
    """计算表格的总列数，通过分析整个表格结构来处理rowspan和colspan

    Args:
        soup: BeautifulSoup解析的表格

    Returns:
        int: 表格的总列数
    """
    rows = soup.find_all("tr")
    if not rows:
        return 0

    # 创建一个矩阵来跟踪每个位置的占用情况
    max_cols = 0
    occupied = {}  # {row_idx: {col_idx: True}}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])

        if row_idx not in occupied:
            occupied[row_idx] = {}

        for cell in cells:
            # 找到下一个未被占用的列位置
            while col_idx in occupied[row_idx]:
                col_idx += 1

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            # 标记被这个单元格占用的所有位置
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True

            col_idx += colspan
            max_cols = max(max_cols, col_idx)

    return max_cols


def calculate_row_columns(row):
    """
    计算表格行的实际列数，考虑colspan属性

    Args:
        row: BeautifulSoup的tr元素对象

    Returns:
        int: 行的实际列数
    """
    cells = row.find_all(["td", "th"])
    column_count = 0

    for cell in cells:
        colspan = int(cell.get("colspan", 1))
        column_count += colspan

    return column_count


def calculate_visual_columns(row):
    """
    计算表格行的视觉列数（实际td/th单元格数量，不考虑colspan）

    Args:
        row: BeautifulSoup的tr元素对象

    Returns:
        int: 行的视觉列数（实际单元格数）
    """
    cells = row.find_all(["td", "th"])
    return len(cells)


def detect_table_headers(soup1, soup2, max_header_rows=5):
    """
    检测并比较两个表格的表头

    Args:
        soup1: 第一个表格的BeautifulSoup对象
        soup2: 第二个表格的BeautifulSoup对象
        max_header_rows: 最大可能的表头行数

    Returns:
        tuple: (表头行数, 表头是否一致, 表头文本列表)
    """
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for i in range(min_rows):
        # 提取当前行的所有单元格
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])

        # 检查两行的结构和内容是否一致
        structure_match = True

        # 首先检查单元格数量
        if len(cells1) != len(cells2):
            structure_match = False
        else:
            # 然后检查单元格的属性和内容
            for cell1, cell2 in zip(cells1, cells2):
                colspan1 = int(cell1.get("colspan", 1))
                rowspan1 = int(cell1.get("rowspan", 1))
                colspan2 = int(cell2.get("colspan", 1))
                rowspan2 = int(cell2.get("rowspan", 1))

                # 去除所有空白字符（包括空格、换行、制表符等）
                text1 = ''.join(full_to_half(cell1.get_text()).split())
                text2 = ''.join(full_to_half(cell2.get_text()).split())

                if colspan1 != colspan2 or rowspan1 != rowspan2 or text1 != text2:
                    structure_match = False
                    break

        if structure_match:
            header_rows += 1
            row_texts = [full_to_half(cell.get_text().strip()) for cell in cells1]
            header_texts.append(row_texts)  # 添加表头文本
        else:
            headers_match = header_rows > 0  # 只有当至少匹配了一行时，才认为表头匹配
            break

    # 如果没有找到匹配的表头行，则返回失败
    if header_rows == 0:
        headers_match = False

    return header_rows, headers_match, header_texts


def can_merge_tables(current_table_block, previous_table_block):
    """判断两个表格是否可以合并"""
    # 检查表格是否有caption和footnote
    # 如果有TABLE_CAPTION类型的块,检查是否至少有一个以"(续)"结尾
    caption_blocks = [block for block in current_table_block["blocks"] if block["type"] == BlockType.TABLE_CAPTION]
    if caption_blocks:
        # 如果所有caption都不以"(续)"、"(续表)"、"(continued)"或"(cont.)"结尾,则不合并

        if not any(
                any(full_to_half(merge_para_with_text(block).strip()).lower().endswith(marker.lower())
                    for marker in CONTINUATION_MARKERS)
                for block in caption_blocks
        ):
            return False, None, None, None, None

    if any(block["type"] == BlockType.TABLE_FOOTNOTE for block in previous_table_block["blocks"]):
        return False, None, None, None, None

    # 获取两个表格的HTML内容
    current_html = ""
    previous_html = ""

    for block in current_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
            current_html = block["lines"][0]["spans"][0].get("html", "")

    for block in previous_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
            previous_html = block["lines"][0]["spans"][0].get("html", "")

    if not current_html or not previous_html:
        return False, None, None, None, None

    # 检查表格宽度差异
    x0_t1, y0_t1, x1_t1, y1_t1 = current_table_block["bbox"]
    x0_t2, y0_t2, x1_t2, y1_t2 = previous_table_block["bbox"]
    table1_width = x1_t1 - x0_t1
    table2_width = x1_t2 - x0_t2

    if abs(table1_width - table2_width) / min(table1_width, table2_width) >= 0.1:
        return False, None, None, None, None

    # 解析HTML并检查表格结构
    soup1 = BeautifulSoup(previous_html, "html.parser")
    soup2 = BeautifulSoup(current_html, "html.parser")

    # 检查整体列数匹配
    table_cols1 = calculate_table_total_columns(soup1)
    table_cols2 = calculate_table_total_columns(soup2)
    # logger.debug(f"Table columns - Previous: {table_cols1}, Current: {table_cols2}")
    tables_match = table_cols1 == table_cols2

    # 检查首末行列数匹配
    rows_match = check_rows_match(soup1, soup2)

    return (tables_match or rows_match), soup1, soup2, current_html, previous_html


def check_rows_match(soup1, soup2):
    """检查表格行是否匹配"""
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    if not (rows1 and rows2):
        return False

    # 获取第一个表的最后一行数据行
    last_row = None
    for row in reversed(rows1):
        if row.find_all(["td", "th"]):
            last_row = row
            break

    # 检测表头行数，以便获取第二个表的首个数据行
    header_count, _, _ = detect_table_headers(soup1, soup2)

    # 获取第二个表的首个数据行
    first_data_row = None
    if len(rows2) > header_count:
        first_data_row = rows2[header_count]  # 第一个非表头行

    if not (last_row and first_data_row):
        return False

    # 计算实际列数（考虑colspan）和视觉列数
    last_row_cols = calculate_row_columns(last_row)
    first_row_cols = calculate_row_columns(first_data_row)
    last_row_visual_cols = calculate_visual_columns(last_row)
    first_row_visual_cols = calculate_visual_columns(first_data_row)

    # logger.debug(f"行列数 - 前表最后一行: {last_row_cols}(视觉列数:{last_row_visual_cols}), 当前表首行: {first_row_cols}(视觉列数:{first_row_visual_cols})")

    # 同时考虑实际列数匹配和视觉列数匹配
    return last_row_cols == first_row_cols or last_row_visual_cols == first_row_visual_cols


def check_row_columns_match(row1, row2):
    # 逐个cell检测colspan属性是否一致
    cells1 = row1.find_all(["td", "th"])
    cells2 = row2.find_all(["td", "th"])
    if len(cells1) != len(cells2):
        return False
    for cell1, cell2 in zip(cells1, cells2):
        colspan1 = int(cell1.get("colspan", 1))
        colspan2 = int(cell2.get("colspan", 1))
        if colspan1 != colspan2:
            return False
    return True


def adjust_table_rows_colspan(rows, start_idx, end_idx,
                              reference_structure, reference_visual_cols,
                              target_cols, current_cols, reference_row):
    """调整表格行的colspan属性以匹配目标列数

    Args:
        rows: 表格行列表
        start_idx: 起始行索引
        end_idx: 结束行索引（不包含）
        reference_structure: 参考行的colspan结构列表
        reference_visual_cols: 参考行的视觉列数
        target_cols: 目标总列数
        current_cols: 当前总列数
        reference_row: 参考行对象
    """
    for i in range(start_idx, end_idx):
        row = rows[i]
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        current_row_cols = calculate_row_columns(row)
        if current_row_cols >= target_cols:
            continue

        # 检查是否与参考行结构匹配
        if calculate_visual_columns(row) == reference_visual_cols and check_row_columns_match(row, reference_row):
            # 尝试应用参考结构
            if len(cells) <= len(reference_structure):
                for j, cell in enumerate(cells):
                    if j < len(reference_structure) and reference_structure[j] > 1:
                        cell["colspan"] = str(reference_structure[j])
        else:
            # 扩展最后一个单元格以填补列数差异
            last_cell = cells[-1]
            current_last_span = int(last_cell.get("colspan", 1))
            last_cell["colspan"] = str(current_last_span + (target_cols - current_cols))


def perform_table_merge(soup1, soup2, previous_table_block, wait_merge_table_footnotes):
    """执行表格合并操作"""
    # 检测表头有几行，并确认表头内容是否一致
    header_count, headers_match, header_texts = detect_table_headers(soup1, soup2)
    # logger.debug(f"检测到表头行数: {header_count}, 表头匹配: {headers_match}")
    # logger.debug(f"表头内容: {header_texts}")

    # 找到第一个表格的tbody，如果没有则查找table元素
    tbody1 = soup1.find("tbody") or soup1.find("table")

    # 获取表1和表2的所有行
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")


    if rows1 and rows2 and header_count < len(rows2):
        # 获取表1最后一行和表2第一个非表头行
        last_row1 = rows1[-1]
        first_data_row2 = rows2[header_count]

        # 计算表格总列数
        table_cols1 = calculate_table_total_columns(soup1)
        table_cols2 = calculate_table_total_columns(soup2)
        if table_cols1 >= table_cols2:
            reference_structure = [int(cell.get("colspan", 1)) for cell in last_row1.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(last_row1)
            # 以表1的最后一行为参考，调整表2的行
            adjust_table_rows_colspan(
                rows2, header_count, len(rows2),
                reference_structure, reference_visual_cols,
                table_cols1, table_cols2, first_data_row2
            )

        else:  # table_cols2 > table_cols1
            reference_structure = [int(cell.get("colspan", 1)) for cell in first_data_row2.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(first_data_row2)
            # 以表2的第一个数据行为参考，调整表1的行
            adjust_table_rows_colspan(
                rows1, 0, len(rows1),
                reference_structure, reference_visual_cols,
                table_cols2, table_cols1, last_row1
            )

    # 将第二个表格的行添加到第一个表格中
    if tbody1:
        tbody2 = soup2.find("tbody") or soup2.find("table")
        if tbody2:
            # 将第二个表格的行添加到第一个表格中（跳过表头行）
            for row in rows2[header_count:]:
                row.extract()
                tbody1.append(row)

    # 添加待合并表格的footnote到前一个表格中
    for table_footnote in wait_merge_table_footnotes:
        temp_table_footnote = table_footnote.copy()
        temp_table_footnote[SplitFlag.CROSS_PAGE] = True
        previous_table_block["blocks"].append(temp_table_footnote)

    return str(soup1)


def merge_table(page_info_list):
    """合并跨页表格"""
    # 倒序遍历每一页
    for page_idx in range(len(page_info_list) - 1, -1, -1):
        # 跳过第一页，因为它没有前一页
        if page_idx == 0:
            continue

        page_info = page_info_list[page_idx]
        previous_page_info = page_info_list[page_idx - 1]

        # 检查当前页是否有表格块
        if not (page_info["para_blocks"] and page_info["para_blocks"][0]["type"] == BlockType.TABLE):
            continue

        current_table_block = page_info["para_blocks"][0]

        # 检查上一页是否有表格块
        if not (previous_page_info["para_blocks"] and previous_page_info["para_blocks"][-1]["type"] == BlockType.TABLE):
            continue

        previous_table_block = previous_page_info["para_blocks"][-1]

        # 收集待合并表格的footnote
        wait_merge_table_footnotes = [
            block for block in current_table_block["blocks"]
            if block["type"] == BlockType.TABLE_FOOTNOTE
        ]

        # 检查两个表格是否可以合并
        can_merge, soup1, soup2, current_html, previous_html = can_merge_tables(
            current_table_block, previous_table_block
        )

        if not can_merge:
            continue

        # 执行表格合并
        merged_html = perform_table_merge(
            soup1, soup2, previous_table_block, wait_merge_table_footnotes
        )

        # 更新previous_table_block的html
        for block in previous_table_block["blocks"]:
            if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
                block["lines"][0]["spans"][0]["html"] = merged_html
                break

        # 删除当前页的table
        for block in current_table_block["blocks"]:
            block['lines'] = []
            block[SplitFlag.LINES_DELETED] = True