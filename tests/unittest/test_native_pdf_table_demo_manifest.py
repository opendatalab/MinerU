"""验证 demo/pdfs 少线表高置信门的可移植真实文档真值。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from mineru.model.flash import PdfModel
from mineru.utils.native_pdf_table import (
    NativeTableInput,
    coerce_native_table_rectangles,
    coerce_native_table_rules,
    recover_native_pdf_table,
)
from mineru.utils.pdf_document import PDFDocument


_PROJECT_ROOT = Path(__file__).parents[2]
_MANIFEST_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "native_pdf_table_demo_manifest.json"


def _is_html_table(content: object) -> bool:
    """判断表体是否采用 Native HTML 输出。"""

    return isinstance(content, str) and content.lstrip().lower().startswith("<table")


def _html_shape(content: str) -> tuple[int, int]:
    """计算 HTML 表格的物理行数和最大展开列数。"""

    soup = BeautifulSoup(content, "html.parser")
    rows = soup.find_all("tr")
    cols = max(
        (sum(max(1, int(cell.get("colspan", 1))) for cell in row.find_all(["td", "th"], recursive=False)) for row in rows),
        default=0,
    )
    return len(rows), cols


def _html_cell_truth(content: str) -> list[dict[str, Any]]:
    """把 HTML 单元格展开为带网格位置和 span 的稳定记录。"""

    soup = BeautifulSoup(content, "html.parser")
    occupied: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(soup.find_all("tr")):
        col_index = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (row_index, col_index) in occupied:
                col_index += 1
            rowspan = max(1, int(cell.get("rowspan", 1)))
            colspan = max(1, int(cell.get("colspan", 1)))
            records.append(
                {
                    "row": row_index,
                    "col": col_index,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "text": cell.get_text(),
                }
            )
            for covered_row in range(row_index, row_index + rowspan):
                for covered_col in range(col_index, col_index + colspan):
                    occupied.add((covered_row, covered_col))
            col_index += colspan
    return records


def _native_table_input(
    document: PDFDocument,
    target: dict[str, Any],
) -> NativeTableInput:
    """按归一化真值 bbox 构造共享 Native Table 输入。"""

    page = document[target["page_index"]]
    width, height = page.size
    left, top, right, bottom = target["bbox"]
    return NativeTableInput(
        table_bbox=(
            left * width,
            top * height,
            right * width,
            bottom * height,
        ),
        page_size=page.size,
        angle=0,
        chars=tuple(page.get_chars()),
        drawing_lines=coerce_native_table_rules(page.get_drawing_lines()),
        rectangles=coerce_native_table_rectangles(page.get_path_infos()),
    )


def _load_manifest() -> dict[str, Any]:
    """读取仓库相对路径的 demo 少线表真值。"""

    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_demo_sparse_table_confidence_manifest() -> None:
    """验证少线目标逐格正确且全目录 HTML/投影总数保持固定。"""

    manifest = _load_manifest()
    source_root = _PROJECT_ROOT / manifest["source_root"]
    targets_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in manifest["tables"]:
        targets_by_file[target["file"]].append(target)

    observed_html = 0
    observed_projection = 0
    observed_tables = 0
    for pdf_path in sorted(source_root.glob("*.pdf")):
        file_name = pdf_path.name
        targets = targets_by_file.get(file_name, [])
        pdf_path = source_root / file_name
        assert pdf_path.is_file()
        with PDFDocument(pdf_path.read_bytes()) as pdf_document:
            model_list = PdfModel().predict(pdf_document)
            for page in model_list:
                for block in page:
                    if block.get("type") != "table":
                        continue
                    observed_tables += 1
                    if _is_html_table(block.get("content")):
                        observed_html += 1
                    else:
                        observed_projection += 1
            for target in targets:
                page_index = target["page_index"]
                page_tables = [block for block in model_list[page_index] if block.get("type") == "table"]
                block = page_tables[target["table_index"]]
                assert block["bbox"] == target["bbox"]
                content = block["content"]
                assert target["expected_output"] == "html"
                assert _is_html_table(content)
                assert _html_shape(content) == (target["rows"], target["cols"])
                assert hashlib.sha256(content.encode("utf-8")).hexdigest() == target["content_sha256"]
                if "cell_truth" in target:
                    assert _html_cell_truth(content) == target["cell_truth"]
                    result = recover_native_pdf_table(
                        _native_table_input(
                            pdf_document,
                            target,
                        )
                    )
                    assert result is not None
                    assert result.source == target["expected_source"]

    totals = manifest["totals"]
    assert observed_tables == totals["tables"]
    assert observed_html == totals["expected_html"]
    assert observed_projection == totals["expected_projection"]
