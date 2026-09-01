from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from mineru.model.flash.pdf.document import PDFDocument
from mineru.model.flash.pdf.pipeline import _analyze_native_document


_PROJECT_ROOT = Path(__file__).parents[2]
_EXPECTATION_PATH = (
    _PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "flash_layout_semantic_expectations.json"
)
_NATURAL_TEXT_TYPES = {"text", "doc_title", "paragraph_title"}


def _visible_text(value: Any) -> str:
    """递归提取真实 Flash block 与 InlineSpan 的可见文本。"""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_visible_text(item) for item in value)
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, (str, list, dict)):
            return _visible_text(content)
        for field_name in ("text", "latex", "html"):
            field_value = value.get(field_name)
            if isinstance(field_value, str):
                return field_value
    return ""


def _normalized_text(value: Any) -> str:
    """移除排版空白，保留语义字符供版本化期望比较。"""

    return re.sub(r"\s+", "", _visible_text(value))


@lru_cache(maxsize=1)
def _expectation() -> dict[str, Any]:
    """读取中文论文 Flash 语义期望并校验源文件指纹。"""

    payload = json.loads(_EXPECTATION_PATH.read_text(encoding="utf-8"))
    document = payload["documents"][0]
    source = _PROJECT_ROOT / document["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == document["sha256"]
    return document


@lru_cache(maxsize=1)
def _pages() -> tuple[tuple[dict[str, Any], ...], ...]:
    """只解析一次真实中文论文，供本文件全部语义断言复用。"""

    source = _PROJECT_ROOT / _expectation()["path"]
    with PDFDocument(str(source)) as document:
        pages = _analyze_native_document(document)
    return tuple(tuple(page) for page in pages)


def _typed_texts(block_type: str) -> Counter[tuple[int, str]]:
    """按页号和规范文本统计指定类型，保留重复项检测能力。"""

    return Counter(
        (page_index, _normalized_text(block.get("content")))
        for page_index, page in enumerate(_pages())
        for block in page
        if block.get("type") == block_type
    )


def test_chinese_paper_matches_versioned_title_semantics() -> None:
    """验证双语文档标题和四十个章节标题与人工视觉金标完全一致。"""

    expectation = _expectation()
    expected_doc_titles = Counter(
        (item["page_index"], item["text"])
        for item in expectation["doc_titles"]
    )
    expected_paragraph_titles = Counter(
        (item["page_index"], item["text"])
        for item in expectation["paragraph_titles"]
    )

    assert _typed_texts("doc_title") == expected_doc_titles
    assert _typed_texts("paragraph_title") == expected_paragraph_titles
    title_text = "\n".join(
        text
        for _page_index, text in _typed_texts("paragraph_title")
    )
    assert all(
        fragment not in title_text
        for fragment in expectation["forbidden_title_fragments"]
    )


def test_chinese_paper_recovers_all_numbered_formula_blocks() -> None:
    """验证式一至式十四唯一成块，且公式内容不吸收相邻说明句。"""

    expectation = _expectation()
    equations = [
        block
        for page in _pages()
        for block in page
        if block.get("type") == "equation"
    ]
    tag_counts: Counter[str] = Counter()
    contents = []
    for block in equations:
        content = _normalized_text(block.get("content"))
        contents.append(content)
        match = re.search(r"\\tag\{([^}]+)\}", content)
        assert match is not None
        tag_counts[match.group(1)] += 1

    assert tag_counts == Counter(expectation["equation_tags"])
    assert all(
        fragment not in content
        for content in contents
        for fragment in expectation["forbidden_equation_fragments"]
    )


def test_chinese_paper_preserves_visual_controls_and_unique_blocks() -> None:
    """验证表图页码数量、双栏归属和顶层自然文本无近乎完全重叠。"""

    expectation = _expectation()
    counts = Counter(
        str(block.get("type"))
        for page in _pages()
        for block in page
    )
    assert all(
        counts[block_type] == expected_count
        for block_type, expected_count in expectation["control_counts"].items()
    )

    first_page = _pages()[0]
    introduction = next(
        block
        for block in first_page
        if block.get("type") == "paragraph_title"
        and _normalized_text(block.get("content")) == "0引言"
    )
    introduction_bottom = float(introduction["bbox"][3])
    assert not any(
        block.get("type") == "text"
        and float(block["bbox"][1]) >= introduction_bottom
        and float(block["bbox"][1]) < 0.84
        and float(block["bbox"][0]) < 0.48
        and float(block["bbox"][2]) > 0.52
        for block in first_page
    )

    overlaps = []
    for page_index, page in enumerate(_pages()):
        blocks = [
            block
            for block in page
            if block.get("type") in _NATURAL_TEXT_TYPES
            and isinstance(block.get("bbox"), list)
        ]
        for first_index, first in enumerate(blocks):
            first_bbox = [float(value) for value in first["bbox"]]
            first_area = max(0.0, first_bbox[2] - first_bbox[0]) * max(
                0.0,
                first_bbox[3] - first_bbox[1],
            )
            for second in blocks[first_index + 1 :]:
                second_bbox = [float(value) for value in second["bbox"]]
                second_area = max(0.0, second_bbox[2] - second_bbox[0]) * max(
                    0.0,
                    second_bbox[3] - second_bbox[1],
                )
                intersection = max(
                    0.0,
                    min(first_bbox[2], second_bbox[2])
                    - max(first_bbox[0], second_bbox[0]),
                ) * max(
                    0.0,
                    min(first_bbox[3], second_bbox[3])
                    - max(first_bbox[1], second_bbox[1]),
                )
                overlap = intersection / max(
                    1e-9,
                    min(first_area, second_area),
                )
                if overlap >= 0.95:
                    overlaps.append(
                        (
                            page_index,
                            _normalized_text(first.get("content")),
                            _normalized_text(second.get("content")),
                        )
                    )
    assert overlaps == []
