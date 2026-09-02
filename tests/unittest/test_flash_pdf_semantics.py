from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from mineru.model.flash.pdf.document import PDFDocument
from mineru.model.flash.pdf.pipeline import _analyze_native_document
from scripts.review_flash_layout_geometry import (
    _geometry_summary_mismatch,
    _page_bbox_fingerprint,
    _page_fingerprint,
)


_PROJECT_ROOT = Path(__file__).parents[2]
_EXPECTATION_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "flash_layout_semantic_expectations.json"
_GEOMETRY_MANIFEST_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "flash_layout_geometry_manifest.json"
_BLOCK_EXPECTATION_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "flash_layout_block_expectations.json"
_FROZEN_SOIL_PATH = _PROJECT_ROOT / "demo" / "pdfs" / "中文论文.pdf"
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


def _normalized_text(
    value: Any,
    *,
    nfkc: bool = False,
) -> str:
    """移除排版空白，保留语义字符供版本化期望比较。"""

    normalized = re.sub(r"\s+", "", _visible_text(value))
    return unicodedata.normalize("NFKC", normalized) if nfkc else normalized


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


@lru_cache(maxsize=1)
def _frozen_soil_pages() -> tuple[tuple[dict[str, Any], ...], ...]:
    """只解析一次中文论文1，供 canonical 几何兼容性断言复用。"""

    with PDFDocument(str(_FROZEN_SOIL_PATH)) as document:
        pages = _analyze_native_document(document)
    return tuple(tuple(page) for page in pages)


@lru_cache(maxsize=None)
def _additional_chinese_paper_pages(
    relative_path: str,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """按版本化相对路径缓存新增中文论文 Flash 页面。"""

    source = _PROJECT_ROOT / relative_path
    with PDFDocument(str(source)) as document:
        pages = _analyze_native_document(document)
    return tuple(tuple(page) for page in pages)


@lru_cache(maxsize=1)
def _block_expectations() -> tuple[dict[str, Any], ...]:
    """读取四篇中文论文的版本化 block 分组与类型库存期望。"""

    payload = json.loads(
        _BLOCK_EXPECTATION_PATH.read_text(encoding="utf-8"),
    )
    assert payload["schema_version"] == 1
    for document in payload["documents"]:
        source = _PROJECT_ROOT / document["path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == document["sha256"]
    return tuple(payload["documents"])


def _pages_for_expectation(
    expectation: dict[str, Any],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """按版本化路径返回已缓存的真实 Flash 页面。"""

    if expectation["path"] == "demo/pdfs/中文论文.pdf":
        return _frozen_soil_pages()
    if expectation["path"] == "demo/pdfs/中文论文2.pdf":
        return _pages()
    return _additional_chinese_paper_pages(
        str(expectation["path"]),
    )


def _block_containing_fragment(
    page: tuple[dict[str, Any], ...],
    fragment: str,
    *,
    block_type: str | None = None,
    nfkc: bool = False,
) -> tuple[int, dict[str, Any]]:
    """返回唯一包含规范化片段的顶层 block 及其页内索引。"""

    normalized_fragment = _normalized_text(
        fragment,
        nfkc=nfkc,
    )
    matches = [
        (index, block)
        for index, block in enumerate(page)
        if (block_type is None or block.get("type") == block_type)
        and normalized_fragment
        in _normalized_text(
            block.get("content"),
            nfkc=nfkc,
        )
    ]
    assert len(matches) == 1, (
        fragment,
        block_type,
        [
            (
                index,
                block.get("type"),
                _normalized_text(
                    block.get("content"),
                    nfkc=nfkc,
                ),
            )
            for index, block in matches
        ],
    )
    return matches[0]


def test_frozen_soil_paper_keeps_tracked_semantic_and_bbox_gold() -> None:
    """验证无长文档门槛时中文论文1仍逐页保持已有语义与 bbox 指纹。"""

    manifest = json.loads(
        _GEOMETRY_MANIFEST_PATH.read_text(encoding="utf-8"),
    )
    expected = next(document for document in manifest["documents"] if document["path"] == "demo/pdfs/中文论文.pdf")
    assert hashlib.sha256(_FROZEN_SOIL_PATH.read_bytes()).hexdigest() == expected["sha256"]
    pages = _frozen_soil_pages()
    assert [_page_fingerprint(list(page)) for page in pages] == [page["fingerprint"] for page in expected["pages"]]
    assert [_page_bbox_fingerprint(list(page)) for page in pages] == [page["bbox_fingerprint"] for page in expected["pages"]]


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
    expected_doc_titles = Counter((item["page_index"], item["text"]) for item in expectation["doc_titles"])
    expected_paragraph_titles = Counter((item["page_index"], item["text"]) for item in expectation["paragraph_titles"])

    assert _typed_texts("doc_title") == expected_doc_titles
    assert _typed_texts("paragraph_title") == expected_paragraph_titles
    title_text = "\n".join(text for _page_index, text in _typed_texts("paragraph_title"))
    assert all(fragment not in title_text for fragment in expectation["forbidden_title_fragments"])


def test_chinese_paper_recovers_all_numbered_formula_blocks() -> None:
    """验证式一至式十四唯一成块，且公式内容不吸收相邻说明句。"""

    expectation = _expectation()
    equations = [block for page in _pages() for block in page if block.get("type") == "equation"]
    tag_counts: Counter[str] = Counter()
    contents = []
    for block in equations:
        content = _normalized_text(block.get("content"))
        contents.append(content)
        match = re.search(r"\\tag\{([^}]+)\}", content)
        assert match is not None
        tag_counts[match.group(1)] += 1

    assert tag_counts == Counter(expectation["equation_tags"])
    assert all(fragment not in content for content in contents for fragment in expectation["forbidden_equation_fragments"])


def test_chinese_paper_preserves_visual_controls_and_unique_blocks() -> None:
    """验证表图页码数量、双栏归属和顶层自然文本无近乎完全重叠。"""

    expectation = _expectation()
    counts = Counter(str(block.get("type")) for page in _pages() for block in page)
    assert all(counts[block_type] == expected_count for block_type, expected_count in expectation["control_counts"].items())

    first_page = _pages()[0]
    introduction = next(
        block
        for block in first_page
        if block.get("type") == "paragraph_title" and _normalized_text(block.get("content")) == "0引言"
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
        blocks = [block for block in page if block.get("type") in _NATURAL_TEXT_TYPES and isinstance(block.get("bbox"), list)]
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
                    min(first_bbox[2], second_bbox[2]) - max(first_bbox[0], second_bbox[0]),
                ) * max(
                    0.0,
                    min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]),
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


def test_chinese_papers_match_versioned_block_group_expectations() -> None:
    """验证四篇真实论文的类型库存、指定合并和指定拆分与人工审阅期望一致。"""

    for expectation in _block_expectations():
        pages = _pages_for_expectation(expectation)
        normalize_nfkc = expectation.get("normalize_nfkc") is True
        counts = Counter(str(block.get("type")) for page in pages for block in page)
        assert counts == Counter(expectation["type_counts"])

        for item in expectation.get("exact_typed_text", []):
            page = pages[item["page_index"]]
            matches = [
                block
                for block in page
                if block.get("type") == item["type"]
                and _normalized_text(
                    block.get("content"),
                    nfkc=normalize_nfkc,
                )
                == _normalized_text(
                    item["text"],
                    nfkc=normalize_nfkc,
                )
            ]
            assert len(matches) == 1, item

        for group in expectation.get("same_block_groups", []):
            page = pages[group["page_index"]]
            matched_indices = {
                _block_containing_fragment(
                    page,
                    fragment,
                    block_type=group["type"],
                    nfkc=normalize_nfkc,
                )[0]
                for fragment in group["fragments"]
            }
            assert len(matched_indices) == 1, group

        for group in expectation.get("different_block_groups", []):
            page = pages[group["page_index"]]
            matched_indices = {
                _block_containing_fragment(
                    page,
                    fragment,
                    nfkc=normalize_nfkc,
                )[0]
                for fragment in group["fragments"]
            }
            assert len(matched_indices) == len(group["fragments"]), group

        for item in expectation.get("forbidden_type_fragments", []):
            page = pages[item["page_index"]]
            normalized_fragment = _normalized_text(
                item["fragment"],
                nfkc=normalize_nfkc,
            )
            assert not any(
                block.get("type") == item["type"]
                and normalized_fragment
                in _normalized_text(
                    block.get("content"),
                    nfkc=normalize_nfkc,
                )
                for block in page
            ), item


def test_chinese_paper_four_third_page_upper_band_inventory() -> None:
    """验证中文论文4第三页原假大表区域恢复为指定的文本、图表和标题库存。"""

    pages = _additional_chinese_paper_pages(
        "demo/pdfs/中文论文4.pdf",
    )
    upper_band = [
        block
        for block in pages[2]
        if block.get("type") != "header"
        and isinstance(block.get("bbox"), list)
        and float(block["bbox"][3]) <= 0.56
    ]

    assert Counter(str(block.get("type")) for block in upper_band) == Counter(
        {
            "paragraph_title": 3,
            "text": 4,
            "image": 1,
            "caption": 2,
            "table": 1,
        }
    )


def test_chinese_paper_continuation_caption_precedes_tight_table_body() -> None:
    """验证第3页续表 caption 不进入 HTML，且其边界位于收紧后的表体上方。"""

    page = _pages()[2]
    _caption_index, caption = _block_containing_fragment(
        page,
        "续表",
        block_type="caption",
    )
    table = next(block for block in page if block.get("type") == "table")

    assert _normalized_text(caption.get("content")) == "续表"
    assert "续表" not in str(table.get("content") or "")
    assert float(caption["bbox"][3]) < float(table["bbox"][1])


def test_flash_layout_geometry_summary_comparison_is_strict() -> None:
    """验证几何摘要缺失或任一计数漂移都会形成独立门禁失败。"""

    expected = {
        "expected_geometry_summary": {
            "repaired_chars": 10,
            "repaired_lines": 2,
        }
    }
    actual = {
        "geometry_summary": {
            "repaired_chars": 10,
            "repaired_lines": 2,
        }
    }

    assert _geometry_summary_mismatch("sample.pdf", expected, actual) is None
    assert _geometry_summary_mismatch("sample.pdf", {}, actual) == {
        "file": "sample.pdf",
        "reason": "geometry_summary_expectation_missing",
    }
    assert _geometry_summary_mismatch(
        "sample.pdf",
        expected,
        {
            "geometry_summary": {
                "repaired_chars": 10,
                "repaired_lines": 1,
            }
        },
    ) == {
        "file": "sample.pdf",
        "reason": "geometry_summary_mismatch",
        "expected": expected["expected_geometry_summary"],
        "actual": {
            "repaired_chars": 10,
            "repaired_lines": 1,
        },
    }
