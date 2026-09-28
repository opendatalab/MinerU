"""验证普通、旋转及 Rust 原生回填共用非 CJK 墨迹词界规则。"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from docvortex.document.pdf import PDFDocument
from docvortex.document.pdf.text._contracts import Bbox
from mineru.backend.analysis.pdf.text import native
from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
from mineru.types import ContentType


def _char(text: str, index: int, x: float) -> dict[str, Any]:
    """构造短英文或 CJK 字符，故意使 loose 框覆盖真实留白。"""
    return {
        "char": text,
        "char_idx": index,
        "bbox": Bbox([x, 0, x + 10, 10]),
        "tight_bbox": (x, 1, x + 4, 9),
        "origin": (x, 10),
        "writing_angle": 0.0,
        "rotation": 0.0,
        "font": {"name": "Fixture", "size": 10.0, "flags": 0, "weight": 400},
    }


@pytest.mark.parametrize("text, expected", [("AIML", "AI ML"), ("中文", "中文"), ("中A", "中A"), ("A文", "A文")])
def test_short_span_needs_no_reference_samples(text: str, expected: str) -> None:
    """无需同行参考样本即可补短英文，CJK 边界保持旧输出。"""
    positions = [0, 5, 13, 18] if len(text) == 4 else [0, 8]
    chars = [_char(char, index, positions[index]) for index, char in enumerate(text)]
    span = _AnalyzeSpan(ContentType.TEXT, (0, 0, 40, 10), metadata={"chars": chars})
    native.chars_to_content(span, detect_scripts=False)
    assert span.content == expected


def test_code_span_keeps_old_content() -> None:
    """被代码块认领的 span 不启用新增墨迹词界。"""
    span = _AnalyzeSpan(
        ContentType.TEXT,
        (0, 0, 30, 10),
        metadata={"chars": [_char("A", 0, 0), _char("B", 1, 8)], "_native_tight_spacing": False},
    )
    native.chars_to_content(span, detect_scripts=False)
    assert span.content == "AB"


def test_virtual_page_fill_keeps_actual_code_region() -> None:
    """实际页面代码区域的禁用标记不会被虚拟整页文本框覆盖。"""
    from mineru.backend.analysis.pdf.text.content import _protect_code_span_spacing
    from mineru.types import BlockType
    from unittest.mock import Mock

    chars = [_char("A", 0, 0), _char("B", 1, 8)]
    span = _AnalyzeSpan(ContentType.TEXT, (0, 0, 30, 10))
    _protect_code_span_spacing([span], [{"type": BlockType.CODE, "bbox": (0, 0, 0.5, 0.5)}], (100, 100))
    page = Mock()
    page.get_char_count.return_value = 2
    native.txt_spans_extract(
        page,
        [span],
        None,
        1.0,
        [(0, 0, 100, 100, None, None, None, BlockType.TEXT)],
        [],
        page_chars=chars,
        detect_scripts=False,
    )
    assert span.content == "AB"
    assert span.metadata["_native_tight_spacing"] is False


def test_vertical_line_fill_uses_tight_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """强制命中竖向 line/span 回填分支，验证没有绕过共享词界规则。"""
    import math
    from unittest.mock import Mock
    from PIL import Image
    from mineru.types import BlockType

    chars = [_char("A", 0, 0), _char("B", 1, 8)]
    for char, y in zip(chars, [0, 8]):
        char.update(
            bbox=Bbox([40, y, 50, y + 10]),
            tight_bbox=(41, y, 49, y + 4),
            origin=(40, y),
            rotation=math.pi / 2,
            writing_angle=math.pi / 2,
        )
    line = {"rotation": math.pi / 2, "bbox": Bbox([40, 0, 50, 30]), "spans": [{"text": "AB", "chars": chars}]}
    monkeypatch.setattr(native, "get_lines_from_chars", Mock(return_value=[line]))
    page = Mock()
    page.get_char_count.return_value = 2
    target = _AnalyzeSpan(ContentType.TEXT, (40, 0, 50, 30))
    spans = [target, _AnalyzeSpan(ContentType.TEXT, (60, 60, 80, 68)), _AnalyzeSpan(ContentType.TEXT, (60, 80, 80, 88))]
    with Image.new("RGB", (100, 100), "white") as image:
        native.txt_spans_extract(
            page,
            spans,
            image,
            1.0,
            [(0, 0, 100, 100, None, None, None, BlockType.TEXT)],
            [],
            page_chars=chars,
            detect_scripts=False,
        )
    assert target.content == "A B"


@pytest.mark.parametrize("enabled", [True, False])
def test_real_snapshot_content_matches_python(enabled: bool) -> None:
    """真实论文逐字符验证 Rust 物化与 Python 回退一致，且确实执行 Rust 内容快路径。"""
    from docvortex._compute_backend import get_native

    extension = get_native()
    if extension is None:
        pytest.skip("requires native extension")
    path = Path(__file__).resolve().parents[3] / "docvortex/demo/pdfs/中文论文2.pdf"
    with PDFDocument(path.read_bytes()) as document:
        owner = document[0].get_text_snapshot()
        width, height = document[0].size
    geometry = owner.materialize_geometry()
    results = []
    before = extension.text_snapshot_stats()[2]
    for current_owner in (None, owner):
        span = _AnalyzeSpan(
            ContentType.TEXT,
            (0.0, 0.0, width, height),
            metadata={"chars": [], "height": height, "width": width, "_native_tight_spacing": enabled},
        )
        native.fill_char_in_spans(
            [span],
            deepcopy(geometry.chars),
            10.0,
            tight_bboxes=geometry.tight_bboxes,
            origins=geometry.origins,
            detect_scripts=False,
            _native_text=current_owner,
        )
        results.append(span.content)
    assert extension.text_snapshot_stats()[2] == before + 1
    assert results[0] == results[1]
    assert ("Large Language Models" in results[0]) is enabled
