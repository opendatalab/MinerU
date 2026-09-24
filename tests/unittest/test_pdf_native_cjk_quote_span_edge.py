from __future__ import annotations

import pytest

from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
from mineru.backend.analysis.pdf.text.native import calculate_char_in_span, fill_char_in_spans
from mineru.types import ContentType

SPAN_BBOX = (0.0, 0.0, 100.0, 20.0)
RIGHT_EDGE_CHAR_BBOX = [98.0, 2.0, 106.0, 18.0]
LEFT_EDGE_CHAR_BBOX = [-6.0, 2.0, 2.0, 18.0]
CJK_LINE_STOP_CLOSERS = ("」", "』", "〉", "〕", "〗")
CJK_LINE_START_OPENERS = ("〈", "〔", "〖")


def _fillable_span(bbox: tuple[float, float, float, float] = SPAN_BBOX) -> _AnalyzeSpan:
    """构造已初始化字符容器的可回填文本 Span。"""
    return _AnalyzeSpan(
        type=ContentType.TEXT,
        bbox=bbox,
        score=1.0,
        metadata={
            "chars": [],
            "height": bbox[3] - bbox[1],
            "width": bbox[2] - bbox[0],
        },
    )


@pytest.mark.parametrize("char", CJK_LINE_STOP_CLOSERS)
def test_calculate_char_in_span_accepts_cjk_closers_past_right_edge(char: str) -> None:
    """右边缘略超出 span 的 CJK 结束引号/括弧应与已支持的 》 同样命中。"""
    assert calculate_char_in_span(RIGHT_EDGE_CHAR_BBOX, SPAN_BBOX, "》") is True
    assert calculate_char_in_span(RIGHT_EDGE_CHAR_BBOX, SPAN_BBOX, char) is True


@pytest.mark.parametrize("char", CJK_LINE_START_OPENERS)
def test_calculate_char_in_span_accepts_cjk_openers_past_left_edge(char: str) -> None:
    """左边缘略超出 span 的 CJK 起始括弧应与已支持的 「 同样命中。"""
    assert calculate_char_in_span(LEFT_EDGE_CHAR_BBOX, SPAN_BBOX, "「") is True
    assert calculate_char_in_span(LEFT_EDGE_CHAR_BBOX, SPAN_BBOX, char) is True


def test_calculate_char_in_span_rejects_letter_at_closing_quote_geometry() -> None:
    """同一越界几何下，非标点汉字仍不能靠 LINE_STOP 边缘规则进入 span。"""
    assert calculate_char_in_span(RIGHT_EDGE_CHAR_BBOX, SPAN_BBOX, "好") is False


def test_fill_char_in_spans_keeps_dialogue_closing_corner_quote() -> None:
    """行末 」 中心略超出 span 右边界时仍应回填为 他说「好」。"""
    span = _fillable_span()
    chars = [
        {"char": "他", "bbox": [42.0, 2.0, 60.0, 18.0], "char_idx": 0},
        {"char": "说", "bbox": [60.0, 2.0, 78.0, 18.0], "char_idx": 1},
        {"char": "「", "bbox": [78.0, 2.0, 87.0, 18.0], "char_idx": 2},
        {"char": "好", "bbox": [87.0, 2.0, 98.0, 18.0], "char_idx": 3},
        {"char": "」", "bbox": [98.0, 2.0, 106.0, 18.0], "char_idx": 4},
    ]
    assert calculate_char_in_span(chars[-1]["bbox"], span.bbox, "」") is True
    fill_char_in_spans([span], chars, median_span_height=20.0, detect_scripts=False)  # type: ignore[arg-type]
    assert span.content == "他说「好」"


def test_fill_char_in_spans_keeps_left_edge_white_bracket_opener() -> None:
    """行首 〈 中心略超出 span 左边界时仍应回填，而不是丢掉起始括弧。"""
    span = _fillable_span()
    chars = [
        {"char": "〈", "bbox": [-6.0, 2.0, 2.0, 18.0], "char_idx": 0},
        {"char": "好", "bbox": [2.0, 2.0, 20.0, 18.0], "char_idx": 1},
    ]
    assert calculate_char_in_span(chars[0]["bbox"], span.bbox, "〈") is True
    fill_char_in_spans([span], chars, median_span_height=20.0, detect_scripts=False)  # type: ignore[arg-type]
    assert span.content == "〈好"
