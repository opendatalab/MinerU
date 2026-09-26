"""真实字符快照的 Rust 归属与原 Python 规则逐字符差分。"""

from copy import deepcopy
from dataclasses import asdict
from io import BytesIO
import random

import pytest
from reportlab.pdfgen.canvas import Canvas

from docvortex.document.pdf import PDFDocument
from mineru.backend.analysis.pdf.text import native
from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
from mineru.types import ContentType


@pytest.fixture
def owned_text():
    """生成包含边界标点、空白及多行的 PDF，关闭原文档后再验证归属。"""
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(320, 100))
    for y, text in [(80, "(A B), C! [D]"), (50, "E F? G: H;"), (20, "I J. K-L / M")]:
        canvas.drawString(10, y, text)
    canvas.save()
    with PDFDocument(stream.getvalue()) as document:
        owner = document[0].get_text_snapshot()
    if owner is None or not hasattr(owner, "assign_spans"):
        pytest.skip("Native span assignment unavailable")
    assert owner.supports_span_matching()
    return owner


@pytest.mark.parametrize("seed", range(24))
def test_owned_assignments_match_python_for_overlapping_spans(owned_text, monkeypatch, seed):
    """在随机重叠框、负坐标与相同框中比对原始 char_idx 归属、文本及后置 OCR 判定。"""
    randomizer = random.Random(seed)
    boxes = [(0.0, 0.0, 100.0, 100.0), (0.0, 0.0, 100.0, 100.0)]
    for _ in range(14):
        x, y = randomizer.uniform(-15, 100), randomizer.uniform(-10, 95)
        boxes.append((x, y, x + randomizer.uniform(2, 90), y + randomizer.uniform(2, 30)))
    original = native.chars_to_content
    observed = []

    def capture(span, **kwargs):
        """在原内容组装删除临时字符前记录归属，不替换实际规则。"""
        observed.append((span.bbox, [char["char_idx"] for char in span.metadata["chars"]]))
        original(span, **kwargs)

    monkeypatch.setattr(native, "chars_to_content", capture)
    results = []
    assignments = []
    for owner in [None, owned_text]:
        geometry = owned_text.materialize_geometry()
        spans = [
            _AnalyzeSpan(ContentType.TEXT, box, metadata={"chars": [], "height": box[3] - box[1], "width": box[2] - box[0]})
            for box in boxes
        ]
        observed.clear()
        pending = native.fill_char_in_spans(
            spans,
            geometry.chars,
            12.0,
            tight_bboxes=geometry.tight_bboxes,
            origins=geometry.origins,
            detect_scripts=False,
            **({"_native_text": owner} if owner else {}),
        )
        assignments.append(deepcopy(observed))
        results.append(([asdict(span) for span in spans], [spans.index(span) for span in pending]))
    assert assignments[0] == assignments[1]
    assert results[0] == results[1]


def test_owned_assignment_keeps_replaced_rules_on_reference_path(owned_text, monkeypatch):
    """调用方替换归属规则时不偷偷使用原生版本，保持参考计算能力。"""

    def reject(*args):
        """让所有字符失配，验证规则替换确实被执行。"""
        return False

    monkeypatch.setattr(native, "calculate_char_in_span", reject)
    assert native._owned_span_assignments(owned_text, [_AnalyzeSpan(ContentType.TEXT, (0.0, 0.0, 100.0, 100.0))], 12.0) is None


def test_rotated_snapshot_explicitly_retains_reference_selection():
    """整行倾斜水印仍由原有组行规则裁决，不能把未迁移角度当成可忽略字符。"""
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(120, 120))
    canvas.rotate(20)
    canvas.drawString(20, 50, "WATERMARK")
    canvas.save()
    with PDFDocument(stream.getvalue()) as document:
        owner = document[0].get_text_snapshot()
    if owner is None or not hasattr(owner, "assign_spans"):
        pytest.skip("Native span assignment unavailable")
    assert not owner.supports_span_matching()
    assert owner.assign_spans([(0.0, 0.0, 120.0, 120.0)], 10.0, native.LINE_STOP_FLAG, native.LINE_START_FLAG, 0.33) is None
