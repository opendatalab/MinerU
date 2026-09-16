"""覆盖四档位同步/异步几何传递、Flash 图片公式与缓存汇总。"""

from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pypdf import PdfReader, PdfWriter
import pytest
from reportlab.pdfgen.canvas import Canvas

from docvortex.document.pdf import PDFDocument
from docvortex.document.pdf.layout import merge_layout_extensions
from docvortex.schema import DocumentProperties, MiddleJson, ModelJson
from mineru.backend import analyze
from mineru.backend.analysis.pdf import pipeline
from mineru.backend.postprocess import document as postprocess
from mineru.doclib.background.compaction import Compaction
from mineru.doclib.services.parse_svc import parse_batch_json_path
from mineru.render import PdfLayout, PdfRenderOptions, RenderFormat, render, render_pdf


def _source() -> bytes:
    """生成不同尺寸及空白页的真实 PDF，用于检查几何生命周期。"""
    output = BytesIO()
    canvas = Canvas(output)
    for width, height in [(400, 600), (320, 480), (600, 400)]:
        canvas.setPageSize((width, height))
        canvas.showPage()
    canvas.save()
    writer = PdfWriter(clone_from=BytesIO(output.getvalue()))
    writer.pages[1].rotate(90)
    rotated = BytesIO()
    writer.write(rotated)
    return rotated.getvalue()


def _model_pages(effort: str) -> list[list[dict]]:
    """按上游契约提供 Flash 空公式及其他档位公式文本，几何采集与导出仍走真实实现。"""
    image = BytesIO()
    Image.new("RGB", (80, 20), "black").save(image, "PNG")
    return [
        [
            {"type": "text", "content": "invalid without line geometry", "bbox": (0.1, 0.1, 0.2, 0.2), "angle": 270},
            {
                "type": "equation",
                "bbox": (0.1, 0.1, 0.9, 0.2),
                "angle": 90,
                "content": "" if effort == "flash" else "x^2",
                "image_base64": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode(),
            },
        ],
        [],
        [{"type": "text", "bbox": (0.1, 0.1, 0.9, 0.3), "content": [{"type": "text", "content": "last page"}]}],
    ]


@pytest.mark.parametrize("effort,tier", [("flash", "flash"), ("medium", "basic"), ("high", "standard"), ("xhigh", "advanced")])
@pytest.mark.parametrize("parse_mode", ["txt", "ocr"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_all_tiers_preserve_geometry_to_pdf(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
    tier: str,
    parse_mode: str,
    asynchronous: bool,
) -> None:
    """同一结果经过完整同步/异步门面后保留源页映射，并可显式按原布局导出。"""
    events = []

    def prepare(state: object, data: bytes, requested_effort: str, mode: str, vlm_config: object) -> None:
        """打开真实 PDF，但用占位上下文代替昂贵推理模型。"""
        state.document = PDFDocument(data)
        state.parse_mode = mode
        state.flash_txt_mode = requested_effort == "flash" and mode == "txt"
        state.hybrid_model = SimpleNamespace(device="cpu")
        state.predictor = object()

    def windows(*args: object, **kwargs: object) -> list[list[dict]]:
        """返回独立 raw pages，模拟三个窗口最终汇总的结果。"""
        return deepcopy(_model_pages(effort))

    async def async_windows(*args: object, **kwargs: object) -> list[list[dict]]:
        """覆盖原生异步分支，不把异步 API 误测为纯同步代理。"""
        events.append("native_async")
        return windows()

    monkeypatch.setattr(pipeline, "_prepare_analysis", prepare)
    monkeypatch.setattr(pipeline, "process_pdf_windows", windows)
    monkeypatch.setattr(pipeline, "aio_process_pdf_windows", async_windows)
    monkeypatch.setattr(pipeline, "release_document", lambda *args: None)
    monkeypatch.setattr("mineru.model.vlm.client.uses_native_async_vlm", lambda *args: True)
    monkeypatch.setattr(postprocess, "apply_llm_aided_postprocess", lambda *args: None)

    async def no_enhancement(*args: object) -> None:
        """关闭与几何验证无关的外部 LLM 服务。"""

    monkeypatch.setattr(postprocess, "aio_apply_llm_aided_postprocess", no_enhancement)
    kwargs = {
        "effort": effort,
        "parse_mode": parse_mode,
        "page_index_map": [1, 4, 5],
        "source_properties": DocumentProperties(),
    }
    middle, model = (
        asyncio.run(analyze.aio_doc_analyze(_source(), **kwargs)) if asynchronous else analyze.doc_analyze(_source(), **kwargs)
    )
    assert model.extensions["mineru"] == {"tier": tier, "parse_mode": parse_mode}
    expected = [
        {"page_idx": idx, "width_pt": width, "height_pt": height}
        for idx, (width, height) in zip([1, 4, 5], [(400, 600), (480, 320), (600, 400)], strict=True)
    ]
    expected[0]["image_rotations"] = {"0": 90}
    assert middle.extensions["docvortex_layout"]["pages"] == model.extensions["docvortex_layout"]["pages"] == expected
    assert model.pages[0][0]["content"] == ("" if effort == "flash" else "x^2")
    assert ModelJson.from_json(model.to_json()).to_json() == model.to_json()
    assert MiddleJson.from_json(middle.to_json()).to_json() == middle.to_json()
    payload = render(middle, RenderFormat.PDF, options=PdfRenderOptions(layout=PdfLayout.ORIGINAL))
    direct = render_pdf(middle, layout=PdfLayout.ORIGINAL)
    assert [page.extract_text() for page in PdfReader(BytesIO(payload)).pages] == [
        page.extract_text() for page in PdfReader(BytesIO(direct)).pages
    ]
    assert [tuple(page.mediabox)[2:] for page in PdfReader(BytesIO(payload)).pages] == [(400, 600), (480, 320), (600, 400)]
    if asynchronous and effort in {"high", "xhigh"}:
        assert events == ["native_async"]


def test_compaction_merges_page_geometry_with_latest_content(tmp_path: Path) -> None:
    """不同批次页几何可合并，重复页的图片方向与最新正文一同更新。"""
    sha = "a" * 64
    for page_range, done_at, indices in [("1-2", 1, [0, 1]), ("2-3", 2, [1, 2])]:
        model = ModelJson(
            pages=[[] for _ in indices],
            page_index_map=indices,
            metadata={"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
            extensions={
                "mineru": {"tier": "standard", "parse_mode": "txt"},
                "docvortex_layout": {
                    "version": 1,
                    "pages": [
                        {
                            "page_idx": idx,
                            "width_pt": 400,
                            "height_pt": 600,
                            "image_rotations": {"0": 90 if done_at == 1 else 270},
                        }
                        for idx in indices
                    ],
                },
            },
        )
        middle = postprocess.build_middle_json(model)
        path = Path(parse_batch_json_path(str(tmp_path), sha, "standard", page_range, done_at))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(middle.to_json())
    compaction = Compaction(db=None, interval_sec=600, data_dir=str(tmp_path))
    rows = [{"page_range": "2-3", "done_at": 2}, {"page_range": "1-2", "done_at": 1}]
    loaded = compaction._load_batch_payloads(sha, "standard", rows)
    assert loaded is not None
    pages_by_index, envelope = loaded
    pages = envelope["extensions"]["docvortex_layout"]["pages"]
    assert [page["page_idx"] for page in pages] == [0, 1, 2]
    assert [page["image_rotations"]["0"] for page in pages] == [90, 270, 270]
    assert len(list(tmp_path.rglob("*.json"))) == 2
    before = deepcopy(envelope)
    written = compaction._write_compacted_json_files(sha, "standard", ["1-3"], 2, pages_by_index, envelope)
    assert written is not None and len(written) == 1
    replayed = MiddleJson.from_json(Path(next(iter(written))).read_text())
    assert replayed.extensions == envelope["extensions"]
    assert envelope == before
    assert len(PdfReader(BytesIO(render_pdf(replayed, layout=PdfLayout.ORIGINAL))).pages) == 3


def test_old_replacement_batch_does_not_keep_stale_rotation() -> None:
    """较新正文没有几何时，不从较旧同页借用可能对应错误块索引的旋转信息。"""
    old = {
        "docvortex_layout": {
            "version": 1,
            "pages": [
                {"page_idx": 0, "width_pt": 400, "height_pt": 600},
                {"page_idx": 1, "width_pt": 400, "height_pt": 600},
            ],
        }
    }
    original = json.dumps(old)
    merged = merge_layout_extensions(old, {}, [1])
    assert [page["page_idx"] for page in merged["docvortex_layout"]["pages"]] == [0]
    assert json.dumps(old) == original
