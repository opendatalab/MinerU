"""验证真实 TXT 窗口的 owned 快照共享、旧调用语义和释放边界。"""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from weakref import ref

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from docvortex.document.pdf import Bbox, PDFDocument, PDFPage, PDFPageTextGeometry, PDFPageVectorGeometry
from mineru.backend.analysis.pdf import snapshots, tables, window
from mineru.backend.analysis.pdf.text import content, native
from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
from mineru.types import BlockType, ContentType


def _pdf() -> bytes:
    """提供真实三行文字和矩形路径，满足方向投票且无需模型。"""
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(200, 200))
    for y in (150, 120, 90):
        canvas.drawString(20, y, "Table text 123")
    canvas.rect(10, 70, 180, 100)
    canvas.save()
    return stream.getvalue()


def _require_owned(page: PDFPage) -> Any:
    """参考后端跳过真实扩展断言，Rust 模式必须具备 owned 入口。"""
    owner = page.get_text_snapshot()
    if owner is None:
        pytest.skip("owned text snapshot unavailable in reference backend")
    return owner


class _Payload:
    """提供可弱引用的资源，用于观察窗口是否真正释放强引用。"""


def _state(page: PDFPage, cache: snapshots.PageSnapshotCache) -> window._WindowInputs:
    """构造不持有图片的窗口，聚焦关闭和取消的资源所有权。"""
    return window._WindowInputs(
        window._ProcessingWindow(0, 1, 0, 0),
        [],
        [page],
        [],
        [],
        [],
        [],
        [],
        [],
        [None],
        [None],
        page_snapshots=cache,
    )


def test_fake_pages_and_explicit_geometry_do_not_probe_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """MagicMock 的伪造接口和显式可变几何不得进入新快照通道。"""
    fake = MagicMock(spec=PDFPage)
    assert snapshots.create_page_snapshot_cache([fake], enabled=True) is None
    assert snapshots.get_page_snapshot_entry([None], 0, fake) is None
    fake.get_text_snapshot.assert_not_called()
    with PDFDocument(_pdf()) as document:
        page = document[0]
        geometry = PDFPageTextGeometry(chars=[], tight_bboxes={}, origins={})
        vectors = PDFPageVectorGeometry((), ())
        blocked = MagicMock(side_effect=AssertionError("explicit geometry entered snapshot"))
        monkeypatch.setattr(page, "get_text_snapshot", blocked)
        cache = [None]
        assert snapshots.get_page_snapshot_entry(cache, 0, page, geometry=geometry) is None
        assert snapshots.get_page_snapshot_entry(cache, 0, page, vector_geometry=vectors) is None
        blocked.assert_not_called()


def test_unsupported_owned_probe_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """不支持的原生入口每页只探测一次，不将 None 当作未探测反复读取。"""
    with PDFDocument(_pdf()) as document:
        page = document[0]
        getter = MagicMock(return_value=None)
        monkeypatch.setattr(page, "get_text_snapshot", getter)
        cache = [None]
        assert snapshots.get_page_snapshot_entry(cache, 0, page) is None
        assert snapshots.get_page_snapshot_entry(cache, 0, page) is None
        getter.assert_called_once_with()
        snapshots.clear_page_snapshot_cache(cache)
        assert cache == []


@pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
def test_non_table_window_does_not_eagerly_materialize_snapshots(effort: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """没有表格时保持 VLM 前无原生字符工作，三个 TXT 档位只分配空缓存。"""
    with PDFDocument(_pdf()) as document:
        page = document[0]
        monkeypatch.setattr(window, "_get_window_pdf_pages", lambda *_: [page])
        owner = MagicMock(side_effect=AssertionError("eager owner"))
        full = MagicMock(side_effect=AssertionError("eager full snapshot"))
        monkeypatch.setattr(page, "get_text_snapshot", owner)
        monkeypatch.setattr(page, "get_snapshot", full)
        image = Image.new("RGB", (200, 200))
        monkeypatch.setattr(window, "load_images_from_pdf_bytes_range", lambda **_: [{"img_pil": image, "scale": 1.0}])
        model = MagicMock()
        model.layout_model.batch_predict.return_value = [[]]
        state = window._prepare_pdf_window(
            b"",
            document,
            window._ProcessingWindow(0, 1, 0, 0),
            page_count=1,
            effort=effort,
            parse_mode="txt",
            hybrid_model=model,
        )
        assert state.page_snapshots == [None]
        owner.assert_not_called()
        full.assert_not_called()
        state.close()
        assert state.page_snapshots == []


def test_direction_table_and_body_share_owner_without_legacy_repacking(monkeypatch: pytest.MonkeyPatch) -> None:
    """方向只读轻量粗行，表格不回填旧缓存，正文直接消费同一完整快照。"""
    with PDFDocument(_pdf()) as document, Image.new("RGB", (200, 200)) as image:
        page = document[0]
        owner = _require_owned(page)
        owner_getter = MagicMock(wraps=page.get_text_snapshot)
        full_getter = MagicMock(wraps=page.get_snapshot)
        legacy_getter = MagicMock(side_effect=AssertionError("legacy geometry materialized before owned prepare"))
        monkeypatch.setattr(page, "get_text_snapshot", owner_getter)
        monkeypatch.setattr(page, "get_snapshot", full_getter)
        monkeypatch.setattr(page, "get_chars_with_geometry", legacy_getter)
        cache = snapshots.create_page_snapshot_cache([page], enabled=True)
        geometries: list[PDFPageTextGeometry | None] = [None]
        vectors: list[PDFPageVectorGeometry | None] = [None]
        layout = {"label": "table", "bbox": [0, 0, 200, 200]}
        images = [{"img_pil": image, "scale": 1.0}]
        assert (
            tables._resolve_txt_table_orientations(
                [{"page_idx": 0, "layout_item": layout}],
                [page],
                images,
                geometries,
                page_snapshots=cache,
            )
            == []
        )
        assert cache[0].text_owner is owner
        assert cache[0].full_snapshot is None
        assert geometries == [None]
        full_getter.assert_not_called()
        legacy_getter.assert_not_called()
        monkeypatch.setattr(tables, "recover_table_region", lambda *_args, **_kwargs: None)
        summary = tables._apply_native_txt_table_priority(
            [[{"type": "table", "bbox": [0.0, 0.0, 1.0, 1.0]}]],
            [[layout]],
            [page],
            images,
            effort="high",
            page_text_geometries=geometries,
            page_vector_geometries=vectors,
            page_snapshots=cache,
        )
        assert summary.errors == 0 and summary.total == 1
        assert geometries == [None] and vectors == [None]
        full = cache[0].full_snapshot
        assert full is not None
        captured = []
        original = content.prepare_text_evidence

        def prepare(*args: Any, **kwargs: Any) -> Any:
            """确认正文直接走 snapshot 参数，保留真实 DocVortex 证据计算。"""
            assert kwargs.get("snapshot") is full and "geometry" not in kwargs
            result = original(*args, **kwargs)
            captured.append(result.geometry)
            return result

        def fill(*args: Any, page_text_geometry: Any, **kwargs: Any) -> list:
            """字符回填使用证据联合物化的同一几何，禁止再次打开文本页。"""
            assert page_text_geometry is captured[0]
            return []

        monkeypatch.setattr(content, "prepare_text_evidence", prepare)
        monkeypatch.setattr(content, "_fill_native_pdf_text_spans", fill)
        monkeypatch.setattr(content, "_group_page_spans_by_block", lambda *_: {})
        monkeypatch.setattr(content, "_apply_window_post_ocr", lambda *_: None)
        monkeypatch.setattr(content, "_apply_block_content_and_line_metadata", lambda *_: None)
        monkeypatch.setattr(content, "apply_text_evidence", lambda *_: None)
        content._fill_window_block_content_and_lines(
            images,
            [page],
            [[]],
            [[]],
            [[]],
            "txt",
            "high",
            set(),
            MagicMock(),
            geometries,
            page_vector_geometries=vectors,
            page_snapshots=cache,
        )
        assert geometries[0] is captured[0]
        assert cache == [None]
        owner_getter.assert_called_once_with()
        full_getter.assert_called_once_with()
        legacy_getter.assert_not_called()


def test_explicit_geometry_mutation_wins_over_cached_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """已经提供的可变字符即使伴随 owned 缓存也必须原样交给表格入口。"""
    with PDFDocument(_pdf()) as document, Image.new("RGB", (200, 200)) as image:
        page = document[0]
        owner = _require_owned(page)
        geometry = page.get_chars_with_geometry()
        geometry.chars[0]["char"] = "changed"
        cache = [snapshots._PageSnapshotEntry(ref(page), owner)]
        captured = []
        original = tables.prepare_table_page

        def prepare(*args: Any, **kwargs: Any) -> Any:
            """保留显式对象身份和修改，不能以旧 snapshot 偷换它。"""
            assert kwargs["geometry"] is geometry and "snapshot" not in kwargs
            assert kwargs["geometry"].chars[0]["char"] == "changed"
            result = original(*args, **kwargs)
            captured.append(result)
            return result

        monkeypatch.setattr(tables, "prepare_table_page", prepare)
        monkeypatch.setattr(tables, "recover_table_region", lambda *_args, **_kwargs: None)
        tables._apply_native_txt_table_priority(
            [[{"type": "table", "bbox": [0.0, 0.0, 1.0, 1.0]}]],
            [[]],
            [page],
            [{"img_pil": image, "scale": 1}],
            effort="high",
            page_text_geometries=[geometry],
            page_snapshots=cache,
        )
        assert captured[0].geometry is geometry
        snapshots.clear_page_snapshot_cache(cache)


def test_window_close_releases_payload_even_when_entry_is_still_referenced() -> None:
    """关闭窗口必须清除条目中的强引用，不只清空外层列表。"""
    with PDFDocument(_pdf()) as document:
        page = document[0]
        owner, full = _Payload(), _Payload()
        owner_ref, full_ref = ref(owner), ref(full)
        entry = snapshots._PageSnapshotEntry(ref(page), owner, full)
        cache = [entry]
        state = _state(page, cache)
        del owner, full
        assert owner_ref() is not None and full_ref() is not None
        state.close()
        assert owner_ref() is None and full_ref() is None
        assert entry.text_owner is None and entry.full_snapshot is None and cache == []


def test_prepare_failure_releases_snapshot_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """准备尚未返回 state 时失败也释放 owned 强引用并关闭已渲染图片。"""
    with PDFDocument(_pdf()) as document:
        page = document[0]
        owner = _Payload()
        owner_ref = ref(owner)
        entry = snapshots._PageSnapshotEntry(ref(page), owner)
        cache = [entry]
        del owner
        monkeypatch.setattr(window, "_get_window_pdf_pages", lambda *_: [page])
        monkeypatch.setattr(window, "create_page_snapshot_cache", lambda *_args, **_kwargs: cache)
        image = Image.new("RGB", (10, 10))
        monkeypatch.setattr(window, "load_images_from_pdf_bytes_range", lambda **_: [{"img_pil": image, "scale": 1}])
        model = MagicMock()
        model.layout_model.batch_predict.side_effect = RuntimeError("layout failed")
        with pytest.raises(RuntimeError, match="layout failed"):
            window._prepare_pdf_window(
                b"",
                document,
                window._ProcessingWindow(0, 1, 0, 0),
                page_count=1,
                effort="high",
                parse_mode="txt",
                hybrid_model=model,
            )
        assert cache == [] and owner_ref() is None


def test_async_cancellation_releases_owned_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """VLM 等待期间取消，原有 finally 路径须清空快照并释放数据。"""
    with PDFDocument(_pdf()) as document:
        page = document[0]
        owner = _Payload()
        owner_ref = ref(owner)
        cache = [snapshots._PageSnapshotEntry(ref(page), owner)]
        state = _state(page, cache)
        del owner
        monkeypatch.setattr(window, "_prepare_locked_window", lambda *_args, **_kwargs: state)
        monkeypatch.setattr(window, "_inference_options", lambda *_args: {})
        monkeypatch.setattr(window, "get_document_render_session", lambda *_args: None)
        monkeypatch.setattr(window, "trim_process_heap", lambda: None)

        async def run() -> None:
            """等待推理真正开始再取消，避免只覆盖尚未取得资源的情况。"""
            entered = asyncio.Event()

            async def infer(**kwargs: Any) -> None:
                """模拟持有窗口的远端推理等待。"""
                entered.set()
                await asyncio.Event().wait()

            predictor = MagicMock()
            predictor.aio_batch_extract_with_layout = AsyncMock(side_effect=infer)
            task = asyncio.create_task(
                window.aio_process_pdf_windows(
                    b"",
                    document,
                    effort="high",
                    parse_mode="txt",
                    image_analysis=False,
                    hybrid_model=MagicMock(),
                    vlm_predictor=predictor,
                )
            )
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        assert cache == [] and owner_ref() is None


def test_standard_text_does_not_eagerly_group_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """普通无旋转文字即使进入回填函数也不提前生成基础行。"""
    page = MagicMock()
    page.get_char_count.return_value = 1
    blocked = MagicMock(side_effect=AssertionError("eager grouping"))
    monkeypatch.setattr(native, "get_lines_from_chars", blocked)
    char = {"char": "A", "char_idx": 0, "rotation": 0.0, "bbox": Bbox([0.0, 0.0, 5.0, 10.0]), "font": {}}
    assert native.txt_spans_extract(page, [], object(), 1.0, [], [], page_chars=[char]) == []
    blocked.assert_not_called()


def test_rotated_and_vertical_fill_share_lazy_grouping(monkeypatch: pytest.MonkeyPatch) -> None:
    """局部旋转筛选和竖排回填共用一次基于真实可变字符的组行。"""
    chars = [
        {"char": "A", "char_idx": 0, "rotation": 0.0, "bbox": Bbox([0.0, 0.0, 5.0, 10.0]), "font": {}},
        {"char": "B", "char_idx": 1, "rotation": 0.1, "bbox": Bbox([5.0, 0.0, 10.0, 10.0]), "font": {}},
    ]
    grouped = [{"bbox": Bbox([30.0, 0.0, 35.0, 100.0]), "rotation": 0.0, "spans": [{"text": "AB", "chars": chars}]}]
    reader = MagicMock(return_value=grouped)
    monkeypatch.setattr(native, "get_lines_from_chars", reader)
    monkeypatch.setattr(native, "fill_char_in_spans", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(native, "_prepare_post_ocr_spans", lambda _needed, spans, *_: spans)
    page = MagicMock()
    page.get_char_count.return_value = 2
    spans = [
        _AnalyzeSpan(ContentType.TEXT, (0.0, 0.0, 20.0, 10.0)),
        _AnalyzeSpan(ContentType.TEXT, (0.0, 15.0, 20.0, 25.0)),
        _AnalyzeSpan(ContentType.TEXT, (30.0, 0.0, 35.0, 100.0)),
    ]
    native.txt_spans_extract(
        page, spans, object(), 1.0, [(0.0, 0.0, 200.0, 200.0, None, None, None, BlockType.TEXT)], [], page_chars=chars
    )
    reader.assert_called_once_with(chars)
    assert spans[2].content == "AB"


def test_raw_character_limit_precedes_owned_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """正文继续按原始字符数限制回退，不先创建快照或用去重后的数量替代。"""
    with PDFDocument(_pdf()) as document, Image.new("RGB", (200, 200)) as image:
        page = document[0]
        monkeypatch.setattr(page, "get_char_count", lambda: native.MAX_NATIVE_TEXT_CHARS_PER_PAGE + 1)
        blocked = MagicMock(side_effect=AssertionError("high character count opened owned snapshot"))
        monkeypatch.setattr(page, "get_text_snapshot", blocked)
        monkeypatch.setattr(content, "prepare_text_evidence", blocked)
        monkeypatch.setattr(content, "_fill_native_pdf_text_spans", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(content, "_group_page_spans_by_block", lambda *_: {})
        monkeypatch.setattr(content, "_apply_window_post_ocr", lambda *_: None)
        monkeypatch.setattr(content, "_apply_block_content_and_line_metadata", lambda *_: None)
        monkeypatch.setattr(content, "apply_text_evidence", lambda *_: None)
        cache = [None]
        content._fill_window_block_content_and_lines(
            [{"img_pil": image, "scale": 1.0}],
            [page],
            [[]],
            [[]],
            [[]],
            "txt",
            "xhigh",
            set(),
            MagicMock(),
            [None],
            page_snapshots=cache,
        )
        assert cache == [None]
        blocked.assert_not_called()
