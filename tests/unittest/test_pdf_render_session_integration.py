"""双仓真实渲染会话接入验证，模型用固定空结果隔离。"""

import asyncio
from typing import Any
from io import BytesIO
from types import SimpleNamespace
import pytest
from reportlab.pdfgen.canvas import Canvas
from docvortex.document.pdf import PDFDocument
from docvortex.document.pdf.render_session import shutdown_pdf_render_sessions
from mineru.backend.analysis.pdf import images, window


@pytest.fixture
def pdf_input() -> bytes:
    """生成两页不同尺寸 PDF，便于检测窗口错误复用。"""
    output = BytesIO()
    canvas = Canvas(output, pagesize=(80, 100))
    for index in range(2):
        canvas.setPageSize((80 + index * 10, 100))
        canvas.drawString(5, 40, f"Window {index}")
        canvas.showPage()
    canvas.save()
    return output.getvalue()


@pytest.fixture
def model_stub(monkeypatch: pytest.MonkeyPatch) -> Any:
    """只替换模型结果与回填，保留真实文档、窗口、渲染和关闭逻辑。"""
    monkeypatch.setenv("DOCVORTEX_PDF_RENDER_BACKEND", "session")
    monkeypatch.setenv("MINERU_PROCESSING_WINDOW_SIZE", "1")
    monkeypatch.setenv("MINERU_PDF_RENDER_THREADS", "1")

    def layout(pages: Any, **kwargs: Any) -> Any:
        """为空布局返回逐页容器，不加载神经模型。"""
        return [[] for _ in pages]

    def finish(state: Any, result: Any, **kwargs: Any) -> Any:
        """返回已经隔离的模型结果，保留窗口图像的实际生命周期。"""
        return result

    monkeypatch.setattr(window, "_finish_pdf_window", finish)
    monkeypatch.setattr(window, "trim_process_heap", lambda: None)
    yield SimpleNamespace(device="cpu", layout_model=SimpleNamespace(batch_predict=layout))
    shutdown_pdf_render_sessions()


def test_sync_windows_share_document_session(pdf_input: bytes, model_stub: SimpleNamespace) -> None:
    """同步两个窗口仅打开一次 worker 文档，由外层文档统一关闭。"""
    with PDFDocument(pdf_input) as document:
        result = window.process_pdf_windows(
            pdf_input,
            document,
            effort="medium",
            parse_mode="ocr",
            image_analysis=False,
            flash_txt_mode=False,
            hybrid_model=model_stub,
            vlm_predictor=None,
        )
        session = document.get_render_session()
        assert result == [[], []]
        assert len(session.worker_diagnostics) == 1
        assert not session._closed
    assert session._closed and session.close_diagnostics[0]["input_released"]


def test_async_windows_share_document_session(pdf_input: bytes, model_stub: SimpleNamespace) -> None:
    """异步两个窗口复用同一 worker 文档且保持页面顺序。"""

    async def predict(**kwargs: Any) -> Any:
        """返回与输入窗口对应的空模型输出。"""
        return [[] for _ in kwargs["images"]]

    with PDFDocument(pdf_input) as document:
        result = asyncio.run(
            window.aio_process_pdf_windows(
                pdf_input,
                document,
                effort="xhigh",
                parse_mode="ocr",
                image_analysis=False,
                hybrid_model=model_stub,
                vlm_predictor=SimpleNamespace(aio_batch_two_step_extract=predict),
            )
        )
        session = document.get_render_session()
        assert result == [[], []]
        assert len(session.worker_diagnostics) == 1
    assert session._closed


def test_async_cancel_reclaims_session_and_pixels(pdf_input: bytes, model_stub: SimpleNamespace) -> None:
    """取消异步推理时会话租约及时归还，窗口像素在外层文档关闭前释放。"""
    states = []
    original = window._prepare_locked_window

    def prepare(*args: Any, **kwargs: Any) -> Any:
        """记录真实窗口以检查取消后的资源容器。"""
        state = original(*args, **kwargs)
        states.append(state)
        return state

    async def scenario(document: PDFDocument) -> None:
        """在真实页图已准备后取消等待中的推理。"""
        ready = asyncio.Event()

        async def predict(**kwargs: Any) -> Any:
            """阻塞推理以制造可确定的异步取消边界。"""
            ready.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            window.aio_process_pdf_windows(
                pdf_input,
                document,
                effort="xhigh",
                parse_mode="ocr",
                image_analysis=False,
                hybrid_model=model_stub,
                vlm_predictor=SimpleNamespace(aio_batch_two_step_extract=predict),
            )
        )
        await asyncio.wait_for(ready.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    from unittest.mock import patch

    with PDFDocument(pdf_input) as document, patch.object(window, "_prepare_locked_window", prepare):
        asyncio.run(scenario(document))
        assert document.get_render_session()._closed
        assert states and states[0].images_list == [] and (states[0].np_images == [])


def test_host_timeout_closes_document_session(
    pdf_input: bytes, model_stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """宿主 timeout 显式传到引擎，失败清理会话而非静默回退。"""
    with PDFDocument(pdf_input) as document:
        session = document.get_render_session()
        session.render(image_type="base64_img")
        receive = session._receive

        def expired_response(worker, request_id, deadline, **kwargs):
            """用确定过期的响应截止时间覆盖各平台时钟粒度，保留真实失败清理。"""
            return receive(worker, request_id, float("-inf"), **kwargs)

        monkeypatch.setattr(session, "_receive", expired_response)
        with pytest.raises(TimeoutError):
            images.load_images_from_pdf_bytes_range(pdf_input, document=document, timeout=1.0)
        assert document.get_render_session()._closed


def test_requested_session_requires_engine_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """明确选择 session 但引擎不具备接口时直接报错。"""
    monkeypatch.setenv("DOCVORTEX_PDF_RENDER_BACKEND", "session")
    with pytest.raises(RuntimeError, match="get_render_session"):
        images.get_document_render_session(SimpleNamespace())
    monkeypatch.delattr(images._images, "get_pdf_render_backend")
    with pytest.raises(RuntimeError, match="does not support"):
        images.load_images_from_pdf_bytes_range(b"pdf")


def test_async_cancel_interrupts_active_render_wait(
    pdf_input: bytes, model_stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """取消准备阶段时主动终止渲染等待，不等待完整渲染超时才清理。"""
    import threading

    with PDFDocument(pdf_input) as document:
        session = images.get_document_render_session(document)
        ready = threading.Event()
        original = session._receive

        def receive(worker: Any, request_id: Any, deadline: Any, **kwargs: Any) -> Any:
            """在真实 worker 请求之后制造可取消的确定等待点。"""
            ready.set()
            session._cancelled.wait(10)
            return original(worker, request_id, deadline, **kwargs)

        monkeypatch.setattr(session, "_receive", receive)

        async def scenario() -> None:
            """渲染开始后取消协程并确认线程和输入资源均已退出。"""
            task = asyncio.create_task(
                window.aio_process_pdf_windows(
                    pdf_input,
                    document,
                    effort="xhigh",
                    parse_mode="ocr",
                    image_analysis=False,
                    hybrid_model=model_stub,
                    vlm_predictor=None,
                )
            )
            assert await asyncio.to_thread(ready.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)

        asyncio.run(scenario())
        assert session._closed and (not session._input.exists()) and (not session._workers)


def test_flash_visual_windows_share_session(
    pdf_input: bytes, model_stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flash 补图使用文档会话，两个稀疏窗口不重复打开 worker 文档。"""
    import docvortex.analyzers.native as native

    def predict(document: PDFDocument) -> Any:
        """固定两页图像块，用真实裁图验证 Flash 接入。"""
        return [[{"type": "image", "bbox": [0.1, 0.1, 0.5, 0.5]}] for _ in range(document.page_count)]

    monkeypatch.setattr(native, "PdfModel", lambda: SimpleNamespace(predict=predict))
    with PDFDocument(pdf_input) as document:
        result = window.process_pdf_windows(
            pdf_input,
            document,
            effort="flash",
            parse_mode="txt",
            image_analysis=False,
            flash_txt_mode=True,
            hybrid_model=None,
            vlm_predictor=None,
        )
        session = document.get_render_session()
        assert len(session.worker_diagnostics) == 1
        assert all((page[0]["image_base64"] for page in result))
    assert session._closed


def test_flash_text_only_does_not_create_session(
    pdf_input: bytes, model_stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """显式 session 模式下，纯文本 Flash 仍不创建映射输入或渲染租约。"""
    import docvortex.analyzers.native as native

    def predict(document: PDFDocument) -> Any:
        """固定无视觉素材的页面，隔离模型对会话惰性行为的影响。"""
        return [[{"type": "text", "bbox": [0.1, 0.1, 0.5, 0.5]}] for _ in range(document.page_count)]

    monkeypatch.setattr(native, "PdfModel", lambda: SimpleNamespace(predict=predict))
    with PDFDocument(pdf_input) as document:
        window.process_pdf_windows(
            pdf_input,
            document,
            effort="flash",
            parse_mode="txt",
            image_analysis=False,
            flash_txt_mode=True,
            hybrid_model=None,
            vlm_predictor=None,
        )
        assert document._render_session is None
