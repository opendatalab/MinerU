"""上传级输入准备、复用、清理及无源 PDF 素材消费的回归。"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image
from docvortex.document.pdf import PDFDocument

from mineru.kit.gradio.document_io import LocalGradioDocumentIO
from mineru.kit.gradio.sources import PreparedSource, SessionSources, prepare_source
from mineru.kit.gradio.artifacts import persist_parse_result
from mineru.parser.base import ParseResult
from test_kit_gradio import _middle_json, _pdf_bytes


def test_pdf_preparation_preserves_bytes_and_image_has_one_pdf(tmp_path: Path) -> None:
    """PDF 原字节保存，图片预览与解析使用同一文件并保留原始类型。"""
    pdf = tmp_path / "中文 #?.pdf"
    pdf.write_bytes(_pdf_bytes(3))
    prepared = prepare_source(pdf, source_id="a" * 32, output_root=tmp_path)
    assert prepared.source_path.read_bytes() == pdf.read_bytes()
    assert prepared.parse_path == prepared.preview_pdf_path == prepared.source_path
    assert not list(prepared.root.glob("origin.pdf"))
    image = tmp_path / "photo.png"
    Image.new("RGBA", (40, 60), (255, 0, 0, 128)).save(image)
    prepared = prepare_source(image, source_id="b" * 32, output_root=tmp_path)
    assert prepared.file_suffix == "png"
    assert prepared.preview_pdf_path == prepared.parse_path != prepared.source_path
    assert len(list(prepared.root.glob("*.pdf"))) == 1
    assert PreparedSource.from_state(prepared.as_state(), source_root=tmp_path / "gradio/sources") == prepared


def test_shared_preparation_and_active_lease_cleanup(tmp_path: Path) -> None:
    """并发准备只执行一次，切换会话不能提前删除正在解析的文件。"""
    image = tmp_path / "one.png"
    Image.new("RGB", (20, 30), "red").save(image)
    io = LocalGradioDocumentIO()
    original = io.prepare_source
    io.prepare_source = Mock(wraps=original)

    async def scenario() -> None:
        """真实生成 PDF 并交错持有、失效和释放租期。"""
        sources = SessionSources(io, tmp_path)
        first, second = await asyncio.gather(*(sources.prepare("s", image, "a" * 32, 1) for _ in range(2)))
        assert first == second and io.prepare_source.call_count == 1
        acquired, lease = await sources.acquire("s", image)
        assert acquired == first and io.prepare_source.call_count == 1
        await sources.retire("s", revision=2)
        assert first.root.is_dir()
        await sources.release(lease)
        assert not first.root.exists()
        with pytest.raises(ValueError, match="expired"):
            await sources.prepare("s", image, "b" * 32, 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode,orientation", [("RGB", 6), ("CMYK", 1), ("L", 1)])
def test_image_preparation_keeps_public_conversion_rules(tmp_path: Path, mode: str, orientation: int) -> None:
    """EXIF 转正、CMYK 和灰度输入都由公共转换实现生成单页 PDF。"""
    source = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[274] = orientation
    with Image.new(mode, (40, 60)) as image:
        image.save(source, exif=exif)
    prepared = prepare_source(source, source_id="c" * 32, output_root=tmp_path)
    with PDFDocument(prepared.parse_path.read_bytes()) as document:
        assert document.page_count == 1
        width, height = document.page_size(0)
        assert width / height == pytest.approx(1.5 if orientation == 6 else 2 / 3, rel=0.001)


def test_repeated_image_conversion_reuses_prepared_pdf_bytes(tmp_path: Path) -> None:
    """上传准备、两次解析和预览共用一份 PDF，原图身份继续禁止图片选页。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mineru.kit.gradio.conversion import ConversionInput, ConversionRun, ConversionSuccess
    from mineru.kit.gradio.conversion_service import ConversionService

    image = tmp_path / "photo.png"
    Image.new("RGB", (20, 30), "red").save(image)
    io = LocalGradioDocumentIO()
    io.prepare_source = Mock(wraps=io.prepare_source)
    client = SimpleNamespace(parse_file=AsyncMock(return_value=ParseResult(middle_json=_middle_json())))

    async def scenario() -> None:
        """完整执行服务回调，比较实际交给解析客户端的路径和字节。"""
        service = ConversionService(
            client,
            tier_choices=("basic",),
            output_root=tmp_path,
            max_pages=1,
            slot=asyncio.Semaphore(1),
            document_io=io,
        )
        prepared = await service.sources.prepare("session", image, "a" * 32, 1)
        before = prepared.parse_path.read_bytes()
        for identity in ("first", "second"):
            run = ConversionRun(identity)
            outcome = await service.execute(
                run,
                ConversionInput(str(image), 0, "5,2", True, "session", prepared.source_id),
                public_base_url="http://test.local",
            )
            assert isinstance(outcome, ConversionSuccess)
            assert outcome.output.preview.pdf_path == prepared.preview_pdf_path
            assert outcome.output.preview.image_path is None
        assert io.prepare_source.call_count == 1
        assert client.parse_file.await_count == 2
        for call in client.parse_file.await_args_list:
            assert call.args[0] == prepared.preview_pdf_path
            assert call.args[0].read_bytes() == before
            assert call.kwargs["page_range"] == ""
            assert call.kwargs["ocr_mode"] == "auto"

    asyncio.run(scenario())


def test_successful_result_retains_source_but_failed_prepare_can_retry(tmp_path: Path) -> None:
    """成功结果引用的输入保留；失败准备清理目录且允许新修订重试。"""
    image = tmp_path / "bad.png"
    image.write_bytes(b"invalid")

    async def scenario() -> None:
        """生成失败不泄漏半成品目录，成功后的清除不破坏已有文件输出。"""
        sources = SessionSources(LocalGradioDocumentIO(), tmp_path)
        with pytest.raises(Exception):
            await sources.prepare("s", image, "a" * 32, 1)
        assert not (tmp_path / "gradio/sources" / ("a" * 32)).exists()
        Image.new("RGB", (20, 30), "blue").save(image)
        prepared = await sources.prepare("s", image, "b" * 32, 2)
        _, lease = await sources.acquire("s", image)
        await sources.release(lease, retain=True)
        await sources.retire("s")
        assert prepared.parse_path.is_file()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", [OSError, asyncio.CancelledError])
def test_api_retry_replaces_failed_preparation_task(tmp_path: Path, failure_type: type[BaseException]) -> None:
    """同会话 API 重试会重建失败或取消的准备任务，并继续合并并发重试。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    io = LocalGradioDocumentIO()
    io.prepare_source = Mock(wraps=io.prepare_source, side_effect=failure_type("preparation failed"))

    async def scenario() -> None:
        """故障恢复后沿用 API 会话和输入路径，显式浏览器身份仍禁止被替换。"""
        registry = SessionSources(io, tmp_path)
        with pytest.raises(failure_type):
            await registry.acquire("api-session", source)
        failed_id, failed_lease = registry.current["api-session"]
        with pytest.raises(failure_type):
            await registry.acquire("api-session", source, source_id=failed_id)
        assert io.prepare_source.call_count == 1 and failed_lease.users == 0
        io.prepare_source.side_effect = None
        first, second = await asyncio.gather(*(registry.acquire("api-session", source) for _ in range(2)))
        prepared, lease = first
        assert first == second and prepared.parse_path.is_file()
        assert prepared.source_id != failed_id and failed_lease.retired
        assert io.prepare_source.call_count == 2 and lease.users == 2
        await registry.release(lease)
        await registry.release(second[1])
        await registry.retire("api-session")
        await asyncio.gather(*registry.cleanups)
        assert not prepared.root.exists()

    asyncio.run(scenario())


def test_prepared_source_rejects_path_escape(tmp_path: Path) -> None:
    """共享源目录只能包含本次输入文件，不能扩大为任意 output root 路径。"""
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(_pdf_bytes())
    prepared = prepare_source(pdf, source_id="a" * 32, output_root=tmp_path)
    state = prepared.as_state()
    state["parse_path"] = str(tmp_path / "another.pdf")
    with pytest.raises(ValueError, match="escapes"):
        PreparedSource.from_state(state, source_root=tmp_path / "gradio/sources")


def test_materialization_never_opens_pdf_and_bbox_only_is_legal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """只有语义 bbox 的块合法，已有素材保存不允许打开或渲染任何源 PDF。"""
    forbidden = Mock(side_effect=AssertionError("WebUI materialization must not open PDF"))
    monkeypatch.setattr(PDFDocument, "__init__", forbidden)
    middle = _middle_json(page_indices=(4, 8))
    middle.pages[1].blocks[1].content[0].image_base64 = None
    result = persist_parse_result(
        ParseResult(middle_json=middle), tmp_path / "unavailable.pdf", output_root=tmp_path, page_range="5,9"
    )
    forbidden.assert_not_called()
    assert result.initial_page_idx == 4
    assert len(list((result.root / "images").iterdir())) == 1
    assert not list(result.root.glob("*.pdf"))
    assert middle.pages[0].blocks[1].content[0].image_base64 is not None


def test_materialization_reports_invalid_payload_and_missing_file(tmp_path: Path) -> None:
    """实际声明的素材必须存在且有效，不借助 bbox 掩盖缺失或损坏。"""
    middle = _middle_json()
    body = middle.pages[0].blocks[1].content[0]
    body.image_base64 = "data:image/png;base64,aW52YWxpZA=="
    with pytest.raises(ValueError):
        persist_parse_result(ParseResult(middle_json=middle), tmp_path / "source.pdf", output_root=tmp_path, page_range="")
    body.image_base64 = None
    body.image_path = "images/missing.png"
    with pytest.raises(FileNotFoundError):
        persist_parse_result(ParseResult(middle_json=middle), tmp_path / "source.pdf", output_root=tmp_path, page_range="")


def test_new_source_ready_does_not_wait_for_retired_preparation(tmp_path: Path) -> None:
    """旧图片准备未退出时，新 PDF 仍可就绪，旧目录在线程结束后清理。"""
    import threading

    started = threading.Event()
    finish = threading.Event()
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    io = LocalGradioDocumentIO()
    original = io.prepare_source

    def delayed(path: Path, *, source_id: str, output_root: Path) -> PreparedSource:
        """只阻塞第一份准备，模拟快速替换大图片的场景。"""
        if source_id == "a" * 32:
            started.set()
            assert finish.wait(5)
        return original(path, source_id=source_id, output_root=output_root)

    io.prepare_source = delayed

    async def scenario() -> None:
        """新请求不依赖旧准备退出，失效回执也不能重新成为当前输入。"""
        registry = SessionSources(io, tmp_path)
        old = asyncio.create_task(registry.prepare("s", source, "a" * 32, 1))
        await asyncio.to_thread(started.wait, 2)
        try:
            current = await asyncio.wait_for(registry.prepare("s", source, "b" * 32, 2), timeout=1)
            assert current.source_id == "b" * 32 and not old.done()
        finally:
            finish.set()
        with pytest.raises(ValueError, match="expired"):
            await old
        await asyncio.gather(*registry.cleanups)
        assert not (tmp_path / "gradio/sources" / ("a" * 32)).exists()
        await registry.retire("s")

    asyncio.run(scenario())


def test_expired_conversion_cannot_replace_current_prepared_source(tmp_path: Path) -> None:
    """旧转换请求即使晚于新上传到达，也不能重新准备旧文件并覆盖当前身份。"""
    source = tmp_path / "one.pdf"
    source.write_bytes(_pdf_bytes())

    async def scenario() -> None:
        """过期 source_id 在取得解析租期之前失败，当前记录保持不变。"""
        registry = SessionSources(LocalGradioDocumentIO(), tmp_path)
        await registry.prepare("s", source, "a" * 32, 1)
        current = await registry.prepare("s", source, "b" * 32, 2)
        with pytest.raises(ValueError, match="expired"):
            await registry.acquire("s", source, source_id="a" * 32)
        assert registry.current["s"][0] == current.source_id
        await registry.retire("s")
        await asyncio.gather(*registry.cleanups)

    asyncio.run(scenario())


def test_cold_image_preparation_and_pdf_metadata_import_together(tmp_path: Path) -> None:
    """冷进程首次准备图片与读取 PDF 页数时保持导入锁顺序一致。"""
    Image.new("RGB", (20, 30), "red").save(tmp_path / "image.png")
    (tmp_path / "source.pdf").write_bytes(_pdf_bytes())
    code = """
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from mineru.kit.gradio.sources import prepare_source
from mineru.kit.gradio.page_range import pdf_page_metadata
root = Path(sys.argv[1])
with ThreadPoolExecutor(2) as pool:
    image = pool.submit(prepare_source, root / 'image.png', source_id='a' * 32, output_root=root)
    metadata = pool.submit(pdf_page_metadata, str(root / 'source.pdf'))
    assert image.result().parse_path.is_file()
    assert metadata.result()['page_count'] == 1
"""
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
