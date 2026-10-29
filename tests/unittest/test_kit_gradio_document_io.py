"""验证宿主文档操作的完整回调接入、应用隔离及取消边界。"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from mineru.kit.gradio import GradioDocumentIO, build_gradio_app
from mineru.kit.gradio.artifacts import RunArtifacts
from mineru.kit.gradio.client import V1ServerCapabilities
from mineru.kit.gradio.conversion import ConversionCancelled, ConversionInput, ConversionRun, ConversionSuccess
from mineru.kit.gradio.conversion_service import ConversionService
from mineru.kit.gradio.document_io import LocalGradioDocumentIO
from mineru.kit.gradio.page_range import PdfPageMetadata
from mineru.parser.base import ParseResult
from tests.unittest.test_kit_gradio import _middle_json, _pdf_bytes


class _RecordingDocumentIO(LocalGradioDocumentIO):
    """保留真实文档操作，只记录参数并允许单次失败注入。"""

    def __init__(self) -> None:
        """每个宿主实例独立保存调用记录和错误开关。"""
        self.calls: list[tuple[object, ...]] = []
        self.fail_at: Literal["range", "persist"] | None = None

    def pdf_page_metadata(self, file_path: str | None) -> PdfPageMetadata:
        """记录上传回调实际选择的宿主，再读取真实页数。"""
        self.calls.append(("metadata", file_path))
        return super().pdf_page_metadata(file_path)

    def effective_page_range(
        self,
        path: str | Path | None,
        raw_page_range: str | None,
        *,
        max_pages: int | None = None,
    ) -> str:
        """记录原始选页文本和上限，必要时模拟宿主预检失败。"""
        self.calls.append(("range", path, raw_page_range, max_pages))
        if self.fail_at == "range":
            raise ValueError("host range unavailable")
        return super().effective_page_range(path, raw_page_range, max_pages=max_pages)

    def persist_parse_result(
        self,
        result: ParseResult,
        source_path: Path,
        *,
        output_root: Path,
        page_range: str,
    ) -> RunArtifacts:
        """记录输出目录及规范化范围，不在记录中延长解析树生命周期。"""
        self.calls.append(("persist", source_path, output_root, page_range))
        if self.fail_at == "persist":
            raise OSError("host persistence unavailable")
        return super().persist_parse_result(result, source_path, output_root=output_root, page_range=page_range)


@pytest.mark.parametrize("callback_name", ["convert_handler", "convert_ui"])
def test_callbacks_share_injected_io_without_crossing_app_instances(tmp_path: Path, callback_name: str) -> None:
    """两个真实界面交错执行元数据和转换回调，验证宿主实例、选页和输出互不串用。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes(page_count=2))
    hosts = [_RecordingDocumentIO(), _RecordingDocumentIO()]
    clients = [
        SimpleNamespace(parse_file=AsyncMock(return_value=ParseResult(middle_json=_middle_json(page_indices=(index,)))))
        for index in range(2)
    ]
    capabilities = V1ServerCapabilities("http://localhost:1", ("flash",), ("default",), ("upload",))
    apps = [
        build_gradio_app(
            client,
            capabilities,
            output_root=tmp_path / f"output-{index}",
            enable_example=False,
            max_pages=index + 1,
            document_io=host,
        )
        for index, (client, host) in enumerate(zip(clients, hosts))
    ]

    async def scenario() -> None:
        """第二个应用构建后再运行第一个，防止全局替换被最终实例覆盖。"""
        for index, (app, host, client) in enumerate(zip(apps, hosts, clients)):
            callbacks = {function.name: function.fn for function in app.fns.values()}
            metadata = json.loads(callbacks["read_page_metadata"](str(source)))
            assert metadata == {"path": str(source), "page_count": 2, "error": ""}
            raw_range = f" {index + 1} "
            if callback_name == "convert_handler":
                outputs = await callbacks[callback_name](str(source), 0, raw_range, True)
                state = outputs[6]
            else:
                ticket = json.dumps({"run_id": f"{index + 1:032x}", "revision": 1})
                request = SimpleNamespace(session_hash="same-session")
                receipt = json.loads(await callbacks[callback_name](str(source), 0, raw_range, True, ticket, request))
                outputs = receipt["outputs"]
                state = {"root": str(Path(outputs[2]["value"]["path"]).parent)}
                assert Path(state["root"]).name == outputs[6]
                assert receipt["run_id"] == f"{index + 1:032x}"
            assert "Completed" in outputs[0] and outputs[1] and outputs[-1]
            assert Path(state["root"]).parent == tmp_path / f"output-{index}" / "gradio"
            assert host.calls == [
                ("metadata", str(source)),
                ("range", source.resolve(), raw_range, index + 1),
                ("persist", source.resolve(), tmp_path / f"output-{index}", str(index + 1)),
            ]
            client.parse_file.assert_awaited_once()
            assert client.parse_file.call_args.kwargs["page_range"] == str(index + 1)
            assert client.parse_file.call_args.kwargs["ocr_mode"] == "ocr"

    try:
        asyncio.run(scenario())
    finally:
        for app in apps:
            app.close()


@pytest.mark.parametrize("stage", ["range", "persist"])
def test_host_failure_returns_error_and_allows_next_conversion(tmp_path: Path, stage: Literal["range", "persist"]) -> None:
    """宿主页范围或保存失败不能生成可下载状态，也不能阻塞下一次转换。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    host = _RecordingDocumentIO()
    host.fail_at = stage
    client = SimpleNamespace(parse_file=AsyncMock(return_value=ParseResult(middle_json=_middle_json(with_image=False))))
    app = build_gradio_app(
        client,
        V1ServerCapabilities("http://localhost:1", ("flash",), ("default",), ("upload",)),
        output_root=tmp_path / "output",
        enable_example=False,
        document_io=host,
    )
    convert = next(function.fn for function in app.fns.values() if function.name == "convert_handler")

    async def scenario() -> None:
        """首次失败后解除故障，下一次请求必须正常完成。"""
        failed = await convert(str(source), 0, "1")
        error = "host range unavailable" if stage == "range" else "host persistence unavailable"
        assert f"Failed: {error}" in failed[0] and failed[6] is None
        assert client.parse_file.await_count == (0 if stage == "range" else 1)
        assert all(update["interactive"] is False for update in failed[8:15])
        host.fail_at = None
        completed = await convert(str(source), 0, "1")
        assert "Completed" in completed[0] and completed[6]

    try:
        asyncio.run(scenario())
    finally:
        app.close()


@pytest.mark.parametrize("stage", ["range", "persist"])
def test_cancelled_host_work_retains_slot_until_sync_operation_finishes(tmp_path: Path, stage: str) -> None:
    """注入的同步预检和保存被取消时，仍等待线程退出后才接纳下一份转换。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    started, release = threading.Event(), threading.Event()

    class BlockingDocumentIO(LocalGradioDocumentIO):
        """仅阻塞第一次指定操作，验证取消与后续请求的真实互斥。"""

        def wait_once(self, operation: str) -> None:
            """在可控事件上等待，异常路径也由测试最终释放。"""
            if operation == stage and not started.is_set():
                started.set()
                assert release.wait(10)

        def effective_page_range(
            self, path: str | Path | None, raw_page_range: str | None, *, max_pages: int | None = None
        ) -> str:
            """预检完成前持续占用转换槽。"""
            self.wait_once("range")
            return super().effective_page_range(path, raw_page_range, max_pages=max_pages)

        def persist_parse_result(
            self, result: ParseResult, source_path: Path, *, output_root: Path, page_range: str
        ) -> RunArtifacts:
            """保存完成前持续占用转换槽。"""
            self.wait_once("persist")
            return super().persist_parse_result(result, source_path, output_root=output_root, page_range=page_range)

    async def scenario() -> None:
        """失效旧任务并排队新任务，确认完成后只有新结果可被消费。"""
        slot = asyncio.Semaphore(1)
        host: GradioDocumentIO = BlockingDocumentIO()
        client = SimpleNamespace(parse_file=AsyncMock(return_value=ParseResult(middle_json=_middle_json(with_image=False))))
        service = ConversionService(
            client, tier_choices=("flash",), output_root=tmp_path / "output", max_pages=1, slot=slot, document_io=host
        )
        old_run, new_run = ConversionRun("old"), ConversionRun("new")
        inputs = ConversionInput(str(source), 0, "1")
        first = asyncio.create_task(service.execute(old_run, inputs, public_base_url="http://localhost:7860"))
        tasks = [first]
        try:
            assert await asyncio.to_thread(started.wait, 3)
            old_run.cancel()
            second = asyncio.create_task(service.execute(new_run, inputs, public_base_url="http://localhost:7860"))
            tasks.append(second)
            await asyncio.sleep(0.03)
            assert slot.locked() and not first.done() and not second.done()
            assert client.parse_file.await_count == (0 if stage == "range" else 1)
            assert "Queued locally" in new_run.state.message
            release.set()
            assert isinstance(await first, ConversionCancelled)
            assert isinstance(await second, ConversionSuccess)
            assert old_run.artifacts is None and new_run.artifacts is not None and not slot.locked()
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())
