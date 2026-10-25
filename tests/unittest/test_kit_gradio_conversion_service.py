"""不构建 Gradio 界面的转换服务行为与结果生命周期回归。"""

from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from mineru.kit.gradio.conversion import (
    ConversionCancelled,
    ConversionFailure,
    ConversionInput,
    ConversionRun,
    ConversionSuccess,
)
from mineru.kit.gradio.conversion_service import ConversionService
from mineru.kit.gradio.status import STATUS_COMPLETED, STATUS_PROCESSING_ON_SERVER, ParseStatusUpdate
from mineru.parser.base import ParseResult
from tests.unittest.test_kit_gradio import _middle_json


class _ParseClient:
    """记录服务实际选择的参数，只通过弱引用观察解析结果生命周期。"""

    def __init__(self, *, cancel: bool = False) -> None:
        """保存可控取消行为与调用记录，不缓存解析结果。"""
        self.cancel = cancel
        self.calls: list[tuple[Path, str, str, str]] = []
        self.result_ref: weakref.ReferenceType[ParseResult] | None = None

    async def parse_file(
        self,
        path: Path,
        *,
        tier: str,
        page_range: str,
        ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
        status_callback: Callable[[str | ParseStatusUpdate], None] | None = None,
    ) -> ParseResult:
        """构造一次性解析结果，避免模拟客户端自身延长结果生命周期。"""
        self.calls.append((path, tier, page_range, ocr_mode))
        if self.cancel:
            raise asyncio.CancelledError
        if status_callback is not None:
            status_callback(STATUS_PROCESSING_ON_SERVER)
        result = ParseResult(middle_json=_middle_json(with_image=False, file_suffix="csv"))
        self.result_ref = weakref.ref(result)
        return result


def test_service_returns_materialized_output_without_retaining_parse_result(tmp_path: Path) -> None:
    """直接转换 CSV，验证实际档位、具名结果和会话缓存均不保留解析树。"""
    source = tmp_path / "source.csv"
    source.write_text("name,value\na,1\n", encoding="utf-8")
    client = _ParseClient()
    run = ConversionRun("direct-conversion")

    async def scenario() -> None:
        """原生格式强制 Flash，忽略 PDF 选页与 OCR，并物化可下载结果。"""
        service = ConversionService(
            client,
            tier_choices=("flash", "basic"),
            output_root=tmp_path / "output",
            max_pages=10,
            slot=asyncio.Semaphore(1),
        )
        outcome = await service.execute(
            run,
            ConversionInput(str(source), 1, "1-3", True),
            public_base_url="https://example.test/mineru",
        )
        assert isinstance(outcome, ConversionSuccess)
        assert client.calls == [(source.resolve(), "flash", "", "auto")]
        output = outcome.output
        assert output.artifacts.middle_json_path.is_file()
        assert output.artifacts.structured_content_path.read_text(encoding="utf-8") == output.structured_json
        assert output.html
        assert output.preview.show_ready_placeholder and not output.preview.preserve_source
        assert output.preview.pdf_path is None and output.preview.image_path is None
        assert run.artifacts == output.artifacts.as_state()
        assert run.task is None and run.state.message == STATUS_COMPLETED
        await asyncio.sleep(0)
        gc.collect()
        assert client.result_ref is not None and client.result_ref() is None

    asyncio.run(scenario())


def test_service_rejects_input_before_waiting_for_slot(tmp_path: Path) -> None:
    """无文件请求直接形成失败结果，不等待已占用的解析槽。"""
    client = _ParseClient()
    run = ConversionRun("invalid-input")

    async def scenario() -> None:
        """使用没有空位的执行槽，确保输入错误不进入排队和解析。"""
        service = ConversionService(
            client,
            tier_choices=("flash",),
            output_root=tmp_path,
            max_pages=None,
            slot=asyncio.Semaphore(0),
        )
        outcome = await asyncio.wait_for(
            service.execute(run, ConversionInput(None, 0, ""), public_base_url="https://example.test"),
            timeout=1,
        )
        assert isinstance(outcome, ConversionFailure) and outcome.message == "No input file"
        assert run.terminal and run.state.message == "Failed: No input file"
        assert run.task is None and run.artifacts is None and not client.calls

    asyncio.run(scenario())


def test_service_returns_cancelled_outcome_and_releases_slot(tmp_path: Path) -> None:
    """解析取消只返回取消结果，不能生成成功素材或留下执行槽。"""
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    run = ConversionRun("cancelled-input")

    async def scenario() -> None:
        """等待客户端取消传播完成后检查任务句柄、素材状态和槽位。"""
        slot = asyncio.Semaphore(1)
        service = ConversionService(
            _ParseClient(cancel=True),
            tier_choices=("flash",),
            output_root=tmp_path / "output",
            max_pages=None,
            slot=slot,
        )
        outcome = await service.execute(run, ConversionInput(str(source), 0, ""), public_base_url="https://example.test")
        assert isinstance(outcome, ConversionCancelled)
        assert run.cancelled and run.task is None and run.artifacts is None
        assert not slot.locked()
        assert not (tmp_path / "output").exists()

    asyncio.run(scenario())
