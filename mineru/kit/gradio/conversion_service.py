"""WebUI 转换业务：输入校验、解析排队、素材持久化和预览准备。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from loguru import logger

from ...errors import MineruError
from ...filetypes import (
    FLASH_ONLY_PARSE_EXTENSIONS,
    HTML_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MHTML_EXTENSIONS,
    OFFICE_EXTENSIONS,
    PARSEABLE_EXTENSIONS,
    PDF_EXTENSIONS,
)
from ...parser.base import ParseResult
from ...types import Tier
from .artifacts import RunArtifacts, render_html_preview
from .conversion import (
    ConversionCancelled,
    ConversionFailure,
    ConversionInput,
    ConversionOutcome,
    ConversionOutput,
    ConversionRun,
    ConversionSuccess,
    PreviewPlan,
    await_task_completion,
    run_sync_output,
)
from .document_io import GradioDocumentIO, LocalGradioDocumentIO
from .page_range import validate_max_pages
from .status import (
    STATUS_COMPLETED,
    STATUS_PREPARING_REQUEST,
    STATUS_PROCESSING_OUTPUT,
    STATUS_QUEUED_LOCALLY,
    ParseStatusUpdate,
)


class ParseClient(Protocol):
    """转换流程所需的最小解析接口，兼容远端与进程内客户端。"""

    async def parse_file(
        self,
        path: Path,
        *,
        tier: str,
        page_range: str,
        ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
        status_callback: Callable[[str | ParseStatusUpdate], None] | None = None,
    ) -> ParseResult:
        """解析输入文件，进度通过回调传递，取消遵守客户端已有执行边界。"""
        ...


@dataclass(frozen=True)
class _ValidatedInput:
    """保存已校验的源文件与实际档位，避免排队后重复判断输入。"""

    source_path: Path
    suffix: str
    tier: Tier


class ConversionService:
    """执行转换并返回具名结果，不接收 Gradio 请求或生成组件更新。"""

    def __init__(
        self,
        client: ParseClient,
        *,
        tier_choices: Sequence[Tier],
        output_root: Path,
        max_pages: int | None,
        slot: asyncio.Semaphore,
        document_io: GradioDocumentIO | None = None,
    ) -> None:
        """显式接收解析能力、产物目录、页数限制和共享执行槽。"""
        validate_max_pages(max_pages)
        self.client = client
        self.tier_choices = tuple(tier_choices)
        self.output_root = output_root
        self.max_pages = max_pages
        self.slot = slot
        self.document_io = document_io if document_io is not None else LocalGradioDocumentIO()

    async def execute(
        self,
        run: ConversionRun,
        inputs: ConversionInput,
        *,
        public_base_url: str,
    ) -> ConversionOutcome:
        """返回成功、失败或取消结果，并在所有退出路径回收后台任务句柄。"""
        try:
            validated = self._validate_input(inputs)
        except ValueError as exc:
            return self._failure(run, str(exc))
        loop = asyncio.get_running_loop()

        def emit(message: str | ParseStatusUpdate) -> None:
            """记录通知实际发生时间，再由事件循环串行更新会话快照。"""
            at = time.monotonic()
            loop.call_soon_threadsafe(lambda: run.publish(message, at=at))

        task = asyncio.create_task(self._convert(run, inputs, validated, emit, public_base_url=public_base_url))
        run.task = task
        try:
            output = await asyncio.shield(task)
            if run.cancelled:
                return ConversionCancelled()
            run.publish(STATUS_COMPLETED)
            return ConversionSuccess(output)
        except asyncio.CancelledError:
            run.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await await_task_completion(task)
            return ConversionCancelled()
        except Exception as exc:
            logger.exception("WebUI conversion failed run_id={}", run.run_id)
            message = f"{exc.code}: {exc}" if isinstance(exc, MineruError) else str(exc)
            return self._failure(run, message)
        finally:
            run.task = None

    def _validate_input(self, inputs: ConversionInput) -> _ValidatedInput:
        """在进入队列前校验文件和滑块位置，并应用原生格式的 Flash 限制。"""
        if not inputs.file_path:
            raise ValueError("No input file")
        source_path = Path(inputs.file_path).resolve()
        if not source_path.is_file():
            raise ValueError("input file does not exist")
        suffix = source_path.suffix.lower().lstrip(".")
        if suffix not in PARSEABLE_EXTENSIONS:
            raise ValueError(f"unsupported file type '.{suffix}'")
        if suffix in FLASH_ONLY_PARSE_EXTENSIONS:
            # 原生文档独立于 PDF/图片滑块，隐藏控件的旧值不能影响提交。
            selected_tier: Tier = "flash"
        else:
            position = inputs.tier_position
            if isinstance(position, bool) or position not in range(len(self.tier_choices)):
                raise ValueError("Invalid tier slider position")
            selected_tier = self.tier_choices[int(position)]
        return _ValidatedInput(source_path, suffix, selected_tier)

    async def _convert(
        self,
        run: ConversionRun,
        inputs: ConversionInput,
        validated: _ValidatedInput,
        emit: Callable[[str | ParseStatusUpdate], None],
        *,
        public_base_url: str,
    ) -> ConversionOutput:
        """同步工作退出前始终持有执行槽，输出完成后只缓存素材路径。"""
        if self.slot.locked():
            run.publish(STATUS_QUEUED_LOCALLY)
        async with self.slot:
            run.publish(STATUS_PREPARING_REQUEST)
            page_text = await run_sync_output(
                self.document_io.effective_page_range, validated.source_path, inputs.page_range, max_pages=self.max_pages
            )
            result = await self.client.parse_file(
                validated.source_path,
                tier=validated.tier,
                page_range=page_text,
                ocr_mode="ocr" if validated.suffix in PDF_EXTENSIONS and inputs.force_ocr else "auto",
                status_callback=emit,
            )
            # 先消费本轮已发出的解析通知，保持阶段和解析耗时的顺序。
            await asyncio.sleep(0)
            run.publish(STATUS_PROCESSING_OUTPUT)
            output_started = time.monotonic()
            artifacts = await run_sync_output(
                self.document_io.persist_parse_result,
                result,
                validated.source_path,
                output_root=self.output_root,
                page_range=page_text,
            )
            rendered_html = await run_sync_output(render_html_preview, artifacts, public_base_url=public_base_url)
            structured_json = await run_sync_output(artifacts.structured_content_path.read_text, encoding="utf-8")
            run.artifacts = artifacts.as_state()
            logger.debug(
                "WebUI output ready run_id={} artifacts={} elapsed={:.3f}s html_bytes={} json_bytes={} ready_at={:.3f}",
                run.run_id,
                artifacts.root.name,
                time.monotonic() - output_started,
                len(rendered_html.encode("utf-8")),
                len(structured_json.encode("utf-8")),
                time.time(),
            )
            return ConversionOutput(
                artifacts=artifacts,
                html=rendered_html,
                structured_json=structured_json,
                preview=_prepare_preview(validated.suffix, artifacts),
            )

    @staticmethod
    def _failure(run: ConversionRun, message: str) -> ConversionFailure:
        """将最终错误补充到失败快照，保留已有成功和取消终态保护。"""
        run.publish(f"Failed: {message}", final_failure=True)
        return ConversionFailure(message)


def _prepare_preview(suffix: str, artifacts: RunArtifacts) -> PreviewPlan:
    """仅描述结果预览数据，保留已有 Office、OFD、EPUB 和网页源预览。"""
    if suffix in OFFICE_EXTENSIONS | HTML_EXTENSIONS | MHTML_EXTENSIONS | {"ofd", "epub"}:
        return PreviewPlan(preserve_source=True)
    preview_path = artifacts.layout_pdf_path or artifacts.origin_pdf_path
    return PreviewPlan(
        preserve_source=False,
        pdf_path=preview_path,
        image_path=artifacts.source_path if suffix in IMAGE_EXTENSIONS and preview_path is None else None,
        show_ready_placeholder=preview_path is None,
    )


__all__ = ["ConversionService", "ParseClient"]
