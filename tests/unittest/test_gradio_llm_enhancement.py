"""验证输出增强的语义结果、计时、超时、取消和公开界面参数。"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mineru.backend.postprocess import llm_client
from mineru.config import LLMAidedConfig
from mineru.kit.gradio import app as gradio_app
from mineru.kit.gradio import conversion_service, llm_enhancement
from mineru.kit.gradio.client import V1ServerCapabilities
from mineru.kit.gradio.conversion import ConversionInput, ConversionRun, ConversionSuccess
from mineru.kit.gradio.conversion import ConversionFailure
from mineru.kit.gradio.conversion_service import ConversionService
from mineru.kit.gradio.llm_enhancement import enhance_outputs
from mineru.kit.gradio.status import STATUS_PROCESSING_ON_SERVER, STATUS_PROCESSING_OUTPUT, ParseStatusUpdate
from mineru.parser.base import ParseResult
from mineru.types import BlockType, DocTitleBlock, MiddleJson, PageInfo, ParagraphTitleBlock
from _span_test_utils import inline
from test_kit_gradio import _middle_json as simple_middle_json
from test_kit_gradio_conversion_service import _ParseClient
from test_llm_aided_postprocess import _cross_page_table_pages, _llm_config, _merged_row_texts, _middle_json


def test_advanced_options_hover_keyboard_and_file_switches() -> None:
    """运行真实前端控制器，验证悬停延迟、键盘、触摸和清除后的显隐。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend state tests")
    result = subprocess.run(
        [node, str(Path(__file__).with_name("test_kit_gradio_advanced_options.cjs"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class _SDK:
    """模拟真实 SDK 请求边界，保留请求参数、取消和关闭证据。"""

    def __init__(self) -> None:
        """默认返回合法结果，可按标题内容阻塞某一个请求。"""
        self.chat = SimpleNamespace(completions=self)
        self.requests: list[dict[str, Any]] = []
        self.options: dict[str, Any] = {}
        self.block_title = ""
        self.invalid = False
        self.started = asyncio.Event()
        self.cancelled = False
        self.closed = False

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        """模拟标题、表格合法响应和可取消的慢请求。"""
        self.requests.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if self.block_title and self.block_title in prompt:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        content = "invalid" if self.invalid else "[1, 0]" if "JSON 整数数组" in prompt else '{"0": 3}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    async def close(self) -> None:
        """记录客户端关闭，不启动任何网络连接。"""
        self.closed = True


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> _SDK:
    """替换底层 SDK，仍执行真实的重试、校验、共享并发和写回。"""
    instance = _SDK()

    def create_sdk(**kwargs: Any) -> _SDK:
        """记录 SDK 级超时和重试配置。"""
        instance.options = kwargs
        return instance

    monkeypatch.setattr(llm_client, "AsyncOpenAI", create_sdk)
    return instance


def _title_pages(indices: tuple[int, ...]) -> MiddleJson:
    """用两个独立标题组覆盖选页、部分成功和超时场景。"""
    pages = [
        PageInfo(
            page_idx=page_idx,
            blocks=[
                DocTitleBlock(type=BlockType.DOC_TITLE, index=0, content=inline(f"Document {i}"), level=1),
                ParagraphTitleBlock(type=BlockType.PARAGRAPH_TITLE, index=1, content=inline(f"Section {i}"), level=2),
            ],
        )
        for i, page_idx in enumerate(indices)
    ]
    return _middle_json(pages, is_full_document=False)


@pytest.mark.parametrize("indices", [(0,), (0, 19), (4, 19), (0, 4, 19)])
def test_output_enhancement_levels_partial_documents_without_changing_identity(sdk: _SDK, indices: tuple[int, ...]) -> None:
    """所有选页均可分级，但不伪造整本标识和原页号。"""
    middle = _title_pages(indices)
    warnings = asyncio.run(
        enhance_outputs(middle, _llm_config(title=True), title_leveling=None, cross_page_table_cell_merge=None)
    )
    assert warnings == ()
    assert not middle.is_full_document
    assert tuple(page.page_idx for page in middle.pages) == indices
    assert [page.blocks[1].level for page in middle.pages] == [3] * len(indices)
    assert sdk.options["timeout"] == 20 and sdk.options["max_retries"] == 0
    assert sdk.closed


@pytest.mark.parametrize("title,cells", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_switches_change_real_semantics(sdk: _SDK, title: bool, cells: bool) -> None:
    """检查标题层级和真实渲染行，不只断言函数被调用。"""
    pages = _cross_page_table_pages()
    pages.append(_title_pages((2,)).pages[0])
    middle = _middle_json(pages)
    warnings = asyncio.run(
        enhance_outputs(middle, _llm_config(title=True, table=True), title_leveling=title, cross_page_table_cell_merge=cells)
    )
    assert warnings == ()
    assert middle.pages[2].blocks[1].level == (3 if title else 2)
    assert middle.pages[1].blocks[0].cell_merge == ([1, 0] if cells else None)
    if cells:
        assert _merged_row_texts(middle) == [["H1", "H2"], ["A B", "X"], ["Y"]]
    assert len(sdk.requests) == int(title) + int(cells)
    assert sdk.closed is (title or cells)


def test_timeout_keeps_completed_title_groups_and_closes_client(sdk: _SDK, monkeypatch: pytest.MonkeyPatch) -> None:
    """同一功能的一个候选超时，不丢弃先完成的候选，也不遗留后台请求。"""
    sdk.block_title = "Section 1"
    monkeypatch.setattr(llm_enhancement, "ENHANCEMENT_TIMEOUT_SECONDS", 0.05)
    middle = _title_pages((0, 1))
    warnings = asyncio.run(
        enhance_outputs(middle, _llm_config(title=True), title_leveling=True, cross_page_table_cell_merge=False)
    )
    assert warnings == ("llm_timeout", "llm_title_incomplete")
    assert [page.blocks[1].level for page in middle.pages] == [3, 2]
    assert sdk.cancelled and sdk.closed


def test_user_cancellation_propagates_and_drains_requests(sdk: _SDK) -> None:
    """用户取消不会被当成普通增强失败，未完成请求和客户端均回收。"""
    sdk.block_title = "Section 0"

    async def scenario() -> None:
        """在请求进入后取消，检查取消真正完成才返回。"""
        task = asyncio.create_task(
            enhance_outputs(_title_pages((0,)), _llm_config(title=True), title_leveling=True, cross_page_table_cell_merge=False)
        )
        await sdk.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sdk.cancelled and sdk.closed

    asyncio.run(scenario())


def test_invalid_response_retries_three_times_then_preserves_original(sdk: _SDK) -> None:
    """无效响应仅按业务层重试三次，输出保留原层级并提示失败功能。"""
    sdk.invalid = True
    middle = _title_pages((0,))
    warnings = asyncio.run(
        enhance_outputs(middle, _llm_config(title=True), title_leveling=True, cross_page_table_cell_merge=False)
    )
    assert warnings == ("llm_title_incomplete",)
    assert middle.pages[0].blocks[1].level == 2
    assert len(sdk.requests) == 3 and sdk.closed


def test_empty_candidates_and_missing_credentials_do_not_send_requests(sdk: _SDK) -> None:
    """无候选不是失败；密钥缺失只降级，不创建客户端或发送请求。"""
    empty = _middle_json([])
    assert (
        asyncio.run(
            enhance_outputs(empty, _llm_config(title=True, table=True), title_leveling=True, cross_page_table_cell_merge=True)
        )
        == ()
    )
    sdk.options.clear()
    warnings = asyncio.run(enhance_outputs(empty, LLMAidedConfig(), title_leveling=True, cross_page_table_cell_merge=True))
    assert warnings == ("llm_unavailable",)
    assert sdk.requests == [] and sdk.options == {}


def test_non_pdf_skips_enhancement_even_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """服务端格式守卫覆盖直接 API 请求，非适用文件没有增强调用和提示。"""
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")

    async def forbidden(*args: Any, **kwargs: Any) -> tuple[str, ...]:
        """非适用格式不得进入增强或创建客户端。"""
        pytest.fail("non-PDF must not enter LLM enhancement")

    monkeypatch.setattr(conversion_service, "enhance_outputs", forbidden)

    async def scenario() -> None:
        """通过完整服务路径产生真实文件并检查没有警告。"""
        run = ConversionRun("non-pdf")
        service = ConversionService(
            _ParseClient(),
            tier_choices=("flash",),
            output_root=tmp_path,
            max_pages=20,
            slot=asyncio.Semaphore(1),
            llm_aided_config=_llm_config(title=True, table=True),
        )
        outcome = await service.execute(
            run, ConversionInput(str(source), 0, "", False, True, True), public_base_url="http://localhost"
        )
        assert isinstance(outcome, ConversionSuccess)
        assert not run.state.warning_keys

    asyncio.run(scenario())


@pytest.mark.parametrize("extension", ["pdf", "png"])
def test_enhancement_runs_after_parse_timer_and_before_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extension: str
) -> None:
    """用受控增强等待验证 PDF 和图片的时序，并检查落盘与预览采用增强结果。"""
    source = tmp_path / f"source.{extension}"
    source.touch()
    run = ConversionRun("output-timing")
    entered = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []
    # CSV 语义避免测试依赖 PDF 渲染，输入后缀仍真实走 PDF/图片增强分支。
    result = ParseResult(middle_json=simple_middle_json(with_image=False, file_suffix="csv"))

    async def parse_file(*args: Any, **kwargs: Any) -> ParseResult:
        """发出真实解析阶段和固定服务端耗时。"""
        order.append("parse")
        kwargs["status_callback"](STATUS_PROCESSING_ON_SERVER)
        kwargs["status_callback"](ParseStatusUpdate(STATUS_PROCESSING_OUTPUT, duration_ms=1250))
        return result

    async def enhance(*args: Any, **kwargs: Any) -> tuple[str, ...]:
        """暂停增强并确认计时已经停止，再写入可由真实持久化观察的标题。"""
        assert run.state.message == STATUS_PROCESSING_OUTPUT and run.state._processing_started is None
        order.append("llm")
        entered.set()
        await release.wait()
        result.middle_json.pages[0].blocks[0] = ParagraphTitleBlock(
            type=BlockType.PARAGRAPH_TITLE, index=0, content=inline("Enhanced title"), level=3
        )
        return ("llm_cells_incomplete",)

    class DocumentIO:
        """复用真实保存过程，同时隔离源文件 PDF 操作。"""

        def effective_page_range(self, *args: Any, **kwargs: Any) -> str:
            """返回固定范围，避免本测试依赖源 PDF 有效性。"""
            return ""

        def persist_parse_result(self, parsed: ParseResult, original: Path, **kwargs: Any) -> Any:
            """用真实 CSV 文件保存同一增强语义，验证预览与下载的数据源。"""
            from mineru.kit.gradio.artifacts import persist_parse_result

            order.append("persist")
            actual_source = tmp_path / "actual.csv"
            actual_source.write_text("a,b\n1,2\n", encoding="utf-8")
            return persist_parse_result(parsed, actual_source, **kwargs)

    monkeypatch.setattr(conversion_service, "enhance_outputs", enhance)

    async def scenario() -> None:
        """增强阻塞期间没有落盘；完成后仍显示解析的 1.25 秒。"""
        service = ConversionService(
            SimpleNamespace(parse_file=parse_file),
            tier_choices=("flash",),
            output_root=tmp_path,
            max_pages=20,
            slot=asyncio.Semaphore(1),
            document_io=DocumentIO(),
            llm_aided_config=_llm_config(title=True, table=True),
        )
        task = asyncio.create_task(
            service.execute(run, ConversionInput(str(source), 0, ""), public_base_url="http://localhost")
        )
        await entered.wait()
        assert order == ["parse", "llm"] and run.state.server_elapsed == 1.25
        release.set()
        outcome = await task
        assert isinstance(outcome, ConversionSuccess)
        assert order == ["parse", "llm", "persist"]
        assert "1.25" in run.state.render() and "部分跨页单元格" in run.state.render()
        assert "### Enhanced title" in outcome.output.artifacts.markdown_path.read_text()
        assert "Enhanced title" in outcome.output.html and "Enhanced title" in outcome.output.structured_json
        assert json.loads(outcome.output.artifacts.middle_json_path.read_text())["pages"][0]["blocks"][0]["level"] == 3

    asyncio.run(scenario())


def test_configured_app_appends_optional_inputs_without_exposing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """实际 Gradio 配置保留原输入顺序、追加可选开关，并且不序列化密钥。"""
    config = _llm_config(title=True, table=True)
    config.api_key = "test-private-credential"
    demo = gradio_app.build_gradio_app(
        SimpleNamespace(),
        V1ServerCapabilities("in-process", ("flash",), ("default",), ("upload",)),
        output_root=tmp_path,
        enable_example=False,
        llm_aided_config=config,
    )
    try:
        convert = next(fn for fn in demo.fns.values() if fn.name == "convert_handler")
        assert len(convert.inputs) == 6
        assert convert.inputs[-2].value is True and convert.inputs[-1].value is True
        assert "mineru-title-leveling" in convert.inputs[-2].elem_classes
        assert "mineru-cell-merge" in convert.inputs[-1].elem_classes
        assert config.api_key not in json.dumps(demo.config, default=str)
        api = demo.get_api_info()["named_endpoints"][f"/{convert.api_name}"]["parameters"]
        assert [parameter["parameter_name"] for parameter in api[-2:]] == ["title_leveling", "cross_page_table_cell_merge"]
        assert all(parameter["parameter_has_default"] and parameter["parameter_default"] is True for parameter in api[-2:])
        received: list[ConversionInput] = []

        async def capture(
            self: ConversionService, run: ConversionRun, inputs: ConversionInput, **kwargs: Any
        ) -> ConversionFailure:
            """截取真实 Gradio 预处理后的输入，验证旧四参数调用正确使用新增默认值。"""
            received.append(inputs)
            return ConversionFailure("test request")

        monkeypatch.setattr(ConversionService, "execute", capture)

        async def api_call() -> None:
            """经官方客户端补默认值、Gradio 参数注入执行一次旧形式调用。"""
            from gradio_client.utils import construct_args

            arguments = construct_args(api, (None, 0, "", False), {})
            await demo.process_api(convert, arguments)

        asyncio.run(api_call())
        assert received[0].title_leveling is True and received[0].cross_page_table_cell_merge is True
        clear = next(block for block in demo.blocks.values() if block.__class__.__name__ == "ClearButton")
        for event in demo.config["dependencies"]:
            if (clear._id, "click") in event["targets"]:
                assert not set(event["outputs"]) & {component._id for component in convert.inputs[-2:]}
    finally:
        demo.close()
