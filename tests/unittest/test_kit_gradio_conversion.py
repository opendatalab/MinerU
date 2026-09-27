"""完整响应、任务隔离和不可取消输出线程的回归测试。"""

from __future__ import annotations

import asyncio
import inspect
import json
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from mineru.kit.gradio import app as gradio_app
from mineru.kit.gradio.client import V1ArtifactError, V1ServerCapabilities
from mineru.kit.gradio.conversion import ConversionRun, SessionConversions
from mineru.kit.gradio.status import STATUS_COMPLETED, STATUS_PROCESSING_ON_SERVER
from mineru.parser.base import ParseResult
from tests.unittest.test_kit_gradio import _middle_json, _pdf_bytes


def _application(tmp_path: Path, client: Any) -> Any:
    """使用真实组件注册转换和轮询事件，仅替换解析服务。"""
    return gradio_app.build_gradio_app(
        client,
        V1ServerCapabilities("http://unused.test", ("flash", "basic"), ("zip",), ("file_id",)),
        output_root=tmp_path / "output",
        enable_example=False,
    )


def _callback(app: Any, name: str) -> Any:
    """按名称读取事件，避免依赖组件数量和内部编号。"""
    return next(fn.fn for fn in app.fns.values() if fn.name == name)


def _ticket(index: int) -> str:
    """生成有序浏览器提交，用于模拟响应乱序和取消先到。"""
    return json.dumps({"run_id": f"{index:032x}", "revision": index})


def _conversions(app: Any) -> SessionConversions:
    """读取公开回调持有的注册表，验证请求结束后的资源所有权。"""
    return inspect.getclosurevars(_callback(app, "convert_handler")).nonlocals["conversions"]


@pytest.mark.parametrize("with_session", [False, True])
@pytest.mark.parametrize("outcome", ["success", "input_error", "parse_error", "cancelled"])
def test_public_conversion_releases_runs(tmp_path: Path, with_session: bool, outcome: str) -> None:
    """公开请求的所有退出路径都释放任务，成功响应中的素材仍能独立下载。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    client = SimpleNamespace(parse_file=AsyncMock(return_value=ParseResult(middle_json=_middle_json(with_image=False))))
    if outcome == "parse_error":
        client.parse_file.side_effect = V1ArtifactError("controlled failure")
    elif outcome == "cancelled":
        client.parse_file.side_effect = asyncio.CancelledError()
    app = _application(tmp_path, client)
    convert = _callback(app, "convert_handler")
    downloader = _callback(app, "handler")
    conversions = _conversions(app)

    async def scenario() -> None:
        """连续使用不同 API 会话，确认注册表不会随请求数量增长。"""
        for index in range(3):
            request = SimpleNamespace(session_hash=f"api-{index}") if with_session else None
            response = await convert(None if outcome == "input_error" else str(source), 0, "", False, request)
            assert len(response) == 16
            assert conversions.runs == {}
            assert conversions.revisions == {}
            if outcome == "success":
                state = response[6]
                assert state and Path(state["root"]).is_dir()
                path, receipt = downloader(state, json.dumps({"run_id": response[7]}), request)
                assert path and Path(path).is_file() and not json.loads(receipt)["error"]
            elif outcome != "cancelled":
                assert "Failed:" in response[0] and response[6] is None

    asyncio.run(scenario())


def test_public_cleanup_preserves_replacement_ui_run(tmp_path: Path) -> None:
    """旧公开请求结束时只清理自身，不能删除同会话的新浏览器任务。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    first_started, second_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0

    async def parse_file(*_args: Any, **_kwargs: Any) -> ParseResult:
        """让旧请求等待取消，新任务保持运行直到检查完旧请求的清理。"""
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await asyncio.Future()
        second_started.set()
        await release.wait()
        return ParseResult(middle_json=_middle_json(with_image=False))

    app = _application(tmp_path, SimpleNamespace(parse_file=parse_file))
    public, ui, poll = (_callback(app, name) for name in ("convert_handler", "convert_ui", "read_conversion_status"))
    request = SimpleNamespace(session_hash="replacement")

    async def scenario() -> None:
        """以真实回调触发替换，并验证新任务完成后的快照仍可轮询。"""
        first = asyncio.create_task(public(str(source), 0, "", False, request))
        second = None
        try:
            await asyncio.wait_for(first_started.wait(), 3)
            second = asyncio.create_task(ui(str(source), 0, "", False, _ticket(1), request))
            await asyncio.wait_for(second_started.wait(), 3)
            await asyncio.wait_for(first, 3)
            assert _conversions(app).current("replacement", f"{1:032x}") is not None
            assert not second.done()
            release.set()
            receipt = json.loads(await asyncio.wait_for(second, 3))
            snapshot = json.loads(poll(_ticket(1), request))
            assert snapshot["terminal"] and snapshot["sequence"] == receipt["sequence"]
            assert "Completed" in receipt["outputs"][0]
        finally:
            release.set()
            first.cancel()
            if second is not None:
                second.cancel()
            await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["failed", "canceled"])
@pytest.mark.parametrize("browser", [False, True])
def test_final_error_replaces_provisional_failure(tmp_path: Path, status: str, browser: bool) -> None:
    """失败通知先到达时，公开响应和浏览器回执仍包含最终错误详情。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    provisional = f"Failed: server task {status}"
    detail = "page_range_invalid: requested page does not exist"

    async def parse_file(*_args: Any, status_callback: Any, **_kwargs: Any) -> ParseResult:
        """模拟 HTTP 客户端关闭时让出事件循环，确保通用失败先被发布。"""
        status_callback(provisional)
        await asyncio.sleep(0)
        raise V1ArtifactError(detail, code="page_range_invalid")

    app = _application(tmp_path, SimpleNamespace(parse_file=parse_file))
    request = SimpleNamespace(session_hash="failure-detail")

    async def scenario() -> None:
        """最终回执可覆盖已到达的失败快照，随后普通通知不能覆盖错误。"""
        if browser:
            receipt = json.loads(await _callback(app, "convert_ui")(str(source), 0, "", False, _ticket(1), request))
            response = receipt["outputs"]
            run = _conversions(app).runs[request.session_hash]
            assert detail in run.snapshot and run.state.message == f"Failed: {detail}"
            snapshot = json.loads(_callback(app, "read_conversion_status")(_ticket(1), request))
            assert snapshot["sequence"] == receipt["sequence"] and snapshot["terminal"]
            run.publish(provisional)
            assert run.state.message == f"Failed: {detail}"
        else:
            response = await _callback(app, "convert_handler")(str(source), 0, "", False, request)
        assert detail in response[0] and "server task" not in response[0]
        assert response[6] == ("" if browser else None)

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", ["failed", "completed", "cancelled"])
def test_final_failure_respects_terminal_boundaries(terminal: str) -> None:
    """明确的最终失败只能替换失败，成功和取消终态仍拒绝任何更新。"""
    run = ConversionRun("terminal-boundary")
    if terminal == "cancelled":
        run.cancel()
    else:
        run.publish("Failed: provisional" if terminal == "failed" else STATUS_COMPLETED)
    previous_snapshot = run.snapshot
    previous_sequence = run.state.sequence
    run.publish("Failed: final detail", final_failure=True)
    if terminal == "failed":
        assert run.state.message == "Failed: final detail"
        assert run.state.sequence == previous_sequence + 1
        assert json.loads(run.snapshot)["terminal"]
    else:
        assert run.snapshot == previous_snapshot and run.state.sequence == previous_sequence
    final_snapshot = run.snapshot
    run.publish(STATUS_PROCESSING_ON_SERVER, final_failure=True)
    run.publish("Failed: late callback")
    assert run.snapshot == final_snapshot


def test_frontend_conversion_receipts() -> None:
    """执行真实前端脚本，覆盖迟到状态、清除、失败和重复回执。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    result = subprocess.run([node, str(Path(__file__).with_suffix(".cjs"))], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_conversion_and_poll_are_full_non_generator_responses(tmp_path: Path) -> None:
    """通过 Gradio process_api 反复序列化，确认结果不会创建生成器差分缓存。"""
    from gradio import Request
    from gradio.data_classes import FileData
    from gradio.state_holder import SessionState
    from gradio.utils import get_upload_folder

    source = tmp_path / "中文 #1.pdf"
    source.write_bytes(_pdf_bytes())
    client = SimpleNamespace(parse_file=AsyncMock(return_value=ParseResult(middle_json=_middle_json(with_image=False))))
    app = _application(tmp_path, client)
    ui = next(fn for fn in app.fns.values() if fn.name == "convert_ui")
    poll = next(fn for fn in app.fns.values() if fn.name == "read_conversion_status")
    public = next(fn for fn in app.fns.values() if fn.name == "convert_handler")
    assert all(inspect.iscoroutinefunction(fn.fn) for fn in (ui, public))
    assert all(not inspect.isasyncgenfunction(fn.fn) for fn in (ui, public, poll))
    assert poll.queue is False and len(public.inputs) == 4 and len(public.outputs) == 16
    # 模拟已上传的文件，保留 Gradio 实际输入预处理和输出序列化。
    uploaded = Path(get_upload_folder()) / tmp_path.name / source.name
    uploaded.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, uploaded)
    file_input = FileData(path=str(uploaded)).model_dump()
    request = Request(session_hash="serial", headers={"host": "localhost:7860"})
    session = SessionState(app)

    async def scenario() -> None:
        """连续成功和失败仍只返回完整回执，并保持终态及下载所有权。"""
        for index in range(1, 31):
            failed = index % 5 == 0
            client.parse_file.side_effect = RuntimeError("controlled failure") if failed else None
            response = await app.process_api(
                ui,
                [file_input, index % 2, "", False, _ticket(index)],
                state=session,
                request=request,
                session_hash="serial",
                event_id=str(index),
            )
            assert response["is_generating"] is False and response["iterator"] is None
            receipt = json.loads(response["data"][0])
            assert receipt["run_id"] == f"{index:032x}"
            assert len(receipt["outputs"]) == 15
            status = await app.process_api(poll, [_ticket(index)], state=session, request=request, session_hash="serial")
            snapshot = json.loads(status["data"][0])
            assert snapshot["terminal"] and snapshot["sequence"] == receipt["sequence"]
            if failed:
                assert "Failed:" in receipt["outputs"][0]
            else:
                file = receipt["outputs"][2]["value"]
                assert file["meta"]["_type"] == "gradio.FileData"
                assert unquote(file["url"]).endswith(file["path"]) and Path(file["path"]).is_file()
                downloader = next(fn.fn for fn in app.fns.values() if fn.name == "handler")
                path, download_receipt = downloader(None, json.dumps({"run_id": receipt["outputs"][6]}), request)
                assert path and not json.loads(download_receipt)["error"]
            assert not app.pending_diff_streams

    asyncio.run(scenario())


def test_cancelled_sync_output_keeps_slot_until_thread_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """输出线程被取消后继续持槽，旧回执失效且新任务只能在线程退出后开始。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    started, release = threading.Event(), threading.Event()
    persist = gradio_app.persist_parse_result
    client = SimpleNamespace(parse_file=AsyncMock(return_value=ParseResult(middle_json=_middle_json(with_image=False))))
    calls = 0

    def slow_output(*args: Any, **kwargs: Any) -> Any:
        """第一次输出停在同步线程中，取消也不能提前释放该工作。"""
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(10)
        return persist(*args, **kwargs)

    monkeypatch.setattr(gradio_app, "persist_parse_result", slow_output)
    app = _application(tmp_path, client)
    convert, poll, cancel = (_callback(app, name) for name in ("convert_ui", "read_conversion_status", "cancel_ui_conversion"))
    request = SimpleNamespace(session_hash="same-session")

    async def scenario() -> None:
        """多次取消不能使同步工作逃逸到执行槽外，也不能取消较新的任务。"""
        first = asyncio.create_task(convert(str(source), 0, "", False, _ticket(1), request))
        pending: list[asyncio.Task[Any]] = [first]
        try:
            assert await asyncio.to_thread(started.wait, 3)
            assert "Preparing outputs" in poll(_ticket(1), request)
            cancellation = asyncio.create_task(cancel(_ticket(1), request))
            pending.append(cancellation)
            await asyncio.sleep(0.01)
            assert poll(_ticket(1), request) == "" and not first.done()
            second = asyncio.create_task(convert(str(source), 0, "", False, _ticket(2), request))
            pending.append(second)
            await asyncio.sleep(0.01)
            assert "Queued locally" in poll(_ticket(2), request)
            # 模拟 Gradio 取消与显式取消同时到达。
            first.cancel()
            await cancel(_ticket(1), request)
            await asyncio.sleep(0.01)
            assert client.parse_file.await_count == 1 and not first.done()
            release.set()
            assert await first == ""
            await cancellation
            result = json.loads(await second)
            assert result["run_id"] == f"{2:032x}" and "Completed" in result["outputs"][0]
            assert client.parse_file.await_count == 2
            # 取消先于排队提交到达时，旧提交不能重新建立状态。
            await cancel(_ticket(3), request)
            assert await convert(str(source), 0, "", False, _ticket(3), request) == ""
            assert client.parse_file.await_count == 2
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(scenario())
