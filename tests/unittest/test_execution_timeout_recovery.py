"""执行硬期限、父进程监督和真实 worker 重建的针对性验收。"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner
from fastapi.testclient import TestClient
from task_test_utils import submit_task, task_url, wait_task

from mineru.kit.gradio.client import ManagedLocalRouter, V1ArtifactClient, V1ArtifactError
from mineru.kit.router import RouterSettings, create_app
from mineru.kit.router.workers import ManagedLocalWorker, WorkerPool, WorkerState, parse_local_gpus
from mineru.parser.api_client import MinerUApiParser
from mineru.parser.api_server import ExecutionStore, _ExecutionRecord, _ExecutionRequest
from mineru.parser.execution_limits import ExecutionLease
from mineru.parser.process_control import ManagedProcessControl, ManagedProcessControlWatcher
from mineru.parser.process_tree import _group_running, process_group_options, stop_process_tree
from mineru.tasks.errors import ApiServerError


def record(name: str) -> _ExecutionRecord:
    """构造独立执行身份，避免使用父 Task ID 代替执行 ID。"""
    return _ExecutionRecord(
        name,
        _ExecutionRequest(
            source={"type": "url", "url": "https://example.test/a.pdf"},
            file_id=f"file-{name}",
            file_name=f"{name}.pdf",
            tier="flash",
        ),
    )


def test_cancel_cleanup_keeps_deadline_and_fails_queued_execution() -> None:
    """用户取消后清理仍卡住时监督继续有效，等待者不能提前进入同一实例。"""

    async def scenario() -> None:
        """用清理门闩验证公开取消状态与实际退出之间的差别。"""
        store = ExecutionStore(task_timeout_seconds=0.05)
        first, second = record("active"), record("queued")
        store._executions = {first.id: first, second.id: second}
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        called = []

        async def work() -> None:
            """模拟请求被取消后尚未完成的底层清理。"""
            first.status = "running"
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaning.set()
                await release.wait()

        async def queued() -> None:
            """记录不应获准执行的下一份输入。"""
            called.append(True)

        store.start_task(first, work)
        await entered.wait()
        store.start_task(second, queued)
        store.cancel(first.id)
        await cleaning.wait()
        await asyncio.sleep(0.15)
        assert first.status == "canceled" and first.id in store._execution_leases
        assert second.status == "failed" and second.error.code == "worker_restarted"
        assert not called and store._semaphore._value == 0
        with pytest.raises(ApiServerError) as error:
            store.require_ready()
        assert error.value.error.code == "task_timeout"
        release.set()
        await store.shutdown()
        assert not store._execution_leases and store._semaphore._value == 1

    asyncio.run(scenario())


def test_preparation_and_queue_do_not_start_execution_deadline() -> None:
    """输入准备超过执行预算仍不算推理超时，只有取得执行名额后才启动期限。"""

    async def scenario() -> None:
        """用受控准备阶段核对计时起点和正常收尾。"""
        store = ExecutionStore(task_timeout_seconds=0.04)
        item = record("prepare")
        store._executions[item.id] = item
        gate = asyncio.Event()

        async def prepare() -> None:
            """等待准备完成，不占执行名额。"""
            await gate.wait()

        async def work() -> None:
            """正常执行完成后不触发恢复。"""
            item.status = "completed"

        store.start_task(item, work, prepare=prepare)
        await asyncio.sleep(0.08)
        assert not store._execution_leases
        store.require_ready()
        gate.set()
        await asyncio.gather(*store._tasks.values())
        await asyncio.sleep(0.08)
        store.require_ready()
        await store.shutdown()

    asyncio.run(scenario())


def test_control_reports_keep_timeout_after_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实 IPC 上报保留已经发生的超时，正常完成则撤销执行期限。"""
    parent = ManagedProcessControl.create(execution_reports=True)
    parent.start_accepting()
    for key, value in parent.child_env().items():
        monkeypatch.setenv(key, value)
    child = ManagedProcessControlWatcher.from_environment(lambda: None)
    child.start()
    try:
        normal = ExecutionLease("normal", "file-normal", time.monotonic() + 10)
        failed = ExecutionLease("failed", "file-failed", time.monotonic() - 1)
        child.execution_started(normal)
        child.execution_finished(normal.execution_id)
        child.execution_started(failed)
        child.execution_timed_out(failed)
        child.execution_finished(failed.execution_id)
        until = time.monotonic() + 3
        while time.monotonic() < until and not parent.expired_executions(time.monotonic()):
            time.sleep(0.01)
        assert parent.expired_executions(time.monotonic()) == (failed,)
    finally:
        child.close()
        parent.close()


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_invalid_deadlines_are_rejected(value: float) -> None:
    """所有 Python 配置入口一致拒绝无法执行的时间预算。"""
    for factory in (ExecutionStore, RouterSettings, ManagedLocalRouter):
        with pytest.raises(ValueError):
            factory(task_timeout_seconds=value)
    with pytest.raises(ValueError):
        MinerUApiParser(poll_timeout_seconds=value)


def test_inherit_mode_does_not_probe_or_rewrite_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """托管 WebUI 继承设备环境，不把多卡环境展开成多个模型实例。"""
    from mineru.kit.router import workers

    def forbidden() -> None:
        """禁止 Router 为 inherit 模式读取自身设备。"""
        raise AssertionError("unexpected device probe")

    monkeypatch.setattr(workers, "get_device", forbidden)
    assert parse_local_gpus("inherit") == [None]


@pytest.mark.parametrize("command", ["webui", "router", "api-server"])
@pytest.mark.parametrize("value", [None, 600, 0, -1])
def test_timeout_option_reaches_execution_owner(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    value: int | None,
) -> None:
    """通过正式 CLI 检查默认值、覆盖、非法输入和 Router reload 配置，不启动网络服务。"""
    from mineru.kit.main import app
    from mineru.kit.gradio import app as gradio_app
    from mineru.kit.router import cli as router_cli
    from mineru.parser import api_server

    captured = {}

    def capture_webui(**options: object) -> None:
        """记录 WebUI 交给托管适配器的完整配置。"""
        captured["timeout"] = options["task_timeout_seconds"]

    def capture_router(settings: RouterSettings) -> object:
        """记录 Router 设置，后续检查其环境序列化。"""
        captured["timeout"] = settings.task_timeout_seconds
        return object()

    def capture_api(*, args: list[str], **options: object) -> None:
        """解析 CLI 显式传入 API 入口的参数，避免重复实现默认值。"""
        captured["timeout"] = int(args[args.index("--task-timeout") + 1])

    monkeypatch.setattr(gradio_app, "launch_gradio", capture_webui)
    monkeypatch.setattr(router_cli, "create_app", capture_router)
    monkeypatch.setattr(router_cli, "run_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_server.main, "main", capture_api)
    with patch.dict(os.environ):
        result = CliRunner().invoke(app, [command, *(["--task-timeout", str(value)] if value is not None else [])])
        if value is not None and value <= 0:
            assert result.exit_code != 0 and not captured
        else:
            assert result.exit_code == 0, result.output
            assert captured["timeout"] == (3600 if value is None else value)
            if command == "router":
                assert RouterSettings.from_env().task_timeout_seconds == captured["timeout"]


def test_recovery_budget_stops_after_three_start_failures() -> None:
    """反复启动失败会熔断，不无限创建新模型进程。"""

    async def scenario() -> None:
        """仅控制进程创建失败，保留真实池的失败计数和监督状态。"""
        starts = []

        async def start(client: httpx.AsyncClient) -> None:
            """模拟启动配置或设备持续不可用。"""
            starts.append(True)
            raise RuntimeError("cannot start")

        local = SimpleNamespace(process=None, start=start)
        pool = WorkerPool(RouterSettings(local_gpus="none"))
        worker = WorkerState("local", "", "local", local_worker=local)
        pool._workers[worker.worker_id] = worker
        try:
            for _ in range(5):
                worker.retry_at = 0
                await pool.refresh(worker)
            assert len(starts) == 3 and worker.recovery_exhausted and not worker.healthy
        finally:
            await pool.client.aclose()

    asyncio.run(scenario())


def test_managed_wait_reports_recovery_and_does_not_submit() -> None:
    """恢复等待仅探测健康；启动预算耗尽时不制造新的问题文件重试。"""

    async def scenario() -> None:
        """固定不可用响应核对等待消息和最终错误。"""
        requests = []

        def handle(request: httpx.Request) -> httpx.Response:
            """记录网络行为，模拟 Router 正在重建唯一 worker。"""
            requests.append((request.method, request.url.path))
            return httpx.Response(503, json={"error": {"code": "upstream_unavailable", "message": "recovering"}})

        client = V1ArtifactClient(
            api_url="http://router", managed=True, recovery_timeout_seconds=0.01, transport=httpx.MockTransport(handle)
        )
        messages = []
        with pytest.raises(V1ArtifactError, match="recovery failed"):
            await client.wait_for_ready(messages.append)
        assert messages and set(requests) == {("GET", "/v1/health")}

    asyncio.run(scenario())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("budget", [None, 1])
def test_sdk_wait_budget_uses_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    asynchronous: bool,
    budget: int | None,
) -> None:
    """HTTP 耗时计入客户端预算，无限等待模式仍能消费真实终态与 ZIP。"""
    from mineru.parser import api_client
    from tests.unittest.test_tasks_sdk import bind_transport, task_snapshot, zip_output

    now = [0.0]
    queries = []
    source = tmp_path / "file.pdf"
    source.write_bytes(b"pdf")

    def handle(request: httpx.Request) -> httpx.Response:
        """用虚拟耗时模拟慢查询，避免将一小时测试变成真实长等待。"""
        if request.url.host == "storage.test":
            return httpx.Response(200, content=zip_output())
        if request.method == "POST":
            return httpx.Response(202, json=task_snapshot("pending"))
        queries.append(request)
        now[0] += 0.6
        state = "completed" if len(queries) == 4 else "pending"
        return httpx.Response(200, json=task_snapshot(state))

    bind_transport(monkeypatch, handle)
    monkeypatch.setattr(api_client, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _: None))
    parser = MinerUApiParser(api_url="http://api.test", poll_timeout_seconds=budget)

    def invoke() -> object:
        """同步和异步路径使用同一组 HTTP 响应与时钟。"""
        return asyncio.run(parser.parse_async(source)) if asynchronous else parser.parse(source)

    if budget is None:
        assert len(invoke().pages) == 1
        assert len(queries) == 4
    else:
        with pytest.raises(api_client._V1APIError, match="timeout"):
            invoke()
        assert len(queries) == 3


def test_supervisor_waits_for_completion_notification_and_replaces_once() -> None:
    """健康 HTTP 始终成功时仍能恢复真实超期；边界完成通知可撤销误报。"""

    async def scenario() -> None:
        """用可控停止门闩核对独立监督、同代去重和等待清理期间拒绝接单。"""
        leases = [ExecutionLease("boundary", "file-boundary", 0)]
        invalidated, stops = [], []
        release = asyncio.Event()

        async def stop() -> None:
            """保留慢速回收窗口，重复监督不能再次执行替换。"""
            stops.append(True)
            await release.wait()
            local.process = None

        async def start(client: httpx.AsyncClient) -> None:
            """发布新的健康实例。"""
            local.control = None
            local.process = SimpleNamespace(poll=lambda: None)
            local.base_url = "http://new"

        async def invalidate(worker: WorkerState) -> None:
            """记录旧任务失效通知，不等待进程停止。"""
            invalidated.append((worker.generation, set(worker.timed_out_file_ids)))

        local = SimpleNamespace(
            process=SimpleNamespace(poll=lambda: None),
            base_url="http://old",
            control=SimpleNamespace(expired_executions=lambda now: tuple(leases)),
            start=start,
            stop=stop,
        )
        pool = WorkerPool(
            RouterSettings(local_gpus="none"),
            on_local_worker_replaced=invalidate,
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok", "data": []})),
        )
        worker = WorkerState("local", "http://old", "local", local_worker=local, healthy=True, generation=1)
        pool._workers[worker.worker_id] = worker
        try:
            await pool.supervise_once()
            leases.clear()
            await pool.supervise_once()
            assert not invalidated and worker.healthy
            leases.extend([ExecutionLease("hang-a", "file-a", 0), ExecutionLease("hang-b", "file-b", 0)])
            await pool.supervise_once()
            worker.expired_observed_at = dict.fromkeys(worker.expired_observed_at, time.monotonic() - 1)
            await pool.supervise_once()
            await asyncio.sleep(0)
            await pool.supervise_once()
            assert invalidated == [(1, {"file-a", "file-b"})]
            assert len(stops) == 1 and not worker.healthy
            release.set()
            await worker.recovery_task
            assert worker.generation == 2 and worker.healthy
        finally:
            release.set()
            await pool.close()

    asyncio.run(scenario())


def test_failed_process_cleanup_preserves_worker_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    """强制回收失败必须保留进程句柄，禁止把仍存活的实例当作已经释放。"""
    from mineru.kit.router import workers

    def fail(*args: object, **kwargs: object) -> None:
        """在真正终止边界模拟系统拒绝回收。"""
        raise RuntimeError("still alive")

    monkeypatch.setattr(workers, "stop_process_tree", fail)
    worker = ManagedLocalWorker("local", "127.0.0.1", None, RouterSettings())
    process, control = object(), object()
    worker.process, worker.control = process, control
    with pytest.raises(RuntimeError, match="still alive"):
        asyncio.run(worker.stop())
    assert worker.process is process and worker.control is control


@pytest.mark.skipif(os.name == "nt", reason="POSIX 真实进程组验收")
def test_process_tree_stops_nested_sessions(tmp_path: Path) -> None:
    """真实父子进程分别 setsid，回收父 Router 时也必须终止另一个 worker 进程组。"""
    marker = tmp_path / "child-pid"
    code = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'],start_new_session=True); "
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid));time.sleep(90)"
    )
    process = subprocess.Popen([sys.executable, "-c", code], **process_group_options())
    try:
        until = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < until:
            time.sleep(0.01)
        child_pid = int(marker.read_text())
        stop_process_tree(process, None, timeout_seconds=2, grouped=True)
        assert process.poll() is not None and not _group_running(child_pid)
    finally:
        if process.poll() is None:
            stop_process_tree(process, None, timeout_seconds=2, grouped=True)


def test_linux_proc_discovery_does_not_require_ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """精简容器通过 proc 解析含空格和括号的进程名，忽略扫描期间已消失的进程。"""
    from mineru.parser import process_tree

    for pid, text in (("123", "123 (worker (name)) S 12 123 123 0"), ("124", "124 (child) Z 123 123 123 0")):
        directory = tmp_path / pid
        directory.mkdir()
        (directory / "stat").write_text(text)
    (tmp_path / "125").mkdir()
    scan = os.scandir
    monkeypatch.setattr(process_tree.sys, "platform", "linux")
    monkeypatch.setattr(process_tree.os, "scandir", lambda path: scan(tmp_path))
    assert sorted(process_tree._process_rows()) == [(123, 12, 123, "S"), (124, 123, 123, "Z")]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object 实机验收")
def test_windows_job_reaps_children_after_parent_exit(tmp_path: Path) -> None:
    """父进程已退出时，Job 仍持有后代，关闭不能退化为只检查 Popen.poll。"""
    from mineru.parser.windows_process_job import create_process_job

    gate = tmp_path / "assigned"
    code = (
        "import subprocess,sys,time,pathlib\n"
        f"while not pathlib.Path({str(gate)!r}).exists(): time.sleep(.01)\n"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'])\n"
    )
    process = subprocess.Popen([sys.executable, "-c", code], **process_group_options())
    job = create_process_job(process)
    try:
        gate.touch()
        process.wait(timeout=5)
        stop_process_tree(process, None, timeout_seconds=2, grouped=True, process_job=job)
        assert job._handle is None
    finally:
        job.stop(time.monotonic() + 3)


@pytest.mark.skipif(os.name == "nt", reason="POSIX 托管子进程故障注入")
@pytest.mark.parametrize("failure", ["await", "cleanup", "event_loop"])
def test_real_worker_timeout_then_real_pdf_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """独立 API 子进程内注入明确阶段的停滞，Router 重建后真实解析仓库 PDF 并检查产物。"""
    marker = tmp_path / "entered.json"
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    monkeypatch.setenv("MINERU_CONFIG", str(config))
    source = Path("demo/pdfs/demo1.pdf").resolve()
    script = Path("tests/fixtures/stalled_parse_worker.py").resolve()

    def command(self: ManagedLocalWorker, port: int, upload_dir: Path) -> list[str]:
        """仅替换测试进程入口，仍使用真实 worker 启停、控制通道和公开 Tasks 协议。"""
        return [sys.executable, str(script), str(marker), failure, str(port), str(upload_dir)]

    monkeypatch.setattr(ManagedLocalWorker, "command", command)
    settings = RouterSettings(
        local_gpus="inherit",
        worker_tier="flash",
        task_timeout_seconds=1,
        worker_startup_timeout_seconds=30,
        worker_refresh_interval_seconds=0.1,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        worker = app.state.worker_pool.workers[0]
        old_pid = worker.local_worker.process.pid
        old_temporary_root = Path(worker.local_worker.temp_dir.name)
        submitted_at = time.monotonic()
        first = submit_task(client, source, tier="flash", ocr_mode="txt")
        until = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < until:
            time.sleep(0.01)
        entered = json.loads(marker.read_text())
        assert entered == {"pid": old_pid, "phase": failure}
        if failure == "cleanup":
            canceled = client.post(task_url(client, first) + "/cancel")
            assert canceled.status_code == 200
        terminal = wait_task(client, first, timeout=8)
        assert terminal["status"] == ("canceled" if failure == "cleanup" else "failed")
        if failure != "cleanup":
            assert terminal["files"][0]["error"]["code"] == "task_timeout"
            assert time.monotonic() - submitted_at < 3, "失败状态不能等到旧进程回收后才发布"
        until = time.monotonic() + 25
        while time.monotonic() < until and (worker.generation == 1 or not worker.healthy):
            time.sleep(0.05)
        assert worker.generation == 2 and worker.healthy
        assert worker.local_worker.process.pid != old_pid and not _group_running(old_pid)
        assert not old_temporary_root.exists()
        second = submit_task(client, source, tier="flash", ocr_mode="txt")
        done = wait_task(client, second, timeout=20)
        assert done["status"] == "completed", done
        result = client.get(done["files"][0]["result_url"])
        assert result.status_code == 200
        with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
            names = archive.namelist()
            middle = next(name for name in names if name.endswith("middle.json") or name.endswith("middle_json.json"))
            payload = json.loads(archive.read(middle))
            assert len(payload["pages"]) == 13
            markdown = next(name for name in names if name.endswith(".md"))
            assert len(archive.read(markdown)) > 100
            assert any("images/" in name for name in names)
