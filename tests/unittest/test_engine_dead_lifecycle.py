"""明确引擎死亡、任务清理和 Router 恢复回归。"""

from __future__ import annotations

from task_test_utils import task_url

from task_test_utils import submit_task, wait_task

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from mineru.model.vlm.errors import EngineDeadError, is_fatal_engine_error
from mineru.parser import api_server
from mineru.kit.router import workers


def test_fatal_classifier_uses_types_and_exception_chain() -> None:
    """普通 EngineCore 文本和网络超时不是死亡；明确类型及 cause 是死亡。"""
    assert not is_fatal_engine_error(RuntimeError("EngineCore request invalid"))
    assert not is_fatal_engine_error(httpx.ReadTimeout("enginecore timeout"))
    fatal_type = type("EngineDeadError", (RuntimeError,), {"__module__": "vllm.v1.engine.exceptions"})
    wrapper = ValueError("wrapped")
    wrapper.__cause__ = fatal_type("dead")
    assert is_fatal_engine_error(wrapper)
    assert is_fatal_engine_error(EngineDeadError("dead"))


def test_api_engine_failure_drains_running_and_queued_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """死亡后新提交及健康返回 503，排队失败、运行清理、用户取消保持 canceled。"""
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *args: None)
    entered: list[str] = []
    cleanup: list[str] = []
    gate: asyncio.Event | None = None

    async def infer(path: str, **kwargs: Any) -> None:
        """第一份输入等待触发致命错误，其余运行任务在取消后异步清理。"""
        name = str(Path(path).read_bytes(), "ascii")
        entered.append(name)
        try:
            await gate.wait()
            if name == "fatal":
                raise EngineDeadError("local engine died")
            await asyncio.sleep(60)
        finally:
            await asyncio.sleep(0.02)
            cleanup.append(name)

    monkeypatch.setattr(api_server, "parse_async", infer)
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",), concurrency=2)
    with TestClient(app) as client:

        async def make_gate() -> None:
            """在所属应用循环创建门闩。"""
            nonlocal gate
            gate = asyncio.Event()

        client.portal.call(make_gate)
        jobs = []
        for name in ("fatal", "running", "pending", "user-cancel"):
            source = tmp_path / f"{name}.pdf"
            source.write_bytes(name.encode())
            jobs.append(submit_task(client, source, tier="flash"))
        assert client.post(task_url(client, jobs[-1]) + "/cancel").status_code == 200
        client.portal.call(gate.set)
        for job in jobs:
            wait_task(client, job)

        async def drain_tasks() -> None:
            """等待公开父任务完成资源清理后检查内部执行槽。"""
            await asyncio.gather(
                *(
                    app.state.execution_store.wait_for_terminal(file.execution_id, 2)
                    for record in app.state.task_store._records.values()
                    for file in record.file_states
                )
            )
            await asyncio.gather(*(record.operation for record in app.state.task_store._records.values()))

        client.portal.call(drain_tasks)
        assert entered == ["fatal", "running"]
        assert set(cleanup) == {"fatal", "running"}
        assert [client.get(task_url(client, job)).json()["status"] for job in jobs] == [
            "failed",
            "failed",
            "failed",
            "canceled",
        ]
        for path, method in (("/v1/health", "get"), ("/v1/tasks", "post")):
            response = getattr(client, method)(
                path,
                **({"json": {"tier": "flash", "files": [{"file_name": "new.pdf", "bytes": 3}]}} if method == "post" else {}),
            )
            assert response.status_code == 503 and response.json().get("error")["code"] == "engine_dead"
        assert app.state.execution_store._semaphore._value == 2
        assert not app.state.execution_store._tasks
        assert not app.state.file_store._reserved_output_ids
        for job in jobs:
            file_id = job["files"][0]["file_id"]
            assert client.get(f"/v1/files/{file_id}").status_code == 404


def test_router_single_generation_replacement_and_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """健康中的明确死亡触发一次替换，失败重试服从退避，外部 worker 不重启。"""

    async def scenario() -> None:
        """在同一事件循环并发刷新，验证锁和 generation 隔离。"""
        now = [0.0]
        monkeypatch.setattr(workers.time, "monotonic", lambda: now[0])
        local = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), base_url="http://old")
        starts: list[int] = []
        stops: list[int] = []
        failures = [True]

        async def stop() -> None:
            """停止旧进程，保留启动失败时可重试的空状态。"""
            stops.append(1)
            local.process = None

        async def start(client: Any) -> None:
            """首轮启动失败，后续轮次成功。"""
            starts.append(1)
            if failures[0]:
                raise RuntimeError("startup failed")
            local.process = SimpleNamespace(poll=lambda: None)
            local.base_url = "http://new"

        local.stop, local.start = stop, start

        def handle(request: httpx.Request) -> httpx.Response:
            """旧进程报告 engine_dead，新进程正常报告能力。"""
            if request.url.host == "old":
                return httpx.Response(503, json={"error": {"code": "engine_dead"}})
            return httpx.Response(200, json={"status": "ok", "data": []})

        pool = workers.WorkerPool(workers.RouterSettings(local_gpus="none"), transport=httpx.MockTransport(handle))
        state = workers.WorkerState("local", "http://old", "local", local_worker=local, generation=1)
        try:
            await pool.refresh(state)
            assert not state.healthy and state.restart_requested
            await asyncio.gather(*(pool.refresh(state) for _ in range(3)))
            assert len(starts) == len(stops) == 1 and state.generation == 2
            assert state.retry_at == 5
            now[0] = 5
            await pool.refresh(state)
            assert state.retry_at == 15
            failures[0] = False
            now[0] = 15
            await asyncio.gather(*(pool.refresh(state) for _ in range(3)))
            assert len(stops) == 1 and state.healthy and state.generation == 2
            remote = workers.WorkerState("remote", "http://old", "remote")
            await pool.refresh(remote)
            assert not remote.healthy and not remote.restart_requested
        finally:
            await pool.client.aclose()

    asyncio.run(scenario())


def test_old_generation_failure_snapshot_remains_queryable() -> None:
    """旧 generation 未完成任务返回失败快照，不将查询发给新进程。"""
    from test_v1_router import _FakeV1Upstream, _make_router

    upstream = _FakeV1Upstream("worker-a", ("standard",))
    app, _ = _make_router(upstream)
    with TestClient(app) as client:
        pool, registry = app.state.worker_pool, app.state.registry
        local = SimpleNamespace(process=SimpleNamespace(poll=lambda: 1), base_url="http://old")

        async def stop() -> None:
            """清除旧进程对象。"""
            local.process = None

        async def start(client: Any) -> None:
            """启动一个可以返回正常 health 的新进程。"""
            local.process = SimpleNamespace(poll=lambda: None)
            local.base_url = "http://worker-a"

        local.stop, local.start = stop, start
        state = workers.WorkerState("local-lost", "http://old", "local", local_worker=local, generation=1)
        pool._workers[state.worker_id] = state
        from task_test_utils import router_execution

        route = router_execution(
            registry, worker_id=state.worker_id, upstream_id="job-old", file_id="file-old", file_name="old.pdf"
        )
        route.payload = {
            "task_id": "job-old",
            "status": "running",
            "files": [{"status": "pending"}],
            "progress": {"total": 1, "completed": 0},
        }
        client.portal.call(pool.refresh, state)
        from mineru.kit.router.tasks import read_task_execution

        async def snapshot() -> Any:
            """旧执行只读取已保存的失败信息，不请求替换后的 worker。"""
            return await read_task_execution(route, pool, registry)

        execution = client.portal.call(snapshot)
        assert execution.status == "failed"
        assert execution.error is not None and execution.error.code == "upstream_task_lost"
        pool._workers = {state.worker_id: state}
        reservation = pool.reserve_job(tier="standard", required_sources={"upload"})
        assert reservation is not None and reservation.worker is state
        reservation.release()
        assert state.active_jobs == 0


def test_old_generation_engine_dead_response_does_not_kill_replacement() -> None:
    """旧请求迟到报告死亡时，不得把健康的新 generation 再次标记死亡。"""
    from mineru.kit.router.proxy import request_upstream

    async def scenario() -> None:
        """响应处理之前模拟替换已发布，验证 notify 的 generation 条件。"""
        state = workers.WorkerState(
            "local",
            "http://old",
            "local",
            generation=1,
            healthy=True,
            local_worker=SimpleNamespace(process=SimpleNamespace(poll=lambda: None)),
        )

        def handle(request: httpx.Request) -> httpx.Response:
            """旧网络请求返回前发布新代，随后返回旧代的 engine_dead。"""
            state.generation = 2
            state.base_url = "http://new"
            state.healthy = True
            return httpx.Response(503, json={"error": {"code": "engine_dead"}})

        pool = workers.WorkerPool(workers.RouterSettings(local_gpus="none"), transport=httpx.MockTransport(handle))
        try:
            response = await request_upstream(pool, state, "GET", "/v1/health")
            assert response.status_code == 503
            assert state.healthy and not state.restart_requested
        finally:
            await pool.client.aclose()

    asyncio.run(scenario())


def test_fatal_preload_uses_engine_dead_code() -> None:
    """预加载期间死亡与推理期间死亡共享同一稳定错误码。"""
    assert api_server._classify_model_preload_error(EngineDeadError("dead")) == ("engine_dead", "dead")
