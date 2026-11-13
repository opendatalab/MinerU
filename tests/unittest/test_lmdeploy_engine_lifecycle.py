"""后端真实异常契约、同步运行时失效与服务空闲期健康闭环。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from mineru_vl_utils.vlm_client.base_client import EngineHealth, EngineUnavailableError, ServerError
from task_test_utils import submit_task, task_url, wait_task

from mineru.model.runtime.ownership import RuntimeOwner
from mineru.model.vlm.errors import EngineDeadError, EngineNotReadyError, is_fatal_engine_error
from mineru.model.vlm.handles import RuntimeHandle
from mineru.model.vlm.runtime import ModelSingleton
from mineru.model.vlm.text_recognition import recognize_text_batch
from mineru.parser import api_server


class Client:
    """保留真实同步识别调用与关闭接口的后端替身。"""

    def __init__(self) -> None:
        """记录调用、可控故障及关闭状态。"""
        self.error: Exception | None = None
        self.calls = 0
        self.closed = False

    def batch_content_extract(self, images: list[Any], *, types: str) -> list[str]:
        """在真实识别边界抛出指定后端异常或返回成功结果。"""
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [str(item) for item in images]

    def content_extract(self, image: Any, *, type: str) -> str:
        """单片补试仅供可恢复故障对照使用。"""
        return str(image)

    async def aclose(self) -> None:
        """模拟客户端资源排空，底层引擎关闭仍由宿主执行。"""
        self.closed = True


@pytest.mark.parametrize("fatal", [False, True])
def test_real_server_error_inheritance_controls_retry(fatal: bool) -> None:
    """真实 ServerError 子类必须依赖致命分类才能禁用补试，普通错误继续恢复。"""
    error = EngineUnavailableError("dead") if fatal else ServerError("bad crop")
    client = SimpleNamespace(batch_content_extract=MagicMock(side_effect=error), content_extract=MagicMock(return_value="ok"))
    if fatal:
        with pytest.raises(EngineUnavailableError) as raised:
            recognize_text_batch(client, [1])
        assert raised.value is error
        client.content_extract.assert_not_called()
    else:
        assert recognize_text_batch(client, [1]) == ["ok"]
        client.content_extract.assert_called_once()


def test_fatal_runtime_notifies_all_owners_and_rejects_cache_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """共享引擎故障通知全部宿主，缓存命中不重建、不调用已经死亡的客户端。"""
    client = Client()
    original = EngineUnavailableError("original failure", error_message="backend details")
    client.error = original
    handle = RuntimeHandle(client, backend="lmdeploy-engine")
    owners = [RuntimeOwner(), RuntimeOwner()]
    handle.owners.update(owners)
    pool = ModelSingleton()
    key = ("test-lmdeploy",)
    monkeypatch.setattr(ModelSingleton, "_models", {key: handle})
    factory = MagicMock()
    try:
        with pytest.raises(EngineUnavailableError):
            handle.batch_recognize_text([1])
        assert all(owner.engine_error == "original failure" for owner in owners)
        client.error = None
        with pytest.raises(EngineDeadError) as raised:
            handle.batch_recognize_text([2])
        assert raised.value.__cause__ is original
        with pytest.raises(EngineDeadError):
            pool._get_or_initialize(key, factory, native_async=False, backend="lmdeploy-engine", max_concurrency=1)
        factory.assert_not_called()
        assert client.calls == 1
    finally:
        handle.shutdown()


def test_temporary_probe_failure_recovers_but_fatal_state_never_recovers() -> None:
    """探测超时可以恢复，明确死亡即使之后出现健康结果也不能复活。"""
    client = Client()
    status = EngineHealth("unavailable", "probe timed out")

    def probe(timeout: float) -> EngineHealth:
        """按测试状态返回真实探测协议。"""
        return status

    handle = RuntimeHandle(client, backend="lmdeploy-engine", probe_engine=probe)
    owner = RuntimeOwner()
    handle.owners.add(owner)
    try:
        with pytest.raises(EngineNotReadyError):
            handle.batch_recognize_text([1])
        assert client.calls == 0
        assert owner.engine_error is None
        assert owner.engine_unavailable == "probe timed out"
        status = EngineHealth("healthy")
        assert handle.batch_recognize_text([2]) == ["2"]
        assert owner.engine_unavailable is None
        status = EngineHealth("dead", "loop stopped")
        with pytest.raises(EngineDeadError):
            handle.probe_health()
        status = EngineHealth("healthy")
        with pytest.raises(EngineDeadError):
            handle.batch_recognize_text([3])
        assert client.calls == 1
    finally:
        handle.shutdown()


def test_one_engine_recovery_does_not_clear_another_engine_unavailability() -> None:
    """同一宿主拥有多个模型时，各自暂时状态不可互相覆盖。"""
    owner = RuntimeOwner()
    first, second = object(), object()
    owner.set_engine_unavailable(first, "one")
    owner.set_engine_unavailable(second, "two")
    owner.set_engine_unavailable(first, None)
    assert owner.engine_unavailable == "two"


def test_close_waits_for_active_health_probe() -> None:
    """关闭与健康探测相交时，必须先结束探测租约再关闭底层资源。"""
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    client = Client()

    def probe(timeout: float) -> EngineHealth:
        """阻塞探测以核对关闭动作的先后关系。"""
        entered.set()
        assert release.wait(2)
        assert not closed.is_set()
        return EngineHealth("healthy")

    handle = RuntimeHandle(client, backend="lmdeploy-engine", probe_engine=probe, close_engine=closed.set)
    with ThreadPoolExecutor(2) as executor:
        checking = executor.submit(handle.probe_health)
        assert entered.wait(2)
        closing = executor.submit(handle.shutdown)
        assert not closed.wait(0.03)
        release.set()
        checking.result(timeout=2)
        closing.result(timeout=2)
    assert client.closed and closed.is_set()


def test_idle_server_detects_death_and_transient_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """无需新解析请求，周期探测就能撤销能力；超时恢复后才重新允许接单。"""
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *args: None)
    monkeypatch.setattr(api_server, "_ENGINE_HEALTH_INTERVAL_SECONDS", 0.01)
    status = EngineHealth("healthy")

    def probe(timeout: float) -> EngineHealth:
        """模拟已加载的引擎健康变化。"""
        return status

    handle = RuntimeHandle(Client(), backend="lmdeploy-engine", probe_engine=probe)
    monkeypatch.setattr(ModelSingleton, "_models", {("idle-engine",): handle})
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",))
    try:
        with TestClient(app) as client:
            owner = app.state.execution_store.runtime_owner
            handle.owners.add(owner)
            handle.pinned = True
            owner.has_engine_health_probe = True

            async def wait_state(code: str | None) -> None:
                """等待监测任务完成状态传播，不使用长时间阻塞睡眠。"""
                async with asyncio.timeout(2):
                    while True:
                        current = (
                            "engine_dead" if owner.engine_error else "engine_unavailable" if owner.engine_unavailable else None
                        )
                        if current == code:
                            return
                        await asyncio.sleep(0.005)

            status = EngineHealth("unavailable", "timeout")
            client.portal.call(wait_state, "engine_unavailable")
            response = client.get("/v1/health")
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "engine_unavailable"
            assert owner.engine_error is None
            status = EngineHealth("healthy")
            client.portal.call(wait_state, None)
            assert client.get("/v1/health").status_code == 200
            status = EngineHealth("dead", "engine main loop stopped")
            client.portal.call(wait_state, "engine_dead")
            for endpoint in ("/v1/health", "/v1/tiers", "/v1/models"):
                response = client.get(endpoint)
                assert response.status_code == 503
                assert response.json()["error"]["code"] == "engine_dead"
            response = client.post("/v1/tasks", json={"tier": "flash", "files": [{"file_name": "new.pdf", "bytes": 3}]})
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "engine_dead"
    finally:
        handle.shutdown()


def test_probe_does_not_initialize_lazy_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有创建 VLM 的应用不会为了健康查询访问模型池或下载模型。"""
    probe = MagicMock(side_effect=AssertionError("must not access model pool"))
    monkeypatch.setattr(ModelSingleton, "probe_owner_health", probe)
    api_server._probe_initialized_engines(RuntimeOwner())
    probe.assert_not_called()


def test_monitor_drains_running_and_queued_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """后台健康发现死亡时也必须清理已接纳任务，不能等待某次推理主动报错。"""
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *args: None)
    monkeypatch.setattr(api_server, "_ENGINE_HEALTH_INTERVAL_SECONDS", 0.01)
    stopped = threading.Event()
    entered = threading.Event()
    cleaned = threading.Event()

    def probe(timeout: float) -> EngineHealth:
        """由外部信号模拟后台引擎主循环独立停止。"""
        return EngineHealth("dead", "background loop stopped") if stopped.is_set() else EngineHealth("healthy")

    async def infer(path: str, **kwargs: Any) -> None:
        """保持请求在途，只有健康管理者取消它才会结束。"""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.01)
            cleaned.set()

    monkeypatch.setattr(api_server, "parse_async", infer)
    handle = RuntimeHandle(Client(), backend="lmdeploy-engine", probe_engine=probe)
    handle.pinned = True
    monkeypatch.setattr(ModelSingleton, "_models", {("engine",): handle})
    app = api_server.create_app(upload_dir=str(tmp_path / "uploads"), tiers=("flash",), concurrency=1)
    try:
        with TestClient(app) as client:
            owner = app.state.execution_store.runtime_owner
            handle.owners.add(owner)
            owner.has_engine_health_probe = True
            jobs = []
            for name in ["running", "queued"]:
                path = tmp_path / f"{name}.pdf"
                path.write_bytes(name.encode())
                jobs.append(submit_task(client, path, tier="flash"))
            assert entered.wait(2)
            stopped.set()
            for job in jobs:
                result = wait_task(client, job)
                assert result["status"] == "failed"
                assert result["files"][0]["error"]["code"] == "engine_dead"
                assert client.get(task_url(client, job)).status_code == 200
            assert cleaned.wait(2)
    finally:
        handle.shutdown()


def test_fatal_engine_chain_keeps_original_backend_signal() -> None:
    """上层包装异常仍能识别后端致命原因。"""
    error = ValueError("wrapper")
    error.__cause__ = EngineUnavailableError("original")
    assert is_fatal_engine_error(error)
