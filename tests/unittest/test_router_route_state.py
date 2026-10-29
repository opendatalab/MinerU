"""验证 Router 有类型状态与响应缓存、worker 预占和结果保留的边界。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import httpx
import pytest
from task_test_utils import router_execution

from mineru.kit.router import tasks
from mineru.kit.router.proxy import rewrite_file_payload
from mineru.kit.router.resources import ExecutionRoute, FileRoute, ResourceRegistry, SourceFileStore
from mineru.kit.router.route_state import ExecutionRouteState
from mineru.kit.router.workers import JobReservation, RouterSettings, WorkerPool, WorkerState


@pytest.mark.parametrize("release_before_replacement", [False, True])
def test_execution_state_releases_only_its_own_generation(release_before_replacement: bool) -> None:
    """所有终态入口归还同一预占；重复和迟到释放不能扣除新 generation 的活动名额。"""
    registry = ResourceRegistry(retention_seconds=0)
    worker = WorkerState("worker", "http://worker", "remote", generation=1, active_jobs=1)
    reservation = JobReservation(worker, worker.generation)
    route = router_execution(registry, worker_id=worker.worker_id, upstream_id="old", reservation=reservation)
    assert isinstance(route, ExecutionRoute)
    if release_before_replacement:
        route.state.release_reservation()
        assert worker.active_jobs == 0
    worker.generation += 1
    worker.active_jobs = 1
    replacement = JobReservation(worker, worker.generation)
    route.state.release_reservation()
    route.state.release_reservation()
    assert route.state.reservation is None and reservation.released
    assert worker.active_jobs == 1 and not replacement.released
    replacement.release()
    assert worker.active_jobs == 0


def test_file_response_refresh_preserves_lease_pins_and_unknown_fields() -> None:
    """HTTP 响应原样保留扩展，重复查询刷新缓存不会覆盖内部租约或传输固定状态。"""
    registry = ResourceRegistry(retention_seconds=0)
    worker = WorkerState("worker", "http://worker", "remote")
    route = registry.register_file(owner_scope="owner", worker_id=worker.worker_id, upstream_id="upstream")
    assert isinstance(route, FileRoute)
    route.lease.held = True
    route.lease.retain_until = time.time() + 60
    until = route.lease.retain_until
    upstream = {"id": "upstream", "purpose": "parse_output", "extensions": {"values": [1]}, "pins": 999}
    with pytest.raises(ValueError, match="interrupted transfer"):
        with registry.pin_route(route), registry.pin_route(route):
            response = rewrite_file_payload(upstream, worker, registry, "owner")
            assert response == {**upstream, "id": route.public_id}
            assert registry.find("file", route.public_id) is route
            assert route.pins == 2 and route.lease.held and route.lease.retain_until == until
            response["extensions"]["values"].append(2)
            assert route.payload["extensions"]["values"] == upstream["extensions"]["values"] == [1]
            raise ValueError("interrupted transfer")
    assert route.pins == 0 and route.lease.held and route.lease.retain_until == until
    with pytest.raises(AttributeError):
        route.kind = "job"  # type: ignore[misc]


def test_conflicting_execution_registration_keeps_existing_state() -> None:
    """上游标识冲突时不能替换已有快照或预占所有者，正反索引仍指向原记录。"""
    registry = ResourceRegistry(retention_seconds=0)
    worker = WorkerState("worker", "http://worker", "remote", active_jobs=1)
    reservation = JobReservation(worker, worker.generation)
    route = router_execution(registry, worker_id=worker.worker_id, upstream_id="upstream", reservation=reservation)
    state = route.state
    with pytest.raises(ValueError, match="another public ID"):
        registry.register_execution(
            owner_scope=route.owner_scope,
            worker_id=worker.worker_id,
            upstream_id=route.upstream_id,
            state=ExecutionRouteState(execution=replace(state.execution, execution_id="job-conflict")),
        )
    assert registry.get("job", route.public_id) is route
    assert registry.find_upstream("job", route.owner_scope, worker.worker_id, route.upstream_id) is route
    assert route.state is state and route.state.reservation is reservation and worker.active_jobs == 1
    route.state.release_reservation()


@pytest.mark.parametrize("release", [False, True])
def test_lease_timeout_releases_expired_state_without_affecting_other_workers(
    monkeypatch: pytest.MonkeyPatch, release: bool
) -> None:
    """一个 worker 超时仍清理到期认证和结果持有，其他 worker 正常续租且传输固定独立生效。"""
    monkeypatch.setattr(tasks, "_TASK_LEASE_REFRESH_TIMEOUT_SECONDS", 0.05)

    async def scenario() -> None:
        """使用真实异步 HTTP 客户端执行分组续租，只有请求传输使用受控替身。"""
        seen: dict[str, dict[str, str]] = {}
        blocked = asyncio.Event()

        async def request_upstream(request: httpx.Request) -> httpx.Response:
            """令第一个 worker 阻塞到超时，第二个立即应答并记录独立认证头。"""
            seen[request.url.host] = dict(request.headers)
            if request.url.host == "slow":
                await blocked.wait()
            return httpx.Response(200, json={})

        pool = WorkerPool(RouterSettings(local_gpus="none"), transport=httpx.MockTransport(request_upstream))
        sources = SourceFileStore(retention_seconds=10)
        registry = ResourceRegistry(retention_seconds=10)
        try:
            slow, fast = WorkerState("slow", "http://slow", "remote"), WorkerState("fast", "http://fast", "remote")
            pool._workers = {worker.worker_id: worker for worker in (slow, fast)}
            result = registry.register_file(owner_scope="owner", worker_id="slow", upstream_id="result")
            result.payload = {"purpose": "parse_output", "expires_at": time.time() - 1}
            result.lease.held = True
            slow_route = router_execution(registry, owner_scope="owner", worker_id="slow", upstream_id="slow-task")
            fast_route = router_execution(registry, owner_scope="owner", worker_id="fast", upstream_id="fast-task")
            for route in (slow_route, fast_route):
                route.payload = {"status": "completed", "files": []}
                route.lease.held = True
                route.lease.retain_until = time.time() + 60
                route.state.upstream_headers = {"authorization": f"Bearer {route.worker_id}"}
            slow_route.lease.retain_until = None if release else time.time() - 1
            slow_route.state.result_file_id = result.public_id
            with registry.pin_route(result):
                await tasks.maintain_task_result_leases(registry, pool, release=release)
                assert not slow_route.lease.held and not slow_route.state.upstream_headers
                assert not result.lease.held and result.pins == 1
                assert fast_route.lease.held is not release
                assert fast_route.lease.refreshed_at > 0
                assert fast_route.state.upstream_headers == ({} if release else {"authorization": "Bearer fast"})
                assert seen["slow"]["authorization"] == "Bearer slow"
                assert seen["slow"]["x-mineru-task-lease-seconds"] == "0"
                assert seen["fast"]["authorization"] == "Bearer fast"
                assert seen["fast"]["x-mineru-task-lease-seconds"] == ("0" if release else "120")
                registry.collect_expired(sources)
                assert registry.get("file", result.public_id) is result
            registry.collect_expired(sources)
            with pytest.raises(KeyError):
                registry.get("file", result.public_id)
        finally:
            sources.close()
            await pool.client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["timeout", "http_error", "transport_error"])
def test_failed_lease_refresh_does_not_block_other_tasks_on_same_worker(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """同一 worker 的单个请求超时或失败时，健康任务每轮仍能续租，失败不记为成功。"""
    monkeypatch.setattr(tasks, "_TASK_LEASE_REFRESH_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(tasks, "_TASK_LEASE_REQUEST_TIMEOUT_SECONDS", 0.02)

    async def scenario() -> None:
        """通过真实请求路径连续维护四轮，覆盖超时、HTTP 错误和传输异常。"""
        seen: list[str] = []
        blocked = asyncio.Event()

        async def request_upstream(request: httpx.Request) -> httpx.Response:
            """首个任务持续失败，后续任务立即响应并保留请求记录。"""
            seen.append(request.url.path)
            if request.url.path.endswith("/slow-task"):
                if failure == "timeout":
                    await blocked.wait()
                elif failure == "transport_error":
                    raise httpx.ConnectError("worker unavailable", request=request)
                return httpx.Response(503, json={})
            return httpx.Response(200, json={})

        pool = WorkerPool(RouterSettings(local_gpus="none"), transport=httpx.MockTransport(request_upstream))
        worker = WorkerState("worker", "http://worker", "remote")
        pool._workers = {worker.worker_id: worker}
        registry = ResourceRegistry(retention_seconds=60)
        slow = router_execution(registry, worker_id=worker.worker_id, upstream_id="slow-task")
        healthy = router_execution(registry, worker_id=worker.worker_id, upstream_id="healthy-task")
        for route in (slow, healthy):
            route.lease.held = True
            route.lease.retain_until = time.time() + 60
        try:
            for round_number in range(1, 5):
                previous_refresh = healthy.lease.refreshed_at
                await tasks.maintain_task_result_leases(registry, pool)
                assert seen.count("/v1/tasks/healthy-task") == round_number
                # Windows 单调时钟的相邻读数可能相等；逐轮请求次数验证实际续租已执行。
                assert healthy.lease.refreshed_at > 0
                assert healthy.lease.refreshed_at >= previous_refresh
                assert slow.lease.refreshed_at == 0
        finally:
            await pool.client.aclose()

    asyncio.run(scenario())


def test_lease_refresh_rotates_after_worker_budget_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """整轮预算耗尽时保留尝试顺序，多条阻塞路由不能在后续轮次持续抢占健康任务。"""
    monkeypatch.setattr(tasks, "_TASK_LEASE_REFRESH_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(tasks, "_TASK_LEASE_REQUEST_TIMEOUT_SECONDS", 0.5)

    async def scenario() -> None:
        """每轮只容纳一条阻塞请求，验证尚未尝试的队尾任务最终得到续租。"""
        seen: list[str] = []
        blocked = asyncio.Event()

        async def request_upstream(request: httpx.Request) -> httpx.Response:
            """让两条慢路由一直等待，健康路由则立即成功。"""
            seen.append(request.url.path)
            if not request.url.path.endswith("/healthy-task"):
                await blocked.wait()
            return httpx.Response(200, json={})

        pool = WorkerPool(RouterSettings(local_gpus="none"), transport=httpx.MockTransport(request_upstream))
        worker = WorkerState("worker", "http://worker", "remote")
        pool._workers = {worker.worker_id: worker}
        registry = ResourceRegistry(retention_seconds=60)
        routes = [
            router_execution(registry, worker_id=worker.worker_id, upstream_id=name)
            for name in ("slow-first", "slow-second", "healthy-task")
        ]
        for route in routes:
            route.lease.held = True
            route.lease.retain_until = time.time() + 60
        try:
            for _ in range(3):
                await tasks.maintain_task_result_leases(registry, pool)
            assert seen[:3] == ["/v1/tasks/slow-first", "/v1/tasks/slow-second", "/v1/tasks/healthy-task"]
            assert routes[-1].lease.refreshed_at > 0
            assert all(route.lease.refreshed_at == 0 for route in routes[:-1])
        finally:
            await pool.client.aclose()

    asyncio.run(scenario())
