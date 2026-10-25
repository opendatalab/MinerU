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
