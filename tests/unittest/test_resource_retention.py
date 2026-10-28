"""API 资源 TTL、共享字节租约、存储重启与 Router 缓存回收回归。"""

from __future__ import annotations


import asyncio
import hashlib
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from task_test_utils import execution_request, router_execution

from mineru.parser import api_server
from mineru.parser.api_server import FileStore, ExecutionStore
from mineru.tasks.errors import ApiServerError
from mineru.tasks import execution
from mineru.tasks.execution import execution_timestamp
from mineru.kit.router.resources import ResourceRegistry, SourceFileStore, StoredSourceFile
from mineru.utils.retention import resolve_retention_seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """只替换墙上时间，不影响事件循环超时和任务调度。"""
    value = [2_000_000_000]
    monkeypatch.setattr(api_server.time, "time", lambda: value[0])
    monkeypatch.setattr(execution, "datetime", SimpleNamespace(now=lambda zone: datetime.fromtimestamp(value[0], zone)))
    return value


def _source(store: FileStore, tmp_path: Path, data: bytes = b"document") -> tuple[str, str]:
    """注册真实共享 blob，并使磁盘年龄能由测试墙上时间控制。"""
    path = tmp_path / "input.pdf"
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    file_id = store.register_source_file("input.pdf", path, sha)
    os.utime(store._blob_abs(sha), (0, 0))
    return file_id, sha


def test_retention_config_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """API 参数优先于统一环境变量；默认一天、零禁用、非法配置拒绝启动。"""
    monkeypatch.delenv("MINERU_API_RETENTION_SECONDS", raising=False)
    assert resolve_retention_seconds() == 86400
    monkeypatch.setenv("MINERU_API_RETENTION_SECONDS", "172800")
    assert resolve_retention_seconds() == 172800 and resolve_retention_seconds(0) == 0
    for raw in ("-1", "bad"):
        monkeypatch.setenv("MINERU_API_RETENTION_SECONDS", raw)
        with pytest.raises(ValueError):
            resolve_retention_seconds()


def test_shared_hash_and_queue_leases_survive_expired_views(tmp_path: Path, clock: list[int]) -> None:
    """两文件、两任务共享同一哈希，视图到期后最后一个任务释放前字节不能删除。"""
    store = FileStore(tmp_path / "files", retention_seconds=10)
    first, sha = _source(store, tmp_path)
    second, _ = _source(store, tmp_path)
    leases = [store.acquire_inputs([file]) for file in (first, second)]
    clock[0] += 11
    store.collect_expired()
    assert not store._files and store._blob_abs(sha).is_file()
    with pytest.raises(ApiServerError) as exc:
        store.get_file(first)
    assert exc.value.status_code == 404
    assert store.pin_cached_blob(sha) is None
    store.release_inputs(leases[0])
    store.collect_expired()
    assert store._blob_abs(sha).is_file()
    data = asyncio.run(api_server._extract_bytes(api_server._StoredSource(file_id=second), store, input_blobs=leases[1]))
    assert data.data == b"document"
    store.release_inputs(leases[1])
    store.collect_expired()
    assert not store._blob_abs(sha).exists() and not store._pins


def test_output_retention_starts_after_cleanup(tmp_path: Path, clock: list[int]) -> None:
    """长任务的产物和输入受到保护；完成后清理事件与产物按同一期限删除。"""

    async def scenario() -> None:
        """手动控制运行、排队、完成和回收的交错顺序。"""
        files = FileStore(tmp_path / "files", retention_seconds=10)
        source, sha = _source(files, tmp_path)
        jobs = ExecutionStore(concurrency=1, retention_seconds=10)
        gate = asyncio.Event()
        records = [
            jobs.create(execution_request(files, {"type": "file_id", "file_id": source}, tier="flash"), files) for _ in range(2)
        ]
        outputs: list[str] = []

        async def work(rec: Any) -> None:
            """创建早期产物，然后等待；第二个排队任务读取已过期视图的租约。"""
            rec.status = "running"
            output = files.create_file_for_output("out.md", b"result", job_id=rec.id)
            outputs.append(output)
            await gate.wait()
            extracted = await api_server._extract_bytes(rec.request.source, files, input_blobs=rec.input_blobs)
            assert extracted.data == b"document"
            rec.status = "completed"

        for rec in records:
            jobs.start_task(rec, lambda rec=rec: work(rec))
        await asyncio.sleep(0)
        clock[0] += 100
        jobs.collect_expired()
        files.collect_expired()
        assert len(jobs._executions) == 2 and len(outputs) == 1 and files._blob_abs(sha).exists()
        assert files.get_file(outputs[0]).expires_at is None
        gate.set()
        for rec in records:
            await jobs.wait_for_terminal(rec.id, 2)
        assert not files._pins
        assert all(files.get_file(file).expires_at == clock[0] + 10 for file in outputs)
        clock[0] += 11
        for file in outputs:
            with pytest.raises(ApiServerError) as exc:
                files.read_file_data(file)
            assert exc.value.status_code == 404
        jobs.collect_expired()
        files.collect_expired()
        assert not jobs._executions and not jobs._completion_events and not files._files
        assert not any(files._blobs.glob("??/*"))
        await jobs.shutdown()

    asyncio.run(scenario())


def test_cached_preflight_pins_and_last_reference_delete(tmp_path: Path, clock: list[int]) -> None:
    """删除最后文件与哈希复用交错时，预检租约保护路径直到提交或拒绝完成。"""
    store = FileStore(tmp_path / "files", retention_seconds=10)
    file_id, sha = _source(store, tmp_path)
    path = store.pin_cached_blob(sha)
    store.delete_file(file_id)
    store.collect_expired()
    assert path.is_file()
    replacement = store.register_source_file("new.pdf", path, sha)
    store.unpin_cached_blob(sha)
    store.collect_expired()
    assert store.get_file(replacement).sha256sum == sha
    store.delete_file(replacement)
    store.collect_expired()
    assert not path.exists()


def test_startup_orphan_cleanup_and_directory_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: list[int]
) -> None:
    """重启清理旧无引用受管文件；未知路径保留，第二实例不能占用相同目录。"""
    root = tmp_path / "files"
    orphan_store = FileStore(root, retention_seconds=10)
    _, sha = _source(orphan_store, tmp_path)
    unknown = root / "blobs/unknown.txt"
    unknown.write_text("keep")
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *args: None)
    app = api_server.create_app(upload_dir=str(root), tiers=("flash",), retention_seconds=10)
    with TestClient(app):
        assert not orphan_store._blob_abs(sha).exists() and unknown.exists()
        other = api_server.create_app(upload_dir=str(root), tiers=("flash",), retention_seconds=10)
        with pytest.raises(RuntimeError, match="already owned"):
            with TestClient(other):
                pass
    with TestClient(api_server.create_app(upload_dir=str(root), tiers=("flash",), retention_seconds=10)):
        pass


def test_disabled_collection_preserves_blobs_and_jobs(tmp_path: Path, clock: list[int]) -> None:
    """零策略保留源字节和产物，显式上传期限仍由其本身控制。"""
    files = FileStore(tmp_path / "files", retention_seconds=0)
    file, sha = _source(files, tmp_path)
    out = files.create_file_for_output("out.md", b"output")
    clock[0] += 1_000_000
    files.collect_expired()
    assert files.get_file(file).expires_at is None and files.get_file(out).expires_at is None
    assert files._blob_abs(sha).exists()


def test_router_expiry_cache_pins_and_alias_cleanup(tmp_path: Path, clock: list[int]) -> None:
    """Router 到期立即隐藏资源，复制租约保留路径，扫描同步清除公共与反向索引。"""
    sources = SourceFileStore(retention_seconds=10)
    registry = ResourceRegistry(retention_seconds=10)
    path = sources._root / "input"
    path.write_bytes(b"input")
    stored = StoredSourceFile(path, 5, hashlib.sha256(b"input").hexdigest(), "application/pdf")
    file = registry.register_file(owner_scope="one", worker_id="worker", upstream_id="file-up")
    sources.bind_source(file.public_id, stored, owner_scope="one", expires_at=clock[0] + 10)
    registry.alias_upstream(file, worker_id="copy", upstream_id="copy-file")
    job = router_execution(registry, owner_scope="one", worker_id="worker", upstream_id="job-up", file_id=file.public_id)
    job.payload = {"status": "running", "files": [{"file_id": file.public_id}]}
    with sources.pin(stored):
        clock[0] += 11
        registry.collect_expired(sources)
        assert registry.find("file", file.public_id) is None
        assert sources.find_hash("one", stored.sha256sum) is None
        assert path.exists() and registry.find("job", job.public_id) is job
        job.payload["status"] = "failed"
        job.payload["finished_at"] = execution_timestamp()
        registry.collect_expired(sources)
        assert registry.find_upstream("file", "one", "copy", "copy-file") is None
        assert path.exists()
    assert not path.exists()
    clock[0] += 11
    registry.collect_expired(sources)
    assert not registry._by_public["job"] and not registry._by_upstream
    sources.close()


def test_queued_cancel_releases_input_before_execution(tmp_path: Path, clock: list[int]) -> None:
    """首次调度前取消的任务同样释放输入租约及完成事件，不进入解析。"""

    async def scenario() -> None:
        """取消早于 run_owned 的首次执行，覆盖没有 finally 执行机会的分支。"""
        files = FileStore(tmp_path / "files", retention_seconds=10)
        source, sha = _source(files, tmp_path)
        jobs = ExecutionStore(retention_seconds=10)
        req = execution_request(files, {"type": "file_id", "file_id": source}, tier="flash")
        rec = jobs.create(req, files)

        async def forbidden() -> None:
            """排队取消后不允许调用此操作。"""
            raise AssertionError("must not execute")

        jobs.start_task(rec, forbidden)
        jobs.cancel(rec.id)
        await jobs.wait_for_terminal(rec.id, 2)
        assert rec.status == "canceled" and not files._pins
        clock[0] += 11
        files.collect_expired()
        jobs.collect_expired()
        assert not files._blob_abs(sha).exists() and not jobs._completion_events
        await jobs.shutdown()

    asyncio.run(scenario())


def test_hash_preflight_rejection_releases_request_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """清单 hash 命中后的声明大小校验失败不会泄漏准备阶段的 blob 引用。"""
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *args: None)
    app = api_server.create_app(upload_dir=str(tmp_path / "files"), tiers=("flash",))
    with TestClient(app) as client:
        _, sha = _source(app.state.file_store, tmp_path)
        response = client.post(
            "/v1/tasks",
            json={
                "tier": "flash",
                "files": [{"file_name": "input.pdf", "bytes": 3, "sha256sum": sha}],
            },
        )
        assert response.status_code == 400
        assert response.json().get("error")["code"] == "upload_size_mismatch"
        assert not app.state.file_store._pins and not app.state.execution_store._executions


def test_gc_interleaves_with_blob_writes_and_downloads(tmp_path: Path, clock: list[int]) -> None:
    """多个线程写相同哈希、下载和删除时，回收只允许产生规范 404，不出现残缺字节。"""
    from concurrent.futures import ThreadPoolExecutor

    files = FileStore(tmp_path / "files", retention_seconds=10)
    data = b"download" * 10000

    def work(index: int) -> None:
        """每个下载视图都有独立引用，扫描不能删除另一个线程刚写完的文件。"""
        file = files.create_file_for_output(f"out-{index}.md", data)
        files.collect_expired()
        assert files.read_file_data(file) == data
        files.delete_file(file)
        files.collect_expired()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(20)))
    clock[0] += 11
    files.collect_expired()
    assert not files._files and not files._pins


def test_router_task_result_lease_keeps_terminal_execution_visible(clock: list[int]) -> None:
    """父任务持有结果租约时，内部执行终态不会提前被回收。"""
    registry = ResourceRegistry(retention_seconds=10)
    route = router_execution(registry, owner_scope="one", worker_id="worker", upstream_id="child")
    route.payload = {"status": "completed"}
    route.lease.held = True
    clock[0] += 20
    assert registry.find("job", route.public_id) is route
    route.lease.retain_until = clock[0] + 10
    clock[0] += 11
    assert registry.find("job", route.public_id) is None


def test_router_usage_survives_expiry_collection_and_worker_removal(clock: list[int]) -> None:
    """内部逐文件执行计数在过期和 worker 移除后仍保留，并隔离调用方。"""
    registry = ResourceRegistry(retention_seconds=10)
    sources = SourceFileStore(retention_seconds=10)
    try:
        for index in range(2):
            route = router_execution(registry, owner_scope="owner", worker_id="worker", upstream_id=str(index))
            route.payload = {"status": "completed", "files": [{"status": "completed"}]}
            route.finalized_at = execution_timestamp()
        before = registry.usage("owner")
        assert before == {"jobs_created": 2, "files_processed": 2}
        clock[0] += 11
        registry.collect_expired(sources)
        assert registry.usage("owner") == before and not registry.list("job", include_expired=True)
        route = router_execution(registry, owner_scope="owner", worker_id="worker", upstream_id="new")
        route.payload = {"status": "running", "files": [{"status": "running"}]}
        registry.remove_worker("worker")
        assert registry.usage("owner") == {"jobs_created": 3, "files_processed": 2}
        assert registry.usage("anonymous") == {"jobs_created": 0, "files_processed": 0}
    finally:
        sources.close()


def test_retention_keeps_cumulative_usage(tmp_path: Path, clock: list[int]) -> None:
    """元数据回收后已完成的文件、页数与任务累计值不减少。"""
    files = FileStore(tmp_path / "files", retention_seconds=10)
    source, _ = _source(files, tmp_path)
    jobs = ExecutionStore(retention_seconds=10)
    request = execution_request(files, {"type": "file_id", "file_id": source}, tier="flash")
    rec = jobs.create(request, files)
    rec.status = "completed"
    rec.page_range = "1-3"
    rec.finished_at = execution_timestamp()
    before = jobs.usage("anonymous").current
    files.release_inputs(rec.input_blobs)
    clock[0] += 11
    jobs.collect_expired()
    assert jobs.usage("anonymous").current == before
    assert before.jobs_created == 1 and before.pages_processed == 3 and before.files_processed == 1


def test_kit_cli_forwards_explicit_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    """正式 Typer 入口传递零值，覆盖环境变量，并供托管 worker 启动命令使用。"""
    from unittest.mock import MagicMock
    from typer.testing import CliRunner
    from mineru.kit.main import app

    forwarded = MagicMock()
    monkeypatch.setattr(api_server.main, "main", forwarded)
    response = CliRunner().invoke(
        app, ["api-server", "--tier", "flash", "--retention-seconds", "0"], env={"MINERU_API_RETENTION_SECONDS": "7200"}
    )
    assert response.exit_code == 0, response.output
    args = forwarded.call_args.kwargs["args"]
    assert args[args.index("--retention-seconds") + 1] == "0"


def test_external_worker_lost_task_releases_load_and_keeps_snapshot() -> None:
    """worker Tasks 返回 404 时后台收敛负载，并保存失败执行快照。"""
    from test_v1_router import _FakeV1Upstream, _make_router
    from mineru.kit.router.app import _reconcile_jobs_once

    app, _ = _make_router(_FakeV1Upstream("worker-a", ("standard",)))
    with TestClient(app) as client:
        pool = app.state.worker_pool
        reservation = pool.reserve_job(tier="standard", required_sources={"upload"})
        route = router_execution(
            app.state.registry,
            worker_id=reservation.worker.worker_id,
            upstream_id="lost-task",
            file_id="file-lost",
            file_name="lost.pdf",
            reservation=reservation,
        )
        route.payload = {"status": "running", "files": [{"status": "pending"}]}
        client.portal.call(_reconcile_jobs_once, app.state.registry, pool)
        assert reservation.worker.active_jobs == 0
        assert route.payload["status"] == "failed"
        assert route.payload["files"][0].get("error")["code"] == "upstream_task_lost"
