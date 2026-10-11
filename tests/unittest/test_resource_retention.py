"""API 资源 TTL、共享字节租约、存储重启与 Router 缓存回收回归。"""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mineru.parser import api_server
from mineru.parser.api_server import ApiServerError, CreateJobRequest, FileStore, JobStore
from mineru.kit.router.resources import ResourceRegistry, SourceFileStore, StoredSourceFile
from mineru.utils.retention import resolve_retention_seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """只替换墙上时间，不影响事件循环超时和任务调度。"""
    value = [2_000_000_000]
    monkeypatch.setattr(api_server.time, "time", lambda: value[0])
    monkeypatch.setattr(
        JobStore, "_now", staticmethod(lambda: datetime.fromtimestamp(value[0], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    )
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
    data = asyncio.run(api_server._extract_bytes(api_server.FileIdSource(file_id=second), store, input_blobs=leases[1]))
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
        jobs = JobStore(concurrency=1, retention_seconds=10)
        req = CreateJobRequest.model_validate({"tier": "flash", "files": [{"source": {"type": "file_id", "file_id": source}}]})
        gate = asyncio.Event()
        records = [jobs.create(req, files) for _ in range(2)]
        outputs: list[str] = []

        async def work(rec: Any) -> None:
            """创建早期产物，然后等待；第二个排队任务读取已过期视图的租约。"""
            rec.status = "running"
            output = files.create_file_for_output("out.md", b"result", job_id=rec.id)
            outputs.append(output)
            await gate.wait()
            extracted = await api_server._extract_bytes(req.files[0].source, files, input_blobs=rec.input_blobs)
            assert extracted.data == b"document"
            rec.status = "completed"

        for rec in records:
            jobs.start_task(rec, lambda rec=rec: work(rec))
        await asyncio.sleep(0)
        clock[0] += 100
        jobs.collect_expired()
        files.collect_expired()
        assert len(jobs._jobs) == 2 and len(outputs) == 1 and files._blob_abs(sha).exists()
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
        assert not jobs._jobs and not jobs._completion_events and not files._files
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
    app = api_server.create_app(upload_dir=str(root), tier="flash", retention_seconds=10)
    with TestClient(app):
        assert not orphan_store._blob_abs(sha).exists() and unknown.exists()
        other = api_server.create_app(upload_dir=str(root), tier="flash", retention_seconds=10)
        with pytest.raises(RuntimeError, match="already owned"):
            with TestClient(other):
                pass
    with TestClient(api_server.create_app(upload_dir=str(root), tier="flash", retention_seconds=10)):
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
    file = registry.register("file", owner_scope="one", worker_id="worker", upstream_id="file-up")
    sources.bind_source(file.public_id, stored, owner_scope="one", expires_at=clock[0] + 10)
    registry.alias_upstream(file, worker_id="copy", upstream_id="copy-file")
    job = registry.register("job", owner_scope="one", worker_id="worker", upstream_id="job-up")
    job.metadata["payload"] = {"status": "running", "files": [{"file_id": file.public_id}]}
    with sources.pin(stored):
        clock[0] += 11
        registry.collect_expired(sources)
        assert registry.find("file", file.public_id) is None
        assert sources.find_hash("one", stored.sha256sum) is None
        assert path.exists() and registry.find("job", job.public_id) is job
        job.metadata["payload"]["status"] = "failed"
        job.metadata["payload"]["finished_at"] = JobStore._now()
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
        jobs = JobStore(retention_seconds=10)
        req = CreateJobRequest.model_validate({"tier": "flash", "files": [{"source": {"type": "file_id", "file_id": source}}]})
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
    """哈希命中后的 Content-Length 校验失败不会泄漏请求持有的 blob 引用。"""
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *args: None)
    app = api_server.create_app(upload_dir=str(tmp_path / "files"), tier="flash")
    with TestClient(app) as client:
        _, sha = _source(app.state.file_store, tmp_path)
        response = client.post(
            f"/v1/tasks?filename=input.pdf&sha256sum={sha}&tier=flash",
            content=b"bad",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 400
        assert not app.state.file_store._pins and not app.state.job_store._jobs


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


def test_router_pending_copy_cleanup_keeps_terminal_job_visible(clock: list[int]) -> None:
    """副本尚未清理完成时，任务不应开始保留期或从用户列表消失。"""
    from mineru.kit.router.resources import CopiedInputFile

    registry = ResourceRegistry(retention_seconds=10)
    route = registry.register("job", owner_scope="one", worker_id="worker", upstream_id="job-up")
    route.metadata["payload"] = {"status": "failed", "finished_at": JobStore._now()}
    route.metadata["copied_inputs"] = [CopiedInputFile("file-one", "one", "worker", "file-copy")]
    clock[0] += 11
    assert registry.find("job", route.public_id) is route
    assert registry.list("job") == [route]
    assert registry.list("job", include_expired=True) == [route]


def test_router_retention_starts_after_copy_cleanup_retry(clock: list[int]) -> None:
    """上游早已结束但副本删除失败时，重试成功后仍享有完整保留期。"""
    from test_v1_router import _FakeV1Upstream, _make_router, _upload_file
    from mineru.kit.router.app import _reconcile_jobs_once

    first = _FakeV1Upstream("worker-a", ("standard",))
    second = _FakeV1Upstream("worker-b", ("advanced",))
    app, _ = _make_router(first, second)
    app.state.registry.retention_seconds = 10
    headers = {"authorization": "Bearer retained-copy"}
    with TestClient(app) as client:
        source = _upload_file(client, token="retained-copy", content=b"pdf")
        created = client.post(
            "/v1/parse/jobs",
            headers=headers,
            json={"tier": "advanced", "files": [{"source": {"type": "file_id", "file_id": source}}]},
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        route = app.state.registry.get("job", job_id)
        copied = route.metadata["copied_inputs"][0]
        second.fail_file_delete_once.add(copied.upstream_file_id)
        assert client.get(f"/v1/parse/jobs/{job_id}", headers=headers).status_code == 200
        route.metadata["payload"]["finished_at"] = JobStore._now()
        assert route.metadata.get("copied_inputs")
        clock[0] += 11
        app.state.registry.collect_expired(app.state.source_store)
        assert app.state.registry.find("job", job_id) is route
        assert client.get("/v1/parse/jobs", headers=headers).json()["data"][0]["job_id"] == job_id
        client.portal.call(_reconcile_jobs_once, app.state.registry, app.state.worker_pool)
        assert not route.metadata.get("copied_inputs")
        assert copied.upstream_file_id not in second.files
        clock[0] += 9
        assert client.get(f"/v1/parse/jobs/{job_id}", headers=headers).status_code == 200
        clock[0] += 1
        assert client.get(f"/v1/parse/jobs/{job_id}", headers=headers).status_code == 404


def test_router_usage_survives_expiry_collection_and_worker_removal(clock: list[int]) -> None:
    """过期隐藏、重复扫描与 worker 移除均不能减少或重复累计调用方用量。"""
    from test_v1_router import _FakeV1Upstream, _make_router

    app, _ = _make_router(_FakeV1Upstream("worker-a", ("standard",)))
    app.state.registry.retention_seconds = 10
    headers = {"authorization": "Bearer lifetime"}
    body = {"tier": "standard", "files": [{"source": {"type": "inline", "name": "demo.pdf", "data": "cGRm"}}]}
    with TestClient(app) as client:
        for _ in range(2):
            created = client.post("/v1/parse/jobs", headers=headers, json=body)
            assert created.status_code == 202
            job_id = created.json()["job_id"]
            assert client.get(f"/v1/parse/jobs/{job_id}", headers=headers).status_code == 200
        before = client.get("/v1/usage", headers=headers).json()["current"]
        assert before == {"pages_processed": 0, "files_processed": 2, "jobs_created": 2}
        clock[0] += 11
        assert client.get("/v1/parse/jobs", headers=headers).json()["data"] == []
        assert client.get("/v1/usage", headers=headers).json()["current"] == before
        for _ in range(2):
            app.state.registry.collect_expired(app.state.source_store)
        assert not app.state.registry.list("job", include_expired=True)
        assert client.get("/v1/usage", headers=headers).json()["current"] == before
        created = client.post("/v1/parse/jobs", headers=headers, json=body)
        assert created.status_code == 202
        route = app.state.registry.get("job", created.json()["job_id"])
        app.state.registry.remove_worker(route.worker_id)
        assert client.get("/v1/usage", headers=headers).json()["current"] == {
            **before,
            "jobs_created": 3,
        }
        assert client.get("/v1/usage").json()["current"] == {
            "pages_processed": 0,
            "files_processed": 0,
            "jobs_created": 0,
        }


def test_retention_keeps_cumulative_usage(tmp_path: Path, clock: list[int]) -> None:
    """元数据回收后已完成的文件、页数与任务累计值不减少。"""
    files = FileStore(tmp_path / "files", retention_seconds=10)
    source, _ = _source(files, tmp_path)
    jobs = JobStore(retention_seconds=10)
    request = CreateJobRequest.model_validate({"tier": "flash", "files": [{"source": {"type": "file_id", "file_id": source}}]})
    rec = jobs.create(request, files)
    rec.status = rec.files[0].status = "completed"
    rec.files[0].page_range = "1-3"
    rec.finished_at = JobStore._now()
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
    """外部 worker 重启丢失内存任务时，Router 不得永久占用名额或丢弃失败历史。"""
    from test_v1_router import _FakeV1Upstream, _make_router
    from mineru.kit.router.app import _reconcile_jobs_once

    upstream = _FakeV1Upstream("worker-a", ("standard",))
    app, _ = _make_router(upstream)
    with TestClient(app) as client:
        created = client.post(
            "/v1/parse/jobs",
            json={"tier": "standard", "files": [{"source": {"type": "inline", "name": "demo.pdf", "data": "cGRm"}}]},
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        route = app.state.registry.get("job", job_id)
        worker = app.state.worker_pool.get(route.worker_id)
        assert worker.active_jobs == 1
        upstream.jobs.clear()
        client.portal.call(_reconcile_jobs_once, app.state.registry, app.state.worker_pool)
        assert worker.active_jobs == 0
        response = client.get(f"/v1/parse/jobs/{job_id}")
        assert response.status_code == 200 and response.json()["status"] == "failed"
        assert response.json()["files"][0]["error"]["code"] == "upstream_job_lost"


def test_lost_task_retries_failed_cross_worker_copy_cleanup() -> None:
    """上游任务丢失后副本删除失败一次，后续后台轮次仍继续回收并释放 alias。"""
    from test_v1_router import _FakeV1Upstream, _make_router, _upload_file
    from mineru.kit.router.app import _reconcile_jobs_once

    first = _FakeV1Upstream("worker-a", ("standard",))
    second = _FakeV1Upstream("worker-b", ("advanced",))
    app, _ = _make_router(first, second)
    headers = {"authorization": "Bearer lost-copy"}
    with TestClient(app) as client:
        source = _upload_file(client, token="lost-copy", content=b"pdf")
        response = client.post(
            "/v1/parse/jobs",
            headers=headers,
            json={"tier": "advanced", "files": [{"source": {"type": "file_id", "file_id": source}}]},
        )
        assert response.status_code == 202
        route = app.state.registry.get("job", response.json()["job_id"])
        copied = route.metadata["copied_inputs"][0]
        second.jobs.clear()
        second.fail_file_delete_once.add(copied.upstream_file_id)
        client.portal.call(_reconcile_jobs_once, app.state.registry, app.state.worker_pool)
        assert route.metadata["upstream_lost"] and route.metadata["copied_inputs"]
        assert second.files.get(copied.upstream_file_id) is not None
        client.portal.call(_reconcile_jobs_once, app.state.registry, app.state.worker_pool)
        assert not route.metadata.get("copied_inputs")
        assert copied.upstream_file_id not in second.files
        assert app.state.registry.find_upstream("file", copied.owner_scope, copied.worker_id, copied.upstream_file_id) is None
