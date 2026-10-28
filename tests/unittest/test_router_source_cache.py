"""私有源缓存不依赖下游 Files 标识，引用和过期由本层任务生命周期管理。"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from task_test_utils import task_url
from test_task_api import HTML, _WorkerTransport, _submit, _terminal, _upload

from mineru.kit.router import RouterSettings, create_app
from mineru.kit.router.resources import SourceFileStore, StoredSourceFile
from mineru.parser import api_server


def test_private_cache_reference_lifetime_and_transfer_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """独立回收只移除到期引用，活跃任务和并发下载均保护同一份字节。"""
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    sources = SourceFileStore(retention_seconds=10)
    try:
        path = sources._root / "shared"
        path.write_bytes(b"source")
        stored = StoredSourceFile(path, 6, hashlib.sha256(b"source").hexdigest(), "application/pdf")
        for reference in ("task-a", "task-b"):
            sources.bind_source(reference, stored, owner_scope="owner")
            sources.hold_reference(reference)
        assert sources.find_hash("stranger", stored.sha256sum) is None
        now[0] += 1000
        sources.collect_expired()
        assert sources.find_hash("owner", stored.sha256sum) is stored
        sources.release_reference("task-a", now[0])
        deadline = sources._file_expiry["task-a"]
        now[0] += 11
        sources.release_reference("task-a", now[0])
        assert sources._file_expiry["task-a"] == deadline
        sources.collect_expired()
        assert set(sources._files) == {"task-b"} and path.exists()
        with sources.pin(stored):
            sources.release_reference("task-b", now[0])
            now[0] += 11
            sources.collect_expired()
            assert not sources._files and not sources._active_references
            assert sources.find_hash("owner", stored.sha256sum) is None
            assert path.exists()
        assert not path.exists() and not sources._pins
    finally:
        sources.close()


def test_private_cache_disabled_retention_and_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    """禁用保留期仍保留已提交源缓存，准备失败回滚只删除自己的引用。"""
    sources = SourceFileStore(retention_seconds=0)
    try:
        path = sources._root / "source"
        path.write_bytes(b"source")
        stored = StoredSourceFile(path, 6, hashlib.sha256(b"source").hexdigest(), "application/pdf")
        for reference in ("committed", "prepared"):
            sources.bind_source(reference, stored, owner_scope="owner")
            sources.hold_reference(reference)
        sources.delete_file("prepared")
        sources.delete_file("prepared")
        sources.release_reference("committed", 1000)
        monkeypatch.setattr(time, "time", lambda: 10_000_000)
        sources.collect_expired()
        assert sources.find_hash("owner", stored.sha256sum) is stored
        assert set(sources._files) == {"committed"} and not sources._active_references
    finally:
        sources.close()


def test_parent_cleanup_releases_sources_after_worker_route_disappears(tmp_path: Path) -> None:
    """执行路由已移除时，父任务仍须结束私有源引用并允许独立缓存回收。"""
    worker = api_server.create_app(upload_dir=str(tmp_path / "worker"), tiers=("flash",))
    app = create_app(
        RouterSettings(upstream_urls=("http://worker",), local_gpus="none", worker_refresh_interval_seconds=0),
        transport=_WorkerTransport({"worker": worker}),
    )
    with TestClient(app) as client:
        task = _submit(client, [("source.html", HTML)], headers={"x-mineru-task-lease-seconds": "120"})
        _upload(client, task, [("source.html", HTML)])
        assert _terminal(client, task)["status"] == "completed"
        record = app.state.task_store._records[task["task_id"]]
        sources = app.state.source_store
        assert sources._active_references
        client.portal.call(app.state.registry.remove, "job", record.file_states[0].execution_id)
        assert client.get(task_url(client, task), headers={"x-mineru-task-lease-seconds": "0"}).status_code == 200

        async def drain() -> None:
            """等待真实父任务清理，确保检查时释放钩子已经执行。"""
            await record.operation

        client.portal.call(drain)
        assert not sources._active_references and not sources._pins
        assert sources._files
        for reference in sources._files:
            assert sources._file_expiry[reference] is not None
            sources._file_expiry[reference] = 0
        paths = {stored.path for stored in sources._files.values()}
        client.portal.call(sources.collect_expired)
        assert not sources._files and all(not path.exists() for path in paths)
