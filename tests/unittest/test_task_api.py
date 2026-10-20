"""真实 API 与双 worker Router 的表单任务、逐文件上传和 ZIP 链接契约。"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mineru.kit.router import RouterSettings
from mineru.kit.router import create_app as create_router
from mineru.parser import api_server, task_api

HTML = b"<!doctype html><html><body><h1>Shared tasks</h1><p>Preserved content.</p></body></html>"
SECOND = b"<h1>Second document</h1><p>Second content.</p>"


class _WorkerTransport(httpx.AsyncBaseTransport):
    """使用真实 worker 路由验证源文件权限和任务生命周期，不以宽松 fake 替代。"""

    def __init__(self, apps: dict[str, FastAPI]) -> None:
        """保存 worker 和真实 ASGI transport，供断连与调度测试观察。"""
        self.apps = apps
        self.transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}
        self.calls: list[tuple[str, str, str]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """按主机选择 worker，并记录真实上游协议。"""
        self.calls.append((request.url.host, request.method, request.url.path))
        return await self.transports[request.url.host].handle_async_request(request)

    async def aclose(self) -> None:
        """等待任务与 Job 清理完成，再释放测试 worker 的解析运行时。"""
        for app in self.apps.values():
            await app.state.task_store.shutdown()
            await app.state.job_store.shutdown()


@pytest.fixture(params=["api", "router"])
def service(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[tuple[TestClient, FastAPI]]:
    """对本地服务与双 worker Router 执行同一份共同协议验收。"""
    if request.param == "api":
        app = api_server.create_app(upload_dir=str(tmp_path / "direct"), tier="flash")
    else:
        workers = {
            host: api_server.create_app(upload_dir=str(tmp_path / host), tier="flash") for host in ("worker-a", "worker-b")
        }
        transport = _WorkerTransport(workers)
        app = create_router(
            RouterSettings(
                upstream_urls=tuple(f"http://{host}" for host in workers), local_gpus="none", worker_refresh_interval_seconds=0
            ),
            transport=transport,
        )
        app.state.test_transport = transport
    with TestClient(app) as client:
        yield client, app


def _submit(
    client: TestClient,
    inputs: list[tuple[str, bytes]],
    *,
    hashes: bool = True,
    headers: dict[str, str] | None = None,
    **options: str | list[str],
) -> dict[str, Any]:
    """只用表单提交真实大小及可选 SHA-256，绝不在创建请求中发送多个文件。"""
    manifest = [
        {
            "client_file_id": f"f{index + 1}",
            "filename": name,
            "bytes": len(data),
            **({"sha256sum": hashlib.sha256(data).hexdigest()} if hashes else {}),
        }
        for index, (name, data) in enumerate(inputs)
    ]
    response = client.post("/v1/tasks", data={"manifest": json.dumps(manifest), "tier": "flash", **options}, headers=headers)
    assert response.status_code == 202, response.text
    return response.json()


def _upload(
    client: TestClient, task: dict[str, Any], inputs: list[tuple[str, bytes]], *, headers: dict[str, str] | None = None
) -> None:
    """仅上传返回链接对应的缺失文件，按 client_file_id 保持同名文件的关联。"""
    for link in task["uploads"]:
        index = int(link["client_file_id"][1:]) - 1
        response = client.put(
            link["upload_url"], content=inputs[index][1], headers={**link["upload_headers"], **(headers or {})}
        )
        assert response.status_code == 200, response.text


def _terminal(client: TestClient, task: dict[str, Any], *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """只轮询状态接口；成功前必须一直没有 result_url。"""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(task["status_url"], headers=headers)
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in task_api.TERMINAL_TASK_STATUSES:
            return payload
        assert payload["result_url"] is None
        time.sleep(0.01)
    pytest.fail("Task did not reach a terminal state")


def _archive(client: TestClient, task: dict[str, Any]) -> dict[str, bytes]:
    """从返回的独立 Files 链接下载并读取真实 ZIP 成员。"""
    assert "/v1/files/" in task["result_url"]
    response = client.get(task["result_url"])
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _job_count(app: FastAPI) -> int:
    """观察唯一执行队列，准备任务不应该产生 Job 或 worker 预占。"""
    if hasattr(app.state, "job_store"):
        return len(app.state.job_store._jobs)
    return len(app.state.registry.list("job"))


@pytest.mark.parametrize("hashes", [False, True])
def test_manifest_uploads_and_zip_only_status(service: tuple[TestClient, FastAPI], hashes: bool) -> None:
    """单文件与批量都经过共同协议，状态只给元数据，ZIP 保留输入次序。"""
    client, app = service
    inputs = [("同名 文档.html", HTML), ("同名 文档.html", SECOND)]
    task = _submit(client, inputs, hashes=hashes, output_formats=["markdown", "middle_json"])
    assert task["status"] == "waiting_upload" and task["result_url"] is None
    assert len(task["uploads"]) == 2 and _job_count(app) == 0
    if hasattr(app.state, "worker_pool"):
        assert [worker.active_jobs for worker in app.state.worker_pool.workers] == [0, 0]
    first = client.put(task["uploads"][0]["upload_url"], content=HTML)
    assert first.status_code == 200
    ready = client.get(task["status_url"]).json()
    assert ready["status"] == "waiting_upload"
    assert [file["upload_status"] for file in ready["files"]] == ["ready", "pending"]
    assert [link["client_file_id"] for link in ready["uploads"]] == ["f2"]
    assert _job_count(app) == 0
    _upload(client, ready, inputs)
    result = _terminal(client, task)
    assert result["status"] == "completed"
    assert result["task_id"] == task["task_id"] and _job_count(app) == 1
    assert all("content" not in file for file in result["files"])
    assert client.get(f"/v1/parse/jobs/{task['task_id']}").json()["job_id"] == task["task_id"]
    members = _archive(client, result)
    assert b"Preserved content" in members["0001/markdown.md"]
    assert b"Second content" in members["0002/markdown.md"]
    manifest = json.loads(members["manifest.json"])
    assert manifest["status"] == "completed"
    assert [item["client_file_id"] for item in manifest["files"]] == ["f1", "f2"]
    assert [item["directory"] for item in manifest["files"]] == ["0001/", "0002/"]
    assert "task_id" not in manifest
    assert all("file_id" not in item and "output_files" not in item for item in manifest["files"])
    assert json.loads(members["0001/middle_json.json"])["schema"] == "docvortex.middle"
    assert client.get(f"/v1/tasks/{task['task_id']}/result").status_code == 404
    assert client.post("/v1/file_parse").status_code == 404
    assert client.post("/tasks").status_code == 404
    assert client.get(f"/v1/files/{result['files'][0]['file_id']}/content").status_code == 403


@pytest.mark.parametrize("hashes", [False, True])
def test_single_multipart_is_asynchronous_and_checks_hash(service: tuple[TestClient, FastAPI], hashes: bool) -> None:
    """单文件直传保留便利性，实际发送的字节必须经过 hash 检查。"""
    client, app = service
    form: dict[str, Any] = {"tier": "flash"}
    if hashes:
        form["manifest"] = json.dumps(
            [
                {
                    "client_file_id": "f1",
                    "filename": "logical.html",
                    "bytes": len(HTML),
                    "sha256sum": hashlib.sha256(HTML).hexdigest(),
                }
            ]
        )
    response = client.post("/v1/tasks", data=form, files={"files": ("source.html", HTML)})
    assert response.status_code == 202, response.text
    result = _terminal(client, response.json())
    assert result["status"] == "completed"
    assert result["files"][0]["filename"] == ("logical.html" if hashes else "source.html")
    assert b"Shared tasks" in _archive(client, result)["markdown.md"]
    before = _job_count(app)
    invalid = client.post(
        "/v1/tasks",
        data={
            "manifest": json.dumps(
                [
                    {
                        "client_file_id": "f1",
                        "filename": "source.html",
                        "bytes": len(HTML),
                        "sha256sum": hashlib.sha256(HTML).hexdigest(),
                    }
                ]
            )
        },
        files={"files": ("source.html", HTML.replace(b"Shared", b"Broken"))},
    )
    assert invalid.status_code == 400 and invalid.json()["error"]["code"] == "file_hash_mismatch"
    assert _job_count(app) == before


class _UnreadableBody(httpx.SyncByteStream):
    """重复上传已确认文件时，读取主体应立即使测试失败。"""

    def __iter__(self) -> Iterator[bytes]:
        """验证服务端幂等返回，既不覆盖已确认字节也不读取重传主体。"""
        raise AssertionError("Ready file body must not be read")
        yield b""  # pragma: no cover


def test_mixed_cache_hit_survives_source_deletion(service: tuple[TestClient, FastAPI]) -> None:
    """命中项在等待其他文件时持有独立引用，原源文件删除不导致缓存竞态。"""
    client, app = service
    initial = _submit(client, [("original.html", HTML)])
    _upload(client, initial, [("original.html", HTML)])
    original = _terminal(client, initial)
    inputs = [("renamed.html", HTML), ("new.html", SECOND)]
    task = _submit(client, inputs)
    assert [link["client_file_id"] for link in task["uploads"]] == ["f2"]
    assert task["files"][0]["upload_status"] == "ready"
    assert client.delete(f"/v1/files/{original['files'][0]['file_id']}").status_code == 200
    duplicate = client.put(
        f"/v1/tasks/{task['task_id']}/uploads/f1",
        content=_UnreadableBody(),
        headers={"content-length": str(len(HTML)), "expect": "100-continue"},
    )
    assert duplicate.status_code == 200 and duplicate.headers["connection"] == "close"
    _upload(client, task, inputs)
    result = _terminal(client, task)
    assert result["status"] == "completed" and result["files"][0]["name"] == "renamed.html"
    cached = _submit(client, inputs)
    assert cached["uploads"] == [] and cached["status"] == "queued"
    assert _terminal(client, cached)["status"] == "completed"
    assert _job_count(app) == 3


def test_no_hash_still_deduplicates_source_storage(service: tuple[TestClient, FastAPI]) -> None:
    """不发送 hash 时仍计算实际内容标识，相同内容只保留一份源字节。"""
    client, app = service
    first = _submit(client, [("first.html", HTML)], hashes=False)
    _upload(client, first, [("first.html", HTML)])
    assert _terminal(client, first)["status"] == "completed"
    second = _submit(client, [("second.html", HTML)], hashes=False)
    assert len(second["uploads"]) == 1
    _upload(client, second, [("second.html", HTML)])
    assert _terminal(client, second)["status"] == "completed"
    records = app.state.task_store._records
    assert records[first["task_id"]].uploads[0].path == records[second["task_id"]].uploads[0].path


def test_waiting_upload_retains_deleted_source_views(service: tuple[TestClient, FastAPI]) -> None:
    """等待其他文件期间删除源视图，不影响任务已经取得的字节引用。"""
    client, app = service
    inputs = [("a.html", HTML), ("b.html", SECOND)]
    task = _submit(client, inputs, hashes=False)
    assert client.put(task["uploads"][0]["upload_url"], content=HTML).status_code == 200
    record = app.state.task_store._records[task["task_id"]]
    if hasattr(app.state, "file_store"):
        file_id = record.uploads[0].file_id
        assert app.state.file_store.get_file(file_id).expires_at is None
        assert client.delete(f"/v1/files/{file_id}").status_code == 200
    else:
        app.state.source_store.delete_file(record.backend._owned_files[0])
    assert record.uploads[0].path.is_file()
    assert client.put(task["uploads"][1]["upload_url"], content=SECOND).status_code == 200
    result = _terminal(client, task)
    assert result["status"] == "completed", result
    assert b"Preserved content" in _archive(client, result)["0001/markdown.md"]


def test_failed_upload_can_retry_without_reuploading_ready_file(service: tuple[TestClient, FastAPI]) -> None:
    """一个文件校验失败只影响该项，其他成功文件保持 ready 且不提前创建 Job。"""
    client, app = service
    inputs = [("a.html", HTML), ("b.html", SECOND)]
    task = _submit(client, inputs)
    assert client.put(task["uploads"][0]["upload_url"], content=HTML).status_code == 200
    failed = client.put(task["uploads"][1]["upload_url"], content=SECOND.replace(b"Second", b"Broken"))
    assert failed.status_code == 400 and failed.json()["error"]["code"] == "file_hash_mismatch"
    mismatch = client.put(task["uploads"][1]["upload_url"], content=SECOND + b"x")
    assert mismatch.status_code == 400 and mismatch.json()["error"]["code"] == "upload_size_mismatch"
    pending = client.get(task["status_url"]).json()
    assert pending["status"] == "waiting_upload" and _job_count(app) == 0
    assert [file["upload_status"] for file in pending["files"]] == ["ready", "pending"]
    assert list(app.state.task_store._records[task["task_id"]].directory.iterdir()) == []
    _upload(client, pending, inputs)
    assert _terminal(client, task)["status"] == "completed" and _job_count(app) == 1


def test_concurrent_last_uploads_and_duplicate_put_submit_once(service: tuple[TestClient, FastAPI]) -> None:
    """并发上传与重复 PUT 共用提交门闩，批量文件只产生一个 Job。"""
    client, app = service
    task = _submit(client, [("a.html", HTML), ("b.html", SECOND)])

    async def upload_all() -> list[httpx.Response]:
        """在服务所属事件循环同时发起两个文件及同一文件的重复请求。"""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as asynchronous:
            return await asyncio.gather(
                asynchronous.put(task["uploads"][0]["upload_url"], content=HTML),
                asynchronous.put(task["uploads"][1]["upload_url"], content=SECOND),
                asynchronous.put(task["uploads"][1]["upload_url"], content=SECOND),
            )

    responses = client.portal.call(upload_all)
    assert all(response.status_code == 200 for response in responses)
    assert _terminal(client, task)["status"] == "completed"
    assert _job_count(app) == 1
    if hasattr(app.state, "worker_pool"):
        assert sum(worker.active_jobs for worker in app.state.worker_pool.workers) == 0


def test_cancel_and_expire_waiting_upload_release_inputs(service: tuple[TestClient, FastAPI]) -> None:
    """取消和过期不需要 worker，准备阶段保存的文件视图及引用都被释放。"""
    client, app = service
    for expired in (False, True):
        task = _submit(client, [("a.html", HTML), ("b.html", SECOND)])
        assert client.put(task["uploads"][0]["upload_url"], content=HTML).status_code == 200
        record = app.state.task_store._records[task["task_id"]]
        if expired:
            record.expires_at = int(time.time()) - 1
            response = client.get(task["status_url"])
            assert response.json()["error"]["code"] == "upload_expired"
        else:
            response = client.delete(task["status_url"])
        assert response.status_code == 200
        assert response.json()["status"] == ("failed" if expired else "canceled")
        assert response.json()["uploads"] == [] and _job_count(app) == 0
        assert not record.directory.exists()
        assert record.backend._owned_files == []
        assert client.put(task["uploads"][1]["upload_url"], content=SECOND).status_code == 409


def test_cancel_running_task_releases_execution_slot(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """运行任务取消经过同一执行队列，HTTP 返回后不存在重复预占或悬挂解析。"""
    client, app = service
    started = threading.Event()

    async def blocked_parse(*args: Any, **kwargs: Any) -> None:
        """保持真实后台解析句柄可取消，避免依赖 GPU 或定时竞态。"""
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(api_server, "parse_async", blocked_parse)
    task = _submit(client, [("a.html", HTML)])
    _upload(client, task, [("a.html", HTML)])
    assert started.wait(5)
    response = client.delete(task["status_url"])
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "canceled"
    assert client.get(task["status_url"]).json()["result_url"] is None
    deadline = time.monotonic() + 5
    operation = app.state.task_store._records[task["task_id"]].operation
    while operation is not None and not operation.done():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    if hasattr(app.state, "worker_pool"):
        assert sum(worker.active_jobs for worker in app.state.worker_pool.workers) == 0
    else:
        assert not app.state.job_store._tasks


def test_status_never_reads_or_rebuilds_zip(service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch) -> None:
    """批量 ZIP 在后台只构建一次，状态与流式下载均不调用整包读取方法。"""
    client, _ = service
    combine = task_api._combine_archives
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> str:
        """统计实际后台打包次数，确认多次查询及下载不重新构建。"""
        nonlocal calls
        calls += 1
        return combine(*args, **kwargs)

    monkeypatch.setattr(task_api, "_combine_archives", counted)
    inputs = [("a.html", HTML), ("b.html", SECOND)]
    task = _submit(client, inputs)
    _upload(client, task, inputs)
    result = _terminal(client, task)
    assert result["status"] == "completed" and calls == 1

    def forbid_read(*args: Any, **kwargs: Any) -> bytes:
        """完成后的状态与下载必须使用元数据或文件流，禁止 read_blob 整包读取。"""
        raise AssertionError("Result body was read into memory")

    monkeypatch.setattr(api_server.FileStore, "read_blob", forbid_read)
    monkeypatch.setattr(api_server.FileStore, "read_file_data", forbid_read)
    for _ in range(3):
        assert client.get(task["status_url"]).json()["result_url"] == result["result_url"]
        assert "manifest.json" in _archive(client, result)
    assert calls == 1


def test_partial_and_packaging_failure_have_honest_results(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """部分解析成功保留错误，打包失败不能发布 completed 或虚假的下载链接。"""
    client, _ = service
    inputs = [("same.html", HTML), ("broken.pdf", b"not a pdf")]
    task = _submit(client, inputs)
    _upload(client, task, inputs)
    result = _terminal(client, task)
    assert result["status"] == "partial" and result["files"][1]["error"]
    members = _archive(client, result)
    assert json.loads(members["manifest.json"])["status"] == "partial"
    assert not any(name.startswith("0002/") for name in members)

    def fail_packaging(*args: Any, **kwargs: Any) -> str:
        """模拟 ZIP 落盘失败，执行状态必须反映产物不可用。"""
        raise OSError("simulated full disk")

    monkeypatch.setattr(task_api, "_combine_archives", fail_packaging)
    good = [("a.html", HTML), ("b.html", SECOND)]
    task = _submit(client, good)
    _upload(client, task, good)
    failed = _terminal(client, task)
    assert failed["status"] == "failed" and failed["result_url"] is None
    assert failed["error"]["code"] == "result_packaging_failed"


def test_input_validation_and_openapi(service: tuple[TestClient, FastAPI]) -> None:
    """新协议只接受表单和单文件直传，OpenAPI 不再宣传旧入口及大小限制。"""
    client, app = service
    valid = {"client_file_id": "f1", "filename": "a.html", "bytes": len(HTML)}
    for manifest in (
        [],
        [valid] * 2,
        [valid] * 101,
        [{**valid, "bytes": 0}],
        [{**valid, "sha256sum": "invalid"}],
        [{**valid, "page_range": "invalid"}],
    ):
        response = client.post("/v1/tasks", data={"manifest": json.dumps(manifest)})
        assert response.status_code == 400, response.text
    for response in (
        client.post("/v1/tasks?tier=flash", data={"manifest": json.dumps([valid])}),
        client.post("/v1/tasks", files=[("files", ("a.html", HTML)), ("files", ("b.html", SECOND))]),
        client.post(
            "/v1/tasks",
            data={"manifest": json.dumps([valid, {**valid, "client_file_id": "f2"}])},
            files={"files": ("a.html", HTML)},
        ),
        client.post("/v1/tasks", files={"files": ("a.html", b"")}),
        client.post("/v1/tasks", data={"manifest": json.dumps([valid]), "wait_timeout": "300"}),
    ):
        assert response.status_code == 400, response.text
    assert client.post("/v1/tasks", json={"files": []}).status_code == 415
    assert client.post("/v1/tasks", content=HTML, headers={"content-type": "application/octet-stream"}).status_code == 415
    assert _job_count(app) == 0
    spec = client.get("/openapi.json").json()
    assert "/v1/file_parse" not in spec["paths"]
    assert "/v1/tasks/{task_id}/result" not in spec["paths"]
    creation = spec["paths"]["/v1/tasks"]["post"]
    assert not creation.get("parameters")
    assert set(creation["requestBody"]["content"]) == {"application/x-www-form-urlencoded", "multipart/form-data"}
    assert "209715200" not in json.dumps(creation)
    limits = client.get("/v1/usage").json()["limits"]
    assert limits["max_file_size_bytes"] is None and limits["max_pages_per_file"] is None
    assert limits["max_files_per_job"] == 100


def test_router_private_hash_cache_and_task_scope(tmp_path: Path) -> None:
    """Router 的缓存、任务与文件地址均隔离调用方，不能仅凭已知 hash 使用他人输入。"""
    workers = {"worker-a": api_server.create_app(upload_dir=str(tmp_path / "worker"), tier="flash")}
    app = create_router(
        RouterSettings(upstream_urls=("http://worker-a",), local_gpus="none", worker_refresh_interval_seconds=0),
        transport=_WorkerTransport(workers),
    )
    with TestClient(app) as client:
        owner, stranger = {"authorization": "Bearer owner"}, {"authorization": "Bearer stranger"}
        task = _submit(client, [("a.html", HTML)], headers=owner)
        _upload(client, task, [("a.html", HTML)], headers=owner)
        result = _terminal(client, task, headers=owner)
        assert client.get(task["status_url"], headers=stranger).status_code == 404
        assert client.get(result["result_url"], headers=stranger).status_code == 404
        other = _submit(client, [("a.html", HTML)], headers=stranger)
        assert len(other["uploads"]) == 1
        assert client.delete(other["status_url"], headers=stranger).status_code == 200
        owned = _submit(client, [("a.html", HTML)], headers=owner)
        assert not owned["uploads"]
        assert _terminal(client, owned, headers=owner)["status"] == "completed"


@pytest.mark.parametrize("root", ["", "/mineru"])
def test_task_links_respect_proxy_root(tmp_path: Path, root: str) -> None:
    """状态、上传与结果链接全部携带部署前缀，客户端不用重建任何地址。"""
    app = api_server.create_app(upload_dir=str(tmp_path), tier="flash", api_key="secret")
    with TestClient(app, root_path=root, headers={"Authorization": "Bearer secret"}) as client:
        manifest = [{"client_file_id": "f1", "filename": "a.html", "bytes": len(HTML)}]
        response = client.post(root + "/v1/tasks", data={"manifest": json.dumps(manifest), "tier": "flash"})
        assert response.status_code == 202
        task = response.json()
        assert task["status_url"].startswith("http://testserver" + root + "/v1/tasks/")
        _upload(client, task, [("a.html", HTML)])
        result = _terminal(client, task)
        assert result["result_url"].startswith("http://testserver" + root + "/v1/files/")
        assert "markdown.md" in _archive(client, result)
