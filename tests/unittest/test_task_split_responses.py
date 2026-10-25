"""验证独立响应、服务端公共文件 ID，以及取消后可直接使用的逐文件终态。"""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_task_api import HTML, SECOND
from test_task_api import service as _service_fixture
from mineru.parser.api_server import FileStore
from mineru.kit.router.resources import ResourceRegistry


service = _service_fixture


def _create(client: TestClient, contents: list[bytes]) -> dict[str, Any]:
    """用同名输入验证服务端按请求顺序分配 ID，不依赖文件名唯一性。"""
    response = client.post(
        "/v1/tasks",
        json={
            "files": [
                {"file_name": "same.html", "bytes": len(content), "sha256sum": hashlib.sha256(content).hexdigest()}
                for content in contents
            ]
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def _wait_files(client: TestClient, created: dict[str, Any], count: int) -> dict[str, Any]:
    """等待指定数量的成功文件，允许同一父任务仍有尚未上传的输入。"""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/v1/tasks/{created['task_id']}")
        assert response.status_code == 200, response.text
        task = response.json()
        if sum(item["status"] == "completed" for item in task["files"]) == count:
            return task
        assert all(item["status"] != "failed" for item in task["files"]), task
        time.sleep(0.01)
    raise AssertionError(f"Files did not complete: {task}")


@pytest.mark.parametrize("field", ["client_file_id", "file_id"])
@pytest.mark.parametrize("value", [None, "file-user"])
def test_creation_rejects_all_caller_file_ids(service: tuple[TestClient, FastAPI], field: str, value: str | None) -> None:
    """客户端不能通过新旧字段注入标识，空值也不能绕过请求模型。"""
    client, app = service
    response = client.post("/v1/tasks", json={"files": [{"file_name": "x.html", "bytes": 1, field: value}]})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert not app.state.task_store._records


def test_split_responses_keep_ids_and_cancellation_results(service: tuple[TestClient, FastAPI]) -> None:
    """完整 HTTP 链路验证预留 ID、独立回执、同名产物和取消后直接下载。"""
    client, _app = service
    contents = [HTML, SECOND, b"<p>not uploaded</p>"]
    created = _create(client, contents)
    assert set(created) == {"task_id", "status", "files"}
    assert client.delete(f"/v1/tasks/{created['task_id']}").status_code == 405
    assert client.get(f"/v1/tasks/{created['task_id']}").json()["status"] == "waiting-file"
    file_ids = [item["file_id"] for item in created["files"]]
    assert len(set(file_ids)) == 3 and all(value.startswith("file-") for value in file_ids)
    for item in created["files"]:
        assert set(item) == {"file_id", "file_name", "status", "upload"}
        for suffix in ("", "/content"):
            assert client.get(f"/v1/files/{item['file_id']}{suffix}").status_code == 404
        assert client.delete(f"/v1/files/{item['file_id']}").status_code == 405
    detail = client.get(f"/v1/tasks/{created['task_id']}").json()
    assert detail["progress"] == {"completed": 0, "failed": 0, "total": 3}
    assert {"started_at", "finished_at", "error"}.isdisjoint(detail)
    for item, content in zip(created["files"][:2], contents):
        response = client.put(item["upload"]["url"], content=content, headers=item["upload"]["headers"])
        assert response.status_code == 200, response.text
        assert set(response.json()) == {"task_id", "file_id", "status"}
        assert response.json()["file_id"] == item["file_id"]
    ready = _wait_files(client, created, 2)
    assert [item["file_id"] for item in ready["files"]] == file_ids
    response = client.post(f"/v1/tasks/{created['task_id']}" + "/cancel")
    assert response.status_code == 200, response.text
    canceled = response.json()
    assert set(canceled) == {"task_id", "status", "finished_at", "files"}
    assert canceled["status"] == "canceled"
    assert [item["status"] for item in canceled["files"]] == ["completed", "completed", "canceled"]
    assert [item["file_id"] for item in canceled["files"]] == file_ids
    assert set(canceled["files"][2]) == {"file_id", "file_name", "status"}
    for item in canceled["files"][:2]:
        assert set(item) == {"file_id", "file_name", "status", "parse", "result_url"}
        assert item["result_url"].endswith(f"/v1/files/{item['file_id']}/content")
        metadata = client.get(f"/v1/files/{item['file_id']}").json()
        assert metadata["id"] == item["file_id"] and metadata["file_name"] == "same.zip"
        downloaded = client.get(item["result_url"])
        assert downloaded.status_code == 200
        with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
            assert archive.read("markdown.md").strip()
    assert client.post(f"/v1/tasks/{created['task_id']}" + "/cancel").status_code == 409
    assert client.delete(f"/v1/files/{file_ids[0]}").status_code == 405
    client.get(f"/v1/tasks/{created['task_id']}")
    assert client.get(f"/v1/files/{file_ids[0]}").status_code == 200
    assert client.get(canceled["files"][0]["result_url"]).status_code == 200
    assert client.get(f"/v1/files/{file_ids[2]}").status_code == 404


def test_cached_source_uses_new_public_result_id(service: tuple[TestClient, FastAPI]) -> None:
    """源字节免传不复用上次任务的公共文件或产物标识。"""
    client, _app = service
    first = _create(client, [HTML])
    client.put(first["files"][0]["upload"]["url"], content=HTML)
    _wait_files(client, first, 1)
    second = _create(client, [HTML])
    assert second["files"][0]["file_id"] != first["files"][0]["file_id"]
    assert "upload" not in second["files"][0]
    completed = _wait_files(client, second, 1)
    assert completed["files"][0]["result_url"].endswith(f"/{second['files'][0]['file_id']}/content")


def test_openapi_separates_operation_schemas(service: tuple[TestClient, FastAPI]) -> None:
    """OpenAPI 显式区分四类成功响应，查询的可选字段不再声明为必填。"""
    client, _app = service
    schema = client.get("/openapi.json").json()
    for path, method, status, model in (
        ("/v1/tasks", "post", "202", "CreateTaskResponse"),
        ("/v1/tasks/{task_id}", "get", "200", "TaskDetailResponse"),
        ("/v1/tasks/{task_id}/cancel", "post", "200", "CancelTaskResponse"),
        ("/v1/tasks/{task_id}/uploads/{file_id}", "put", "200", "UploadFileResponse"),
    ):
        reference = schema["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]["$ref"]
        assert reference.endswith("/" + model)
    assert "delete" not in schema["paths"]["/v1/tasks/{task_id}"]
    assert "delete" not in schema["paths"]["/v1/files/{file_id}"]
    models = schema["components"]["schemas"]
    assert "FileDeletionResponse" not in models
    assert set(models["TaskFileDetail"]["required"]) == {"file_id", "file_name", "status"}
    assert set(models["CreateTaskResponse"]["properties"]) == {"task_id", "status", "files"}
    assert "upload" not in models["CanceledTaskFile"]["properties"]


def test_reserved_output_ids_reject_collisions_and_late_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """故意制造随机碰撞，验证取消、覆盖和已删除结果都不能重新占用旧 ID。"""
    store = FileStore(tmp_path)
    values = iter(["file-first", "file-first", "file-second"])

    def next_id() -> str:
        """把首次碰撞固定为可复现路径，第三个候选 ID 才可分配。"""
        return next(values)

    monkeypatch.setattr(store, "_new_file_id", next_id)
    first, second = store.reserve_file_id(), store.reserve_file_id()
    assert (first, second) == ("file-first", "file-second")
    source = tmp_path / "output.zip"
    source.write_bytes(b"complete output")
    store.discard_file_id(first)
    with pytest.raises(ValueError, match="not reserved"):
        store.create_file_for_output_path("output.zip", source, job_id="job-one", file_id=first)
    assert store.create_file_for_output_path("output.zip", source, job_id="job-two", file_id=second) == second
    with pytest.raises(ValueError, match="already been published"):
        store.create_file_for_output_path("other.zip", source, job_id="job-other", file_id=second)
    store.delete_file(second)
    with pytest.raises(ValueError, match="not reserved"):
        store.create_file_for_output_path("output.zip", source, job_id="job-two", file_id=second)
    store.finalize_outputs("job-two", int(time.time()))
    assert not store._files and not store._pins and not store._reserved_output_ids
    store.forget_file_id(first)
    store.forget_file_id(second)
    assert not store._allocated_file_ids and not store._task_file_ids


def test_router_rejects_rebinding_and_deleted_result_resurrection() -> None:
    """不同 worker 不能覆盖同一公共 ID，已删除映射不能被迟到完成响应复活。"""
    registry = ResourceRegistry()
    public_id = registry.reserve_file_id()
    route = registry.register_file(owner_scope="owner", worker_id="a", upstream_id="file-worker-a", public_id=public_id)
    with pytest.raises(ValueError, match="occupied"):
        registry.register_file(owner_scope="owner", worker_id="b", upstream_id="file-worker-b", public_id=public_id)
    assert registry.find("file", public_id) is route
    registry.remove("file", public_id)
    with pytest.raises(ValueError, match="consumed or discarded"):
        registry.register_file(owner_scope="owner", worker_id="a", upstream_id="file-worker-a", public_id=public_id)
    registry.forget_file_id(public_id)
    assert not registry._allocated_file_ids and not registry._task_file_ids


@pytest.mark.parametrize("service", ["router"], indirect=True)
@pytest.mark.parametrize("changed_field", ["task_id", "file_id"])
def test_router_rejects_mismatched_upload_receipt(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch, changed_field: str
) -> None:
    """下游上传回执串到其他任务或文件时失败，迟到的真实产物也不能注册到公共 ID。"""
    client, app = service
    transport = app.state.test_transport
    original = transport.handle_async_request

    async def mismatched(request: httpx.Request) -> httpx.Response:
        """保留真实 worker 上传和执行，只破坏回执中的身份字段。"""
        response = await original(request)
        if request.method == "PUT" and response.status_code == 200:
            await response.aread()
            payload = response.json()
            payload[changed_field] = "another-resource"
            return httpx.Response(200, json=payload)
        return response

    monkeypatch.setattr(transport, "handle_async_request", mismatched)
    created = _create(client, [HTML])
    file_id = created["files"][0]["file_id"]
    assert client.put(created["files"][0]["upload"]["url"], content=HTML).status_code == 200
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        detail = client.get(f"/v1/tasks/{created['task_id']}").json()
        if detail["status"] == "failed":
            break
        time.sleep(0.01)
    assert detail["status"] == "failed", detail
    assert detail["files"][0]["error"]["code"] == "invalid_upstream_response"
    assert "result_url" not in detail["files"][0]
    assert client.get(f"/v1/files/{file_id}").status_code == 404


def test_task_expiry_releases_only_its_file_id_reservations(service: tuple[TestClient, FastAPI]) -> None:
    """取消任务回收后释放 ID 索引，其他仍等待上传的任务继续持有自己的预留。"""
    client, app = service
    created, other = _create(client, [HTML]), _create(client, [SECOND])
    file_id, other_id = created["files"][0]["file_id"], other["files"][0]["file_id"]
    assert client.post(f"/v1/tasks/{created['task_id']}" + "/cancel").status_code == 200

    async def expire() -> None:
        """在服务所属循环等待取消清理，再模拟保留期结束。"""
        store = app.state.task_store
        record = store._records[created["task_id"]]
        await record.operation
        record.retention_epoch = time.time() - 2
        store.retention_seconds = 1
        store.collect_expired()

    client.portal.call(expire)
    allocator = app.state.file_store if hasattr(app.state, "file_store") else app.state.registry
    assert file_id not in allocator._allocated_file_ids and file_id not in allocator._task_file_ids
    assert other_id in allocator._allocated_file_ids and other_id in allocator._task_file_ids
