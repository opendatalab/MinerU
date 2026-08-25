from __future__ import annotations

import asyncio
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from mineru.kit.router import RouterSettings, create_app
from mineru.kit.router.workers import ManagedLocalWorker, parse_local_gpus, worker_base_url, worker_connect_host
from mineru.parser.api_server import create_app as create_api_server_app


@dataclass
class _FakeV1Upstream:
    """提供 Router 回归测试所需的最小内存 V1 API。"""

    name: str
    tiers: tuple[str, ...]
    upload_counter: int = 0
    file_counter: int = 0
    job_counter: int = 0
    uploads: dict[str, dict[str, Any]] = field(default_factory=dict)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    fail_jobs: bool = False
    source_download_attempts: int = 0
    fail_file_delete_once: set[str] = field(default_factory=set)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        """按请求 path/method 返回真实 V1 shape 或测试错误。"""
        path = request.url.path
        method = request.method
        if path == "/v1/health":
            return self._response(
                request,
                200,
                {
                    "status": "ok",
                    "version": "test",
                    "features": {
                        "webhook": False,
                        "output_formats": ["markdown", "middle_json"],
                        "sources": ["file_id", "url", "inline"],
                    },
                },
            )
        if path == "/v1/tiers":
            return self._response(
                request,
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": tier, "description": f"{self.name}-{tier}", "current_model": f"{self.name}-model"}
                        for tier in self.tiers
                    ],
                },
            )
        if path == "/v1/models":
            return self._response(
                request,
                200,
                {"object": "list", "data": [{"id": f"{self.name}-model", "object": "model", "created": 1}]},
            )
        if path == "/v1/uploads" and method == "POST":
            self.upload_counter += 1
            upload_id = f"upload_{self.name}_{self.upload_counter}"
            body = await self._json_body(request)
            self.uploads[upload_id] = {**body, "content": b"", "status": "pending"}
            return self._response(request, 200, self._upload_payload(upload_id))
        if path.startswith("/v1/uploads/"):
            return await self._handle_upload(request)
        if path.startswith("/v1/files/"):
            return self._handle_file(request)
        if path == "/v1/parse/jobs" and method == "POST":
            return await self._create_job(request)
        if path.startswith("/v1/parse/jobs/"):
            return self._handle_job(request)
        return self._response(request, 404, self._error("not_found", path))

    async def _handle_upload(self, request: httpx.Request) -> httpx.Response:
        """处理 upload 查询、内容写入、完成和取消。"""
        parts = request.url.path.split("/")
        upload_id = parts[3]
        upload = self.uploads.get(upload_id)
        if upload is None:
            return self._response(request, 404, self._error("upload_not_found", upload_id))
        action = parts[4] if len(parts) > 4 else ""
        if request.method == "PUT" and action == "content":
            upload["content"] = await request.aread()
            return httpx.Response(200, request=request)
        if request.method == "POST" and action == "complete":
            upload["status"] = "completed"
            self.file_counter += 1
            file_id = f"file-{self.name}-{self.file_counter}"
            self.files[file_id] = {
                "id": file_id,
                "object": "file",
                "bytes": len(upload["content"]),
                "created_at": self.file_counter,
                "expires_at": None,
                "filename": upload["filename"],
                "purpose": upload.get("purpose", "parse"),
                "sha256sum": upload.get("sha256sum"),
                "content": upload["content"],
            }
            return self._response(request, 200, self._upload_payload(upload_id, file_id=file_id))
        if request.method == "POST" and action == "cancel":
            upload["status"] = "canceled"
            return self._response(request, 200, self._upload_payload(upload_id))
        return self._response(request, 200, self._upload_payload(upload_id))

    def _handle_file(self, request: httpx.Request) -> httpx.Response:
        """处理 file 元数据、内容读取和删除。"""
        parts = request.url.path.split("/")
        file_id = parts[3]
        file_record = self.files.get(file_id)
        if file_record is None:
            return self._response(request, 404, self._error("file_not_found", file_id))
        if len(parts) > 4 and parts[4] == "content":
            if file_record["purpose"] != "parse_output":
                self.source_download_attempts += 1
                return self._response(
                    request,
                    403,
                    self._error("feature_requires_api_key", "Source files cannot be downloaded"),
                )
            return httpx.Response(
                200,
                content=file_record["content"],
                headers={"content-type": "application/octet-stream"},
                request=request,
            )
        if request.method == "DELETE":
            if file_id in self.fail_file_delete_once:
                self.fail_file_delete_once.remove(file_id)
                return self._response(request, 500, self._error("delete_failed", file_id))
            self.files.pop(file_id)
            return self._response(request, 200, {"id": file_id, "object": "file", "deleted": True})
        return self._response(request, 200, {key: value for key, value in file_record.items() if key != "content"})

    async def _create_job(self, request: httpx.Request) -> httpx.Response:
        """创建 queued job，并校验请求中的 file_id 已复制到当前 upstream。"""
        if self.fail_jobs:
            raise httpx.ConnectError("simulated disconnect", request=request)
        body = await self._json_body(request)
        for entry in body.get("files") or []:
            source = entry.get("source") or {}
            if source.get("type") == "file_id" and source.get("file_id") not in self.files:
                return self._response(request, 400, self._error("file_not_found", str(source.get("file_id"))))
        self.job_counter += 1
        job_id = f"job_{self.name}_{self.job_counter}"
        self.jobs[job_id] = {"body": body, "status": "queued"}
        return self._response(request, 202, self._job_payload(job_id))

    def _handle_job(self, request: httpx.Request) -> httpx.Response:
        """查询时完成 job 并生成输出文件，DELETE 时取消 job。"""
        job_id = request.url.path.rsplit("/", 1)[-1]
        job = self.jobs.get(job_id)
        if job is None:
            return self._response(request, 404, self._error("job_not_found", job_id))
        if request.method == "DELETE":
            job["status"] = "canceled"
            return self._response(request, 200, {"job_id": job_id, "status": "canceled", "canceled_at": "2026-01-01T00:00:00Z"})
        if job["status"] == "queued":
            job["status"] = "completed"
            self.file_counter += 1
            output_id = f"file-{self.name}-{self.file_counter}"
            self.files[output_id] = {
                "id": output_id,
                "object": "file",
                "bytes": 6,
                "created_at": self.file_counter,
                "expires_at": None,
                "filename": "result.md",
                "purpose": "parse_output",
                "sha256sum": None,
                "content": b"result",
            }
            job["output_id"] = output_id
        return self._response(request, 200, self._job_payload(job_id))

    def _job_payload(self, job_id: str) -> dict[str, Any]:
        """构造包含输入与可选输出 file ID 的 V1 job response。"""
        job = self.jobs[job_id]
        files: list[dict[str, Any]] = []
        for entry in job["body"].get("files") or []:
            source = entry.get("source") or {}
            file_result: dict[str, Any] = {
                "file_id": source.get("file_id"),
                "name": "demo.pdf",
                "page_range": entry.get("page_range") or "",
                "status": "completed" if job["status"] == "completed" else "queued",
            }
            if job.get("output_id"):
                file_result["output_files"] = {
                    "markdown": {"file_id": job["output_id"], "bytes": 6},
                }
            files.append(file_result)
        return {
            "job_id": job_id,
            "status": job["status"],
            "created_at": "2026-01-01T00:00:00Z",
            "started_at": None,
            "finished_at": None,
            "tier": job["body"].get("tier", "standard"),
            "output_formats": job["body"].get("output_formats", ["markdown"]),
            "access_level": "anonymous",
            "progress": {"completed": 1 if job["status"] == "completed" else 0, "failed": 0, "total": len(files)},
            "files": files,
            "links": {"self": f"/v1/parse/jobs/{job_id}", "cancel": f"/v1/parse/jobs/{job_id}"},
        }

    def _upload_payload(self, upload_id: str, *, file_id: str | None = None) -> dict[str, Any]:
        """构造 pending/completed upload response。"""
        upload = self.uploads[upload_id]
        payload: dict[str, Any] = {
            "id": upload_id,
            "object": "upload",
            "bytes": upload["bytes"],
            "created_at": 1,
            "expires_at": 3601,
            "filename": upload["filename"],
            "purpose": upload.get("purpose", "parse"),
            "mime_type": upload["mime_type"],
            "sha256sum": upload.get("sha256sum"),
            "status": upload["status"],
            "upload_url": f"http://{self.name}/v1/uploads/{upload_id}/content",
            "upload_method": "PUT",
            "upload_headers": {},
        }
        if file_id is not None:
            payload["file"] = {key: value for key, value in self.files[file_id].items() if key != "content"}
        return payload

    @staticmethod
    async def _json_body(request: httpx.Request) -> dict[str, Any]:
        """读取 MockTransport 请求中的 JSON object body。"""
        body = await request.aread()
        return json.loads(body.decode("utf-8")) if body else {}

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        """构造最小 V1 error body。"""
        return {"error": {"type": "invalid_request_error", "code": code, "message": message}}

    @staticmethod
    def _response(request: httpx.Request, status_code: int, payload: dict[str, Any]) -> httpx.Response:
        """创建绑定原请求的 JSON response。"""
        return httpx.Response(status_code, json=payload, request=request)


def _make_router(
    *upstreams: _FakeV1Upstream,
) -> tuple[Any, dict[str, _FakeV1Upstream]]:
    """创建使用 host 分发 MockTransport 的 Router 应用。"""
    by_host = {upstream.name: upstream for upstream in upstreams}

    async def _handler(request: httpx.Request) -> httpx.Response:
        """把请求交给 URL host 对应的 fake upstream。"""
        return await by_host[request.url.host].handle(request)

    settings = RouterSettings(
        upstream_urls=tuple(f"http://{upstream.name}" for upstream in upstreams),
        local_gpus="none",
        worker_refresh_interval_seconds=0,
    )
    return create_app(settings, transport=httpx.MockTransport(_handler)), by_host


def _upload_file(client: TestClient, *, token: str, content: bytes) -> str:
    """通过 Router 完成一次 V1 upload 并返回公共 file ID。"""
    headers = {"authorization": f"Bearer {token}"}
    sha256sum = hashlib.sha256(content).hexdigest()
    create = client.post(
        "/v1/uploads",
        headers=headers,
        json={
            "filename": f"{token}.pdf",
            "bytes": len(content),
            "mime_type": "application/pdf",
            "purpose": "parse",
            "sha256sum": sha256sum,
        },
    )
    assert create.status_code == 200, create.text
    upload_id = create.json()["id"]
    assert create.json()["status"] == "pending"
    assert create.json()["sha256sum"] == sha256sum
    assert create.json()["upload_url"] == f"/v1/uploads/{upload_id}/content"
    assert client.put(f"/v1/uploads/{upload_id}/content", headers=headers, content=content).status_code == 200
    complete = client.post(f"/v1/uploads/{upload_id}/complete", headers=headers, json={})
    assert complete.status_code == 200, complete.text
    return complete.json()["file"]["id"]


def _upload_files_on_distinct_workers(
    client: TestClient,
    router_app: Any,
    *,
    token: str,
) -> list[str]:
    """在同一 caller scope 下轮换 upload worker，获得跨 worker 文件。"""
    pool = router_app.state.worker_pool
    original_select = pool.select
    healthy_workers = pool.healthy_workers()
    upload_index = 0

    def _alternating_select(**kwargs: Any) -> Any:
        """仅对带 affinity 的 Upload 选择轮换 worker，其他选择保持生产逻辑。"""
        nonlocal upload_index
        if kwargs.get("tier") is None and kwargs.get("affinity_key"):
            worker = healthy_workers[upload_index % len(healthy_workers)]
            upload_index += 1
            return worker
        return original_select(**kwargs)

    pool.select = _alternating_select
    files_by_worker: dict[str, str] = {}
    try:
        for index in range(2):
            public_file_id = _upload_file(client, token=token, content=f"pdf-{index}".encode())
            route = router_app.state.registry.get("file", public_file_id)
            files_by_worker.setdefault(route.worker_id, public_file_id)
    finally:
        pool.select = original_select
    assert len(files_by_worker) == 2
    return list(files_by_worker.values())


def test_v1_router_aggregates_capabilities_and_routes_cross_worker_job() -> None:
    """验证完整 V1 主链、跨 worker 输入复制和公共资源 ID 重写。"""
    first = _FakeV1Upstream("worker-a", ("basic", "standard"))
    second = _FakeV1Upstream("worker-b", ("standard", "advanced"))
    router_app, _ = _make_router(first, second)
    staged_path: Path | None = None
    headers = {"authorization": "Bearer main-cross-worker"}

    with TestClient(router_app) as client:
        health = client.get("/v1/health")
        tiers = client.get("/v1/tiers")
        models = client.get("/v1/models")
        assert health.status_code == 200
        assert client.get("/health").status_code == 404
        assert client.get("/tasks").status_code == 404
        assert client.post("/file_parse").status_code == 404
        assert [item["id"] for item in tiers.json()["data"]] == ["basic", "standard", "advanced"]
        assert {item["id"] for item in models.json()["data"]} == {"worker-a-model", "worker-b-model"}

        public_file_ids = _upload_files_on_distinct_workers(
            client,
            router_app,
            token="main-cross-worker",
        )
        stored = router_app.state.source_store.find_file(public_file_ids[0])
        assert stored is not None
        staged_path = stored.path
        assert staged_path.is_file()

        created = client.post(
            "/v1/parse/jobs",
            headers=headers,
            json={
                "tier": "standard",
                "files": [{"source": {"type": "file_id", "file_id": file_id}} for file_id in public_file_ids],
                "output_formats": ["markdown"],
            },
        )
        assert created.status_code == 202, created.text
        assert [item["file_id"] for item in created.json()["files"]] == public_file_ids
        job_id = created.json()["job_id"]
        job_route = router_app.state.registry.get("job", job_id)
        copied_inputs = list(job_route.metadata.get("copied_inputs") or [])
        assert len(copied_inputs) == 1
        copied_upstream_id = copied_inputs[0].upstream_file_id
        target = first if job_route.worker_id == "remote-1" else second
        assert copied_upstream_id in target.files

        completed = client.get(f"/v1/parse/jobs/{job_id}", headers=headers)
        assert completed.status_code == 200
        assert completed.json()["status"] == "completed"
        assert [item["file_id"] for item in completed.json()["files"]] == public_file_ids
        output_id = completed.json()["files"][0]["output_files"]["markdown"]["file_id"]
        assert output_id.startswith("file-")
        assert client.get(f"/v1/files/{output_id}/content", headers=headers).content == b"result"
        assert job_route.metadata.get("copied_inputs") is None
        assert copied_upstream_id not in target.files

        job_list = client.get("/v1/parse/jobs", headers=headers).json()
        file_list = client.get("/v1/files", headers=headers).json()
        usage = client.get("/v1/usage", headers=headers).json()
        assert [item["job_id"] for item in job_list["data"]] == [job_id]
        assert {item["id"] for item in file_list["data"]}.issuperset(set(public_file_ids))
        assert output_id in {item["id"] for item in file_list["data"]}
        assert usage["current"]["jobs_created"] == 1
        assert usage["current"]["files_processed"] == 2

    selected_job = first.jobs or second.jobs
    submitted_files = [entry["source"]["file_id"] for entry in next(iter(selected_job.values()))["body"]["files"]]
    selected_files = first.files if first.jobs else second.files
    assert copied_upstream_id in submitted_files
    assert copied_upstream_id not in selected_files
    assert first.source_download_attempts == 0
    assert second.source_download_attempts == 0
    assert staged_path is not None
    assert not staged_path.exists()


def test_real_v1_api_rejects_public_download_of_parse_source(tmp_path: Path) -> None:
    """验证真实 api-server 保持普通 parse 输入不可公开下载的安全边界。"""
    data = b"%PDF-1.7"
    api_app = create_api_server_app(upload_dir=tmp_path, tier="flash")

    with TestClient(api_app) as client:
        created = client.post(
            "/v1/uploads",
            json={
                "filename": "input.pdf",
                "bytes": len(data),
                "mime_type": "application/pdf",
                "purpose": "parse",
            },
        )
        upload_id = created.json()["id"]
        uploaded = client.put(
            f"/v1/uploads/{upload_id}/content",
            content=data,
            headers={"content-type": "application/octet-stream"},
        )
        completed = client.post(f"/v1/uploads/{upload_id}/complete", json={})
        file_id = completed.json()["file"]["id"]
        downloaded = client.get(f"/v1/files/{file_id}/content")

    assert uploaded.status_code == 200
    assert completed.json()["file"]["purpose"] == "parse"
    assert downloaded.status_code == 403
    assert downloaded.json()["error"]["message"] == "Source files cannot be downloaded"


def test_router_discards_staged_upload_with_wrong_sha256() -> None:
    """验证 Router 在 SHA256 不匹配时删除暂存输入且不转发字节。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",))
    router_app, _ = _make_router(upstream)
    expected_sha256 = hashlib.sha256(b"ok").hexdigest()

    with TestClient(router_app) as client:
        created = client.post(
            "/v1/uploads",
            json={
                "filename": "input.pdf",
                "bytes": 2,
                "mime_type": "application/pdf",
                "purpose": "parse",
                "sha256sum": expected_sha256,
            },
        )
        upload_id = created.json()["id"]
        uploaded = client.put(
            f"/v1/uploads/{upload_id}/content",
            content=b"no",
            headers={"content-type": "application/octet-stream"},
        )

        assert uploaded.status_code == 400
        assert uploaded.json()["error"]["code"] == "upload_sha256_mismatch"
        assert router_app.state.source_store.find_upload(upload_id) is None
        assert next(iter(upstream.uploads.values()))["content"] == b""


def test_completed_upload_rejects_content_rewrite_without_touching_bound_source() -> None:
    """验证 completed Upload 的后续 PUT 返回 409 且绑定源文件保持不可变。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",))
    router_app, _ = _make_router(upstream)
    original = b"original"

    with TestClient(router_app) as client:
        created = client.post(
            "/v1/uploads",
            json={
                "filename": "input.pdf",
                "bytes": len(original),
                "mime_type": "application/pdf",
                "purpose": "parse",
            },
        )
        upload_id = created.json()["id"]
        client.put(f"/v1/uploads/{upload_id}/content", content=original)
        completed = client.post(f"/v1/uploads/{upload_id}/complete", json={})
        file_id = completed.json()["file"]["id"]
        stored = router_app.state.source_store.find_file(file_id)
        assert stored is not None

        rewritten = client.put(f"/v1/uploads/{upload_id}/content", content=b"modified")

        assert rewritten.status_code == 409
        assert rewritten.json()["error"]["code"] == "upload_already_completed"
        assert stored.path.read_bytes() == original


def test_terminal_job_read_survives_deleted_output_metadata() -> None:
    """验证 output 删除后 hydration 失败不会阻断 completed Job 读取。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",))
    router_app, _ = _make_router(upstream)
    headers = {"authorization": "Bearer output-delete"}

    with TestClient(router_app) as client:
        file_id = _upload_file(client, token="output-delete", content=b"pdf")
        created = client.post(
            "/v1/parse/jobs",
            headers=headers,
            json={"tier": "standard", "files": [{"source": {"type": "file_id", "file_id": file_id}}]},
        )
        job_id = created.json()["job_id"]
        completed = client.get(f"/v1/parse/jobs/{job_id}", headers=headers)
        output_id = completed.json()["files"][0]["output_files"]["markdown"]["file_id"]
        assert client.delete(f"/v1/files/{output_id}", headers=headers).status_code == 200

        reread = client.get(f"/v1/parse/jobs/{job_id}", headers=headers)

        assert reread.status_code == 200
        assert reread.json()["status"] == "completed"
        replacement_output_id = reread.json()["files"][0]["output_files"]["markdown"]["file_id"]
        replacement_route = router_app.state.registry.get("file", replacement_output_id)
        assert replacement_route.metadata["hydration_error"]["status_code"] == 404


def test_router_scopes_cached_resources_to_bearer_identity() -> None:
    """验证 files、jobs、usage 与单资源访问按 bearer caller scope 隔离。"""
    upstream = _FakeV1Upstream("worker-a", ("standard",))
    router_app, _ = _make_router(upstream)
    owner_headers = {"authorization": "Bearer tenant-a"}
    other_headers = {"authorization": "Bearer tenant-b"}

    with TestClient(router_app) as client:
        file_id = _upload_file(client, token="tenant-a", content=b"pdf")
        created = client.post(
            "/v1/parse/jobs",
            headers=owner_headers,
            json={"tier": "standard", "files": [{"source": {"type": "file_id", "file_id": file_id}}]},
        )
        job_id = created.json()["job_id"]

        assert [item["id"] for item in client.get("/v1/files", headers=owner_headers).json()["data"]] == [file_id]
        assert [item["job_id"] for item in client.get("/v1/parse/jobs", headers=owner_headers).json()["data"]] == [job_id]
        assert client.get("/v1/files", headers=other_headers).json()["data"] == []
        assert client.get("/v1/parse/jobs", headers=other_headers).json()["data"] == []
        assert client.get("/v1/files").json()["data"] == []
        assert client.get("/v1/parse/jobs").json()["data"] == []
        assert client.get("/v1/usage", headers=other_headers).json()["current"]["jobs_created"] == 0
        assert client.get(f"/v1/files/{file_id}", headers=other_headers).status_code == 404
        assert client.get(f"/v1/parse/jobs/{job_id}", headers=other_headers).status_code == 404


def test_background_reconciler_finalizes_unpolled_cross_worker_job() -> None:
    """验证无客户端轮询时后台 reconciliation 仍归还负载并回收副本。"""
    first = _FakeV1Upstream("worker-a", ("standard",))
    second = _FakeV1Upstream("worker-b", ("standard",))
    router_app, _ = _make_router(first, second)
    headers = {"authorization": "Bearer reconcile"}

    with TestClient(router_app) as client:
        file_ids = _upload_files_on_distinct_workers(client, router_app, token="reconcile")
        created = client.post(
            "/v1/parse/jobs",
            headers=headers,
            json={
                "tier": "standard",
                "files": [{"source": {"type": "file_id", "file_id": file_id}} for file_id in file_ids],
            },
        )
        job_id = created.json()["job_id"]
        route = router_app.state.registry.get("job", job_id)
        worker = first if route.worker_id == "remote-1" else second
        copied = list(route.metadata["copied_inputs"])

        client.portal.call(router_app.state.reconcile_jobs_once)

        assert route.metadata["payload"]["status"] == "completed"
        assert route.metadata.get("active_counted") is None
        assert route.metadata.get("copied_inputs") is None
        assert route.metadata.get("upstream_headers") is None
        assert router_app.state.worker_pool.get(route.worker_id).active_jobs == 0
        assert copied[0].upstream_file_id not in worker.files
        listed_ids = {item["id"] for item in client.get("/v1/files", headers=headers).json()["data"]}
        output_id = route.metadata["payload"]["files"][0]["output_files"]["markdown"]["file_id"]
        assert output_id in listed_ids


def test_router_reclaims_cross_worker_copy_when_job_is_canceled() -> None:
    """验证取消 Job 时立即删除并解除 cross-worker 输入副本。"""
    first = _FakeV1Upstream("worker-a", ("standard",))
    second = _FakeV1Upstream("worker-b", ("standard",))
    router_app, _ = _make_router(first, second)
    headers = {"authorization": "Bearer cancel-cross-worker"}

    with TestClient(router_app) as client:
        public_file_ids = _upload_files_on_distinct_workers(
            client,
            router_app,
            token="cancel-cross-worker",
        )
        created = client.post(
            "/v1/parse/jobs",
            headers=headers,
            json={
                "tier": "standard",
                "files": [{"source": {"type": "file_id", "file_id": file_id}} for file_id in public_file_ids],
            },
        )
        job_id = created.json()["job_id"]
        job_route = router_app.state.registry.get("job", job_id)
        copied = list(job_route.metadata["copied_inputs"])
        assert len(copied) == 1
        target = first if copied[0].worker_id == "remote-1" else second
        assert copied[0].upstream_file_id in target.files

        canceled = client.delete(f"/v1/parse/jobs/{job_id}", headers=headers)

        assert canceled.status_code == 200
        assert copied[0].upstream_file_id not in target.files
        assert job_route.metadata.get("copied_inputs") is None
        assert (
            router_app.state.registry.find_upstream(
                "file",
                copied[0].owner_scope,
                copied[0].worker_id,
                copied[0].upstream_file_id,
            )
            is None
        )


def test_public_source_delete_retries_terminal_copy_cleanup() -> None:
    """验证终态清理失败后，删除公共源文件会重试目标副本回收。"""
    first = _FakeV1Upstream("worker-a", ("standard",))
    second = _FakeV1Upstream("worker-b", ("standard",))
    router_app, _ = _make_router(first, second)
    headers = {"authorization": "Bearer delete-retry"}

    with TestClient(router_app) as client:
        public_file_ids = _upload_files_on_distinct_workers(
            client,
            router_app,
            token="delete-retry",
        )
        created = client.post(
            "/v1/parse/jobs",
            headers=headers,
            json={
                "tier": "standard",
                "files": [{"source": {"type": "file_id", "file_id": file_id}} for file_id in public_file_ids],
            },
        )
        job_id = created.json()["job_id"]
        job_route = router_app.state.registry.get("job", job_id)
        copied = list(job_route.metadata["copied_inputs"])
        target = first if copied[0].worker_id == "remote-1" else second
        target.fail_file_delete_once.add(copied[0].upstream_file_id)

        completed = client.get(f"/v1/parse/jobs/{job_id}", headers=headers)
        assert completed.status_code == 200
        assert copied[0].upstream_file_id in target.files
        assert job_route.metadata.get("copied_inputs") == copied

        deleted = client.delete(f"/v1/files/{copied[0].source_public_id}", headers=headers)

        assert deleted.status_code == 200
        assert copied[0].upstream_file_id not in target.files
        assert job_route.metadata.get("copied_inputs") is None


def test_v1_router_reports_capability_transport_and_resource_errors() -> None:
    """验证无匹配 tier、upstream 断连和未知资源使用稳定 V1 错误。"""
    upstream = _FakeV1Upstream("worker-a", ("flash",))
    router_app, _ = _make_router(upstream)

    with TestClient(router_app) as client:
        unavailable = client.post(
            "/v1/parse/jobs",
            json={"tier": "standard", "files": [{"source": {"type": "inline", "name": "a.pdf", "data": "AAAA"}}]},
        )
        missing = client.get("/v1/files/file-missing")
        assert unavailable.status_code == 503
        assert unavailable.json()["error"]["code"] == "quality_tier_unavailable"
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "file_not_found"

    failing = _FakeV1Upstream("worker-b", ("standard",), fail_jobs=True)
    failing_app, _ = _make_router(failing)
    with TestClient(failing_app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={"tier": "standard", "files": [{"source": {"type": "inline", "name": "a.pdf", "data": "AAAA"}}]},
        )
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "upstream_unavailable"


def test_managed_router_worker_builds_new_api_server_command(tmp_path: Path) -> None:
    """验证本地 Router worker 仅启动正式 mineru-kit api-server 参数面。"""
    settings = RouterSettings(
        local_gpus="none",
        worker_tier="basic",
        worker_concurrency=3,
        preload_models=True,
    )
    worker = ManagedLocalWorker("local-0", "127.0.0.1", "0", settings)

    command = worker.command(18000, tmp_path)

    assert command[:5] == [command[0], "-m", "mineru.kit.main", "api-server", "--host"]
    assert command[0].endswith("python")
    assert ["--tier", "basic"] == command[command.index("--tier") : command.index("--tier") + 2]
    assert ["--concurrency", "3"] == command[command.index("--concurrency") : command.index("--concurrency") + 2]
    assert "--preload-models" in command
    assert "mineru.cli_old" not in " ".join(command)


@pytest.mark.parametrize(
    ("bind_host", "expected_connect_host", "expected_url"),
    [
        ("", "127.0.0.1", "http://127.0.0.1:18000"),
        ("0.0.0.0", "127.0.0.1", "http://127.0.0.1:18000"),
        ("127.0.0.1", "127.0.0.1", "http://127.0.0.1:18000"),
        ("::", "::1", "http://[::1]:18000"),
        ("::1", "::1", "http://[::1]:18000"),
    ],
)
def test_managed_worker_formats_connectable_ipv4_and_ipv6_urls(
    bind_host: str,
    expected_connect_host: str,
    expected_url: str,
) -> None:
    """验证 wildcard、IPv4 与 IPv6 worker 地址生成合法可连接 URL。"""
    assert worker_connect_host(bind_host) == expected_connect_host
    assert worker_base_url(bind_host, 18000) == expected_url


def test_auto_local_gpus_expands_visible_accelerator_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 auto 在无显式设备 CSV 或值为 all 时枚举全部可见 GPU。"""
    import mineru.kit.router.workers as workers

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(workers, "get_device", lambda: "cuda")
    monkeypatch.setattr(workers, "accelerator_device_count", lambda _device: 4)
    assert parse_local_gpus("auto") == ["0", "1", "2", "3"]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    assert parse_local_gpus("auto") == ["2", "4"]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "all")
    monkeypatch.setattr(workers, "accelerator_device_count", lambda _device: 3)
    assert parse_local_gpus("auto") == ["0", "1", "2"]


def test_compose_router_healthcheck_uses_v1_path() -> None:
    """验证 Router Compose profile 只探测正式 `/v1/health`。"""
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker/compose.yaml").read_text(encoding="utf-8")
    router_section = compose_text.split("  mineru-router:", 1)[1].split("  mineru-gradio:", 1)[0]

    assert "http://localhost:8002/v1/health" in router_section
    assert "http://localhost:8002/health" not in router_section


def test_managed_router_worker_uses_control_shutdown() -> None:
    """验证托管 worker 优先通过 ManagedProcessControl 正常退出。"""

    class _FakeProcess:
        """模拟收到控制关闭后可正常 wait 的子进程。"""

        def __init__(self) -> None:
            """初始化进程存活和信号调用状态。"""
            self.exited = False
            self.terminated = False
            self.killed = False

        def poll(self) -> int | None:
            """返回当前模拟退出码。"""
            return 0 if self.exited else None

        def wait(self, timeout: float | None = None) -> int:
            """模拟控制通道关闭后进程正常退出。"""
            del timeout
            self.exited = True
            return 0

        def terminate(self) -> None:
            """记录不应触发的 terminate。"""
            self.terminated = True

        def kill(self) -> None:
            """记录不应触发的 kill。"""
            self.killed = True

    class _FakeControl:
        """模拟可用的父子进程控制通道。"""

        def __init__(self) -> None:
            """初始化关闭请求和资源关闭状态。"""
            self.requested = False
            self.closed = False

        def request_shutdown(self, timeout_sec: float) -> bool:
            """记录优雅关闭请求并返回成功。"""
            assert timeout_sec > 0
            self.requested = True
            return True

        def close(self) -> None:
            """记录控制通道已释放。"""
            self.closed = True

    worker = ManagedLocalWorker("local-0", "127.0.0.1", None, RouterSettings(local_gpus="none"))
    process = _FakeProcess()
    control = _FakeControl()
    worker.process = process  # type: ignore[assignment]
    worker.control = control  # type: ignore[assignment]

    asyncio.run(worker.stop())

    assert control.requested is True
    assert control.closed is True
    assert process.exited is True
    assert process.terminated is False
    assert process.killed is False


def test_router_console_script_points_to_new_cli() -> None:
    """校验过渡 mineru-router console script 不再指向 cli_old。"""
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["mineru-router"] == "mineru.kit.router.cli:main"
