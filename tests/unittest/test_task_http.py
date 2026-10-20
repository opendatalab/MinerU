"""真实 TCP、客户端脚本及隔离 SaaS/OSS 模拟服务的共同协议验收。"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pypdf import PdfWriter
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from test_http_api_example_script import _Server

from mineru.kit.router import RouterSettings
from mineru.kit.router import create_app as create_router
from mineru.parser import api_server
from mineru.utils.async_utils import run_sync

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "http_task_example.sh"
HTML = b"<h1>HTTP tasks</h1><p>Same source and output.</p>"
TERMINAL = {"completed", "partial", "failed", "canceled"}


class _UploadCounter:
    """统计真实 ASGI 收到的逐文件字节，区分文件上传与元数据预检。"""

    def __init__(self, app: ASGIApp, counts: dict[str, int]) -> None:
        """使用外部计数表，测试可以在断连前确认部分主体已被消费。"""
        self.app = app
        self.counts = counts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """只统计 PUT 主体，不改变服务器的接收时序或断连信号。"""

        async def counted_receive() -> Message:
            """记录真实网络块，再原样传递给服务端上传逻辑。"""
            message = await receive()
            if message["type"] == "http.request":
                path = scope["path"]
                self.counts[path] = self.counts.get(path, 0) + len(message.get("body", b""))
            return message

        if scope["type"] == "http" and scope["method"] == "PUT":
            await self.app(scope, counted_receive, send)
        else:
            await self.app(scope, receive, send)


@dataclass
class _LiveService:
    """保留真实网络地址与应用对象，便于核对上传量和唯一任务记录。"""

    url: str
    app: FastAPI
    counts: dict[str, int]


@pytest.fixture(params=["api", "router"])
def live_service(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[_LiveService]:
    """启动真实 Uvicorn 服务；Router 的两个 worker 同样使用真实 TCP。"""
    servers: list[_Server] = []
    try:
        if request.param == "api":
            app = api_server.create_app(upload_dir=str(tmp_path / "api"), tier="flash", api_key="test-key")
        else:
            for index in range(2):
                worker = _Server(
                    api_server.create_app(upload_dir=str(tmp_path / f"worker-{index}"), tier="flash", api_key="test-key")
                )
                worker.start()
                servers.append(worker)
            app = create_router(
                RouterSettings(
                    upstream_urls=tuple(server.url for server in servers),
                    local_gpus="none",
                    worker_refresh_interval_seconds=0,
                )
            )
        counts: dict[str, int] = {}
        app.add_middleware(_UploadCounter, counts=counts)
        server = _Server(app)
        server.start()
        servers.append(server)
        yield _LiveService(server.url, app, counts)
    finally:
        for server in reversed(servers):
            server.stop()


def _client(service: _LiveService) -> httpx.Client:
    """每个请求经过真实 HTTP 与已配置的 API Key，禁用外部代理环境。"""
    return httpx.Client(base_url=service.url, headers={"Authorization": "Bearer test-key"}, timeout=90, trust_env=False)


def _submit(client: httpx.Client, files: list[tuple[str, bytes]], *, hashes: bool = True, **options: Any) -> dict[str, Any]:
    """以表单提交文件清单，hash 缺省时不放入任何替代标识。"""
    manifest = [
        {
            "client_file_id": f"f{index + 1}",
            "filename": name,
            "bytes": len(data),
            **({"sha256sum": hashlib.sha256(data).hexdigest()} if hashes else {}),
        }
        for index, (name, data) in enumerate(files)
    ]
    response = client.post("/v1/tasks", data={"manifest": json.dumps(manifest), "tier": "flash", "ocr_mode": "txt", **options})
    assert response.status_code == 202, response.text
    return response.json()


def _wait(client: httpx.Client, task: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
    """有界轮询任务元数据，不能调用已删除的 result 路由。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(task["status_url"])
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in TERMINAL:
            return payload
        assert payload["result_url"] is None
        time.sleep(0.02)
    pytest.fail("Task did not complete over HTTP")


def _members(data: bytes) -> dict[str, bytes]:
    """比较 ZIP 成员而非时间戳或压缩包装，验证语义与素材保持不变。"""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _script_env(tmp_path: Path, base: str, **extra: str) -> dict[str, str]:
    """隔离示例脚本环境，只继承执行 Python 与本地字符编码所需变量。"""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "MINERU_API_URL": base,
        "MINERU_API_KEY": "test-key",
        "MINERU_PYTHON": sys.executable,
        "MINERU_TIER": "flash",
        "OCR_MODE": "txt",
        "RESULT_FILE": str(tmp_path / "result.zip"),
        "MAX_POLLS": "100",
        "POLL_INTERVAL": "0.02",
        "REQUEST_TIMEOUT": "30",
        "CONNECT_TIMEOUT": "5",
    }
    env.update(extra)
    return env


def _run_script(env: dict[str, str], *paths: Path) -> subprocess.CompletedProcess[str]:
    """执行用户实际调用的 shell 入口，不在测试里重写客户端逻辑。"""
    return subprocess.run(
        ["bash", str(SCRIPT), *(str(path) for path in paths)], env=env, text=True, capture_output=True, timeout=90
    )


@pytest.mark.parametrize("mode,hashes,count", [("preflight", "1", 2), ("preflight", "0", 1), ("direct", "1", 1)])
def test_documented_client_script(live_service: _LiveService, tmp_path: Path, mode: str, hashes: str, count: int) -> None:
    """同一脚本处理本地、Router、单文件直传和多文件逐项上传。"""
    paths: list[Path] = []
    for index in range(count):
        path = tmp_path / f"测试 文档 {index}.html"
        path.write_text(f"<h1>中文内容 {index}</h1>", encoding="utf-8")
        paths.append(path)
    result = _run_script(_script_env(tmp_path, live_service.url, UPLOAD_MODE=mode, SEND_HASH=hashes), *paths)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "task_id=" in result.stderr
    members = _members((tmp_path / "result.zip").read_bytes())
    assert "中文内容".encode() in members["0001/markdown.md" if count > 1 else "markdown.md"]


def test_hash_preflight_sends_only_missing_file_bytes(live_service: _LiveService) -> None:
    """真实网络计数证明混合命中仅上传缺失文件，全命中不发送任何文件 PUT。"""
    with _client(live_service) as client:
        first = _submit(client, [("a.html", HTML)])
        assert client.put(first["uploads"][0]["upload_url"], content=HTML).status_code == 200
        assert _wait(client, first)["status"] == "completed"
        second_data = b"<h1>second</h1>"
        mixed = _submit(client, [("renamed.html", HTML), ("b.html", second_data)])
        assert [link["client_file_id"] for link in mixed["uploads"]] == ["f2"]
        assert client.put(mixed["uploads"][0]["upload_url"], content=second_data).status_code == 200
        assert _wait(client, mixed)["status"] == "completed"
        base = f"/v1/tasks/{mixed['task_id']}/uploads/"
        assert live_service.counts.get(base + "f1", 0) == 0
        assert live_service.counts[base + "f2"] == len(second_data)
        hit = _submit(client, [("a.html", HTML), ("b.html", second_data)])
        assert hit["uploads"] == []
        assert _wait(client, hit)["status"] == "completed"
        assert not any(hit["task_id"] in path for path in live_service.counts)
        denied = client.post(
            "/v1/tasks",
            headers={"Authorization": "Bearer wrong"},
            data={
                "manifest": json.dumps(
                    [
                        {"client_file_id": "f1", "filename": "a.html", "bytes": len(HTML)},
                    ]
                )
            },
        )
        assert denied.status_code == 401


def test_disconnect_then_resume_keeps_uploaded_files(live_service: _LiveService, tmp_path: Path) -> None:
    """A 成功、B 半途断连后通过同一 task_id 恢复，A 不重传且解析仅执行一次。"""
    second = b"<h1>Resumed file</h1><!--" + b"x" * (256 * 1024) + b"-->"
    first_path, second_path = tmp_path / "a.html", tmp_path / "b.html"
    first_path.write_bytes(HTML)
    second_path.write_bytes(second)
    with _client(live_service) as client:
        task = _submit(client, [(first_path.name, HTML), (second_path.name, second)])
        assert client.put(task["uploads"][0]["upload_url"], content=HTML).status_code == 200
        target = urlsplit(task["uploads"][1]["upload_url"])
        connection = socket.create_connection((target.hostname, target.port), timeout=5)
        prefix = second[:4096]
        try:
            headers = (
                f"PUT {target.path} HTTP/1.1\r\nHost: {target.netloc}\r\nAuthorization: Bearer test-key\r\n"
                f"Content-Type: application/octet-stream\r\nContent-Length: {len(second)}\r\nConnection: close\r\n\r\n"
            )
            connection.sendall(headers.encode() + prefix)
            deadline = time.monotonic() + 5
            while live_service.counts.get(target.path, 0) != len(prefix) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert live_service.counts[target.path] == len(prefix)
        finally:
            connection.close()
        pending = client.get(task["status_url"]).json()
        assert pending["status"] == "waiting_upload"
        assert [item["upload_status"] for item in pending["files"]] == ["ready", "pending"]
        result = _run_script(_script_env(tmp_path, live_service.url, TASK_ID=task["task_id"]), first_path, second_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert live_service.counts[urlsplit(task["uploads"][0]["upload_url"]).path] == len(HTML)
        assert live_service.counts[target.path] == len(prefix) + len(second)
        assert len(live_service.app.state.task_store._records) == 1
        members = _members((tmp_path / "result.zip").read_bytes())
        assert b"Resumed file" in members["0002/markdown.md"]
        final = client.get(task["status_url"]).json()
        assert final["task_id"] == task["task_id"] and final["status"] == "completed"


def test_cancel_response_does_not_wait_for_native_thread_cleanup(
    live_service: _LiveService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取消立即返回状态；不可中断的本地线程仍由后台安全等待并释放资源。"""
    entered, release = threading.Event(), threading.Event()

    async def native_work(path: str, **kwargs: Any) -> None:
        """模拟线程推理的清理边界，取消不能强制关闭线程正在读取的文件。"""
        entered.set()
        await run_sync(release.wait, 10)

    monkeypatch.setattr(api_server, "parse_async", native_work)
    try:
        with _client(live_service) as client:
            task = _submit(client, [("a.html", HTML)])
            assert client.put(task["uploads"][0]["upload_url"], content=HTML).status_code == 200
            assert entered.wait(5)
            canceled = client.delete(task["status_url"], timeout=2)
            assert canceled.status_code == 200 and canceled.json()["status"] == "canceled"
            assert not release.is_set()
            assert client.get(task["status_url"]).json()["result_url"] is None
    finally:
        release.set()


@pytest.mark.parametrize("upload_mode", ["preflight", "direct"])
def test_upload_over_200_mib_is_accepted(
    live_service: _LiveService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upload_mode: str,
) -> None:
    """超过旧上限的完整文件经过真实 HTTP 和 worker 转发，校验后才进入解析。"""
    started = threading.Event()

    async def pause_parser(path: str, **kwargs: Any) -> None:
        """只隔离耗时解析，真实源文件读取、完整上传和转发均保留。"""
        assert Path(path).stat().st_size == 200 * 1024 * 1024 + 4096
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(api_server, "parse_async", pause_parser)
    source = tmp_path / "large.html"
    size = 200 * 1024 * 1024 + 4096
    with source.open("wb") as output:
        output.write(b"<h1>Large upload</h1>")
        output.seek(size - 1)
        output.write(b"\n")
    with _client(live_service) as client:
        if upload_mode == "direct":
            with source.open("rb") as stream:
                response = client.post("/v1/tasks", data={"tier": "flash"}, files={"files": (source.name, stream)})
            assert response.status_code == 202, response.text
            task = response.json()
        else:
            response = client.post(
                "/v1/tasks",
                data={
                    "tier": "flash",
                    "manifest": json.dumps(
                        [
                            {"client_file_id": "large", "filename": source.name, "bytes": size},
                        ]
                    ),
                },
            )
            assert response.status_code == 202, response.text
            task = response.json()
            with source.open("rb") as stream:
                uploaded = client.put(task["uploads"][0]["upload_url"], content=stream, headers={"Content-Length": str(size)})
            assert uploaded.status_code == 200, uploaded.text
            assert live_service.counts[urlsplit(task["uploads"][0]["upload_url"]).path] == size
        assert started.wait(60)
        status = client.get(task["status_url"]).json()
        assert status["files"][0]["upload_status"] == "ready"
        assert client.delete(task["status_url"]).json()["status"] == "canceled"


@pytest.mark.parametrize("page_count", [601, 1001])
def test_large_page_count_pdf_is_not_rejected(live_service: _LiveService, page_count: int) -> None:
    """真实 Flash txt 解析保留超过 SaaS 页数策略的全部 PDF 页面。"""
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    data = buffer.getvalue()
    with _client(live_service) as client:
        task = _submit(client, [("many-pages.pdf", data)])
        assert client.put(task["uploads"][0]["upload_url"], content=data).status_code == 200
        result = _wait(client, task, timeout=90)
        assert result["status"] == "completed", result
        download = client.get(result["result_url"])
        middle = json.loads(_members(download.content)["middle_json.json"])
        assert len(middle["pages"]) == page_count
        assert middle["pages"][-1]["page_idx"] == page_count - 1


@pytest.mark.parametrize("document", ["html", "pdf"])
def test_real_documents_match_jobs_outputs(live_service: _LiveService, document: str) -> None:
    """真实 HTML/PDF 的任务 ZIP 与既有 Jobs 产物逐项一致，不改变中间协议或素材。"""
    if document == "pdf":
        path = Path(__file__).resolve().parents[2] / "examples" / "demo1.pdf"
        if not path.is_file():
            pytest.skip("Local real PDF examples/demo1.pdf is unavailable")
        data, name, page_range = path.read_bytes(), path.name, "1-2"
    else:
        data, name, page_range = HTML, "source.html", ""
    formats = ["markdown", "middle_json", "structured_content", "zip"]
    with _client(live_service) as client:
        task = _submit(client, [(name, data)], page_range=page_range, output_formats=formats)
        assert client.put(task["uploads"][0]["upload_url"], content=data).status_code == 200
        result = _wait(client, task)
        assert result["status"] == "completed", result
        submitted = client.post(
            "/v1/parse/jobs",
            json={
                "tier": "flash",
                "ocr_mode": "txt",
                "output_formats": formats,
                "files": [
                    {"source": {"type": "file_id", "file_id": result["files"][0]["file_id"]}, "page_range": page_range or None}
                ],
            },
        )
        assert submitted.status_code == 202, submitted.text
        deadline = time.monotonic() + 60
        while True:
            baseline = client.get(f"/v1/parse/jobs/{submitted.json()['job_id']}").json()
            if baseline["status"] in TERMINAL:
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert baseline["status"] == "completed", baseline
        first = client.get(result["result_url"])
        second = client.get(f"/v1/files/{baseline['files'][0]['output_files']['zip']['file_id']}/content")
        assert first.status_code == second.status_code == 200
        assert _members(first.content) == _members(second.content)


@dataclass
class _CloudState:
    """模拟仅处理元数据的 SaaS 网关与独立对象存储，覆盖客户端可观察的协议。"""

    gateway_url: str = ""
    storage_url: str = ""
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache: dict[str, bytes] = field(default_factory=dict)
    requests: list[tuple[str, str, str, str | None]] = field(default_factory=list)
    upload_attempts: dict[str, int] = field(default_factory=dict)
    gateway_body_sizes: list[int] = field(default_factory=list)
    transient_upload: bool = False
    lost_upload_response: bool = False
    expired_upload: bool = False
    expired_download: bool = False
    api_download_redirect: bool = False
    forced_status: str | None = None
    invalid_response: str | None = None

    def payload(self, task_id: str) -> dict[str, Any]:
        """只从元数据生成响应，网关不读取对象存储中的结果字节。"""
        record = self.tasks[task_id]
        ready = record["ready"]
        status = (self.forced_status or "completed") if len(ready) == len(record["manifest"]) else "waiting_upload"
        files = [
            {
                **item,
                "name": item["filename"],
                "upload_status": "ready" if item["client_file_id"] in ready else "pending",
                "status": "completed" if status in {"completed", "partial"} else "queued",
            }
            for item in record["manifest"]
        ]
        result_url = None
        if status in {"completed", "partial"}:
            result_url = f"{self.storage_url}/download/{task_id}?signature={record['download_version']}"
            if self.api_download_redirect:
                result_url = f"{self.gateway_url}/redirect/{task_id}"
        uploads = [
            {
                "client_file_id": item["client_file_id"],
                "upload_method": "PUT",
                "upload_url": (
                    f"{self.storage_url}/objects/{task_id}/{item['client_file_id']}?signature={record['upload_version']}"
                ),
                "upload_headers": {"Content-Type": "application/octet-stream", "X-Upload-Token": "storage-token"},
                "expires_at": int(time.time()) + 3600,
            }
            for item in record["manifest"]
            if item["client_file_id"] not in ready
        ]
        payload = {
            "task_id": task_id,
            "status": status,
            "status_url": f"{self.gateway_url}/v1/tasks/{task_id}",
            "result_url": result_url,
            "files": files,
            "uploads": uploads,
            "error": None,
        }
        if self.invalid_response == "missing_id":
            payload["task_id"] = None
        elif self.invalid_response == "unknown_status":
            payload["status"] = "mystery"
        return payload


def _cloud_apps(state: _CloudState) -> tuple[FastAPI, FastAPI]:
    """构造两个独立服务；网关没有 result 路由，也没有文件字节上传路由。"""
    gateway, storage = FastAPI(), FastAPI()

    @gateway.middleware("http")
    async def authorize(request: Request, call_next: Any) -> Response:
        """网关请求必须带 API Key，对象存储请求则另行验证不能携带它。"""
        state.requests.append(("gateway", request.method, request.url.path, request.headers.get("authorization")))
        if request.headers.get("authorization") != "Bearer test-key":
            return JSONResponse({"error": {"code": "invalid_api_key"}}, status_code=401)
        return await call_next(request)

    @gateway.post("/v1/tasks")
    async def create_task(request: Request) -> Response:
        """只接收体积很小的表单元数据，用 hash 判断哪些文件需要独立上传。"""
        assert not request.query_params
        assert request.headers["content-type"].startswith("application/x-www-form-urlencoded")
        body = await request.body()
        state.gateway_body_sizes.append(len(body))
        assert len(body) < 4096
        form = await request.form()
        manifest = json.loads(form["manifest"])
        task_id = f"cloud_{len(state.tasks) + 1}"
        state.tasks[task_id] = {
            "manifest": manifest,
            "ready": {item["client_file_id"] for item in manifest if item.get("sha256sum") in state.cache},
            "upload_version": 0,
            "download_version": 0,
        }
        return JSONResponse(state.payload(task_id), status_code=202)

    @gateway.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str) -> Response:
        """仅返回元数据和 OSS 下载链接，不回源读取或代理 ZIP 内容。"""
        return JSONResponse(state.payload(task_id))

    @gateway.get("/redirect/{task_id}")
    async def redirect(task_id: str) -> Response:
        """模拟同源下载地址重定向到同主机不同端口，检查客户端凭据隔离。"""
        return RedirectResponse(
            f"{state.storage_url}/download/{task_id}?signature={state.tasks[task_id]['download_version']}",
            status_code=302,
        )

    @storage.middleware("http")
    async def no_api_key(request: Request, call_next: Any) -> Response:
        """记录实际对象存储请求，MinerU API Key 不得跨源发送。"""
        state.requests.append(("storage", request.method, request.url.path, request.headers.get("authorization")))
        assert request.headers.get("authorization") is None
        return await call_next(request)

    @storage.put("/objects/{task_id}/{client_file_id}")
    async def upload(task_id: str, client_file_id: str, request: Request) -> Response:
        """只在对象存储接收文件字节，并模拟临时错误、链接过期和提交后响应丢失。"""
        assert request.headers.get("x-upload-token") == "storage-token"
        key = task_id + "/" + client_file_id
        state.upload_attempts[key] = state.upload_attempts.get(key, 0) + 1
        record = state.tasks[task_id]
        if request.query_params.get("signature") != str(record["upload_version"]):
            return Response(status_code=403)
        if state.expired_upload and record["upload_version"] == 0:
            record["upload_version"] = 1
            return Response(status_code=403)
        if state.transient_upload and state.upload_attempts[key] == 1:
            return Response(status_code=503)
        data = await request.body()
        item = next(item for item in record["manifest"] if item["client_file_id"] == client_file_id)
        assert len(data) == item["bytes"]
        actual = hashlib.sha256(data).hexdigest()
        assert item.get("sha256sum") in {None, actual}
        state.cache[actual] = data
        record["ready"].add(client_file_id)
        if state.lost_upload_response:
            return Response(status_code=503)
        return Response(status_code=200)

    @storage.get("/download/{task_id}")
    async def download(task_id: str, request: Request) -> Response:
        """结果字节仅由对象存储返回；过期地址通过再次查询原任务刷新。"""
        record = state.tasks[task_id]
        if request.query_params.get("signature") != str(record["download_version"]):
            return Response(status_code=403)
        if state.expired_download and record["download_version"] == 0:
            record["download_version"] = 1
            return Response(status_code=403)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("result.md", "# OSS result\n")
            archive.writestr("manifest.json", json.dumps({"task_id": task_id, "status": state.forced_status or "completed"}))
        return Response(output.getvalue(), media_type="application/zip")

    return gateway, storage


@pytest.fixture
def cloud() -> Iterator[_CloudState]:
    """分别启动 SaaS 网关和 OSS 模拟服务，保证跨源认证检查经过真实 TCP。"""
    state = _CloudState()
    gateway, storage = _cloud_apps(state)
    api_server_instance, storage_server = _Server(gateway), _Server(storage)
    state.gateway_url, state.storage_url = api_server_instance.url, storage_server.url
    storage_server.start()
    api_server_instance.start()
    try:
        yield state
    finally:
        api_server_instance.stop()
        storage_server.stop()


@pytest.mark.parametrize("cache_mode", ["none", "mixed", "all", "no_hash"])
def test_same_client_works_with_saas_and_oss(cloud: _CloudState, tmp_path: Path, cache_mode: str) -> None:
    """通用客户端只向网关发送清单，文件和 ZIP 都直接走对象存储。"""
    inputs = [tmp_path / "a.html", tmp_path / "b.html"]
    data = [HTML, b"<h1>Other file</h1>" * 10000]
    for path, content in zip(inputs, data):
        path.write_bytes(content)
    if cache_mode in {"mixed", "all", "no_hash"}:
        cloud.cache[hashlib.sha256(data[0]).hexdigest()] = data[0]
    if cache_mode == "all":
        cloud.cache[hashlib.sha256(data[1]).hexdigest()] = data[1]
    result = _run_script(_script_env(tmp_path, cloud.gateway_url, SEND_HASH="0" if cache_mode == "no_hash" else "1"), *inputs)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {"none": 2, "mixed": 1, "all": 0, "no_hash": 2}[cache_mode]
    assert len(cloud.upload_attempts) == expected
    assert len(cloud.tasks) == 1 and max(cloud.gateway_body_sizes) < 4096
    assert not any(surface == "gateway" and method == "PUT" for surface, method, _, _ in cloud.requests)
    assert not any(path.endswith("/result") for _, _, path, _ in cloud.requests)
    assert all(auth is None for surface, _, _, auth in cloud.requests if surface == "storage")
    assert _members((tmp_path / "result.zip").read_bytes())["result.md"] == b"# OSS result\n"


@pytest.mark.parametrize(
    "failure", ["transient_upload", "lost_upload_response", "expired_upload", "expired_download", "api_download_redirect"]
)
def test_client_recovers_links_and_uncertain_uploads(cloud: _CloudState, tmp_path: Path, failure: str) -> None:
    """恢复原任务和刷新地址不创建新任务；跨源重定向不能转发 API Key。"""
    setattr(cloud, failure, True)
    source = tmp_path / "a.html"
    source.write_bytes(HTML)
    result = _run_script(_script_env(tmp_path, cloud.gateway_url), source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(cloud.tasks) == 1
    assert cloud.upload_attempts["cloud_1/f1"] == (2 if failure in {"transient_upload", "expired_upload"} else 1)
    assert all(auth is None for surface, _, _, auth in cloud.requests if surface == "storage")
    assert (tmp_path / "result.zip").is_file()


@pytest.mark.parametrize("invalid", ["missing_id", "unknown_status"])
def test_invalid_success_response_does_not_upload_or_poll(cloud: _CloudState, tmp_path: Path, invalid: str) -> None:
    """非法 HTTP 202 响应不会带着空 ID 或未知状态继续执行。"""
    cloud.invalid_response = invalid
    source = tmp_path / "a.html"
    source.write_bytes(HTML)
    result = _run_script(_script_env(tmp_path, cloud.gateway_url), source)
    assert result.returncode == 1
    assert not cloud.upload_attempts
    assert not any(method == "GET" for _, method, _, _ in cloud.requests)
    assert not (tmp_path / "result.zip").exists()


@pytest.mark.parametrize("status,exit_code", [("partial", 2), ("failed", 1), ("running", 124)])
def test_client_exit_codes_and_poll_budget(cloud: _CloudState, tmp_path: Path, status: str, exit_code: int) -> None:
    """部分成功先保存 ZIP，失败不下载，等待预算耗尽保留原 task_id。"""
    cloud.forced_status = status
    source = tmp_path / "a.html"
    source.write_bytes(HTML)
    result = _run_script(_script_env(tmp_path, cloud.gateway_url, MAX_POLLS="2"), source)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert len(cloud.tasks) == 1 and "task_id=cloud_1" in result.stderr
    assert (tmp_path / "result.zip").exists() == (status == "partial")
    if status == "running":
        assert "continues running" in result.stderr
