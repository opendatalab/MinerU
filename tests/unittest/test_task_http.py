"""真实 TCP、客户端脚本及隔离 SaaS/OSS 模拟服务的共同协议验收。"""

from __future__ import annotations

from task_test_utils import pending_files, task_url

import asyncio
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pypdf import PdfWriter
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from test_http_api_example_script import _Server

from mineru.kit.router import RouterSettings
from mineru.kit.router import create_app as create_router
from mineru.parser import api_server
from mineru.filetypes import batch_effective_parse_tier
from mineru.tasks.contracts import TaskFileInput, UploadTaskOptions
from mineru.tasks.errors import ApiServerError
from mineru.tasks.service import resolve_task_tier
from mineru.types import TIERS, Tier
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


class _WorkerRequests:
    """记录真实 HTTP worker 收到的请求头，验证内外两跳的代理来源边界。"""

    def __init__(self, app: ASGIApp, calls: list[tuple[str, str, dict[str, str]]]) -> None:
        """由测试保存观测结果，不改变应用或 Uvicorn 的代理头处理。"""
        self.app = app
        self.calls = calls

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """原样执行 worker，只读取实际抵达的 HTTP 请求头。"""
        if scope["type"] == "http":
            self.calls.append((scope["method"], scope["path"], dict(Request(scope).headers)))
        await self.app(scope, receive, send)


@dataclass
class _LiveService:
    """保留真实网络地址与应用对象，便于核对上传量和唯一任务记录。"""

    url: str
    app: FastAPI
    counts: dict[str, int]
    execution_apps: list[FastAPI]


@pytest.fixture(params=["api", "router"])
def live_service(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[_LiveService]:
    """启动真实 Uvicorn 服务；Router 的两个 worker 同样使用真实 TCP。"""
    servers: list[_Server] = []
    execution_apps: list[FastAPI] = []
    try:
        if request.param == "api":
            app = api_server.create_app(upload_dir=str(tmp_path / "api"), tiers=("flash",), api_key="test-key")
            execution_apps.append(app)
        else:
            for index in range(2):
                worker_app = api_server.create_app(
                    upload_dir=str(tmp_path / f"worker-{index}"), tiers=("flash",), api_key="test-key"
                )
                execution_apps.append(worker_app)
                worker = _Server(worker_app)
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
        yield _LiveService(server.url, app, counts, execution_apps)
    finally:
        for server in reversed(servers):
            server.stop()


def _client(service: _LiveService) -> httpx.Client:
    """每个请求经过真实 HTTP 与已配置的 API Key，禁用外部代理环境。"""
    return httpx.Client(base_url=service.url, headers={"Authorization": "Bearer test-key"}, timeout=90, trust_env=False)


def _submit(client: httpx.Client, files: list[tuple[str, bytes]], *, hashes: bool = True, **options: Any) -> dict[str, Any]:
    """以 JSON 提交文件清单，hash 缺省时不放入任何替代标识。"""
    manifest = [
        {
            "file_name": name,
            "bytes": len(data),
            **({"sha256sum": hashlib.sha256(data).hexdigest()} if hashes else {}),
        }
        for name, data in files
    ]
    response = client.post("/v1/tasks", json={"files": manifest, "tier": "flash", "ocr_mode": "txt", **options})
    assert response.status_code == 202, response.text
    return response.json()


def _wait(client: httpx.Client, task: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
    """有界轮询任务元数据，不能调用已删除的 result 路由。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(task_url(client, task))
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in TERMINAL:
            return payload
        assert all(file.get("result_url") is None for file in payload["files"] if file["status"] != "completed")
        time.sleep(0.02)
    pytest.fail("Task did not complete over HTTP")


def _members(data: bytes) -> dict[str, bytes]:
    """比较 ZIP 成员而非时间戳或压缩包装，验证语义与素材保持不变。"""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_pdf_without_file_id_keeps_metadata_and_cached_results(live_service: _LiveService, tmp_path: Path) -> None:
    """真实 TCP 上传 PDF，确认自动标识、Files 元数据、附件命名及缓存命中后的正文均正确。"""
    from reportlab.pdfgen.canvas import Canvas

    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=(595, 842))
    canvas.drawString(72, 770, "Tasks field migration")
    canvas.drawString(72, 740, "Generated file IDs stay stable across uploads and queries.")
    canvas.save()
    data = buffer.getvalue()
    (tmp_path / "report.v2.pdf").write_bytes(data)
    payload = {
        "files": [{"file_name": "report.v2.pdf", "bytes": len(data), "sha256sum": hashlib.sha256(data).hexdigest()}],
        "tier": "flash",
        "ocr_mode": "txt",
    }
    with _client(live_service) as client:
        response = client.post("/v1/tasks", json=payload)
        assert response.status_code == 202, response.text
        task = response.json()
        assert task["files"][0]["file_id"] == pending_files(task)[0]["file_id"]
        assert task["files"][0]["file_id"].startswith("file-")
        link = pending_files(task)[0]
        for _ in range(2):
            upload = client.put(link.get("upload").get("url"), content=data, headers=link.get("upload")["headers"])
            assert upload.status_code == 200, upload.text
        completed = _wait(client, task)
        assert completed["status"] == "completed", completed
        item = completed["files"][0]
        assert (
            item["file_id"] == task["files"][0]["file_id"] and item["file_name"] == "report.v2.pdf" and "filename" not in item
        )
        result = client.get(item.get("result_url").removesuffix("/content"))
        assert result.status_code == 200 and result.json()["file_name"] == "report.v2.zip"
        download = client.get(item.get("result_url"))
        assert download.status_code == 200
        assert download.headers["content-disposition"] == 'attachment; filename="report.v2.zip"'
        (tmp_path / "report.v2.zip").write_bytes(download.content)
        members = _members(download.content)
        assert b"Tasks field migration" in members["markdown.md"]
        assert b"Generated file IDs stay stable" in members["markdown.md"]
        assert len(json.loads(members["middle_json.json"])["pages"]) == 1
        cached = client.post("/v1/tasks", json=payload)
        assert cached.status_code == 202, cached.text
        cached_task = cached.json()
        assert pending_files(cached_task) == [] and cached_task["files"][0]["file_id"] != task["files"][0]["file_id"]
        assert cached_task["files"][0]["status"] == "pending" and cached_task["files"][0].get("upload") is None
        cached_result = _wait(client, cached_task)
        assert cached_result["status"] == "completed", cached_result
        cached_download = client.get(cached_result["files"][0].get("result_url"))
        assert cached_download.status_code == 200
        assert _members(cached_download.content)["markdown.md"] == members["markdown.md"]


def test_router_keeps_tls_proxy_headers_on_the_public_hop_only(tmp_path: Path) -> None:
    """TLS 终止代理后的上传、hash 复用及下载保持可用，worker 不继承客户端来源头。"""
    calls: list[tuple[str, str, dict[str, str]]] = []
    worker_app = api_server.create_app(upload_dir=str(tmp_path / "worker"), tiers=("flash",), api_key="test-key")
    worker_app.add_middleware(_WorkerRequests, calls=calls)
    worker = _Server(worker_app)
    servers = [worker]
    proxy_headers = {
        "Forwarded": 'for=203.0.113.10;proto=https;host="mineru.example"',
        "X-Forwarded-Proto": "https",
        "X-Forwarded-For": "203.0.113.10",
        "X-Forwarded-Host": "mineru.example",
        "X-Forwarded-Port": "443",
        "X-Forwarded-Prefix": "/mineru",
        "X-Real-IP": "203.0.113.10",
    }
    try:
        worker.start()
        router = _Server(
            create_router(RouterSettings(upstream_urls=(worker.url,), local_gpus="none", worker_refresh_interval_seconds=0))
        )
        servers.append(router)
        router.start()
        with httpx.Client(
            base_url=router.url,
            trust_env=False,
            timeout=30,
            headers={**proxy_headers, "Authorization": "Bearer test-key", "X-Request-ID": "proxy-regression"},
        ) as client:
            for index in range(2):
                task = _submit(client, [(f"document-{index}.html", HTML)])
                assert "status_url" not in task
                assert all(urlsplit(item.get("upload").get("url")).scheme == "https" for item in pending_files(task))
                assert len(pending_files(task)) == (1 if index == 0 else 0)
                for upload in pending_files(task):
                    # 以本地 HTTP 模拟 TLS 终止代理到 Router 的后段连接，保留真实入站代理头。
                    response = client.put(urlsplit(upload.get("upload").get("url")).path, content=HTML)
                    assert response.status_code == 200, response.text
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    response = client.get(urlsplit(task_url(client, task)).path)
                    assert response.status_code == 200, response.text
                    result = response.json()
                    if result["status"] in TERMINAL:
                        break
                    time.sleep(0.01)
                else:
                    pytest.fail("Task did not finish through the TLS proxy hop")
                assert result["status"] == "completed", result
                result_url = result["files"][0].get("result_url")
                assert urlsplit(result_url).scheme == "https"
                response = client.get(urlsplit(result_url).path)
                assert response.status_code == 200, response.text
                assert b"Same source and output" in _members(response.content)["markdown.md"]
                ranged = client.get(urlsplit(result_url).path, headers={"Range": "bytes=0-31"})
                assert ranged.status_code == 206
                assert ranged.content == response.content[:32]
        forwarded_calls = [call for call in calls if call[2].get("x-request-id") == "proxy-regression"]
        assert any(method == "POST" and path == "/v1/tasks" for method, path, _ in forwarded_calls)
        assert any(method == "PUT" for method, _, _ in forwarded_calls)
        assert any(path.endswith("/content") for _, path, _ in forwarded_calls)
        assert any(headers.get("range") == "bytes=0-31" for _, _, headers in forwarded_calls)
        for _, _, headers in forwarded_calls:
            assert headers["authorization"] == "Bearer test-key"
            assert not set(headers).intersection(key.lower() for key in proxy_headers)
    finally:
        for server in reversed(servers):
            server.stop()


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
        "RESULT_DIR": str(tmp_path / "results"),
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


@pytest.mark.parametrize(
    "formats,hashes,count",
    [("default", "1", 2), ("default", "0", 1), ("html", "1", 1), ("docx", "1", 1), ("default,latex,html,docx", "1", 2)],
)
def test_documented_client_script(live_service: _LiveService, tmp_path: Path, formats: str, hashes: str, count: int) -> None:
    """同一客户端以 JSON 创建任务，按文件保存 ZIP、HTML 或 DOCX。"""
    paths: list[Path] = []
    for index in range(count):
        path = tmp_path / f"测试 文档 {index}.html"
        path.write_text(f"<h1>中文内容 {index}</h1>", encoding="utf-8")
        paths.append(path)
    result = _run_script(_script_env(tmp_path, live_service.url, OUTPUT_FORMATS=formats, SEND_HASH=hashes), *paths)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "task_id=" in result.stderr
    for path in paths:
        extension = formats if formats in {"html", "docx"} else "zip"
        output = next((tmp_path / "results").glob(f"*/{path.stem}.{extension}"))
        assert output.is_file()
        if extension == "zip":
            assert "中文内容".encode() in _members(output.read_bytes())["markdown.md"]
        elif extension == "html":
            assert "中文内容" in output.read_text()
        else:
            assert "word/document.xml" in _members(output.read_bytes())


def test_hash_preflight_sends_only_missing_file_bytes(live_service: _LiveService) -> None:
    """真实网络计数证明混合命中仅上传缺失文件，全命中不发送任何文件 PUT。"""
    with _client(live_service) as client:
        first = _submit(client, [("a.html", HTML)])
        assert client.put(pending_files(first)[0].get("upload").get("url"), content=HTML).status_code == 200
        assert _wait(client, first)["status"] == "completed"
        second_data = b"<h1>second</h1>"
        mixed = _submit(client, [("renamed.html", HTML), ("b.html", second_data)])
        assert [link["file_id"] for link in pending_files(mixed)] == [mixed["files"][1]["file_id"]]
        assert client.put(pending_files(mixed)[0].get("upload").get("url"), content=second_data).status_code == 200
        assert _wait(client, mixed)["status"] == "completed"
        base = f"/v1/tasks/{mixed['task_id']}/uploads/"
        assert live_service.counts.get(base + mixed["files"][0]["file_id"], 0) == 0
        assert live_service.counts[base + mixed["files"][1]["file_id"]] == len(second_data)
        hit = _submit(client, [("a.html", HTML), ("b.html", second_data)])
        assert pending_files(hit) == []
        assert _wait(client, hit)["status"] == "completed"
        assert not any(hit["task_id"] in path for path in live_service.counts)
        denied = client.post(
            "/v1/tasks",
            headers={"Authorization": "Bearer wrong"},
            json={
                "files": [
                    {"file_name": "a.html", "bytes": len(HTML)},
                ]
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
        assert client.put(pending_files(task)[0].get("upload").get("url"), content=HTML).status_code == 200
        target = urlsplit(pending_files(task)[1].get("upload").get("url"))
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
        pending = client.get(task_url(client, task)).json()
        assert pending["status"] in {"waiting-file", "pending", "running"}
        assert pending["files"][0]["status"] in {"pending", "running", "completed"}
        assert pending["files"][1]["status"] == "waiting-file"
        result = _run_script(_script_env(tmp_path, live_service.url, TASK_ID=task["task_id"]), first_path, second_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert live_service.counts[urlsplit(pending_files(task)[0].get("upload").get("url")).path] == len(HTML)
        assert live_service.counts[target.path] == len(prefix) + len(second)
        assert len(live_service.app.state.task_store._records) == 1
        members = _members((tmp_path / "results" / task["files"][1]["file_id"] / "b.zip").read_bytes())
        assert b"Resumed file" in members["markdown.md"]
        final = client.get(task_url(client, task)).json()
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
            assert client.put(pending_files(task)[0].get("upload").get("url"), content=HTML).status_code == 200
            assert entered.wait(5)
            canceled = client.post(task_url(client, task) + "/cancel", timeout=2)
            assert canceled.status_code == 200 and canceled.json()["status"] == "canceled"
            assert not release.is_set()
            assert client.get(task_url(client, task)).json()["files"][0].get("result_url") is None
    finally:
        release.set()


def test_upload_over_200_mib_is_accepted(live_service: _LiveService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """超过旧上限的完整文件经过真实 HTTP 和 worker 转发，校验后才解析。"""
    started = threading.Event()

    async def pause_parser(path: str, **kwargs: Any) -> None:
        """只隔离耗时解析，保留完整源文件接收、转发和字节校验。"""
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
        response = client.post(
            "/v1/tasks",
            json={"tier": "flash", "files": [{"file_name": source.name, "bytes": size}]},
        )
        assert response.status_code == 202, response.text
        task = response.json()
        with source.open("rb") as stream:
            uploaded = client.put(
                pending_files(task)[0].get("upload").get("url"), content=stream, headers={"Content-Length": str(size)}
            )
        assert uploaded.status_code == 200, uploaded.text
        assert live_service.counts[urlsplit(pending_files(task)[0].get("upload").get("url")).path] == size
        assert started.wait(60)
        assert client.get(task_url(client, task)).json()["files"][0]["status"] in {"pending", "running", "completed"}
        assert client.post(task_url(client, task) + "/cancel").json()["status"] == "canceled"


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
        assert client.put(pending_files(task)[0].get("upload").get("url"), content=data).status_code == 200
        result = _wait(client, task, timeout=90)
        assert result["status"] == "completed", result
        download = client.get(result["files"][0].get("result_url"))
        middle = json.loads(_members(download.content)["middle_json.json"])
        assert len(middle["pages"]) == page_count
        assert middle["pages"][-1]["page_idx"] == page_count - 1


@pytest.mark.parametrize("document", ["html", "pdf"])
def test_real_documents_match_local_parser_outputs(live_service: _LiveService, document: str) -> None:
    """真实 HTML/PDF 的 Tasks 产物与直接解析逐项一致，检查中间协议和图片字节。"""
    if document == "pdf":
        path = Path(__file__).resolve().parents[2] / "demo" / "pdfs" / "demo1.pdf"
        if not path.is_file():
            pytest.skip("Real PDF demo/pdfs/demo1.pdf is unavailable")
        data, name, page_range = path.read_bytes(), path.name, "1-2"
    else:
        data, name, page_range = HTML, "source.html", ""
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / name
        source.write_bytes(data)
        from mineru.parser import parse
        from mineru.parser.task_outputs import build_task_output
        from docvortex.document.contracts import HtmlSourceContext

        baseline = parse(
            source,
            tier="flash",
            ocr_mode="txt",
            page_range=page_range,
            source_context=HtmlSourceContext() if document == "html" else None,
        )
        expected = build_task_output(baseline, ["default"], Path(directory) / "outputs").read_bytes()
    with _client(live_service) as client:
        task = _submit(client, [(name, data)], page_range=page_range, output_formats=["default"])
        assert client.put(pending_files(task)[0].get("upload").get("url"), content=data).status_code == 200
        result = _wait(client, task)
        assert result["status"] == "completed", result
        output = client.get(result["files"][0].get("result_url"))
        assert output.status_code == 200
        assert _members(output.content) == _members(expected)


@pytest.mark.parametrize("document", ["html", "pdf"])
def test_url_task_over_http_downloads_chunked_source(live_service: _LiveService, document: str) -> None:
    """真实 TCP URL 无需 HEAD 或长度头，保留文件页范围且不向源站发送 API Key。"""
    if document == "pdf":
        sample = Path(__file__).resolve().parents[2] / "demo" / "pdfs" / "demo1.pdf"
        if sample.is_file():
            data = sample.read_bytes()
        else:
            writer = PdfWriter()
            for _ in range(3):
                writer.add_blank_page(width=72, height=72)
            buffer = io.BytesIO()
            writer.write(buffer)
            data = buffer.getvalue()
    else:
        data = HTML
    requests: list[Request] = []
    source = FastAPI()

    @source.api_route("/opaque", methods=["GET", "HEAD"])
    async def download(request: Request) -> StreamingResponse:
        """使用不带 Content-Length 的分块响应，验证只以真实收到的字节计算大小。"""
        requests.append(request)

        async def chunks() -> AsyncIterator[bytes]:
            """分段返回源文件，不借助上传入口或提前探测文件长度。"""
            for offset in range(0, len(data), 8192):
                yield data[offset : offset + 8192]

        return StreamingResponse(chunks(), media_type="application/pdf" if document == "pdf" else "text/html")

    source_server = _Server(source)
    source_server.start()
    try:
        for worker in live_service.execution_apps:
            worker.state.allow_http_source = True
        item = {
            "file_name": f"logical.{document}",
            "url": f"{source_server.url}/opaque?signature=a%2Fb&part=1",
            **({"page_range": "2"} if document == "pdf" else {}),
        }
        with _client(live_service) as client:
            response = client.post(
                "/v1/tasks",
                json={
                    "files": [item],
                    "tier": "flash",
                    "ocr_mode": "txt",
                    "page_range": "1" if document == "pdf" else "",
                },
            )
            assert response.status_code == 202, response.text
            task = response.json()
            assert pending_files(task) == [] and task["files"][0]["status"] == "waiting-file"
            result = _wait(client, task)
            assert result["status"] == "completed", result
            assert result["files"][0].get("bytes") == len(data)
            assert "sha256sum" not in result["files"][0]
            members = _members(client.get(result["files"][0].get("result_url")).content)
            assert live_service.counts == {}
            assert len(requests) == 1 and requests[0].method == "GET"
            assert requests[0].scope["query_string"] == b"signature=a%2Fb&part=1"
            assert "authorization" not in requests[0].headers
            if document == "html":
                # 同一原始 URL 保留 HTML 来源上下文，重复执行的产物应完全一致。
                response = client.post("/v1/tasks", json={"tier": "flash", "ocr_mode": "txt", "files": [item]})
                assert response.status_code == 202, response.text
                baseline_result = _wait(client, response.json())
                assert baseline_result["status"] == "completed"
                baseline_url = baseline_result["files"][0].get("result_url")
            else:
                baseline = _submit(client, [(item["file_name"], data)], page_range="2")
                assert client.put(pending_files(baseline)[0].get("upload").get("url"), content=data).status_code == 200
                baseline_result = _wait(client, baseline)
                assert baseline_result["status"] == "completed"
                baseline_url = baseline_result["files"][0].get("result_url")
            baseline_members = _members(client.get(baseline_url).content)
            assert {k: v for k, v in members.items() if k != "images/"} == {
                k: v for k, v in baseline_members.items() if k not in {"model_output.json", "images/"}
            }
            if document == "pdf":
                middle = json.loads(members["middle_json.json"])
                assert len(middle["pages"]) == 1 and middle["pages"][0]["page_idx"] == 1
    finally:
        source_server.stop()


@dataclass
class _CloudState:
    """模拟仅处理元数据的 SaaS 网关与独立对象存储，覆盖客户端可观察的协议。"""

    gateway_url: str = ""
    storage_url: str = ""
    pdf_tiers: tuple[Tier, ...] = ("standard",)
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
        """只从逐文件元数据生成状态与链接，完成文件不等待其他输入。"""
        record = self.tasks[task_id]
        ready, expired = record["ready"], record["expired"]
        files = []
        for index, item in enumerate(record["files"]):
            file_id = item["file_id"]
            status = "completed" if file_id in ready else "waiting-file"
            if file_id in expired:
                status = "failed"
            if file_id in ready and self.forced_status in {"failed", "running"}:
                status = self.forced_status
            if self.forced_status == "partial" and index == len(record["files"]) - 1 and file_id in ready:
                status = "failed"
            result_url = None
            if status == "completed":
                result_url = f"{self.storage_url}/download/{task_id}/{file_id}?signature={record['download_version']}"
                if self.api_download_redirect:
                    result_url = f"{self.gateway_url}/redirect/{task_id}/{file_id}"
            files.append(
                {
                    "file_id": file_id,
                    "file_name": item["file_name"],
                    "bytes": item.get("bytes"),
                    "url": item.get("url"),
                    "upload": None
                    if file_id in ready | expired
                    else {
                        "url": f"{self.storage_url}/objects/{task_id}/{file_id}?signature=0",
                        "headers": {"Content-Type": "application/octet-stream", "X-Upload-Token": "storage-token"},
                        "expires_at": record["expires_at"],
                    },
                    "status": status,
                    "result_url": result_url,
                    "parse": {
                        "tier": batch_effective_parse_tier(record["tier"], item["file_name"]),
                        "duration_ms": 0,
                        "parser_version": "test",
                    } if status == "completed" else None,
                }
            )
        states = [item["status"] for item in files]
        if "running" in states:
            status = "running"
        elif "waiting-file" in states:
            status = "waiting-file"
        else:
            status = (
                "completed"
                if all(value == "completed" for value in states)
                else ("partial" if "completed" in states else "failed")
            )
        payload = {
            "task_id": task_id,
            "status": status,
            "files": [{key: value for key, value in item.items() if value is not None} for item in files],
            "output_formats": record["output_formats"],
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

    @gateway.get("/v1/health")
    async def health() -> dict[str, Any]:
        """原生支持是共同契约，不增加格式能力字段。"""
        return {
            "status": "ok", "version": "test",
            "features": {"webhook": False, "task_output_formats": ["default"], "sources": ["upload"]},
        }

    @gateway.get("/v1/tiers")
    async def tiers() -> dict[str, Any]:
        """SaaS 默认只发布 PDF/图片 Standard 档位。"""
        return {"object": "list", "data": [{"id": tier, "description": tier} for tier in state.pdf_tiers]}

    @gateway.post("/v1/tasks")
    async def create_task(request: Request) -> Response:
        """只接收体积很小的 JSON 元数据，用 hash 判断哪些文件需要独立上传。"""
        assert not request.query_params
        assert request.headers["content-type"].startswith("application/json")
        body = await request.body()
        state.gateway_body_sizes.append(len(body))
        assert len(body) < 4096
        submitted = await request.json()
        manifest = submitted["files"]
        try:
            selected_tier = resolve_task_tier(
                UploadTaskOptions.model_validate({key: value for key, value in submitted.items() if key != "files"}),
                [TaskFileInput.model_validate(item) for item in manifest], set(state.pdf_tiers),
            )
        except ApiServerError as error:
            return JSONResponse({"error": error.error.model_dump()}, status_code=error.status_code)
        task_id = f"cloud_{len(state.tasks) + 1}"
        assert all({"client_file_id", "file_id"}.isdisjoint(item) for item in manifest)
        manifest = [{**item, "file_id": f"file-{task_id}-{index + 1}"} for index, item in enumerate(manifest)]
        state.tasks[task_id] = {
            "request": submitted,
            "tier": selected_tier,
            "files": manifest,
            "ready": {item["file_id"] for item in manifest if item.get("sha256sum") in state.cache},
            "upload_version": 0,
            "download_version": 0,
            "expired": set(),
            "expires_at": int(time.time()) + 3600,
            "output_formats": submitted["output_formats"],
        }
        detail = state.payload(task_id)
        return JSONResponse(
            {
                "task_id": detail["task_id"],
                "status": detail["status"],
                "files": [
                    {key: value for key, value in item.items() if key in {"file_id", "file_name", "status", "upload"}}
                    for item in detail["files"]
                ],
            },
            status_code=202,
        )

    @gateway.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str) -> Response:
        """仅返回元数据和 OSS 下载链接，不回源读取或代理 ZIP 内容。"""
        return JSONResponse(state.payload(task_id))

    @gateway.get("/redirect/{task_id}/{file_id}")
    async def redirect(task_id: str, file_id: str) -> Response:
        """模拟同源下载地址重定向到同主机不同端口，检查客户端凭据隔离。"""
        return RedirectResponse(
            f"{state.storage_url}/download/{task_id}/{file_id}?signature={state.tasks[task_id]['download_version']}",
            status_code=302,
        )

    @storage.middleware("http")
    async def no_api_key(request: Request, call_next: Any) -> Response:
        """记录实际对象存储请求，MinerU API Key 不得跨源发送。"""
        state.requests.append(("storage", request.method, request.url.path, request.headers.get("authorization")))
        assert request.headers.get("authorization") is None
        return await call_next(request)

    @storage.put("/objects/{task_id}/{file_id}")
    async def upload(task_id: str, file_id: str, request: Request) -> Response:
        """只在对象存储接收文件字节，并模拟临时错误、链接过期和提交后响应丢失。"""
        assert request.headers.get("x-upload-token") == "storage-token"
        key = task_id + "/" + file_id
        state.upload_attempts[key] = state.upload_attempts.get(key, 0) + 1
        record = state.tasks[task_id]
        if request.query_params.get("signature") != str(record["upload_version"]):
            return Response(status_code=403)
        if state.expired_upload and record["upload_version"] == 0:
            record["expired"].add(file_id)
            return Response(status_code=410)
        if state.transient_upload and state.upload_attempts[key] == 1:
            return Response(status_code=503)
        data = await request.body()
        item = next(item for item in record["files"] if item["file_id"] == file_id)
        assert len(data) == item.get("bytes")
        actual = hashlib.sha256(data).hexdigest()
        assert item.get("sha256sum") in {None, actual}
        state.cache[actual] = data
        record["ready"].add(file_id)
        if state.lost_upload_response:
            return Response(status_code=503)
        return Response(status_code=200)

    @storage.get("/download/{task_id}/{file_id}")
    async def download(task_id: str, file_id: str, request: Request) -> Response:
        """结果字节仅由对象存储返回；过期地址通过再次查询原任务刷新。"""
        record = state.tasks[task_id]
        if request.query_params.get("signature") != str(record["download_version"]):
            return Response(status_code=403)
        if state.expired_download and record["download_version"] == 0:
            record["download_version"] = 1
            return Response(status_code=403)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("markdown.md", f"# OSS result {file_id}\n")
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


@pytest.mark.parametrize("tier", [None, "basic", "advanced"])
def test_example_script_forwards_optional_tier(cloud: _CloudState, tmp_path: Path, tier: str | None) -> None:
    """执行真实示例脚本，验证缺省不发送 tier、显式选档不由客户端替换。"""
    cloud.pdf_tiers = TIERS
    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF-script-protocol")
    env = _script_env(tmp_path, cloud.gateway_url)
    if tier is None:
        env.pop("MINERU_TIER")
    else:
        env["MINERU_TIER"] = tier
    result = _run_script(env, source)
    assert result.returncode == 0, result.stdout + result.stderr
    body = cloud.tasks["cloud_1"]["request"]
    if tier is None:
        assert "tier" not in body
    else:
        assert body["tier"] == tier


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
    assert _members((tmp_path / "results/file-cloud_1-1/a.zip").read_bytes())["markdown.md"] == b"# OSS result file-cloud_1-1\n"


def test_saas_standard_only_with_native_and_oss(cloud: _CloudState) -> None:
    """SaaS 仅发布 Standard，混合请求仍逐文件返回 Standard/Flash 并从 OSS 下载。"""
    with httpx.Client(base_url=cloud.gateway_url, headers={"Authorization": "Bearer test-key"}, trust_env=False) as client:
        assert [item["id"] for item in client.get("/v1/tiers").json()["data"]] == ["standard"]
        assert "native_extensions" not in client.get("/v1/health").json()["features"]
        for tier in ("flash", "basic", "advanced"):
            rejected = client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}], "tier": tier})
            assert rejected.status_code == 400 and rejected.json()["error"]["param"] == "tier"
        assert not cloud.tasks
        receipt = client.post("/v1/tasks", json={
            "files": [{"file_name": name, "bytes": 1} for name in ("test.pdf", "test.docx")],
            "output_formats": ["default"],
        })
        assert receipt.status_code == 202, receipt.text
        for item in receipt.json()["files"]:
            upload = item["upload"]
            response = httpx.put(upload["url"], content=b"x", headers=upload["headers"], trust_env=False)
            assert response.status_code == 200
        result = client.get("/v1/tasks/" + receipt.json()["task_id"]).json()
        assert result["status"] == "completed"
        assert [item["parse"]["tier"] for item in result["files"]] == ["standard", "flash"]
        for item in result["files"]:
            download = httpx.get(item["result_url"], trust_env=False)
            assert download.status_code == 200 and "markdown.md" in _members(download.content)


@pytest.mark.parametrize("failure", ["transient_upload", "lost_upload_response", "expired_download", "api_download_redirect"])
def test_client_recovers_links_and_uncertain_uploads(cloud: _CloudState, tmp_path: Path, failure: str) -> None:
    """恢复原任务和刷新地址不创建新任务；跨源重定向不能转发 API Key。"""
    setattr(cloud, failure, True)
    source = tmp_path / "a.html"
    source.write_bytes(HTML)
    result = _run_script(_script_env(tmp_path, cloud.gateway_url), source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(cloud.tasks) == 1
    assert cloud.upload_attempts["cloud_1/file-cloud_1-1"] == (2 if failure == "transient_upload" else 1)
    assert all(auth is None for surface, _, _, auth in cloud.requests if surface == "storage")
    assert (tmp_path / "results/file-cloud_1-1/a.zip").is_file()


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
    assert not (tmp_path / "results/file-cloud_1-1/a.zip").exists()


@pytest.mark.parametrize("status,exit_code", [("partial", 2), ("failed", 1), ("running", 124)])
def test_client_exit_codes_and_poll_budget(cloud: _CloudState, tmp_path: Path, status: str, exit_code: int) -> None:
    """部分成功先保存 ZIP，失败不下载，等待预算耗尽保留原 task_id。"""
    cloud.forced_status = status
    source = tmp_path / "a.html"
    source.write_bytes(HTML)
    second = tmp_path / "b.html"
    second.write_bytes(HTML + b"<!--second-->")
    result = _run_script(_script_env(tmp_path, cloud.gateway_url, MAX_POLLS="2"), source, second)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert len(cloud.tasks) == 1 and "task_id=cloud_1" in result.stderr
    assert (tmp_path / "results/file-cloud_1-1/a.zip").exists() == (status == "partial")
    if status == "running":
        assert "continues running" in result.stderr


def test_client_does_not_renew_expired_upload(cloud: _CloudState, tmp_path: Path) -> None:
    """过期上传是该文件的终态，查询不能换发链接或重新创建任务。"""
    cloud.expired_upload = True
    source = tmp_path / "a.html"
    source.write_bytes(HTML)
    result = _run_script(_script_env(tmp_path, cloud.gateway_url), source)
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(cloud.tasks) == 1 and cloud.upload_attempts == {"cloud_1/file-cloud_1-1": 1}
    assert not (tmp_path / "results/file-cloud_1-1/a.zip").exists()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sdk_hash_cache_skips_second_upload(live_service: _LiveService, tmp_path: Path, asynchronous: bool) -> None:
    """真实 SDK 经 TCP 解析相同文件两次，第二次命中源缓存且没有任何额外 PUT 字节。"""
    from mineru.parser import MinerUApiParser

    source = tmp_path / "sdk.html"
    source.write_bytes(HTML)
    parser = MinerUApiParser(api_url=live_service.url, api_key="test-key", tier="flash", include_images=True)
    first = asyncio.run(parser.parse_async(source)) if asynchronous else parser.parse(source)
    uploaded = sum(live_service.counts.values())
    second = asyncio.run(parser.parse_async(source)) if asynchronous else parser.parse(source)
    assert uploaded == len(HTML) and sum(live_service.counts.values()) == uploaded
    assert first.to_dict() == second.to_dict()
    assert first.markdown() == second.markdown() and "Same source and output" in first.markdown()


@pytest.mark.parametrize("document", ["pdf", "html"])
def test_real_gradio_handler_uses_tasks_and_saves_artifacts(live_service: _LiveService, tmp_path: Path, document: str) -> None:
    """真实界面转换处理器经过 TCP Tasks，检查 PDF 页码、HTML 正文和实际下载文件。"""
    from mineru.kit.gradio.app import build_gradio_app
    from mineru.kit.gradio.client import V1ArtifactClient
    from mineru.kit.gradio.artifacts import render_download
    from mineru.parser.base import ParseResult

    if document == "pdf":
        source = Path(__file__).resolve().parents[2] / "demo/pdfs/demo1.pdf"
        assert source.is_file(), "The real PDF fixture is required for the migration smoke test"
    else:
        source = tmp_path / "gradio.html"
        source.write_bytes(HTML)
    client = V1ArtifactClient(api_url=live_service.url, api_key="test-key")
    capabilities = asyncio.run(client.discover())
    demo = build_gradio_app(client, capabilities, output_root=tmp_path / "gradio", enable_example=False)
    try:
        handler = next(function.fn for function in demo.fns.values() if function.name == "convert_handler")
        converted = asyncio.run(handler(str(source), 0, "1" if document == "pdf" else "", False))
        state = converted[6]
        assert state, converted[0]
        restored = ParseResult.from_json(Path(state["middle_json_path"]).read_text())
        assert len(restored.pages) == 1 and restored.markdown().strip()
        assert restored.pages[0].page_idx == 0
        assert Path(render_download(state, "docx")).is_file()
        if document == "html":
            assert "Same source and output" in restored.markdown()
        assert sum(live_service.counts.values()) == source.stat().st_size
    finally:
        demo.close()
