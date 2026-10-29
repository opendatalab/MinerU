"""本地 API、Router、SaaS/对象存储的真实 TCP 同步包装验收。"""

from __future__ import annotations

import io
import socket
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from test_file_parse import FORMATS, Upstream, artifact_bytes, assert_no_upstream_cancellation
from test_http_api_example_script import _Server
from test_task_http import HTML, _LiveService, live_service as tasks_live_service

from mineru.kit.file_parse import FileParseSettings, create_app

live_service = tasks_live_service


def assert_clean(directory: Path) -> None:
    """TCP 客户端可能先收到末块数据，给服务器的响应 finally 一个有界收尾窗口。"""
    deadline = time.monotonic() + 2
    while list(directory.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not list(directory.iterdir())


@contextmanager
def wrapper(tmp_path: Path, base_url: str, key: str | None) -> Iterator[tuple[httpx.Client, FastAPI]]:
    """启动真正的包装 HTTP 服务，统一测试只改变上游地址和认证。"""
    app = create_app(FileParseSettings(base_url, key, tmp_path, poll_interval=0.02))
    server = _Server(app)
    server.start()
    try:
        with httpx.Client(base_url=server.url, timeout=90, trust_env=False) as client:
            yield client, app
    finally:
        server.stop()


def parse_file(
    client: httpx.Client,
    request_id: str,
    filename: str,
    content: bytes,
    formats: list[str],
) -> httpx.Response:
    """从调用者端发送单文件 multipart，请求格式遵循标准重复字段编码。"""
    return client.post(
        "/v1/file_parse",
        files=[
            ("file", (filename, content)),
            ("request_id", (None, request_id)),
            ("tier", (None, "flash")),
            ("ocr_mode", (None, "txt")),
            *(("output_formats", (None, item)) for item in formats),
        ],
    )


def test_api_and_router_all_formats(live_service: _LiveService, tmp_path: Path) -> None:
    """真实解析与导出全部格式组合，包装返回的字节必须和上游保存的产物相同。"""
    with wrapper(tmp_path / "wrapper", live_service.url, "test-key") as (client, app):
        with httpx.Client(headers={"Authorization": "Bearer test-key"}, timeout=90, trust_env=False) as upstream:
            for index, formats in enumerate(FORMATS):
                response = parse_file(client, f"formats_{index}", "source.html", HTML, formats)
                assert response.status_code == 200, response.text
                task_id = response.headers["x-task-id"]
                task = upstream.get(f"{live_service.url}/v1/tasks/{task_id}").json()
                direct = upstream.get(task["files"][0]["result_url"])
                assert direct.status_code == 200
                assert response.content == direct.content
                assert response.headers["content-type"] == direct.headers["content-type"]
        assert_clean(app.state.service.store.results)
        assert_clean(app.state.service.store.sources)


def cloud_bridge(upstream: Upstream) -> FastAPI:
    """通过独立网络端口暴露模拟 SaaS 协议，数据传输仍走真实 HTTP。"""
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT"])
    async def relay(request: Request) -> Response:
        """只桥接 HTTP，不为包装客户端增加 Jobs、Files 或存储路径假设。"""
        response = await upstream.handle(
            httpx.Request(
                request.method,
                str(request.url),
                headers=request.headers,
                content=await request.body(),
            )
        )
        return Response(response.content, status_code=response.status_code, headers=dict(response.headers))

    return app


def test_saas_storage_all_formats_and_recovery(tmp_path: Path) -> None:
    """API 路径前缀、独立存储端口、重定向和签名刷新均使用同一包装实现。"""
    state = Upstream()
    api, storage = _Server(cloud_bridge(state)), _Server(cloud_bridge(state))
    state.api_url, state.storage_url = api.url + "/api", storage.url
    api.start()
    storage.start()
    try:
        with wrapper(tmp_path, state.api_url, "secret") as (client, app):
            app.state.service.client.retry_delay = 0
            state.redirect = True
            for index, formats in enumerate(FORMATS):
                state.download_expired = True
                response = parse_file(client, f"cloud_{index}", "source.pdf", b"source", formats)
                assert response.status_code == 200, response.text
                assert response.content == artifact_bytes(formats)
            assert_clean(app.state.service.store.results)
            assert_clean(app.state.service.store.sources)
            source = tmp_path / "example.pdf"
            source.write_bytes(b"source")
            output = tmp_path / "example.docx"
            script = Path(__file__).resolve().parents[2] / "scripts" / "http_file_parse_example.py"
            common = [sys.executable, str(script), str(source), "--base-url", str(client.base_url)]
            completed = subprocess.run(
                [*common, "--output-format", "docx", "--output", str(output), "--request-id", "example-success"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            assert output.read_bytes() == artifact_bytes(["docx"])
            state.complete = False
            waiting = subprocess.run(
                [*common, "--wait-timeout", "0", "--request-id", "example-waiting", "--output", str(tmp_path / "pending.zip")],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert waiting.returncode == 124, waiting.stdout + waiting.stderr
            assert "status_url" in waiting.stdout and not (tmp_path / "pending.zip").exists()
        assert len(state.tasks) == 17
        assert_no_upstream_cancellation(state)
    finally:
        storage.stop()
        api.stop()


def illustrated_pdf() -> bytes:
    """生成包含文本、原生图片和图注的真实 PDF，验证附件传输不改变解析产物。"""
    picture = Image.new("RGB", (240, 140), "white")
    drawing = ImageDraw.Draw(picture)
    drawing.rectangle((10, 20, 100, 120), fill="navy")
    drawing.ellipse((130, 20, 230, 120), fill="orange")
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=(595, 842))
    canvas.setFont("Helvetica", 18)
    canvas.drawString(72, 760, "File Parse round trip")
    canvas.setFont("Helvetica", 12)
    canvas.drawString(72, 720, "The same document must reach every download client unchanged.")
    canvas.drawImage(ImageReader(picture), 100, 420, width=360, height=210)
    canvas.drawString(100, 395, "Figure 1. A blue rectangle and an orange circle.")
    canvas.drawString(72, 350, "The figure is embedded in the PDF as a native image.")
    canvas.save()
    return buffer.getvalue()


def test_real_pdf_attachment_matches_upstream(live_service: _LiveService, tmp_path: Path) -> None:
    """真实 PDF 经 API 与 Router 导出四格式后，附件和上游产物逐字节一致且包含图片。"""
    pdf = illustrated_pdf()
    (tmp_path / "illustrated.pdf").write_bytes(pdf)
    with wrapper(tmp_path / "wrapper", live_service.url, "test-key") as (client, _):
        response = parse_file(client, "illustrated", "illustrated.pdf", pdf, ["default", "docx", "html", "latex"])
        assert response.status_code == 200, response.text
        (tmp_path / "wrapper-result.zip").write_bytes(response.content)
        with httpx.Client(headers={"Authorization": "Bearer test-key"}, timeout=90, trust_env=False) as upstream:
            task = upstream.get(f"{live_service.url}/v1/tasks/{response.headers['x-task-id']}").json()
            direct = upstream.get(task["files"][0]["result_url"])
        assert response.content == direct.content
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert "File Parse round trip" in archive.read("markdown.md").decode()
            assert {"document.docx", "document.html", "document.tex"}.issubset(archive.namelist())
            assert any(name.startswith("images/") and not name.endswith("/") for name in archive.namelist())
            assert b"data:image/" in archive.read("document.html")


def test_tcp_disconnect_keeps_accepted_request(tmp_path: Path) -> None:
    """调用者关闭真实 socket 后，完整收到的文件仍上传并可通过请求 ID 找回。"""
    state = Upstream()
    state.upload_delay = 0.2
    api, storage = _Server(cloud_bridge(state)), _Server(cloud_bridge(state))
    state.api_url, state.storage_url = api.url + "/api", storage.url
    api.start()
    storage.start()
    try:
        with wrapper(tmp_path, state.api_url, "secret") as (client, app):
            request = httpx.Request(
                "POST",
                str(client.base_url).rstrip("/") + "/v1/file_parse",
                files={
                    "file": ("source.pdf", b"source"),
                    "request_id": (None, "disconnected"),
                },
            )
            content = request.read()
            headers = "\r\n".join(f"{key}: {value}" for key, value in request.headers.items()).encode()
            address = client.base_url
            with socket.create_connection((address.host, address.port)) as sock:
                sock.sendall(b"POST /v1/file_parse HTTP/1.1\r\n" + headers + b"\r\n\r\n" + content)
                deadline = time.monotonic() + 3
                while "disconnected" not in app.state.service.records and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert "disconnected" in app.state.service.records
            deadline = time.monotonic() + 5
            while True:
                response = client.get("/v1/file_parse/disconnected")
                if response.json().get("status") == "completed" or time.monotonic() > deadline:
                    break
                time.sleep(0.02)
            assert response.json()["status"] == "completed"
            assert client.get(response.json()["result_url"]).content == artifact_bytes(["default"])
            assert state.uploads == 1 and len(state.tasks) == 1
        assert_no_upstream_cancellation(state)
    finally:
        storage.stop()
        api.stop()
