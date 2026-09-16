"""scripts/http_api_example.sh 的行为契约测试（真实 Flash 服务 + 模拟服务，无模型下载、无 GPU）。

覆盖计划验收点：
- 鉴权与同源规则（同源附加 Bearer、外部预签名地址不附加 MinerU Key、保留服务返回的上传头）
- HTTP 200 非法响应（null/缺失 ID、未知状态）不进入轮询
- 有界轮询（MAX_POLLS 耗尽退出 124 且输出 job_id）
- 终态退出码（0/2/3/4），partial 先保存成功产物再退出
- 下载 302 重定向的凭据隔离（跨 host 与同 host 不同端口均不转发 Authorization）
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

# Module-level FastAPI imports: the mock routes annotate `request: Request`, and
# `from __future__ import annotations` resolves string annotations against
# module globals, so function-local imports would break parameter recognition.
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "http_api_example.sh"

TERMINAL_STATUSES = {"completed", "partial", "failed", "canceled"}

REQUIRED_TOOLS = ("bash", "curl", "jq", "python3")

pytestmark = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in REQUIRED_TOOLS),
    reason="bash/curl/jq/python3 are required (CI installs them explicitly)",
)

HTML_DOC = (
    "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Doc</title></head>"
    "<body><h1>Heading</h1><p>Hello MinerU http example.</p></body></html>"
)


# ─── helpers ────────────────────────────────────────────────────────────────


class _Server:
    """Run a FastAPI app on a pre-picked localhost port in a daemon thread."""

    def __init__(self, app: Any, host: str = "127.0.0.1") -> None:
        import uvicorn

        with socket.socket() as sock:
            sock.bind((host, 0))
            self.port = sock.getsockname()[1]
        self.host = host
        self.url = f"http://{host}:{self.port}"
        self._server = uvicorn.Server(uvicorn.Config(app, host=host, port=self.port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout: float = 20.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("test server did not start within timeout")

    def stop(self, timeout: float = 10.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=timeout)


def _script_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """Environment for the script subprocess: isolated from the user's MINERU_* settings."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "TMPDIR": str(tmp_path),
        "MINERU_HOME": str(tmp_path / "mineru-home"),
        # Default to a tmp dir so downloads never leak into the pytest cwd (repo root);
        # an explicit OUTPUT_DIR in extra overrides this.
        "OUTPUT_DIR": str(tmp_path / "out"),
    }
    for key in ("LANG", "LC_ALL"):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update({key: str(value) for key, value in extra.items()})
    return env


def _run_script(env: dict[str, str], *args: str, timeout: float = 90.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


# ─── mock V1 service ────────────────────────────────────────────────────────


class _MockState:
    """Configurable responses plus a record of every received request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, str | None]] = []
        self.create_upload_response: dict[str, Any] | None = None
        self.put_status: int = 200
        self.create_job_response: dict[str, Any] | None = None
        self.job_statuses: list[str] = ["running", "completed"]
        self.terminal_files: list[dict[str, Any]] | None = None
        self.file_content: bytes = b"# MARKDOWN BYTES\n"
        self.file_redirect: str | None = None

    def record(self, method: str, path: str, request: Any) -> None:
        self.requests.append(
            {
                "method": method,
                "path": path,
                "authorization": request.headers.get("authorization"),
                "x_custom_upload": request.headers.get("x-custom-upload"),
            }
        )

    def next_job_status(self) -> str:
        if len(self.job_statuses) > 1:
            return self.job_statuses.pop(0)
        return self.job_statuses[0]

    def requests_to(self, method: str, prefix: str) -> list[dict[str, str | None]]:
        return [r for r in self.requests if r["method"] == method and str(r["path"]).startswith(prefix)]


COMPLETED_FILES: list[dict[str, Any]] = [
    {
        "file_id": "file-1",
        "name": "a.html",
        "status": "completed",
        "output_files": {
            "markdown": {"file_id": "out-md"},
            "zip": {"file_id": "out-zip"},
        },
    }
]

PARTIAL_FILES: list[dict[str, Any]] = [
    COMPLETED_FILES[0],
    {"file_id": "file-2", "name": "b.html", "status": "failed", "error": {"code": "parse_failed", "message": "boom"}},
]


def _build_mock_app(state: _MockState) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/uploads")
    async def create_upload(request: Request) -> Response:
        state.record("POST", "/v1/uploads", request)
        body = state.create_upload_response or {
            "id": "upload_1",
            "status": "completed",
            "upload_url": None,
            "file": {"id": "file-1", "object": "file", "bytes": 16, "filename": "a.html", "purpose": "parse"},
        }
        return JSONResponse(body)

    @app.put("/put-target")
    async def put_target(request: Request) -> Response:
        state.record("PUT", "/put-target", request)
        if state.put_status != 200:
            return JSONResponse({"error": {"code": "invalid_api_key"}}, status_code=state.put_status)
        return Response(status_code=200)

    @app.post("/v1/uploads/{upload_id}/complete")
    async def complete_upload(request: Request, upload_id: str) -> Response:
        state.record("POST", f"/v1/uploads/{upload_id}/complete", request)
        return JSONResponse(
            {
                "id": upload_id,
                "status": "completed",
                "upload_url": None,
                "file": {"id": "file-1", "object": "file", "bytes": 16, "filename": "a.html", "purpose": "parse"},
            }
        )

    @app.post("/v1/parse/jobs")
    async def create_job(request: Request) -> Response:
        state.record("POST", "/v1/parse/jobs", request)
        body = state.create_job_response or {"job_id": "job_1", "status": "queued"}
        return JSONResponse(body, status_code=202)

    @app.get("/v1/parse/jobs/{job_id}")
    async def get_job(request: Request, job_id: str) -> Response:
        state.record("GET", f"/v1/parse/jobs/{job_id}", request)
        status = state.next_job_status()
        if status in TERMINAL_STATUSES:
            return JSONResponse(
                {"job_id": job_id, "status": status, "files": state.terminal_files or COMPLETED_FILES}
            )
        return JSONResponse({"job_id": job_id, "status": status})

    @app.get("/v1/files/{file_id}/content")
    async def file_content(request: Request, file_id: str) -> Response:
        state.record("GET", f"/v1/files/{file_id}/content", request)
        if state.file_redirect:
            return RedirectResponse(state.file_redirect, status_code=302)
        return Response(content=state.file_content, media_type="text/markdown")

    @app.get("/bytes")
    async def redirected_bytes(request: Request) -> Response:
        state.record("GET", "/bytes", request)
        return Response(content=b"REDIRECTED BYTES", media_type="application/octet-stream")

    return app


@pytest.fixture()
def mock_pair():
    """Two mock services: (state_a, state_b, server_a, server_b). Same host, different ports."""
    state_a, state_b = _MockState(), _MockState()
    server_a = _Server(_build_mock_app(state_a))
    server_b = _Server(_build_mock_app(state_b))
    server_a.start()
    server_b.start()
    yield state_a, state_b, server_a, server_b
    server_a.stop()
    server_b.stop()


@pytest.fixture()
def doc(tmp_path: Path) -> Path:
    path = tmp_path / "doc.html"
    path.write_text(HTML_DOC, encoding="utf-8")
    return path


# ─── real Flash service (no model download, no GPU) ─────────────────────────


@pytest.fixture()
def real_server_factory(tmp_path: Path):
    from mineru.parser.api_server import create_app

    servers: list[_Server] = []

    def _make(api_key: str | None = None) -> _Server:
        app = create_app(
            tier="flash",
            preload_models=False,
            upload_dir=str(tmp_path / "uploads"),
            api_key=api_key,
        )
        server = _Server(app)
        server.start()
        servers.append(server)
        return server

    yield _make
    for server in servers:
        server.stop()


def test_real_service_anonymous_round_trip(real_server_factory, tmp_path: Path, doc: Path) -> None:
    server = real_server_factory()
    out = tmp_path / "out"
    proc = _run_script(_script_env(tmp_path, MINERU_API_URL=server.url, OUTPUT_DIR=str(out)), str(doc))
    assert proc.returncode == 0, _output(proc)
    # artifacts are saved under the uploaded file name plus the format extension
    assert (out / "doc.html.md").is_file()
    assert (out / "doc.html.zip").is_file()


def test_real_service_api_key_same_origin_upload(real_server_factory, tmp_path: Path, doc: Path) -> None:
    server = real_server_factory(api_key="secret-key")
    out = tmp_path / "out"
    proc = _run_script(
        _script_env(
            tmp_path,
            MINERU_API_URL=server.url,
            MINERU_API_KEY="secret-key",
            OUTPUT_DIR=str(out),
        ),
        str(doc),
    )
    assert proc.returncode == 0, _output(proc)
    assert (out / "doc.html.md").is_file()


def test_real_service_rejects_unauthenticated_upload_content(real_server_factory, tmp_path: Path, doc: Path) -> None:
    import httpx

    server = real_server_factory(api_key="secret-key")
    create = httpx.post(
        f"{server.url}/v1/uploads",
        headers={"Authorization": "Bearer secret-key"},
        json={"filename": "doc.html", "bytes": len(HTML_DOC.encode()), "mime_type": "text/html", "purpose": "parse"},
        timeout=10,
    )
    assert create.status_code == 200
    upload = create.json()
    assert upload["status"] == "pending"
    unauthenticated = httpx.put(upload["upload_url"], content=HTML_DOC.encode(), timeout=10)
    assert unauthenticated.status_code == 401


def test_real_service_deduplicated_upload_reused(real_server_factory, tmp_path: Path, doc: Path) -> None:
    server = real_server_factory()
    out = tmp_path / "out"
    env = _script_env(tmp_path, MINERU_API_URL=server.url, OUTPUT_DIR=str(out))
    first = _run_script(env, str(doc))
    assert first.returncode == 0, _output(first)
    second = _run_script(env, str(doc))
    assert second.returncode == 0, _output(second)
    assert "deduplicated" in _output(second)


# ─── mock service: protocol and exit-code contract ──────────────────────────


def test_mock_upload_401_stops_before_complete_and_jobs(mock_pair, tmp_path: Path, doc: Path) -> None:
    state_a, _, server_a, _ = mock_pair
    state_a.create_upload_response = {
        "id": "upload_1",
        "status": "pending",
        "upload_url": f"{server_a.url}/put-target",
        "upload_method": "PUT",
        "upload_headers": {"Content-Type": "application/octet-stream"},
    }
    state_a.put_status = 401
    proc = _run_script(_script_env(tmp_path, MINERU_API_URL=server_a.url), str(doc))
    assert proc.returncode == 1
    assert "byte upload failed" in _output(proc)
    assert state_a.requests_to("POST", "/v1/uploads/upload_1/complete") == []
    assert state_a.requests_to("POST", "/v1/parse/jobs") == []
    assert state_a.requests_to("GET", "/v1/parse/jobs/") == []


@pytest.mark.parametrize(
    ("statuses", "terminal_files", "expected_code"),
    [
        (["completed"], COMPLETED_FILES, 0),
        (["running", "completed"], COMPLETED_FILES, 0),
        (["partial"], PARTIAL_FILES, 2),
        (["failed"], COMPLETED_FILES, 3),
        (["canceled"], COMPLETED_FILES, 4),
    ],
)
def test_mock_terminal_exit_codes(
    mock_pair, tmp_path: Path, doc: Path, statuses: list[str], terminal_files: list[dict[str, Any]], expected_code: int
) -> None:
    state_a, _, server_a, _ = mock_pair
    state_a.job_statuses = list(statuses)
    state_a.terminal_files = terminal_files
    state_a.create_upload_response = {
        "id": "upload_1",
        "status": "completed",
        "upload_url": None,
        "file": {"id": "file-1", "object": "file", "bytes": 16, "filename": "a.html", "purpose": "parse"},
    }
    out = tmp_path / "out"
    proc = _run_script(
        _script_env(tmp_path, MINERU_API_URL=server_a.url, OUTPUT_DIR=str(out), POLL_INTERVAL="0"),
        str(doc),
    )
    assert proc.returncode == expected_code, _output(proc)
    if expected_code in (0, 2):
        # partial must save the artifacts of completed files before exiting
        assert (out / "a.html.md").is_file()
        assert (out / "a.html.zip").is_file()


def test_mock_polling_budget_exhaustion(mock_pair, tmp_path: Path, doc: Path) -> None:
    state_a, _, server_a, _ = mock_pair
    state_a.job_statuses = ["running"]
    state_a.create_upload_response = {
        "id": "upload_1",
        "status": "completed",
        "upload_url": None,
        "file": {"id": "file-1", "object": "file", "bytes": 16, "filename": "a.html", "purpose": "parse"},
    }
    proc = _run_script(
        _script_env(
            tmp_path,
            MINERU_API_URL=server_a.url,
            MAX_POLLS="2",
            POLL_INTERVAL="0",
        ),
        str(doc),
    )
    assert proc.returncode == 124
    assert "job_id=job_1" in _output(proc)
    assert "/v1/parse/jobs/job_1" in _output(proc)


@pytest.mark.parametrize(
    ("create_job_body",),
    [
        ({"job_id": None, "status": "queued"},),
        ({"status": "queued"},),
        ({"job_id": "job_1", "status": "exploded"},),
    ],
    ids=["null-job-id", "missing-job-id", "unknown-status"],
)
def test_mock_invalid_create_job_response_never_polls(
    mock_pair, tmp_path: Path, doc: Path, create_job_body: dict[str, Any]
) -> None:
    state_a, _, server_a, _ = mock_pair
    state_a.create_upload_response = {
        "id": "upload_1",
        "status": "completed",
        "upload_url": None,
        "file": {"id": "file-1", "object": "file", "bytes": 16, "filename": "a.html", "purpose": "parse"},
    }
    state_a.create_job_response = create_job_body
    proc = _run_script(
        _script_env(tmp_path, MINERU_API_URL=server_a.url, MINERU_API_KEY="secret", POLL_INTERVAL="0"),
        str(doc),
    )
    assert proc.returncode == 1
    assert state_a.requests_to("GET", "/v1/parse/jobs/") == []


def test_mock_completed_but_missing_artifact_fails(mock_pair, tmp_path: Path, doc: Path) -> None:
    state_a, _, server_a, _ = mock_pair
    state_a.job_statuses = ["completed"]
    state_a.terminal_files = [
        {"file_id": "file-1", "name": "a.html", "status": "completed", "output_files": {"markdown": {"file_id": "out-md"}}}
    ]
    state_a.create_upload_response = {
        "id": "upload_1",
        "status": "completed",
        "upload_url": None,
        "file": {"id": "file-1", "object": "file", "bytes": 16, "filename": "a.html", "purpose": "parse"},
    }
    proc = _run_script(
        _script_env(
            tmp_path,
            MINERU_API_URL=server_a.url,
            OUTPUT_FORMATS="markdown,zip",
            POLL_INTERVAL="0",
        ),
        str(doc),
    )
    assert proc.returncode == 1
    assert "missing" in _output(proc)


# ─── mock service: same-origin credential isolation ─────────────────────────


def _pending_upload(upload_url: str) -> dict[str, Any]:
    return {
        "id": "upload_1",
        "status": "pending",
        "upload_url": upload_url,
        "upload_method": "PUT",
        "upload_headers": {"Content-Type": "application/octet-stream", "x-custom-upload": "token123"},
    }


def test_mock_external_port_upload_keeps_service_headers_without_api_key(mock_pair, tmp_path: Path, doc: Path) -> None:
    """Same host, different port counts as cross-origin: no MinerU API key, service headers kept."""
    state_a, state_b, server_a, server_b = mock_pair
    state_a.create_upload_response = _pending_upload(f"{server_b.url}/put-target")
    state_a.job_statuses = ["completed"]
    proc = _run_script(
        _script_env(
            tmp_path,
            MINERU_API_URL=server_a.url,
            MINERU_API_KEY="secret",
            OUTPUT_DIR=str(tmp_path / "out"),
            POLL_INTERVAL="0",
        ),
        str(doc),
    )
    assert proc.returncode == 0, _output(proc)
    create_calls = state_a.requests_to("POST", "/v1/uploads")
    assert create_calls and create_calls[0]["authorization"] == "Bearer secret"
    put_calls = state_b.requests_to("PUT", "/put-target")
    assert put_calls, "byte upload did not reach the external origin"
    assert put_calls[0]["authorization"] is None
    assert put_calls[0]["x_custom_upload"] == "token123"


def test_mock_external_host_upload_without_api_key(mock_pair, tmp_path: Path, doc: Path) -> None:
    """Different hostname (127.0.0.1 vs localhost) is cross-origin: no MinerU API key."""
    state_a, state_b, server_a, server_b = mock_pair
    state_a.create_upload_response = _pending_upload(f"http://localhost:{server_b.port}/put-target")
    state_a.job_statuses = ["completed"]
    proc = _run_script(
        _script_env(tmp_path, MINERU_API_URL=server_a.url, MINERU_API_KEY="secret", POLL_INTERVAL="0"),
        str(doc),
    )
    assert proc.returncode == 0, _output(proc)
    put_calls = state_b.requests_to("PUT", "/put-target")
    assert put_calls
    assert put_calls[0]["authorization"] is None
    assert put_calls[0]["x_custom_upload"] == "token123"


def test_mock_relative_upload_url_is_same_origin(mock_pair, tmp_path: Path, doc: Path) -> None:
    state_a, _, server_a, _ = mock_pair
    state_a.create_upload_response = _pending_upload("/put-target")
    state_a.job_statuses = ["completed"]
    proc = _run_script(
        _script_env(tmp_path, MINERU_API_URL=server_a.url, MINERU_API_KEY="secret", POLL_INTERVAL="0"),
        str(doc),
    )
    assert proc.returncode == 0, _output(proc)
    put_calls = state_a.requests_to("PUT", "/put-target")
    assert put_calls
    assert put_calls[0]["authorization"] == "Bearer secret"
    assert put_calls[0]["x_custom_upload"] == "token123"


# ─── mock service: download redirects do not leak credentials ───────────────


@pytest.mark.parametrize("redirect_target", ["cross-port", "cross-host"])
def test_mock_download_redirect_isolation(mock_pair, tmp_path: Path, doc: Path, redirect_target: str) -> None:
    state_a, state_b, server_a, server_b = mock_pair
    if redirect_target == "cross-port":
        redirect_url = f"{server_b.url}/bytes"
    else:
        redirect_url = f"http://localhost:{server_b.port}/bytes"
    state_a.create_upload_response = {
        "id": "upload_1",
        "status": "completed",
        "upload_url": None,
        "file": {"id": "file-1", "object": "file", "bytes": 16, "filename": "a.html", "purpose": "parse"},
    }
    state_a.job_statuses = ["completed"]
    state_a.file_redirect = redirect_url
    out = tmp_path / "out"
    proc = _run_script(
        _script_env(
            tmp_path,
            MINERU_API_URL=server_a.url,
            MINERU_API_KEY="secret",
            OUTPUT_DIR=str(out),
            OUTPUT_FORMATS="markdown",
            POLL_INTERVAL="0",
        ),
        str(doc),
    )
    assert proc.returncode == 0, _output(proc)
    assert (out / "a.html.md").read_bytes() == b"REDIRECTED BYTES"
    redirected = state_b.requests_to("GET", "/bytes")
    assert redirected
    assert redirected[0]["authorization"] is None
