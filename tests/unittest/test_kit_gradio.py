from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageStat
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from mineru.filetypes import FLASH_ONLY_PARSE_EXTENSIONS, IMAGE_EXTENSIONS, PARSEABLE_EXTENSIONS
from mineru.kit.commands import gradio as gradio_command
from mineru.kit.gradio import app as gradio_app
from mineru.kit.gradio import client as gradio_client
from mineru.kit.gradio.app import build_gradio_app
from mineru.kit.gradio.artifacts import create_run_artifacts, persist_parse_result, render_download
from mineru.kit.gradio.client import (
    GradioArtifactClient,
    ManagedLocalApiServer,
    STATUS_DOWNLOADING_RESULT,
    V1ArtifactClient,
    V1ArtifactError,
    V1ServerCapabilities,
    normalize_v1_base_url,
)
from mineru.kit.main import app
from mineru.parser import api_client as parser_api_client
from mineru.parser import api_server as parser_api_server
from mineru.parser.base import ParseResult
from mineru.types import BlockType, ImageBlock, ImageBodyBlock, MiddleJson, ModelJson, PageInfo, TextBlock, TextSpan
from mineru.version import __version__
from typer.main import get_command
from typer.testing import CliRunner

runner = CliRunner()


def _pdf_bytes(page_count: int = 1) -> bytes:
    """生成测试所需的最小多页 PDF。"""
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=300)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _colored_pdf_bytes(colors: list[tuple[float, float, float]]) -> bytes:
    """生成每页使用不同纯色背景的 PDF，用于验证裁页映射。"""
    output = io.BytesIO()
    painter = canvas.Canvas(output, pagesize=(120, 160))
    for red, green, blue in colors:
        painter.setFillColorRGB(red, green, blue)
        painter.rect(0, 0, 120, 160, stroke=0, fill=1)
        painter.showPage()
    painter.save()
    return output.getvalue()


def _middle_json(
    *,
    with_image: bool = True,
    file_suffix: str = "pdf",
    page_indices: tuple[int, ...] = (0,),
) -> MiddleJson:
    """构造包含文字和可选视觉块的严格 Middle JSON。"""
    pages: list[PageInfo] = []
    for page_idx in page_indices:
        blocks: list[Any] = [
            TextBlock(
                type=BlockType.TEXT,
                index=page_idx * 10,
                bbox=(0.1, 0.1, 0.8, 0.2) if file_suffix == "pdf" else None,
                content=[TextSpan(type="text", content=f"hello-{page_idx}")],
            )
        ]
        if with_image:
            image_index = page_idx * 10 + 1
            image_body = ImageBodyBlock(
                type=BlockType.IMAGE_BODY,
                index=image_index,
                bbox=(0.1, 0.3, 0.8, 0.8) if file_suffix == "pdf" else None,
                content="",
            )
            blocks.append(
                ImageBlock(
                    type=BlockType.IMAGE,
                    index=image_index,
                    bbox=(0.1, 0.3, 0.8, 0.8) if file_suffix == "pdf" else None,
                    content=[image_body],
                )
            )
        pages.append(PageInfo(page_idx=page_idx, blocks=blocks))
    return MiddleJson(
        pages=pages,
        is_full_document=page_indices == tuple(range(len(page_indices))),
        file_suffix=file_suffix,  # type: ignore[arg-type]
        effort="flash",
        parse_mode="txt",
        mineru_version=__version__,
    )


def _httpx_proxy_for_test_client(
    test_client: TestClient,
    request_log: list[tuple[str, str, str | None]],
) -> SimpleNamespace:
    """把异步 httpx 调用桥接到已启动 lifespan 的 FastAPI TestClient。"""

    class AsyncClientAdapter:
        """为 V1 async client 提供最小的 TestClient 异步适配层。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """忽略真实网络参数并复用当前 TestClient。"""

        async def __aenter__(self) -> "AsyncClientAdapter":
            """进入异步上下文。"""
            return self

        async def __aexit__(self, *_args: object) -> None:
            """退出异步上下文，不关闭外层 TestClient。"""

        async def get(self, url: str, **kwargs: Any) -> httpx.Response:
            """通过 TestClient 执行 GET。"""
            return self._request("GET", url, **kwargs)

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            """通过 TestClient 执行 POST。"""
            return self._request("POST", url, **kwargs)

        async def put(self, url: str, **kwargs: Any) -> httpx.Response:
            """通过 TestClient 执行 PUT。"""
            return self._request("PUT", url, **kwargs)

        def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
            """记录鉴权信息并把绝对 URL 转换为 ASGI 路径。"""
            headers = kwargs.get("headers") or {}
            parsed = httpx.URL(url)
            request_log.append((method, parsed.path, headers.get("Authorization")))
            return test_client.request(method, parsed.raw_path.decode("ascii"), **kwargs)

    return SimpleNamespace(
        AsyncClient=AsyncClientAdapter,
        Client=httpx.Client,
        HTTPError=httpx.HTTPError,
        InvalidURL=httpx.InvalidURL,
        Response=httpx.Response,
        Timeout=httpx.Timeout,
        TimeoutException=httpx.TimeoutException,
        TransportError=httpx.TransportError,
        URL=httpx.URL,
    )


def test_gradio_command_is_registered_and_help_is_available() -> None:
    """验证 mineru-kit 根命令注册 Gradio 且帮助不启动服务。"""
    result = runner.invoke(app, ["--help"])
    gradio_result = runner.invoke(app, ["gradio", "--help"])
    assert result.exit_code == 0
    assert gradio_result.exit_code == 0
    assert "gradio" in result.output
    assert "--api-url" in gradio_result.output
    assert "--api-server-tier" in gradio_result.output
    assert "Disable Advanced on" not in gradio_result.output
    assert "Disable Flash on" not in gradio_result.output


def test_gradio_managed_tier_disable_option_names_are_removed() -> None:
    """验证禁用档位的参数完整删除，不仅从帮助中隐藏。"""
    root_command = get_command(app)
    command = root_command.get_command(None, "gradio")
    assert command is not None
    options = {parameter.name: tuple(parameter.opts) for parameter in command.params}

    assert "api_server_no_flash" not in options
    assert "api_server_no_advanced" not in options


@pytest.mark.parametrize(
    "flag",
    [
        "--api-server-no-flash",
        "--no-flash",
        "--api-server-no-advanced",
        "--no-advanced",
    ],
)
@pytest.mark.parametrize("standalone", [False, True])
def test_gradio_rejects_removed_tier_disable_options(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    standalone: bool,
) -> None:
    """验证两个命令入口在启动服务前拒绝所有已移除的禁用档位参数。"""
    launch = Mock()
    monkeypatch.setattr(gradio_app, "launch_gradio", launch)

    if standalone:
        monkeypatch.setattr(sys, "argv", ["mineru-gradio", flag])
        with pytest.raises(SystemExit) as error:
            gradio_command.main()
        assert error.value.code == 2
    else:
        result = runner.invoke(app, ["gradio", flag])
        assert result.exit_code == 2, result.output
        assert "No such option" in result.output
    launch.assert_not_called()


@pytest.mark.parametrize(("port_args", "expected_port"), [([], None), (["--server-port", "7861"], 7861)])
def test_gradio_command_preserves_automatic_or_explicit_port(
    monkeypatch: pytest.MonkeyPatch, port_args: list[str], expected_port: int | None
) -> None:
    """验证省略端口时保留 Gradio 原生自动选择，显式端口则优先于环境变量。"""
    launch = Mock()
    monkeypatch.setenv("GRADIO_SERVER_PORT", "17860")
    monkeypatch.setattr(gradio_app, "launch_gradio", launch)
    result = runner.invoke(app, ["gradio", *port_args])
    assert result.exit_code == 0, result.output
    launch.assert_called_once()
    assert launch.call_args.kwargs["server_port"] == expected_port


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_gradio_command_rejects_invalid_explicit_port(monkeypatch: pytest.MonkeyPatch, port: int) -> None:
    """验证显式端口仍遵循 TCP 范围约束，并在启动服务前报错。"""
    launch = Mock()
    monkeypatch.setattr(gradio_app, "launch_gradio", launch)
    result = runner.invoke(app, ["gradio", "--server-port", str(port)])
    assert result.exit_code == 1
    assert "server_port must be between 1 and 65535" in result.output
    launch.assert_not_called()


def test_importing_kit_cli_does_not_import_optional_gradio_runtime() -> None:
    """验证查看通用 CLI 时不会提前导入 Gradio 或 gradio-pdf。"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mineru.kit.main; assert 'gradio' not in sys.modules; assert 'gradio_pdf' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_gradio_command_reports_optional_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证未安装 Gradio 时命令给出可执行安装建议。"""
    original_find_spec = gradio_command.importlib.util.find_spec

    def fake_find_spec(name: str) -> Any:
        """只模拟 Gradio 相关模块缺失，保留其他模块探测。"""
        if name in {"gradio", "gradio_pdf"}:
            return None
        return original_find_spec(name)

    monkeypatch.setattr(gradio_command.importlib.util, "find_spec", fake_find_spec)
    result = runner.invoke(app, ["gradio"])
    assert result.exit_code == 1
    assert "mineru[gradio]" in result.output


def test_gradio_url_normalization_and_capability_dataclass() -> None:
    """验证 V1 base URL 规范化及能力快照字段。"""
    assert normalize_v1_base_url("127.0.0.1:8000/") == "http://127.0.0.1:8000"
    with pytest.raises(V1ArtifactError, match="must not be empty"):
        normalize_v1_base_url("   ")
    with pytest.raises(V1ArtifactError, match="credentials, query, or fragment"):
        normalize_v1_base_url("https://user:secret@example.test/api")
    with pytest.raises(V1ArtifactError, match="Invalid V1 API URL"):
        normalize_v1_base_url("http://example.test:not-a-port")
    capabilities = V1ServerCapabilities("http://example.test", ("flash",), ("zip",), ("file_id",))
    assert capabilities.tiers == ("flash",)
    assert capabilities.output_formats == ("zip",)


def test_v1_client_uses_environment_api_key_when_cli_value_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 API Key 的环境变量回退不会要求页面输入密钥。"""
    monkeypatch.setenv("MINERU_API_KEY", "from-env")
    client = V1ArtifactClient(api_url="http://example.test")
    assert client.api_key == "from-env"


def test_v1_client_discovers_health_and_tiers_and_sends_api_key() -> None:
    """验证能力发现只访问新的 V1 endpoint，并携带 Bearer API Key。"""
    requests: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """返回最小健康和 tier 响应。"""
        requests.append((request.method, request.url.path))
        assert request.headers["authorization"] == "Bearer secret"
        if request.url.path == "/api/v1/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "features": {
                        "output_formats": ["markdown", "zip"],
                        "sources": ["file_id", "inline"],
                    },
                },
                request=request,
            )
        if request.url.path == "/api/v1/tiers":
            return httpx.Response(200, json={"data": [{"id": "flash"}]}, request=request)
        return httpx.Response(404, request=request)

    client = V1ArtifactClient(
        api_url="http://example.test/api",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )
    capabilities = asyncio.run(client.discover())
    assert capabilities.base_url == "http://example.test/api"
    assert capabilities.tiers == ("flash",)
    assert requests == [("GET", "/api/v1/health"), ("GET", "/api/v1/tiers")]


def test_v1_client_discovers_api_server_without_advanced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证真实 API server 禁用 Advanced 后，Gradio 能力快照不会重新补回该档位。"""
    monkeypatch.setattr(parser_api_server, "_preflight_tier_dependencies", Mock())
    api = parser_api_server.create_app(
        upload_dir=str(tmp_path / "api"),
        tier="standard",
        no_advanced=True,
    )
    request_log: list[tuple[str, str, str | None]] = []

    with TestClient(api, base_url="http://testserver") as test_client:
        proxy = _httpx_proxy_for_test_client(test_client, request_log)
        monkeypatch.setattr(gradio_client, "httpx", proxy)
        capabilities = asyncio.run(V1ArtifactClient(api_url="http://testserver").discover())

    assert capabilities.tiers == ("flash", "basic", "standard")
    assert [path for _method, path, _authorization in request_log] == ["/v1/health", "/v1/tiers"]


def test_v1_client_rejects_server_without_zip() -> None:
    """验证缺少 ZIP 能力时在 UI 启动前给出稳定错误。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        """返回没有 ZIP 的健康 payload。"""
        if request.url.path.endswith("/health"):
            return httpx.Response(
                200,
                json={"status": "ok", "features": {"output_formats": ["middle_json"], "sources": ["file_id"]}},
                request=request,
            )
        return httpx.Response(200, json={"data": [{"id": "flash"}]}, request=request)

    client = V1ArtifactClient(api_url="http://example.test", transport=httpx.MockTransport(handler))
    with pytest.raises(V1ArtifactError, match="does not advertise zip"):
        asyncio.run(client.discover())


@pytest.mark.parametrize(
    ("sources", "tiers", "error_code"),
    [
        (("inline", "url"), ("flash",), "unsupported_source"),
        (("file_id",), ("experimental",), "tier_unavailable"),
    ],
)
def test_v1_client_rejects_missing_upload_or_tier_capability(
    sources: tuple[str, ...],
    tiers: tuple[str, ...],
    error_code: str,
) -> None:
    """验证能力发现会拒绝无法完成上传或没有受支持 tier 的服务。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        """返回参数化的最小 V1 能力响应。"""
        if request.url.path.endswith("/health"):
            return httpx.Response(
                200,
                json={"status": "ok", "features": {"output_formats": ["zip"], "sources": list(sources)}},
                request=request,
            )
        return httpx.Response(200, json={"data": [{"id": tier} for tier in tiers]}, request=request)

    client = V1ArtifactClient(api_url="http://example.test", transport=httpx.MockTransport(handler))
    with pytest.raises(V1ArtifactError) as error:
        asyncio.run(client.discover())
    assert error.value.code == error_code


def test_v1_client_checks_requested_tier_after_first_discovery(tmp_path: Path) -> None:
    """验证首次 discover 后也会立即拒绝服务未提供的请求 tier。"""
    source = tmp_path / "demo.pdf"
    source.write_bytes(_pdf_bytes())

    async def handler(request: httpx.Request) -> httpx.Response:
        """只声明 Flash tier。"""
        if request.url.path.endswith("/health"):
            return httpx.Response(
                200,
                json={"status": "ok", "features": {"output_formats": ["zip"], "sources": ["file_id"]}},
                request=request,
            )
        return httpx.Response(200, json={"data": [{"id": "flash"}]}, request=request)

    client = V1ArtifactClient(api_url="http://example.test", transport=httpx.MockTransport(handler))
    with pytest.raises(V1ArtifactError) as error:
        asyncio.run(client.parse_file(source, tier="standard", page_range=""))
    assert error.value.code == "tier_unavailable"


def test_v1_client_full_asgi_upload_job_poll_and_zip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """通过真实 V1 ASGI 路由验证上传、任务、轮询、ZIP 与 API Key 链路。"""
    source = tmp_path / "demo.pdf"
    source.write_bytes(_pdf_bytes())
    parse_calls: list[dict[str, Any]] = []

    async def fake_parse_async(path: str, **kwargs: Any) -> ParseResult:
        """替代模型推理，同时保留 server 的真实 job 和打包流程。"""
        parse_calls.append({"path": path, **kwargs})
        return ParseResult(middle_json=_middle_json(with_image=False), _model_output={"raw": "model"})

    monkeypatch.setattr(parser_api_server, "parse_async", fake_parse_async)
    api = parser_api_server.create_app(
        upload_dir=str(tmp_path / "api"),
        tier="flash",
        api_key="secret",
    )
    request_log: list[tuple[str, str, str | None]] = []
    with TestClient(api, base_url="http://testserver") as test_client:
        proxy = _httpx_proxy_for_test_client(test_client, request_log)
        monkeypatch.setattr(gradio_client, "httpx", proxy)
        monkeypatch.setattr(parser_api_client, "httpx", proxy)
        client = V1ArtifactClient(api_url="http://testserver", api_key="secret")
        result = asyncio.run(client.parse_file(source, tier="flash", page_range="1"))

    paths = [path for _method, path, _authorization in request_log]
    assert "/v1/health" in paths
    assert "/v1/tiers" in paths
    assert "/v1/uploads" in paths
    assert any(path.startswith("/v1/uploads/upload_") and path.endswith("/content") for path in paths)
    assert "/v1/parse/jobs" in paths
    assert any(path.startswith("/v1/files/file-") and path.endswith("/content") for path in paths)
    protected_requests = [item for item in request_log if item[1] not in {"/v1/health", "/v1/tiers"}]
    assert protected_requests and all(authorization == "Bearer secret" for _method, _path, authorization in protected_requests)
    assert result._model_output == {"raw": "model"}
    assert parse_calls[0]["page_range"] == "1"


def test_gradio_routes_real_v1_jobs_and_isolates_remote_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """贯通两套真实 V1 路由，验证本地 Flash 不上传远程且环境密钥只用于远程任务。"""
    source = tmp_path / "report.pdf"
    source.write_bytes(_pdf_bytes())
    monkeypatch.setenv("MINERU_API_KEY", "remote-secret")
    monkeypatch.setattr(parser_api_server, "_preflight_tier_dependencies", Mock())

    async def fake_parse_async(path: str, **kwargs: Any) -> ParseResult:
        """只替换模型推理，保留两套服务各自的上传、任务和 ZIP 打包流程。"""
        return ParseResult(middle_json=_middle_json(with_image=False), _model_output={"raw": "model"})

    monkeypatch.setattr(parser_api_server, "parse_async", fake_parse_async)
    remote_api = parser_api_server.create_app(
        upload_dir=str(tmp_path / "remote"), tier="standard", no_flash=True, no_advanced=True, api_key="remote-secret"
    )
    local_api = parser_api_server.create_app(upload_dir=str(tmp_path / "local"), tier="flash")
    requests: list[tuple[str, str, str, str | None]] = []
    with TestClient(remote_api, base_url="http://remote.test") as remote_http:
        with TestClient(local_api, base_url="http://local.test") as local_http:

            async def handle(request: httpx.Request) -> httpx.Response:
                """按目标主机转入对应 ASGI 服务，记录完整请求目的地和鉴权头。"""
                host = request.url.host
                requests.append((host, request.method, request.url.path, request.headers.get("authorization")))
                target = {"remote.test": remote_http, "local.test": local_http}[host]
                response = target.request(
                    request.method,
                    str(request.url),
                    headers=dict(request.headers),
                    content=request.content,
                )
                return httpx.Response(response.status_code, headers=response.headers, content=response.content)

            def async_client(**kwargs: Any) -> httpx.AsyncClient:
                """为能力发现和 API parser 注入同一双服务传输层。"""
                kwargs["transport"] = httpx.MockTransport(handle)
                return httpx.AsyncClient(**kwargs)

            proxy = SimpleNamespace(**{**vars(httpx), "AsyncClient": async_client})
            monkeypatch.setattr(gradio_client, "httpx", proxy)
            monkeypatch.setattr(parser_api_client, "httpx", proxy)
            remote = V1ArtifactClient(api_url="http://remote.test")
            local = V1ArtifactClient(api_url="http://local.test", api_key="")
            remote_capabilities = asyncio.run(remote.discover())
            local_capabilities = asyncio.run(local.discover())
            routed = GradioArtifactClient(remote, local_flash=local)
            assert routed.capabilities.tiers == ("flash", "basic", "standard")
            assert remote.capabilities is remote_capabilities
            assert remote_capabilities.tiers == ("basic", "standard")
            assert local.capabilities is local_capabilities
            requests.clear()

            flash_result = asyncio.run(routed.parse_file(source, tier="flash", page_range=""))
            assert flash_result._model_output == {"raw": "model"}
            assert requests and all(host == "local.test" and auth is None for host, _method, _path, auth in requests)
            assert any(method == "POST" and path == "/v1/parse/jobs" for _host, method, path, _auth in requests)
            requests.clear()

            remote_result = asyncio.run(routed.parse_file(source, tier="standard", page_range="1"))
            assert remote_result._model_output == {"raw": "model"}
            assert requests and all(
                host == "remote.test" and auth == "Bearer remote-secret" for host, _method, _path, auth in requests
            )
            assert any(method == "POST" and path == "/v1/parse/jobs" for _host, method, path, _auth in requests)


@pytest.mark.parametrize("failure", [None, V1ArtifactError("remote failed", code="parse_failed"), asyncio.CancelledError()])
def test_gradio_router_preserves_remote_flash_results_errors_and_cancellation(failure: BaseException | None) -> None:
    """验证远程已有 Flash 时优先远程，并原样传递结果、失败和取消而不重试本地。"""
    result = ParseResult(middle_json=_middle_json(with_image=False))
    primary = Mock(spec=V1ArtifactClient)
    primary.capabilities = V1ServerCapabilities("http://remote.test", ("standard", "flash"), ("zip",), ("file_id",))
    primary.parse_file = AsyncMock(return_value=result, side_effect=failure)
    local = Mock(spec=V1ArtifactClient)
    routed = GradioArtifactClient(primary, local_flash=local)
    callback = Mock()
    request = routed.parse_file(Path("report.pdf"), tier="flash", page_range="1-2", ocr_mode="ocr", status_callback=callback)
    if failure is None:
        assert asyncio.run(request) is result
    else:
        with pytest.raises(type(failure)) as error:
            asyncio.run(request)
        assert error.value is failure
    primary.parse_file.assert_awaited_once_with(
        Path("report.pdf"), tier="flash", page_range="1-2", ocr_mode="ocr", status_callback=callback
    )
    local.parse_file.assert_not_called()


@pytest.mark.parametrize(("tier", "code"), [("advanced", "tier_unavailable"), ("unknown", "invalid_request")])
def test_gradio_router_rejects_unavailable_tiers(tier: str, code: str) -> None:
    """验证路由层只补充 Flash，拒绝未声明的 Advanced 及未知档位。"""
    primary = Mock(spec=V1ArtifactClient)
    primary.capabilities = V1ServerCapabilities("http://remote.test", ("standard",), ("zip",), ("file_id",))
    local = Mock(spec=V1ArtifactClient)
    local.capabilities = V1ServerCapabilities("http://local.test", ("flash", "advanced"), ("zip",), ("file_id",))
    routed = GradioArtifactClient(primary, local_flash=local)
    assert routed.capabilities.tiers == ("flash", "standard")
    with pytest.raises(V1ArtifactError) as error:
        asyncio.run(routed.parse_file(Path("report.pdf"), tier=tier, page_range=""))
    assert error.value.code == code
    primary.parse_file.assert_not_called()
    local.parse_file.assert_not_called()


def test_api_client_upload_auth_is_limited_to_same_origin() -> None:
    """验证 Bearer Key 会用于同源上传，但不会泄露给外部预签名地址。"""
    same_origin = parser_api_client._same_origin_upload_headers(
        "https://api.example.test/base",
        "https://api.example.test/v1/uploads/1/content",
        {"Content-Type": "application/pdf"},
        {"Authorization": "Bearer secret"},
    )
    external = parser_api_client._same_origin_upload_headers(
        "https://api.example.test/base",
        "https://storage.example.test/upload",
        {"Content-Type": "application/pdf"},
        {"Authorization": "Bearer secret"},
    )
    assert same_origin["Authorization"] == "Bearer secret"
    assert "Authorization" not in external


def test_gradio_ocr_reaches_analysis_through_real_v1_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """贯通 UI、上传、V1 任务与统一解析器，仅替换分析推理并检查连续请求之间的模式隔离。"""
    from mineru.parser import mineru_parser

    source = tmp_path / "report.PDF"
    source.write_bytes(_pdf_bytes())
    received_modes: list[str] = []

    async def fake_analyze(file_bytes: bytes, **kwargs: Any) -> tuple[MiddleJson, ModelJson]:
        """记录真正传入分析层的模式，并生成与模式一致的严格结果和模型输出。"""
        assert file_bytes.startswith(b"%PDF")
        received_modes.append(kwargs["parse_mode"])
        middle = _middle_json(with_image=False)
        middle.parse_mode = "ocr" if kwargs["parse_mode"] == "ocr" else "txt"
        model = ModelJson(
            pages=[[]],
            page_index_map=[],
            file_suffix="pdf",
            effort="flash",
            parse_mode=middle.parse_mode,
            mineru_version=__version__,
        )
        return middle, model

    monkeypatch.setattr(mineru_parser, "aio_doc_analyze", fake_analyze)
    api = parser_api_server.create_app(upload_dir=str(tmp_path / "api"), tier="flash")
    with TestClient(api, base_url="http://testserver") as test_client:
        request_log: list[tuple[str, str, str | None]] = []
        proxy = _httpx_proxy_for_test_client(test_client, request_log)
        monkeypatch.setattr(gradio_client, "httpx", proxy)
        monkeypatch.setattr(parser_api_client, "httpx", proxy)
        client = V1ArtifactClient(api_url="http://testserver")

        async def convert_requests() -> None:
            """同一 Gradio 实例依次强制、自动、强制解析，再通过缺省 SDK 请求验证默认模式。"""
            capabilities = await client.discover()
            demo = build_gradio_app(client, capabilities, output_root=tmp_path / "output", enable_example=False)
            handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")
            for enabled in (True, False, True):
                updates = [update async for update in handler(str(source), 0, "", enabled)]
                state = updates[-1][8]
                assert state is not None, updates[-1][0]
                payload = json.loads(Path(state["middle_json_path"]).read_text())
                assert payload["parse_mode"] == ("ocr" if enabled else "txt")
            default_parser = parser_api_client.MinerUApiParser(api_url="http://testserver", tier="flash")
            result = await default_parser.parse_async(source)
            assert result.middle_json.parse_mode == "txt"

        asyncio.run(convert_requests())
        assert not hasattr(api.state, "ocr_mode")
    assert received_modes == ["ocr", "auto", "ocr", "auto"]
    assert sum(method == "POST" and path == "/v1/parse/jobs" for method, path, _ in request_log) == 4


def test_v1_client_preserves_v1_error_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 V1 error envelope 会保留错误码并清晰传给 UI。"""
    source = tmp_path / "demo.pdf"
    source.write_bytes(_pdf_bytes())

    class FailingParser:
        """模拟 API parser 返回带错误码的失败。"""

        def __init__(self, **_kwargs: Any) -> None:
            """忽略构造参数。"""

        async def parse_async(self, _path: Path, *, page_range: str, status_callback: Any = None) -> ParseResult:
            """抛出与正式 API client 一致的错误形态。"""
            error = RuntimeError(f"bad range: {page_range}")
            error.code = "page_range_invalid"  # type: ignore[attr-defined]
            raise error

    monkeypatch.setattr(gradio_client, "MinerUApiParser", FailingParser)
    client = V1ArtifactClient(api_url="http://example.test")
    client._capabilities = V1ServerCapabilities("http://example.test", ("flash",), ("zip",), ("file_id",))
    with pytest.raises(V1ArtifactError) as error:
        asyncio.run(client.parse_file(source, tier="flash", page_range="999"))
    assert error.value.code == "page_range_invalid"


@pytest.mark.parametrize("ocr_mode", ["auto", "txt", "ocr"])
def test_v1_client_parse_uses_api_parser_zip_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ocr_mode: str) -> None:
    """验证 Gradio client 固定请求带图片和模型输出的 V1 ZIP 结果。"""
    source = tmp_path / "demo.pdf"
    source.write_bytes(_pdf_bytes())
    calls: dict[str, Any] = {}

    class FakeParser:
        """记录 V1 parser 构造和页码参数。"""

        def __init__(self, **kwargs: Any) -> None:
            """保存构造参数，供断言 V1 ZIP 契约。"""
            calls.update(kwargs)

        async def parse_async(self, path: Path, *, page_range: str, status_callback: Any = None) -> ParseResult:
            """返回最小解析结果。"""
            calls["path"] = path
            calls["page_range"] = page_range
            assert statuses == ["Preparing request...", "Submitting task..."]
            assert status_callback is not None
            status_callback("queued")
            status_callback("running")
            status_callback("completed")
            assert statuses[-1] == STATUS_DOWNLOADING_RESULT
            return ParseResult(middle_json=_middle_json(with_image=False))

    monkeypatch.setattr("mineru.kit.gradio.client.MinerUApiParser", FakeParser)
    client = V1ArtifactClient(api_url="http://example.test")
    client._capabilities = V1ServerCapabilities(
        "http://example.test",
        ("standard",),
        ("zip",),
        ("file_id",),
    )
    statuses: list[str] = []
    result = asyncio.run(
        client.parse_file(source, tier="standard", page_range="1-2", ocr_mode=ocr_mode, status_callback=statuses.append)
    )
    assert isinstance(result, ParseResult)
    assert calls["include_images"] is True
    assert calls["include_model_output"] is True
    assert calls["tier"] == "standard"
    assert calls["ocr_mode"] == ocr_mode
    assert calls["page_range"] == "1-2"
    assert statuses[0] == "Preparing request..."
    assert statuses[-1] == STATUS_DOWNLOADING_RESULT
    assert statuses[2:4] == ["Queued on server", "Processing on server..."]


def test_managed_local_api_command_uses_kit_entrypoint() -> None:
    """验证托管 server 使用正式 mineru-kit api-server 命令。"""
    server = ManagedLocalApiServer(
        tier="standard",
        concurrency=3,
        language="en",
        disable_image_analysis=True,
        preload_models=True,
        api_key="secret",
    )
    command = server._command(8123, Path("/tmp/mineru-upload"))
    assert command[:4] == [server._command(8123, Path("/tmp/mineru-upload"))[0], "-m", "mineru.kit.main", "api-server"]
    assert "--tier" in command and command[command.index("--tier") + 1] == "standard"
    assert "--no-flash" not in command
    assert "--no-advanced" not in command
    assert "--preload-models" in command
    assert "--ocr-mode" not in command
    assert command[command.index("--api-key") + 1] == "secret"
    with pytest.raises(TypeError, match="ocr_mode"):
        ManagedLocalApiServer(ocr_mode="ocr")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="api_server_ocr_mode"):
        gradio_app.launch_gradio(api_server_ocr_mode="ocr")  # type: ignore[call-arg]
    for option in ("no_flash", "no_advanced"):
        with pytest.raises(TypeError, match=option):
            ManagedLocalApiServer(**{option: True})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match=f"api_server_{option}"):
            gradio_app.launch_gradio(**{f"api_server_{option}": True})  # type: ignore[arg-type]


def test_managed_local_api_server_cleans_process_control_and_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证托管 server 启停会使用控制通道并清理上传临时目录。"""
    captured: dict[str, Any] = {}

    class FakeControl:
        """记录 ManagedProcessControl 生命周期。"""

        def start_accepting(self) -> None:
            """记录控制通道开始监听。"""
            captured["accepting"] = True

        def child_env(self) -> dict[str, str]:
            """返回测试专用子进程环境。"""
            return {"MINERU_PROCESS_CONTROL": "test"}

        def request_shutdown(self, timeout_sec: float) -> bool:
            """记录优雅关闭请求。"""
            captured["shutdown_timeout"] = timeout_sec
            return True

        def close(self) -> None:
            """记录控制通道关闭。"""
            captured["closed"] = True

    class FakeProcess:
        """模拟能被优雅关闭的 API server 子进程。"""

        def __init__(self) -> None:
            """初始化运行态。"""
            self.returncode: int | None = None

        def poll(self) -> int | None:
            """返回当前进程状态。"""
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            """记录等待并结束模拟进程。"""
            captured["wait_timeout"] = timeout
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            """记录降级终止。"""
            captured["terminated"] = True

        def kill(self) -> None:
            """记录强制终止。"""
            captured["killed"] = True

    fake_control = FakeControl()

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        """记录托管子进程命令和环境。"""
        captured["command"] = command
        captured["popen_kwargs"] = kwargs
        return FakeProcess()

    def fake_create_control() -> FakeControl:
        """返回测试控制通道。"""
        return fake_control

    def fake_wait_until_ready(_server: ManagedLocalApiServer) -> None:
        """跳过真实 HTTP 健康检查。"""

    monkeypatch.setattr(gradio_client.ManagedProcessControl, "create", fake_create_control)
    monkeypatch.setattr(gradio_client.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ManagedLocalApiServer, "_wait_until_ready", fake_wait_until_ready)

    server = ManagedLocalApiServer(tier="flash")
    server.start()
    assert server._temp_dir is not None
    temp_root = Path(server._temp_dir.name)
    assert temp_root.is_dir()
    assert "--allow-local-source" not in captured["command"]
    assert "--no-advanced" not in captured["command"]
    assert captured["accepting"] is True

    server.stop()

    assert not temp_root.exists()
    assert captured["closed"] is True
    assert captured["shutdown_timeout"] == 3.0
    assert "terminated" not in captured


def test_persist_and_render_all_gradio_download_formats(tmp_path: Path) -> None:
    """验证严格 Middle JSON、布局 PDF 和六种下载产物。"""
    source = tmp_path / "报告.pdf"
    source.write_bytes(_pdf_bytes())
    result = ParseResult(middle_json=_middle_json(), _model_output={"raw": "model"})
    artifacts = persist_parse_result(result, source, output_root=tmp_path / "output", page_range="")

    assert artifacts.middle_json_path.read_text(encoding="utf-8").find('"schema_version": "2.0"') >= 0
    assert artifacts.origin_pdf_path == artifacts.root / "origin.pdf"
    assert artifacts.layout_pdf_path == artifacts.root / "layout.pdf"
    assert artifacts.origin_pdf_path is not None and artifacts.origin_pdf_path.is_file()
    assert artifacts.layout_pdf_path is not None and artifacts.layout_pdf_path.is_file()
    assert artifacts.bundle_zip_path.is_file()
    assert list((artifacts.root / "images").glob("*.jpg"))
    with zipfile.ZipFile(artifacts.bundle_zip_path) as archive:
        assert {
            "source.pdf",
            "middle_json.json",
            "markdown.md",
            "structured_content.json",
            "model_output.json",
            "origin.pdf",
            "layout.pdf",
        }.issubset(archive.namelist())

    html_path = Path(render_download(artifacts.as_state(), "html", allowed_root=tmp_path / "output"))
    docx_path = Path(render_download(artifacts.as_state(), "docx", allowed_root=tmp_path / "output"))
    latex_path = Path(render_download(artifacts.as_state(), "latex", allowed_root=tmp_path / "output"))
    epub_path = Path(render_download(artifacts.as_state(), "epub", allowed_root=tmp_path / "output"))
    pdf_path = Path(render_download(artifacts.as_state(), "pdf", allowed_root=tmp_path / "output"))
    assert "<!doctype html>" in html_path.read_text(encoding="utf-8").lower()
    assert "data:image/jpeg;base64," in html_path.read_text(encoding="utf-8")
    assert docx_path.read_bytes().startswith(b"PK")
    assert epub_path.read_bytes().startswith(b"PK")
    assert pdf_path.read_bytes().startswith(b"%PDF")
    with zipfile.ZipFile(latex_path) as archive:
        assert f"{artifacts.stem}.tex" in archive.namelist()
        assert any(name.startswith("images/") for name in archive.namelist())
    layout_page = PdfReader(str(artifacts.layout_pdf_path)).pages[0]
    layout_stream = layout_page.get_contents()
    assert layout_stream is not None
    layout_commands = layout_stream.get_data()
    assert b"20 240 140 30 re" in layout_commands
    assert b"20 60 140 150 re" in layout_commands


def test_persist_image_input_creates_real_origin_and_layout_pdfs(tmp_path: Path) -> None:
    """验证图片输入会生成可读取的 origin/layout PDF，而不是伪装扩展名。"""
    source = tmp_path / "scan.png"
    Image.new("RGB", (64, 48), "white").save(source)
    artifacts = persist_parse_result(
        ParseResult(middle_json=_middle_json(with_image=False)),
        source,
        output_root=tmp_path / "output",
        page_range="",
    )
    assert artifacts.origin_pdf_path is not None
    assert artifacts.layout_pdf_path is not None
    assert artifacts.origin_pdf_path.read_bytes().startswith(b"%PDF")
    assert artifacts.layout_pdf_path.read_bytes().startswith(b"%PDF")
    assert len(PdfReader(str(artifacts.origin_pdf_path)).pages) == 1
    assert len(PdfReader(str(artifacts.layout_pdf_path)).pages) == 1


def test_page_range_image_crops_follow_original_page_indices(tmp_path: Path) -> None:
    """验证抽页后视觉块仍从对应原始页裁图，而不是错位到相邻页。"""
    source = tmp_path / "colors.pdf"
    source.write_bytes(_colored_pdf_bytes([(1, 0, 0), (0, 1, 0), (0, 0, 1)]))
    artifacts = persist_parse_result(
        ParseResult(middle_json=_middle_json(page_indices=(1, 2))),
        source,
        output_root=tmp_path / "output",
        page_range="2-3",
    )
    images = sorted((artifacts.root / "images").glob("*.jpg"))
    assert len(images) == 2
    with Image.open(images[0]) as first_image, Image.open(images[1]) as second_image:
        first_mean = ImageStat.Stat(first_image.convert("RGB")).mean
        second_mean = ImageStat.Stat(second_image.convert("RGB")).mean
    assert first_mean[1] > first_mean[0] + 100 and first_mean[1] > first_mean[2] + 100
    assert second_mean[2] > second_mean[0] + 100 and second_mean[2] > second_mean[1] + 100


def test_persist_page_range_keeps_cropped_origin_page_order(tmp_path: Path) -> None:
    """验证任意 V1 page_range 会同步应用到 origin 和 layout PDF。"""
    source = tmp_path / "three-pages.pdf"
    source.write_bytes(_pdf_bytes(page_count=3))
    artifacts = persist_parse_result(
        ParseResult(middle_json=_middle_json(with_image=False, page_indices=(1, 2))),
        source,
        output_root=tmp_path / "output",
        page_range="2-3",
    )
    assert artifacts.page_indices == (1, 2)
    assert artifacts.origin_pdf_path is not None
    assert len(PdfReader(str(artifacts.origin_pdf_path)).pages) == 2
    assert artifacts.layout_pdf_path is not None
    assert len(PdfReader(str(artifacts.layout_pdf_path)).pages) == 2


def test_run_artifact_paths_are_absolute_and_inside_output_root(tmp_path: Path) -> None:
    """验证每次任务使用独立绝对目录并固定核心产物命名。"""
    source = tmp_path / "复杂 文件名.pdf"
    first = create_run_artifacts(source, tmp_path / "output")
    second = create_run_artifacts(source, tmp_path / "output")
    assert first.root != second.root
    assert first.root.is_absolute()
    assert first.source_path.name == "source.pdf"
    assert first.middle_json_path.name == "middle_json.json"
    assert first.markdown_path.name == "markdown.md"
    assert first.structured_content_path.name == "structured_content.json"
    for path in (
        first.source_path,
        first.middle_json_path,
        first.markdown_path,
        first.structured_content_path,
        first.bundle_zip_path,
        first.downloads_dir,
    ):
        assert path.is_absolute()
        assert path.is_relative_to(first.root)


def test_render_download_rejects_state_outside_allowed_root(tmp_path: Path) -> None:
    """验证浏览器可控 State 不能读取 output root 之外的文件。"""
    source = tmp_path / "demo.pdf"
    source.write_bytes(_pdf_bytes())
    artifacts = persist_parse_result(
        ParseResult(middle_json=_middle_json(with_image=False)),
        source,
        output_root=tmp_path / "output",
        page_range="",
    )
    state = artifacts.as_state()
    state["root"] = str(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes output root"):
        render_download(state, "pdf", allowed_root=tmp_path / "output")

    state = artifacts.as_state()
    state["stem"] = "../../escaped"
    with pytest.raises(ValueError, match="Invalid Gradio artifact stem"):
        render_download(state, "html", allowed_root=tmp_path / "output")
    assert not (tmp_path / "escaped.html").exists()


def test_gradio_file_types_page_range_and_header_follow_new_contract(tmp_path: Path) -> None:
    """验证完整扩展名、PDF-only 页码规则和复用后的 Header。"""
    assert set(gradio_app._supported_file_types()) == {f".{extension}" for extension in PARSEABLE_EXTENSIONS}
    source = tmp_path / "report.PDF"
    source.write_bytes(_pdf_bytes(5))
    assert gradio_app._effective_page_range(source, " 1-3,r1 ", tier="standard") == "1-3,r1"
    assert gradio_app._effective_page_range(source, " 1-3,r1 ", tier="flash") == ""
    assert gradio_app._effective_page_range("report.docx", "1-3", tier="standard") == ""
    header = gradio_app._render_header(gradio_major_version=6)
    assert "mineru-gradio6-header" in header
    assert "mineru-header-popover mineru-model-popover" in header
    assert "mineru-header-popover mineru-paper-popover" in header
    assert "{{HEADER_" not in header


def test_gradio_ocr_control_visibility_reset_and_event_binding(tmp_path: Path) -> None:
    """验证 PDF 专属开关的缺省值、文件重置、清除绑定和独立于 tier 的可见性。"""
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("flash", "standard"), ("zip",), ("file_id",))
    demo = build_gradio_app(Mock(), capabilities, output_root=tmp_path, enable_example=False)
    checkbox = next(block for block in demo.blocks.values() if block.__class__.__name__ == "Checkbox")
    assert checkbox.label == "强制 OCR" and checkbox.value is False and checkbox.visible is False
    update = next(fn for fn in demo.fns.values() if fn.name == "update_ocr_control")
    assert len(update.inputs) == 1 and update.inputs[0].__class__.__name__ == "File"
    for path, visible in (("first.pdf", True), ("second.PDF", True), ("photo.png", False), ("book.docx", False), (None, False)):
        assert update.fn(path) == {"__type__": "update", "value": False, "visible": visible}
    convert = next(fn for fn in demo.fns.values() if fn.name == "convert_handler")
    assert convert.inputs[-1] is checkbox
    slider = next(block for block in demo.blocks.values() if "mineru-tier-slider" in (block.elem_classes or []))
    clear = next(block for block in demo.blocks.values() if block.__class__.__name__ == "ClearButton")
    events = demo.config["dependencies"]
    assert all(checkbox._id not in event["outputs"] for event in events if (slider._id, "input") in event["targets"])
    assert any(checkbox._id in event["outputs"] for event in events if (clear._id, "click") in event["targets"])
    change = next(event for event in events if event["id"] == update._id)
    assert change["queue"] is False and change["trigger_mode"] == "always_last"


@pytest.mark.parametrize("suffix", [".png", ".docx", ".csv"])
def test_gradio_non_pdf_ignores_hidden_force_ocr(tmp_path: Path, suffix: str) -> None:
    """直接调用事件并传入残留 True，非 PDF 请求仍使用 auto。"""
    source = tmp_path / f"source{suffix}"
    source.write_bytes(b"placeholder")
    client = SimpleNamespace(parse_file=AsyncMock(side_effect=V1ArtifactError("stop after request")))
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("zip",), ("file_id",))
    demo = build_gradio_app(client, capabilities, output_root=tmp_path / "output", enable_example=False)
    handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")

    async def convert() -> None:
        """消费完整转换事件，以等待参数进入 client。"""
        updates = [update async for update in handler(str(source), 0, "1-3", True)]
        assert "stop after request" in updates[-1][0]

    asyncio.run(convert())
    client.parse_file.assert_awaited_once()
    assert client.parse_file.call_args.kwargs["ocr_mode"] == "auto"
    assert client.parse_file.call_args.kwargs["page_range"] == ""


def test_build_gradio_app_exposes_three_tabs_and_download_menu(tmp_path: Path) -> None:
    """验证新 UI 不引入旧 backend 控件，并注册三个标签和下载事件。"""
    capabilities = V1ServerCapabilities(
        "http://127.0.0.1:1",
        ("flash", "basic", "standard", "advanced"),
        ("markdown", "middle_json", "structured_content", "zip"),
        ("file_id", "url", "inline"),
    )
    app = build_gradio_app(
        V1ArtifactClient(api_url=capabilities.base_url),
        capabilities,
        output_root=tmp_path,
        enable_example=False,
    )
    tab_labels = [component.label for component in app.blocks.values() if component.__class__.__name__ == "Tab"]
    assert tab_labels == ["Markdown 渲染", "Markdown 源码", "Structured Content 源码"]
    download_labels = [component.label for component in app.blocks.values() if component.__class__.__name__ == "DownloadButton"]
    assert download_labels == ["ZIP", "HTML", "DOCX", "LaTeX bundle", "EPUB", "PDF"]
    download_buttons = [component for component in app.blocks.values() if component.__class__.__name__ == "DownloadButton"]
    assert all(component.interactive is False for component in download_buttons)
    file_inputs = [component for component in app.blocks.values() if component.__class__.__name__ == "File"]
    assert len(file_inputs) == 1
    assert file_inputs[0].file_count == "single"
    assert set(file_inputs[0].file_types) == {f".{extension}" for extension in PARSEABLE_EXTENSIONS}
    preview_handler = next(fn.fn for fn in app.fns.values() if fn.name == "update_file_preview")
    assert preview_handler("report.pdf")[0] == {"__type__": "update", "value": "report.pdf", "visible": True}
    assert preview_handler("report.docx")[0]["visible"] is False
    range_group = next(block for block in app.blocks.values() if "mineru-kit-page-range" in (block.elem_classes or []))
    assert range_group.visible is True
    assert '[data-range-visible="true"]' in app._mineru_kit_css
    assert sum(1 for dependency in app.config["dependencies"] if dependency.get("queue") is True) >= 7


@pytest.mark.parametrize(
    ("tiers", "default_position", "default_tier", "maximum"),
    [
        (("advanced", "flash", "standard", "basic"), 2, "standard", 3),
        (("advanced", "standard", "basic"), 1, "standard", 2),
        (("flash", "basic", "standard"), 2, "standard", 2),
        (("advanced", "flash"), 1, "advanced", 1),
        (("basic", "flash"), 1, "basic", 1),
        (("flash",), 0, "flash", 1),
        (("advanced",), 0, "advanced", 1),
    ],
)
def test_gradio_tier_slider_uses_available_tiers_and_preserves_selection(
    tmp_path: Path, tiers: tuple[str, ...], default_position: int, default_tier: str, maximum: int
) -> None:
    """验证乱序、缺档及单档位下的原生滑块默认值与文件切换、清除行为。"""
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", tiers, ("zip",), ("file_id",))
    demo = build_gradio_app(
        V1ArtifactClient(api_url=capabilities.base_url), capabilities, output_root=tmp_path, enable_example=False
    )
    slider = next(block for block in demo.blocks.values() if block.__class__.__name__ == "Slider")
    label = next(block for block in demo.blocks.values() if "mineru-tier-label" in (block.elem_classes or []))
    assert (slider.minimum, slider.maximum, slider.step, slider.precision) == (0, maximum, 1, 0)
    assert slider.value == default_position
    assert slider.interactive is (len(tiers) > 1)
    assert label.value == f"解析 tier：{default_tier}"
    assert not any(block.__class__.__name__ == "Dropdown" for block in demo.blocks.values())
    label_events = [event for event in demo.config["dependencies"] if (slider._id, "input") in event["targets"]]
    assert len(label_events) == 1
    label_event = label_events[0]
    assert label_event["backend_fn"] is False
    assert label_event["queue"] is False
    assert label_event["trigger_mode"] == "always_last"
    assert label_event["outputs"][7:9] == [slider._id, label._id]
    preference = demo.blocks[label_event["outputs"][9]]
    assert preference.visible is False
    assert json.loads(preference.value) == {"tier": default_tier, "locked": False}
    assert preference._id in label_event["inputs"]
    assert json.dumps(sorted(FLASH_ONLY_PARSE_EXTENSIONS)) in label_event["js"]
    for fn in demo.fns.values():
        if fn.name in {"update_file_preview", "reset_ui"}:
            assert slider not in fn.outputs and label not in fn.outputs
        if fn.name == "convert_handler":
            assert fn.inputs[1] is slider
    assert slider.api_info()["type"] == "integer"


@pytest.mark.parametrize("extension", sorted(FLASH_ONLY_PARSE_EXTENSIONS | IMAGE_EXTENSIONS))
@pytest.mark.parametrize("uppercase", [False, True])
def test_gradio_conversion_enforces_file_type_tier(tmp_path: Path, extension: str, uppercase: bool) -> None:
    """高档位残留值对所有轻量格式固定为 Flash，图片则保留用户选择，后缀不区分大小写。"""
    source = tmp_path / f"source.{extension.upper() if uppercase else extension}"
    source.write_bytes(b"placeholder")
    client = SimpleNamespace(parse_file=AsyncMock(side_effect=V1ArtifactError("stop after request")))
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("standard", "flash"), ("zip",), ("file_id",))
    demo = build_gradio_app(client, capabilities, output_root=tmp_path / "output", enable_example=False)
    handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")

    async def convert() -> None:
        """模拟直接调用事件，并携带不应影响非 PDF 的选页与 OCR 残留值。"""
        updates = [update async for update in handler(str(source), 1, "bad range", True)]
        assert "stop after request" in updates[-1][0]

    asyncio.run(convert())
    client.parse_file.assert_awaited_once()
    kwargs = client.parse_file.call_args.kwargs
    assert kwargs["tier"] == ("flash" if extension in FLASH_ONLY_PARSE_EXTENSIONS else "standard")
    assert kwargs["page_range"] == "" and kwargs["ocr_mode"] == "auto"


@pytest.mark.parametrize("tiers", [("standard",), ("advanced", "basic"), ("basic", "standard", "advanced")])
def test_gradio_flash_only_input_requires_available_flash(tmp_path: Path, tiers: tuple[str, ...]) -> None:
    """服务缺少 Flash 时在事件入口报错，不发起解析请求，也不保留旧下载。"""
    source = tmp_path / "source.csv"
    source.write_text("name,value\na,1\n", encoding="utf-8")
    client = SimpleNamespace(parse_file=AsyncMock())
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", tiers, ("zip",), ("file_id",))
    demo = build_gradio_app(client, capabilities, output_root=tmp_path / "output", enable_example=False)
    handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")

    async def convert() -> list[tuple[Any, ...]]:
        """收集合法位置的错误响应，验证不是滑杆越界触发的拒绝。"""
        return [update async for update in handler(str(source), 0, "")]

    updates = asyncio.run(convert())
    assert "tier_unavailable" in updates[-1][0]
    assert "该格式仅支持 Flash，当前服务不可用" in updates[-1][0]
    assert updates[-1][8] is None
    assert all(item["interactive"] is False and item["value"] is None for item in updates[-1][-6:])
    client.parse_file.assert_not_called()


@pytest.mark.parametrize(
    ("tiers", "position", "expected_tier"),
    [
        (("advanced", "flash", "standard", "basic"), 0, "flash"),
        (("advanced", "flash", "standard", "basic"), 1, "basic"),
        (("advanced", "flash", "standard", "basic"), 2, "standard"),
        (("advanced", "flash", "standard", "basic"), 3, "advanced"),
        (("advanced", "standard", "basic"), 2, "advanced"),
        (("advanced", "flash"), 1, "advanced"),
        (("flash",), 0, "flash"),
    ],
)
@pytest.mark.parametrize("force_ocr", [False, True])
def test_gradio_conversion_forwards_page_range_and_enables_fresh_downloads(
    tmp_path: Path, tiers: tuple[str, ...], position: int, expected_tier: str, force_ocr: bool
) -> None:
    """验证滑块位置映射与 PDF page_range 透传，并只在新结果完成后启用下载。"""
    source = tmp_path / "demo.pdf"
    source.write_bytes(_pdf_bytes())

    class FakeClient:
        """记录 UI 传给 V1 client 的请求。"""

        def __init__(self) -> None:
            """初始化调用记录。"""
            self.calls: list[tuple[Path, str, str]] = []

        async def parse_file(
            self,
            path: Path,
            *,
            tier: str,
            page_range: str,
            ocr_mode: str = "auto",
            status_callback: Any = None,
        ) -> ParseResult:
            """返回最小解析结果并记录页码。"""
            self.calls.append((path, tier, page_range))
            assert ocr_mode == ("ocr" if force_ocr else "auto")
            if status_callback is not None:
                status_callback("Processing on server...")
            return ParseResult(middle_json=_middle_json(with_image=False))

    async def collect_updates(handler: Any) -> list[tuple[Any, ...]]:
        """收集 Gradio 异步生成器的全部增量输出。"""
        return [update async for update in handler(str(source), position, " 1 ", force_ocr)]

    capabilities = V1ServerCapabilities("http://127.0.0.1:1", tiers, ("zip",), ("file_id",))
    client = FakeClient()
    demo = build_gradio_app(
        client,  # type: ignore[arg-type]
        capabilities,
        output_root=tmp_path / "output",
        enable_example=False,
    )
    convert_handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")
    updates = asyncio.run(collect_updates(convert_handler))

    assert client.calls == [(source.resolve(), expected_tier, "" if expected_tier == "flash" else "1")]
    assert len(updates[-1]) == 15
    assert updates[-1][8] is not None
    assert all(update["interactive"] is True and update["value"] is None for update in updates[-1][-6:])
    assert all(update["interactive"] is False and update["value"] is None for update in updates[0][-6:])


@pytest.mark.parametrize(
    ("tiers", "position"),
    [
        (("flash", "basic", "standard", "advanced"), -1),
        (("flash", "basic", "standard", "advanced"), 4),
        (("flash",), 1),
        (("flash", "advanced"), 0.5),
        (("flash", "advanced"), float("nan")),
        (("flash", "advanced"), True),
    ],
)
def test_gradio_conversion_rejects_invalid_tier_position(tmp_path: Path, tiers: tuple[str, ...], position: int | float) -> None:
    """验证非法位置会清除结果且不会向解析服务提交请求，包括单档位的占位上界。"""
    source = tmp_path / "demo.csv"
    source.write_text("name,value\na,1\n", encoding="utf-8")
    client = SimpleNamespace(parse_file=AsyncMock())
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", tiers, ("zip",), ("file_id",))
    demo = build_gradio_app(client, capabilities, output_root=tmp_path, enable_example=False)
    handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")

    async def collect_updates() -> list[tuple[Any, ...]]:
        """收集非法滑块位置对应的错误输出。"""
        return [update async for update in handler(str(source), position, "")]

    updates = asyncio.run(collect_updates())
    assert "Failed: Invalid tier slider position" in updates[-1][0]
    assert updates[-1][8] is None
    assert all(item["interactive"] is False and item["value"] is None for item in updates[-1][-6:])
    client.parse_file.assert_not_called()


def test_gradio_conversion_failure_clears_previous_downloads(tmp_path: Path) -> None:
    """验证解析失败会清空结果 State 并禁用所有旧下载按钮。"""
    source = tmp_path / "demo.pdf"
    source.write_bytes(_pdf_bytes())

    class FailingClient:
        """模拟 V1 解析失败。"""

        async def parse_file(self, *_args: Any, **_kwargs: Any) -> ParseResult:
            """抛出稳定的测试错误。"""
            raise V1ArtifactError("boom", code="parse_failed")

    async def final_update(handler: Any) -> tuple[Any, ...]:
        """返回 Gradio 异步生成器的最后一次更新。"""
        updates = [update async for update in handler(str(source), 0, "")]
        return updates[-1]

    capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("zip",), ("file_id",))
    demo = build_gradio_app(
        FailingClient(),  # type: ignore[arg-type]
        capabilities,
        output_root=tmp_path / "output",
        enable_example=False,
    )
    convert_handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")
    update = asyncio.run(final_update(convert_handler))

    assert "Failed: boom" in update[0]
    assert update[8] is None
    assert all(item["interactive"] is False and item["value"] is None for item in update[-6:])


@pytest.mark.parametrize("explicit_session_cancel", [False, True])
def test_gradio_local_queue_cancellation_releases_slot_and_keeps_sessions_isolated(
    tmp_path: Path, explicit_session_cancel: bool
) -> None:
    """验证等待和运行中的生成器被关闭后释放任务槽，其他会话仍能完成自己的结果。"""
    sources = [tmp_path / f"session-{index}.csv" for index in range(3)]
    for source in sources:
        source.write_text("name,value\na,1\n", encoding="utf-8")

    async def scenario() -> None:
        """模拟两个会话取消后由第三个会话接续执行，不依赖真实服务等待。"""
        finish = asyncio.Event()
        calls: list[Path] = []
        canceled: list[Path] = []

        class WaitingClient:
            """按测试事件控制解析完成时机。"""

            async def parse_file(self, path: Path, *, status_callback: Any, **_kwargs: Any) -> ParseResult:
                """记录实际进入任务槽的会话，并传播本地取消。"""
                calls.append(path)
                status_callback("Processing on server...")
                try:
                    await finish.wait()
                except asyncio.CancelledError:
                    canceled.append(path)
                    raise
                status_callback(STATUS_DOWNLOADING_RESULT)
                return ParseResult(middle_json=_middle_json(with_image=False))

        capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("zip",), ("file_id",))
        demo = build_gradio_app(WaitingClient(), capabilities, output_root=tmp_path / "output", enable_example=False)
        handler_fn = next(fn for fn in demo.fns.values() if fn.name == "convert_handler")
        assert handler_fn.concurrency_limit is None
        # 清除和文件切换都使用 Gradio 的会话级取消事件。
        cancellation_events = [dependency for dependency in demo.config["dependencies"] if dependency["cancels"]]
        assert len(cancellation_events) >= 2
        assert all(handler_fn._id in dependency["cancels"] for dependency in cancellation_events)

        async def advance_until(stream: Any, marker: str) -> tuple[Any, ...]:
            """推进生成器直到目标阶段，并给测试死锁设置明确超时。"""
            while True:
                update = await asyncio.wait_for(anext(stream), timeout=2)
                if marker in update[0]:
                    return update

        requests = [SimpleNamespace(session_hash=f"session-{index}") for index in range(3)]
        first, second, third = [
            handler_fn.fn(str(source), 0, "", request=request) for source, request in zip(sources, requests)
        ]
        cancel_session = next(fn.fn for fn in demo.fns.values() if fn.name == "cancel_session_conversion")
        await advance_until(first, "Processing on server")
        queued = await advance_until(second, "Queued locally.")
        assert all(value == {"__type__": "update"} for value in queued[1:])
        assert calls == [sources[0]]
        if explicit_session_cancel:
            await cancel_session(requests[1])
            with pytest.raises(StopAsyncIteration):
                await anext(second)
        else:
            await second.aclose()
        await advance_until(third, "Queued locally.")
        if explicit_session_cancel:
            await cancel_session(requests[0])
            with pytest.raises(StopAsyncIteration):
                await anext(first)
        else:
            await first.aclose()
        await advance_until(third, "Processing on server")
        assert canceled == [sources[0]]
        assert calls == [sources[0], sources[2]]
        finish.set()
        updates = [update async for update in third]
        final = updates[-1]
        assert "Completed (" in final[0]
        assert final[8]["stem"] == sources[2].stem
        assert all(item["interactive"] is True for item in final[-6:])

    asyncio.run(scenario())


def test_gradio_output_failure_stops_timer_and_allows_next_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证解析完成后的输出整理错误仍显示失败，并释放槽位供下一次转换使用。"""
    source = tmp_path / "source.csv"
    source.write_text("name,value\na,1\n", encoding="utf-8")

    class CompletedClient:
        """模拟已完成远端解析和下载的任务。"""

        async def parse_file(self, _path: Path, *, status_callback: Any, **_kwargs: Any) -> ParseResult:
            """按真实阶段通知后返回解析结果。"""
            status_callback("Processing on server...")
            status_callback(STATUS_DOWNLOADING_RESULT)
            return ParseResult(middle_json=_middle_json(with_image=False))

    def fail_persistence(*_args: Any, **_kwargs: Any) -> Any:
        """模拟产物磁盘写入失败。"""
        raise OSError("output disk unavailable")

    monkeypatch.setattr(gradio_app, "persist_parse_result", fail_persistence)
    capabilities = V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("zip",), ("file_id",))
    demo = build_gradio_app(CompletedClient(), capabilities, output_root=tmp_path, enable_example=False)
    handler = next(fn.fn for fn in demo.fns.values() if fn.name == "convert_handler")

    async def scenario() -> None:
        """连续运行两次，验证异常分支没有泄漏执行槽。"""
        for _ in range(2):
            updates = [update async for update in handler(str(source), 0, "")]
            assert "Failed: output disk unavailable" in updates[-1][0]
            assert "is-error" in updates[-1][0]
            assert updates[-1][8] is None
            assert all(item["interactive"] is False for item in updates[-1][-6:])

    asyncio.run(scenario())
