"""V1 请求级 OCR 参数、输入校验与服务启动配置退役回归。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from mineru.parser import api_client, api_server
from mineru.parser.api_client import MinerUApiParser


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("ocr_mode", [None, "auto", "txt", "ocr"])
def test_api_parser_serializes_request_ocr_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
    ocr_mode: Literal["auto", "txt", "ocr"] | None,
) -> None:
    """通过正式 HTTP 提交流程检查两个入口的显式模式与缺省字段。"""
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    parser = MinerUApiParser(api_url="http://localhost:8000", tier="flash", ocr_mode=ocr_mode)
    parser._source_features = {"local"}
    payloads: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """记录完整任务请求，在产物下载前返回已完成任务。"""
        assert request.method == "POST" and request.url.path == "/v1/parse/jobs"
        payloads.append(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "job_test", "status": "completed"})

    transport = httpx.MockTransport(respond)
    sync_client, async_client = httpx.Client, httpx.AsyncClient

    def create_sync_client(**kwargs: Any) -> httpx.Client:
        """向同步传输注入请求记录器，避免访问真实网络。"""
        return sync_client(transport=transport, **kwargs)

    def create_async_client(**kwargs: Any) -> httpx.AsyncClient:
        """向异步传输注入同一个请求记录器。"""
        return async_client(transport=transport, **kwargs)

    monkeypatch.setattr(api_client.httpx, "Client", create_sync_client)
    monkeypatch.setattr(api_client.httpx, "AsyncClient", create_async_client)
    monkeypatch.setattr(parser, "_build_result", Mock())
    monkeypatch.setattr(parser, "_async_build_result", AsyncMock())
    if use_async:
        asyncio.run(parser.parse_async(source, page_range="1"))
    else:
        parser.parse(source, page_range="1")

    expected: dict[str, Any] = {
        "files": [{"source": {"type": "local", "path": str(source)}, "page_range": "1"}],
        "tier": "flash",
        "output_formats": ["middle_json"],
    }
    if ocr_mode is not None:
        expected["ocr_mode"] = ocr_mode
    assert payloads == [expected]


@pytest.mark.parametrize("value", ["", "forced", True, 1, {}])
def test_api_parser_rejects_invalid_ocr_mode(value: object) -> None:
    """在上传之前拒绝不属于公开模式集合的配置。"""
    with pytest.raises(ValueError, match="Unsupported OCR mode"):
        MinerUApiParser(ocr_mode=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, "", "forced", True, 1, {}])
def test_api_server_rejects_invalid_request_ocr_mode(tmp_path: Path, value: object) -> None:
    """非法模式统一返回带参数定位的 HTTP 400，且不会创建任务。"""
    app = api_server.create_app(upload_dir=str(tmp_path), tier="flash")
    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={
                "files": [{"source": {"type": "inline", "name": "demo.pdf", "data": "JVBERi0xLjcK"}}],
                "ocr_mode": value,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["param"] == "ocr_mode"
    assert app.state.job_store._jobs == {}


def test_api_server_ocr_defaults_to_request_auto_without_startup_configuration(tmp_path: Path) -> None:
    """请求缺省值固定为 auto，应用生命周期不保存全局 OCR 模式。"""
    request = api_server.CreateJobRequest.model_validate({"files": [{"source": {"type": "local", "path": "/tmp/demo.pdf"}}]})
    assert request.ocr_mode == "auto"
    app = api_server.create_app(upload_dir=str(tmp_path), tier="flash")
    assert not hasattr(app.state, "ocr_mode")
    with TestClient(app):
        assert not hasattr(app.state, "ocr_mode")
    with pytest.raises(TypeError, match="ocr_mode"):
        api_server.create_app(ocr_mode="ocr")  # type: ignore[call-arg]
    result = CliRunner().invoke(api_server.main, ["--ocr-mode", "ocr"])
    assert result.exit_code != 0
    assert "No such option: --ocr-mode" in result.output
