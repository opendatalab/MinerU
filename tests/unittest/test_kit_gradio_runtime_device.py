"""验证 WebUI 设备查询的边界、鉴权和真实组件服务接线。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock

import gradio as gr
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mineru.kit.gradio.app import build_gradio_app
from mineru.kit.gradio.client import GradioArtifactClient, V1ArtifactClient, V1ServerCapabilities


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "ok", "runtime": {"device": "cpu"}}, "cpu"),
        ({"status": "ok", "runtime": {"device": "gpu"}}, "gpu"),
        ({"status": "ok"}, None),
        ({"status": "ok", "runtime": None}, None),
        ({"status": "ok", "runtime": []}, None),
        ({"status": "ok", "runtime": {}}, None),
        ({"status": "ok", "runtime": {"device": "mps"}}, None),
        ({"status": "ok", "runtime": {"device": ["gpu"]}}, None),
        ({"status": "error", "runtime": {"device": "gpu"}}, None),
        ([], None),
    ],
)
def test_device_query_only_reads_health_and_keeps_capabilities(payload: object, expected: str | None) -> None:
    """设备查询沿用地址和密钥、设置三秒超时，并接受旧服务或异常字段。"""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """记录真实 HTTP 客户端的单次请求并返回指定 health 响应。"""
        requests.append(request)
        assert request.headers["authorization"] == "Bearer secret"
        assert set(request.extensions["timeout"].values()) == {3}
        return httpx.Response(200, json=payload)

    primary = V1ArtifactClient(api_url="http://example.test/api", api_key="secret", transport=httpx.MockTransport(handler))
    snapshot = V1ServerCapabilities(primary.base_url, ("standard",), ("default",), ("upload",))
    primary._capabilities = snapshot
    routed = GradioArtifactClient(primary)
    assert asyncio.run(routed.read_runtime_device()) == expected
    assert [str(request.url) for request in requests] == ["http://example.test/api/v1/health"]
    assert primary.capabilities is snapshot and routed.capabilities is snapshot


@pytest.mark.parametrize("failure", ["connect", "timeout", "http", "json"])
def test_device_query_failure_is_unknown_without_retry(failure: str) -> None:
    """连接、超时、HTTP 和协议错误只影响设备装饰，不重试也不抛给解析流程。"""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """模拟不同失败边界，并保留网络调用次数供断言。"""
        requests.append(request)
        if failure == "connect":
            raise httpx.ConnectError("offline", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(503 if failure == "http" else 200, text="not JSON")

    client = V1ArtifactClient(api_url="http://example.test", transport=httpx.MockTransport(handler))
    assert asyncio.run(client.read_runtime_device()) is None
    assert len(requests) == 1


@pytest.mark.parametrize("device", ["cpu", "gpu", None])
def test_runtime_loader_is_exposed_as_component_service(tmp_path: Path, device: Literal["cpu", "gpu"] | None) -> None:
    """真实 Gradio 组件服务按页面调用读取器，组装和其他控件服务不会提前读取。"""
    loader = AsyncMock(return_value=device)
    demo = build_gradio_app(
        SimpleNamespace(),
        V1ServerCapabilities("http://localhost:1", ("flash", "standard", "advanced"), ("default",), ("upload",)),
        output_root=tmp_path,
        enable_example=False,
        runtime_device_loader=loader,
    )
    loader.assert_not_awaited()
    tier = next(block for block in demo.blocks.values() if "mineru-tier-control" in (block.elem_classes or []))
    app = gr.mount_gradio_app(FastAPI(), demo, path="/webui")
    body = {"component_id": tier._id, "fn_name": "read_runtime_device", "session_hash": "device-test", "data": []}
    with TestClient(app) as client:
        response = client.post("/webui/gradio_api/component_server", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == device
    loader.assert_awaited_once_with()
    assert tier.props == {"selected_tier": "standard", "locked": False}
