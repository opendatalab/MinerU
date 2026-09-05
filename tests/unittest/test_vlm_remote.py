from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from reportlab.pdfgen import canvas
from typer.testing import CliRunner

from mineru.config import VlmConfig, config
from mineru.model.vlm.client import get_vlm_predictor
from mineru.parser import api_server, parse, parse_async


@dataclass
class _OpenAIServer:
    """记录模拟推理服务的协议请求和可控响应。"""

    url: str = ""
    models: list[str] = field(default_factory=lambda: ["test-model"])
    api_key: str = "test-key"
    chat_status: int = 200
    requests: list[tuple[str, str, dict[str, Any] | None]] = field(default_factory=list)


@pytest.fixture
def openai_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_OpenAIServer]:
    """启动仅监听 loopback 的模拟 OpenAI 服务，并隔离模型缓存与旧环境变量。"""
    from mineru.model.vlm.runtime import ModelSingleton

    monkeypatch.delenv("MINERU_VL_API_KEY", raising=False)
    monkeypatch.delenv("MINERU_VL_MODEL_NAME", raising=False)
    monkeypatch.setattr(config.model, "vlm", VlmConfig())
    monkeypatch.setattr(config.llm_aided.features, "title_leveling", False)
    monkeypatch.setattr(config.llm_aided.features, "cross_page_table_cell_merge", False)
    monkeypatch.setattr(ModelSingleton, "_models", {})
    state = _OpenAIServer()

    class Handler(BaseHTTPRequestHandler):
        """实现模型发现、认证和最小 Chat Completions 协议。"""

        def log_message(self, format: str, *args: object) -> None:
            """关闭测试 HTTP 日志，避免图片请求污染测试输出。"""

        def _respond(self, status: int, payload: dict[str, Any]) -> None:
            """返回带明确长度的 JSON，使客户端可以正常结束响应读取。"""
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            """校验 VLM Key，与解析 API 自身的 Key 独立。"""
            if state.api_key and self.headers.get("Authorization") != f"Bearer {state.api_key}":
                self._respond(401, {"error": {"message": "unauthorized"}})
                return False
            return True

        def do_GET(self) -> None:
            """仅在保留代理前缀的 models 路径返回模型列表。"""
            state.requests.append((self.path, self.headers.get("Authorization", ""), None))
            if not self._authorized():
                return
            if self.path != "/proxy/v1/models":
                self._respond(404, {"error": {"message": "unknown path"}})
                return
            self._respond(200, {"data": [{"id": model} for model in state.models]})

        def do_POST(self) -> None:
            """根据版面或文本提示返回稳定结果，支持失败任务测试。"""
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((self.path, self.headers.get("Authorization", ""), body))
            if not self._authorized():
                return
            if self.path != "/proxy/v1/chat/completions":
                self._respond(404, {"error": {"message": "unknown path"}})
                return
            if state.chat_status != 200:
                self._respond(state.chat_status, {"error": {"message": "inference rejected"}})
                return
            content = "Remote VLM text"
            if "Layout Detection:" in json.dumps(body["messages"]):
                content = "<|box_start|>50 50 950 250<|box_end|><|ref_start|>text<|ref_end|>"
            self._respond(200, {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_port}/proxy/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        for predictor in ModelSingleton._models.values():
            predictor.client._client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _settings(server: _OpenAIServer, **overrides: Any) -> VlmConfig:
    """为模拟服务生成连接配置，允许单个用例覆盖连接字段。"""
    return VlmConfig.model_validate({"server_url": server.url, "api_key": server.api_key, **overrides})


def test_remote_client_protocol_and_cache(openai_server: _OpenAIServer, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实 HTTP 验证模型发现、显式模型、认证、代理路径、参数及缓存隔离。"""
    from mineru.model.vlm import runtime, selector

    local_engine = MagicMock(side_effect=AssertionError("local VLM engine must not load"))
    local_weights = MagicMock(side_effect=AssertionError("local VLM weights must not download"))
    monkeypatch.setattr(selector, "get_vlm_engine", local_engine)
    monkeypatch.setattr(type(runtime.MINERU_2_5_PRO_2605_1_2B), "ensure", local_weights)
    settings = _settings(openai_server, http_timeout=9, max_concurrency=3)
    predictor, backend = get_vlm_predictor(settings)
    assert backend == "http-client"
    assert predictor.client.model_name == "test-model"
    assert predictor.client.max_concurrency == 3
    assert predictor.client._client.timeout.read == 9
    assert predictor.client.predict(Image.new("RGB", (32, 32)), "Text Recognition:") == "Remote VLM text"
    assert get_vlm_predictor(settings)[0] is predictor
    for patch in ({"model": "test-model"}, {"http_timeout": 10}, {"max_concurrency": 4}):
        assert get_vlm_predictor(VlmConfig.model_validate({**settings.model_dump(), **patch}))[0] is not predictor
    openai_server.api_key = "second-key"
    assert get_vlm_predictor(_settings(openai_server))[0] is not predictor
    assert {request[0] for request in openai_server.requests} == {"/proxy/v1/models", "/proxy/v1/chat/completions"}
    body = next(request[2] for request in openai_server.requests if request[2] is not None)
    assert body["model"] == "test-model"
    assert all(request[1] == "Bearer test-key" for request in openai_server.requests[:-1])
    local_engine.assert_not_called()
    local_weights.assert_not_called()


def test_remote_models_and_auth_failures(openai_server: _OpenAIServer) -> None:
    """认证失败、多模型自动发现及不存在模型均显式失败，之后可以用有效模型重试。"""
    with pytest.raises(Exception, match="401"):
        get_vlm_predictor(_settings(openai_server, api_key="wrong-key"))
    openai_server.models = []
    with pytest.raises(Exception, match="exactly one model"):
        get_vlm_predictor(_settings(openai_server))
    openai_server.models = ["first", "second"]
    with pytest.raises(Exception, match="exactly one model"):
        get_vlm_predictor(_settings(openai_server))
    with pytest.raises(Exception, match="not found"):
        get_vlm_predictor(_settings(openai_server, model="missing"))
    predictor, _ = get_vlm_predictor(_settings(openai_server, model="second"))
    assert predictor.client.model_name == "second"


def test_remote_optional_auth_and_concurrent_credentials(openai_server: _OpenAIServer) -> None:
    """允许无鉴权服务，并验证同一服务的不同客户端并发请求不会串用凭据。"""
    openai_server.api_key = ""
    predictor, _ = get_vlm_predictor(_settings(openai_server))
    assert predictor.client.predict(None, "test") == "Remote VLM text"
    assert all(request[1] == "" for request in openai_server.requests)

    def infer(key: str) -> str:
        """使用独立 Key 请求同一上游，返回真实客户端收到的文本。"""
        client, _ = get_vlm_predictor(_settings(openai_server, api_key=key))
        return client.client.predict(None, key)

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(infer, ["first-key", "second-key"])) == ["Remote VLM text", "Remote VLM text"]
    for _path, header, body in openai_server.requests:
        if body is not None and header:
            assert header.removeprefix("Bearer ") in json.dumps(body["messages"])


def test_remote_timeout_propagates(openai_server: _OpenAIServer, monkeypatch: pytest.MonkeyPatch) -> None:
    """请求超时保留原始异常，不选择本地引擎重试。"""
    predictor, _ = get_vlm_predictor(_settings(openai_server, http_timeout=1))
    monkeypatch.setattr(predictor.client._client, "post", MagicMock(side_effect=httpx.ReadTimeout("timed out")))
    with pytest.raises(httpx.ReadTimeout, match="timed out"):
        predictor.client.predict(None, "test")


def test_remote_preflight_and_preload(openai_server: _OpenAIServer, monkeypatch: pytest.MonkeyPatch) -> None:
    """远程预检仅需本地 Hybrid 依赖，预加载和推理复用同一个客户端。"""
    checks: list[str] = []
    local_loads: list[str] = []
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", checks.append)
    monkeypatch.setattr(api_server, "_preload_local_models", local_loads.append)
    settings = _settings(openai_server)
    api_server._preflight_tier_dependencies("standard", settings)
    result = api_server._preload_server_models("standard", language="en", vlm_config=settings)
    assert result.engine == "http-client"
    assert checks == ["basic"]
    assert local_loads == ["en"]
    request_count = len(openai_server.requests)
    get_vlm_predictor(settings)
    assert len(openai_server.requests) == request_count
    api_server._preload_server_models("basic", language="ch", vlm_config=settings)
    assert len(openai_server.requests) == request_count


def test_remote_preload_failure_is_reported(
    openai_server: _OpenAIServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """预加载认证失败通过现有健康和能力错误暴露，未预加载时保持懒连接。"""
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", lambda tier: None)
    settings = _settings(openai_server, api_key="wrong-key")
    lazy = api_server.create_app(upload_dir=str(tmp_path / "lazy"), vlm_config=settings)
    with TestClient(lazy) as client:
        assert client.get("/v1/health").status_code == 200
        assert openai_server.requests == []
    preloaded = api_server.create_app(upload_dir=str(tmp_path / "preload"), vlm_config=settings, preload_models=True)
    with TestClient(preloaded) as client:
        for endpoint in ("health", "tiers", "models"):
            response = client.get(f"/v1/{endpoint}")
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "model_preload_failed"


@pytest.fixture
def hybrid_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离本地神经模型，保留真实 PDF 准备、窗口、HTTP VLM 和 Middle JSON 后处理。"""
    from mineru.backend.analysis.pdf import pipeline, window

    context = MagicMock()
    context.device = "cpu"
    context.layout_model.batch_predict.return_value = [[{"bbox": [20, 20, 250, 80], "label": "text"}]]
    monkeypatch.setattr(
        pipeline,
        "HybridLocalModelContextSingleton",
        MagicMock(return_value=MagicMock(get_model=MagicMock(return_value=context))),
    )
    monkeypatch.setattr(pipeline, "clean_memory", lambda device: None)

    def retain_vlm_text(
        images: object, pages: object, blocks: list[list[dict[str, Any]]], *args: object
    ) -> list[list[dict[str, Any]]]:
        """保留远程正文并模拟 Hybrid 提供的行框，满足 PDF 文本块校验合同。"""
        for page in blocks:
            for block in page:
                if block.get("content"):
                    block["lines"] = [{"bbox": block["bbox"]}]
        return blocks

    monkeypatch.setattr(window, "_process_text_and_formulas", retain_vlm_text)
    monkeypatch.setattr(window, "_apply_seal_ocr", lambda *args: None)
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", lambda tier: None)


def _pdf_input(tmp_path: Path) -> Path:
    """构造单页 PDF fixture，供同步、异步和 API 解析复用。"""
    path = tmp_path / "input.pdf"
    document = canvas.Canvas(str(path), pagesize=(300, 200))
    document.drawString(20, 160, "Source text")
    document.save()
    return path


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("image_input", [False, True])
def test_python_parse_uses_global_remote_vlm(
    openai_server: _OpenAIServer,
    hybrid_stub: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool,
    image_input: bool,
) -> None:
    """同步及异步入口读取全局 VLM 配置，PDF 和图片转 PDF 均产生真实 Middle JSON。"""
    monkeypatch.setattr(config.model, "vlm", _settings(openai_server))
    source = _pdf_input(tmp_path)
    if image_input:
        source = tmp_path / "input.png"
        Image.new("RGB", (300, 200), "white").save(source)
    result = asyncio.run(parse_async(source, ocr_mode="ocr")) if async_mode else parse(source, ocr_mode="ocr")
    assert result.middle_json.file_suffix == "pdf"
    assert result.middle_json.effort == "high"
    assert result.middle_json.pages[0].page_idx == 0
    assert "Remote VLM text" in result.markdown()
    assert any(request[2] is not None for request in openai_server.requests)


def test_kit_parse_uses_global_remote_vlm(
    openai_server: _OpenAIServer, hybrid_stub: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实际执行 mineru-kit parse 命令，验证全局 VLM 配置进入解析并生成 Markdown 文件。"""
    from mineru.kit.main import app

    monkeypatch.setattr(config.model, "vlm", _settings(openai_server))
    output = tmp_path / "result.md"
    result = CliRunner().invoke(app, ["parse", str(_pdf_input(tmp_path)), "-o", str(output), "--ocr-mode", "ocr"])
    assert result.exit_code == 0, result.output
    assert "Remote VLM text" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("tier", ["standard", "advanced"])
@pytest.mark.parametrize("chat_status", [200, 400])
def test_api_job_through_http_vlm_to_middle_json(
    openai_server: _OpenAIServer,
    hybrid_stub: None,
    tmp_path: Path,
    tier: str,
    chat_status: int,
) -> None:
    """API 任务经真实 HTTP VLM 生成 Middle JSON，远程推理错误保持任务失败。"""
    openai_server.chat_status = chat_status
    source = _pdf_input(tmp_path)
    app = api_server.create_app(
        upload_dir=str(tmp_path / "api"),
        allow_local_source=True,
        api_key="inbound-key",
        no_flash=True,
        vlm_config=_settings(openai_server),
    )
    with TestClient(app, headers={"Authorization": "Bearer inbound-key"}) as client:
        submitted = client.post(
            "/v1/parse/jobs",
            json={
                "files": [{"source": {"type": "local", "path": str(source)}}],
                "tier": tier,
                "ocr_mode": "ocr",
                "output_formats": ["middle_json"],
            },
        )
        assert submitted.status_code == 202, submitted.text
        job_id = submitted.json()["job_id"]
        deadline = time.monotonic() + 15
        while True:
            job = client.get(f"/v1/parse/jobs/{job_id}").json()
            if job["status"] in {"completed", "failed", "partial"}:
                break
            assert time.monotonic() < deadline, job
            time.sleep(0.01)
        if chat_status != 200:
            assert job["status"] == "failed", job
            assert job["files"][0]["error"]["code"] == "parse_failed"
            return
        assert job["status"] == "completed", job
        file_id = job["files"][0]["output_files"]["middle_json"]["file_id"]
        output = client.get(f"/v1/files/{file_id}/content")
        assert output.status_code == 200
        assert output.json()["effort"] == ("high" if tier == "standard" else "xhigh")
        assert "Remote VLM text" in output.text
        assert all(request[1] == "Bearer test-key" for request in openai_server.requests)


@pytest.mark.parametrize("effort", ["flash", "medium"])
def test_low_efforts_never_create_remote_client(
    openai_server: _OpenAIServer,
    hybrid_stub: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
) -> None:
    """即使配置了远程服务，Flash 和 Basic 分析也不触发 VLM 客户端构造。"""
    from mineru.backend.analysis.pdf import pipeline

    remote_factory = MagicMock(side_effect=AssertionError("VLM must not initialize"))
    monkeypatch.setattr(pipeline, "get_vlm_predictor", remote_factory)
    monkeypatch.setattr(pipeline, "process_pdf_windows", MagicMock(return_value=[]))
    result = pipeline.analyze_pdf(
        _pdf_input(tmp_path).read_bytes(), effort=effort, parse_mode="txt", vlm_config=_settings(openai_server)
    )
    assert result.effort == effort
    assert openai_server.requests == []
    remote_factory.assert_not_called()
