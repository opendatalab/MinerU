from docvortex.schema import Producer
from mineru.integrations.docvortex import build_metadata
import asyncio
import importlib
import inspect
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from pydantic import ValidationError

import mineru.parser.api_client as api_client
import mineru.parser.api_server as api_server
import mineru.parser.tier as parser_tier
from mineru.config import VlmConfig
from mineru.tasks.contracts import FileParseInfo
from mineru.tasks.execution import TaskExecution
from mineru.parser import MIDDLE_JSON_SCHEMA_VERSION
from mineru.parser.api_client import MinerUApiParser, _pages_from_middle_json, should_trust_env_for_url
from mineru.parser.api_server import (
    FileStore,
    HealthResponse,
    create_app,
    main,
)
from mineru.parser.base import ParseResult
from task_test_utils import execution_request
from task_test_utils import stored_source
from mineru.types import (
    ModelJson,
    DEFAULT_QUALITY_TIER_SELECTION_ORDER,
    DEPLOYMENT_TIERS,
    DeploymentTier,
    PARSING_RULE_TIER_SELECTION_ORDER,
    QUALITY_TIERS,
    SERVER_TIERS,
    SERVER_STARTUP_TIERS,
    TIER_ORDER,
    TIERS,
    TIERS_BY_SERVER_TIER,
    MiddleJson,
    PageInfo,
    Tier,
    select_default_quality_tier,
    select_parsing_rule_tier,
    validate_tier,
)
from mineru.version import __version__

runner = CliRunner()

_REMOVED_DISABLE_TABLE_PARAM = "disable" + "_table"
_REMOVED_DISABLE_FORMULA_PARAM = "disable" + "_formula"
_REMOVED_TABLE_ENABLE_PARAM = "table" + "_enable"
_REMOVED_FORMULA_ENABLE_PARAM = "formula" + "_enable"
_REMOVED_DISABLE_TABLE_OPTION = "--disable-" + "table"
_REMOVED_DISABLE_FORMULA_OPTION = "--disable-" + "formula"
_REMOVED_TABLE_ENABLE_ENV = "MINERU_" + "TABLE" + "_ENABLE"
_REMOVED_FORMULA_ENABLE_ENV = "MINERU_" + "FORMULA" + "_ENABLE"


@pytest.fixture(autouse=True)
def _stable_hardware_recommendation(monkeypatch: pytest.MonkeyPatch) -> None:
    """固定服务契约测试的硬件推荐，设备组合由独立测试覆盖。"""
    monkeypatch.setattr(
        "mineru.model.runtime.tier_recommendation.recommend_tiers", lambda **kwargs: SimpleNamespace(pdf_image_tiers=TIERS)
    )


def _stub_api_server_dependency_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """只跳过服务依赖预检，避免覆盖全局导入而破坏运行时的惰性加载。"""
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", lambda *_args, **_kwargs: None)


def _full_middle_json(*pages: PageInfo) -> MiddleJson:
    """构造未抽页的严格 MiddleJson 测试对象。"""
    return MiddleJson(
        pages=list(pages),
        is_full_document=True,
        metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version=__version__)},
        extensions=build_metadata(
            effort="medium",
            parse_mode="txt",
        ),
    )


def _model_payload() -> dict[str, Any]:
    """构造 ZIP 内的完整分析协议，保留原始模型块用于传输回归。"""

    return ModelJson(
        pages=[[{"raw": "model"}]],
        page_index_map=[],
        metadata={"file_suffix": "pdf", "producer": {"name": "mineru", "version": __version__}},
    ).to_dict()


def _current_payload(pages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """构造严格 Middle JSON 2.0 API envelope。"""
    payload: dict[str, Any] = {
        "schema_version": MIDDLE_JSON_SCHEMA_VERSION,
        "pages": pages,
        "is_full_document": True,
        "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": __version__}},
        "schema": "docvortex.middle",
        "extensions": {"mineru": {"tier": "basic", "parse_mode": "txt"}},
    }
    payload.update(extra)
    return payload


def _current_image_page(image_path: str, *, page_idx: int = 0) -> dict[str, Any]:
    """构造带一个图片 sidecar 的严格 PDF 页面。"""
    bbox = [0.0, 0.0, 0.1, 0.1]
    return {
        "page_idx": page_idx,
        "blocks": [
            {
                "type": "image",
                "index": 0,
                "bbox": bbox,
                "content": [
                    {
                        "type": "image_body",
                        "index": 0,
                        "bbox": bbox,
                        "content": "",
                        "image_path": image_path,
                    }
                ],
            }
        ],
    }


def test_backend_analyze_import_does_not_require_vlm_utils() -> None:
    """校验统一 backend 入口导入阶段不强依赖 VLM 工具包。"""
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys


class BlockVlmUtilsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 阻断 mineru_vl_utils 导入，用来验证 Hybrid basic 的 lazy import 边界。
        if fullname == "mineru_vl_utils" or fullname.startswith("mineru_vl_utils."):
            raise ModuleNotFoundError(f"blocked VLM utility import: {fullname}")
        return None


sys.meta_path.insert(0, BlockVlmUtilsFinder())
import mineru.backend.analyze
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_dependency_preflight_does_not_load_hybrid_runtime_without_torch() -> None:
    """校验 light stack 的依赖预检不会在检查前加载 Hybrid 与 Torch。"""
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import os
import sys

os.environ["MINERU_MODEL_SMALL_BACKEND"] = "onnx"


class BlockTorchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 阻断 Torch 导入，验证依赖预检只加载轻量设备选择模块。
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError(f"blocked Torch import: {fullname}", name=fullname)
        return None


sys.meta_path.insert(0, BlockTorchFinder())
from mineru.parser.tier import required_modules_for_tier

assert required_modules_for_tier("basic") == ["onnxruntime"]
assert "mineru.model.runtime.hybrid" not in sys.modules
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_light_pdf_pipeline_import_does_not_require_torch() -> None:
    """校验 light stack 首次加载 PDF pipeline 时不会导入 Torch。"""
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import builtins
import importlib.util
import os
import sys

os.environ["MINERU_MODEL_SMALL_BACKEND"] = "onnx"
real_import = builtins.__import__
real_find_spec = importlib.util.find_spec


def block_torch_import(name, globals=None, locals=None, fromlist=(), level=0):
    # 阻断真实 Torch 导入，同时允许第三方依赖探测到“未安装”状态。
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError(f"blocked Torch import: {name}", name=name)
    return real_import(name, globals, locals, fromlist, level)


def hide_torch_spec(name, package=None):
    # 模拟 core 安装中不存在 Torch distribution，避免能力探测误判。
    if name == "torch" or name.startswith("torch."):
        return None
    return real_find_spec(name, package)


builtins.__import__ = block_torch_import
importlib.util.find_spec = hide_torch_spec

import mineru.model.runtime.hybrid
import mineru.backend.analysis.pdf.pipeline

assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_hybrid_context_does_not_resolve_full_weights_before_stack_selection() -> None:
    """校验 Context 获取 Layout/MFR 时不提前解析 full-stack 权重路径。"""
    from mineru.model.download import ModelPath
    from mineru.model.runtime.contracts import AtomicModelName
    from mineru.model.runtime.hybrid import HybridLocalModelContext

    class _RecordingAtomManager:
        """记录 Context 交给 atom model 分派器的参数。"""

        def __init__(self) -> None:
            """初始化原子模型请求列表。"""
            self.calls: list[dict[str, Any]] = []

        def get_atom_model(self, atom_model_name: str, **kwargs: Any) -> dict[str, Any]:
            """记录原子模型请求并原样返回。"""
            arguments = {"atom_model_name": atom_model_name, **kwargs}
            self.calls.append(arguments)
            return dict(arguments)

    context = object.__new__(HybridLocalModelContext)
    manager = _RecordingAtomManager()
    context.device = "cpu"
    context.small_backend = "onnx"
    context._resource = None
    context.atom_model_manager = manager  # type: ignore[assignment]

    with patch.object(ModelPath, "ensure", side_effect=AssertionError("full weight path resolved too early")):
        context.get_layout_model()
        context.get_mfr_model()

    assert manager.calls == [
        {"atom_model_name": AtomicModelName.Layout, "device": "cpu", "small_backend": "onnx"},
        {"atom_model_name": AtomicModelName.MFR, "device": "cpu", "small_backend": "onnx"},
    ]


def test_public_parser_import_does_not_load_opencv() -> None:
    """校验公共 parser 导入在无 libGL 的 headless 环境中不加载 OpenCV。"""
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys


class BlockCv2Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 模拟 headless Linux 缺少 libGL 时 cv2 的导入错误。
        if fullname == "cv2" or fullname.startswith("cv2."):
            raise ImportError("libGL.so.1: cannot open shared object file: No such file or directory")
        return None


sys.meta_path.insert(0, BlockCv2Finder())
import mineru.parser

assert "cv2" not in sys.modules
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_hybrid_local_runtime_uses_context_names_and_local_lock_env() -> None:
    """校验 Hybrid 本地模型运行时只暴露 context 命名，并优先读取新的本地锁环境变量。"""
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import os

os.environ["MINERU_ENABLE_LOCAL_MODEL_INFERENCE_LOCKS"] = "true"
os.environ["MINERU_ENABLE_PIPELINE_INFERENCE_LOCKS"] = "false"

from mineru.model.runtime import hybrid as local_model_runtime

assert hasattr(local_model_runtime, "HybridLocalModelContext")
assert hasattr(local_model_runtime, "HybridLocalModelContextSingleton")
for removed_name in [
    "Mineru" + "PipelineModel",
    "Mineru" + "HybridModel",
    "Hybrid" + "ModelSingleton",
]:
    assert not hasattr(local_model_runtime, removed_name)
assert local_model_runtime.LOCAL_MODEL_INFERENCE_LOCKS_ENABLED is True
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_effort_for_tier_rejects_unknown_tier() -> None:
    """校验 effort 由 tier 派生，非法 tier 被拒绝。"""
    from mineru.parser.tier import effort_for_tier

    assert effort_for_tier("basic") == "medium"
    assert effort_for_tier("standard") == "high"
    assert effort_for_tier("advanced") == "xhigh"
    with pytest.raises(ValueError, match="Unsupported tier 'ultra'"):
        effort_for_tier("ultra")


def test_tier_runtime_options_map_hybrid_effort() -> None:
    """校验 tier 到 effort 的共享映射，避免 API/Gradio 分叉维护。"""
    from mineru.parser.tier import runtime_options_for_tier

    assert runtime_options_for_tier("flash").effort == "flash"
    assert runtime_options_for_tier("basic").effort == "medium"
    assert runtime_options_for_tier("standard").effort == "high"
    assert runtime_options_for_tier("advanced").effort == "xhigh"


@pytest.mark.parametrize("module_name", ["torch", "vllm", "lmdeploy", "mlx_vlm", "onnxruntime", "mineru_llama_cpp"])
def test_standard_dependency_error_recommends_backend_extra(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    """各后端缺依赖时保留模块名称，并且只推荐项目中真实存在的安装 extra。"""
    import re

    import tomllib

    monkeypatch.setattr(parser_tier, "installed_distribution_name", lambda: "mineru")

    error = parser_tier.TierDependencyError("standard", [module_name])
    message = str(error)
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    recommended_extras = set(re.findall(r"mineru\[([^\]]+)\]", message))

    assert module_name in message
    assert "mineru[standard]" not in message
    assert recommended_extras == {"torch", "full"}
    assert recommended_extras <= project["optional-dependencies"].keys()
    assert "standard" not in project["optional-dependencies"]


def test_advanced_is_not_a_deployment_dependency_tier() -> None:
    with pytest.raises(ValueError, match="Unsupported deployment tier 'advanced'"):
        parser_tier.required_modules_for_tier("advanced")  # type: ignore[arg-type]


def test_public_tier_literals_match_product_contract() -> None:
    from mineru.parser.tier import runtime_options_for_tier

    assert TIERS == ("flash", "basic", "standard", "advanced")
    assert TIER_ORDER == {"flash": 0, "basic": 1, "standard": 2, "advanced": 3}
    assert validate_tier("advanced") == "advanced"
    with pytest.raises(ValueError, match="Unsupported tier 'ultra'"):
        runtime_options_for_tier("ultra")


def test_server_and_deployment_tier_constants_match_product_contract() -> None:
    assert SERVER_TIERS == ("flash", "basic", "standard", "advanced")
    assert SERVER_STARTUP_TIERS == ("auto", "flash", "basic", "standard", "advanced")
    assert DEPLOYMENT_TIERS == ("basic", "standard")
    assert TIERS_BY_SERVER_TIER == {
        "flash": ("flash",),
        "basic": ("flash", "basic"),
        "standard": ("flash", "basic", "standard"),
        "advanced": ("flash", "basic", "standard", "advanced"),
    }


def test_implicit_tier_selection_excludes_advanced() -> None:
    assert DEFAULT_QUALITY_TIER_SELECTION_ORDER == ("standard", "basic")
    assert QUALITY_TIERS == frozenset(("basic", "standard", "advanced"))
    assert PARSING_RULE_TIER_SELECTION_ORDER == ("standard", "basic", "flash")
    assert select_default_quality_tier(("basic", "advanced", "standard")) == "standard"
    assert select_default_quality_tier(("advanced", "basic")) == "basic"
    assert select_default_quality_tier(("advanced",)) is None
    assert select_parsing_rule_tier(("advanced", "basic", "flash")) == "basic"
    assert select_parsing_rule_tier(("advanced", "flash")) == "flash"


@pytest.mark.parametrize("tier", ["ultra", "experimental"])
def test_api_client_rejects_invalid_tier_names(tier: str) -> None:
    with pytest.raises(ValueError, match=f"Unsupported tier '{tier}'"):
        MinerUApiParser(api_url="http://localhost:8000", tier=tier)  # type: ignore[arg-type]


def test_api_client_uses_env_api_key_when_no_api_key_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_API_KEY", "env-key")

    parser = MinerUApiParser(api_url="https://mineru.net/api", tier="standard")

    assert parser._headers()["Authorization"] == "Bearer env-key"


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object], text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, object]:
        return self._payload


def test_api_client_maps_remote_401_to_invalid_api_key() -> None:
    response = _FakeResponse(
        401,
        {
            "traceId": "trace-1",
            "msgCode": "A0202",
            "msg": "user authenticate failed",
            "data": None,
            "success": False,
            "total": 0,
        },
        text='{"msgCode":"A0202","msg":"user authenticate failed"}',
    )

    with pytest.raises(api_client._V1APIError) as exc_info:
        MinerUApiParser._check(response)

    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.message == "Remote authentication failed: user authenticate failed"
    assert exc_info.value.param == "parse_server.remote.api_key"


def test_api_client_preserves_structured_error_body_on_http_error() -> None:
    response = _FakeResponse(
        401,
        {
            "error": {
                "code": "invalid_api_key",
                "message": "Invalid or missing API key",
                "param": "api_key",
            }
        },
        text='{"error":{"code":"invalid_api_key"}}',
    )

    with pytest.raises(api_client._V1APIError) as exc_info:
        MinerUApiParser._check(response)

    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.message == "Invalid or missing API key"
    assert exc_info.value.param == "api_key"


def test_api_client_ignores_legacy_detail_error_envelope() -> None:
    response = _FakeResponse(
        400,
        {
            "detail": {
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "message": "legacy detail envelope",
                }
            }
        },
        text='{"detail":{"error":{"code":"invalid_request"}}}',
    )

    with pytest.raises(api_client._V1APIError) as exc_info:
        MinerUApiParser._check(response)

    assert exc_info.value.code == "http_error"
    assert exc_info.value.message.startswith("HTTP 400:")


def test_api_client_disables_env_proxy_for_local_network_urls() -> None:
    assert should_trust_env_for_url("http://localhost:8000") is False
    assert should_trust_env_for_url("http://127.0.0.1:8000") is False
    assert should_trust_env_for_url("http://192.168.1.20:8000/api") is False
    assert should_trust_env_for_url("https://mineru.net/api") is True


def test_api_client_poll_timeout_budget_is_one_hour() -> None:
    """验证 v1 job 轮询最大等待预算为 1 小时。"""
    assert api_client._POLL_INTERVAL_SECONDS * api_client._POLL_MAX_ATTEMPTS == 60 * 60


def test_api_client_constructor_exposes_request_ocr_without_legacy_options() -> None:
    """验证 OCR 使用显式请求参数，且不恢复旧 method 或图片分析选项。"""
    parameters = inspect.signature(MinerUApiParser).parameters

    assert parameters["ocr_mode"].default is None
    assert "method" not in parameters
    assert "image_analysis" not in parameters


def test_api_client_accepts_middle_json_current_pages() -> None:
    """验证 API client 恢复严格 2.0 pages envelope；legacy pdf_info 由 accepts_remote_pdf_info 系列覆盖。"""
    pages = _pages_from_middle_json(_current_payload([{"page_idx": 2, "blocks": []}]))

    assert len(pages) == 1
    assert pages[0].page_idx == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"pages": [{"page_idx": 0, "blocks": []}]},
        {"schema_version": "3.0", "pages": [{"page_idx": 0, "blocks": []}]},
    ],
)
def test_api_client_rejects_unknown_schema_version(payload: dict[str, object]) -> None:
    """验证缺失版本与未知版本 payload 仍要求重新解析；pdf_info 走 legacy 迁移单独覆盖。"""
    with pytest.raises((ValueError, api_client._V1APIError), match="reparse the source document"):
        _pages_from_middle_json(payload)


def test_api_server_validation_errors_use_error_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path))
    client = TestClient(app)

    response = client.post("/v1/tasks", json={"files": []})

    assert response.status_code == 400
    body = response.json()
    assert "detail" not in body
    assert "type" not in body["error"]
    assert body.get("error")["code"] == "invalid_request"
    assert "Invalid request" in body.get("error")["message"]


def test_api_server_unhandled_exceptions_use_error_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path))

    @app.get("/boom")
    async def _boom() -> None:
        raise RuntimeError("boom")

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert "detail" not in body
    assert body.get("error") == {
        "code": "internal_error",
        "message": "Internal server error",
    }


def test_create_app_does_not_read_runtime_settings_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    monkeypatch.setenv("MINERU_TIER", "advanced")
    monkeypatch.setenv("MINERU_BACKEND", "hybrid-auto-engine")
    monkeypatch.setenv("MINERU_CONCURRENCY", "9")
    monkeypatch.setenv("MINERU_URL_TIMEOUT", "99")
    monkeypatch.setenv("MINERU_OCR_MODE", "ocr")
    monkeypatch.setenv("MINERU_EFFORT", "high")
    monkeypatch.setenv(_REMOVED_TABLE_ENABLE_ENV, "false")
    monkeypatch.setenv(_REMOVED_FORMULA_ENABLE_ENV, "false")
    monkeypatch.setenv("MINERU_IMAGE_ANALYSIS", "false")

    app = create_app(upload_dir=str(tmp_path))

    assert app.state.model_tier == "standard"
    assert app.state.concurrency == 1
    assert app.state.url_timeout == 60
    assert app.state.allow_http_source is False
    assert not hasattr(app.state, "ocr_mode")
    assert not hasattr(app.state, "language")
    assert app.state.tier_runtime_options["standard"].effort == "high"
    assert app.state.image_analysis is True
    assert not hasattr(app.state, _REMOVED_TABLE_ENABLE_PARAM)
    assert not hasattr(app.state, _REMOVED_FORMULA_ENABLE_PARAM)


def test_api_server_cli_no_longer_exposes_reload() -> None:
    result = runner.invoke(main, ["--reload"])

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--reload" in result.output


def test_usage_counts_completed_files_and_reports_configured_concurrency(tmp_path: Path) -> None:
    """每个真实文件执行计入 jobs_created，成功文件与并发上限保持已有公开语义。"""
    file_store = FileStore(tmp_path / "api-files")
    store = api_server.ExecutionStore(concurrency=2)
    records = []
    for name in ("a1.html", "a2.html", "b1.html"):
        source = tmp_path / name
        source.write_bytes(b"source")
        records.append(store.create(execution_request(file_store, stored_source(file_store, source), tier="flash"), file_store))
    completed, failed, other = records
    completed.status = other.status = "completed"
    failed.status = "failed"

    async def _report_while_busy() -> "api_server.UsageResponse":
        """占用一个名额后确认公开限制仍报告配置上限。"""
        await store._semaphore.acquire()
        return store.usage("anonymous")

    usage = asyncio.run(_report_while_busy())
    assert usage.current.jobs_created == 3
    assert usage.current.files_processed == 2
    assert usage.limits.max_concurrent_jobs == 2


def test_api_server_logs_traceback_when_job_file_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_parse_async(*args: object, **kwargs: object) -> ParseResult:
        raise RuntimeError("boom")

    monkeypatch.setattr("mineru.parser.api_server.parse_async", fake_parse_async)
    file_store = FileStore(tmp_path / "api-files")
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    request = execution_request(file_store, stored_source(file_store, source), tier="standard", output_formats=["default"])
    execution_store = api_server.ExecutionStore()
    rec = execution_store.create(request, file_store)
    mineru_logger = logging.getLogger("mineru")
    api_logger = logging.getLogger("mineru.parser.api_server")
    monkeypatch.setattr(mineru_logger, "disabled", False)
    monkeypatch.setattr(mineru_logger, "propagate", True)
    monkeypatch.setattr(api_logger, "disabled", False)
    monkeypatch.setattr(api_logger, "propagate", True)

    with caplog.at_level(logging.ERROR, logger="mineru.parser.api_server"):
        asyncio.run(api_server._run_execution(rec, file_store, image_analysis=True))

    assert rec.status == "failed"
    assert rec.error is not None
    assert rec.error.message == "boom"
    assert "Parse-server job file failed" in caplog.text
    assert f"job_id={rec.id}" in caplog.text
    assert "RuntimeError: boom" in caplog.text


def test_parse_info_uses_parser_version_not_backend_version() -> None:
    """解析统计包含实际档位、耗时和解析器版本，不接受已删除的模型字段。"""
    parse = FileParseInfo(tier="basic", duration_ms=12, parser_version="3.2.1")
    assert parse.model_dump(exclude_none=True) == {
        "tier": "basic",
        "duration_ms": 12,
        "parser_version": "3.2.1",
    }

    for removed in ("backend_version", "model_used"):
        with pytest.raises(ValidationError):
            FileParseInfo.model_validate({"tier": "basic", removed: "unused"})


@pytest.mark.parametrize("field", ["backend_version", "parser_version"])
def test_health_response_rejects_nonstandard_version_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        HealthResponse.model_validate({"status": "ok", "version": "3.2.1", field: "3.2.1"})


def test_api_server_tier_selects_compatible_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    flash_app = create_app(upload_dir=str(tmp_path / "flash"), tiers=("flash",))
    basic_app = create_app(upload_dir=str(tmp_path / "basic"), tiers=("flash", "basic"))
    standard_app = create_app(upload_dir=str(tmp_path / "standard"), tiers=("flash", "basic", "standard", "advanced"))

    assert flash_app.state.model_tier is None
    assert [tier["id"] for tier in flash_app.state.tiers] == ["flash"]
    assert basic_app.state.model_tier == "basic"
    assert basic_app.state.tier_runtime_options["basic"].effort == "medium"
    assert [tier["id"] for tier in basic_app.state.tiers] == ["flash", "basic"]
    assert standard_app.state.model_tier == "standard"
    assert standard_app.state.tier_runtime_options["standard"].effort == "high"
    assert [tier["id"] for tier in standard_app.state.tiers] == ["flash", "basic", "standard", "advanced"]


def test_api_server_defaults_to_hardware_recommendation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """默认使用推荐集合，模型配置与逐档执行参数保持一致。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path))

    assert app.state.model_tier == "standard"
    assert app.state.default_tier == "standard"
    assert app.state.tier_runtime_options["standard"].effort == "high"
    assert [tier["id"] for tier in app.state.tiers] == ["flash", "basic", "standard", "advanced"]
    assert app.state.tier_runtime_options["flash"].effort == "flash"
    assert app.state.tier_runtime_options["basic"].effort == "medium"
    assert app.state.tier_runtime_options["standard"].effort == "high"
    assert app.state.tier_runtime_options["advanced"].effort == "xhigh"


def test_api_server_standard_no_flash_state_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 Standard 能力上限发布三个模型档位，并可严格禁用 Flash。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tiers=("basic", "standard", "advanced"))

    assert app.state.model_tier == "standard"
    assert app.state.default_tier == "standard"
    assert app.state.tier_runtime_options["standard"].effort == "high"
    assert not hasattr(app.state, "flash_enabled")
    assert [tier["id"] for tier in app.state.tiers] == ["basic", "standard", "advanced"]
    assert app.state.model_ids == [api_server._VIRTUAL_MODEL_ID]
    assert api_server._VIRTUAL_MODEL_ID == f"mineru-{api_server.__version__}"
    assert app.state.tier_runtime_options["basic"].effort == "medium"
    assert app.state.tier_runtime_options["standard"].effort == "high"
    assert app.state.tier_runtime_options["advanced"].effort == "xhigh"


@pytest.mark.parametrize(
    ("tiers", "default"),
    [
        (("flash", "standard"), "standard"),
        (("standard",), "standard"),
        (("basic",), "basic"),
        (("flash",), "flash"),
        (("advanced",), "advanced"),
    ],
)
def test_api_server_explicit_capability_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiers: tuple[Tier, ...], default: Tier | None
) -> None:
    """精确集合独立控制能力发现、执行 runtime 与请求默认档位。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tiers=tiers)
    assert tuple(tier_info["id"] for tier_info in app.state.tiers) == tiers
    assert tuple(app.state.tier_runtime_options) == tiers
    assert app.state.default_tier == default


def test_api_server_standard_http_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 /v1/tiers 与 /v1/models 返回 Standard 上限展开后的全部能力。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tiers=("flash", "basic", "standard", "advanced"))

    with TestClient(app) as client:
        health_response = client.get("/v1/health")
        tiers_response = client.get("/v1/tiers")
        models_response = client.get("/v1/models")
        model_detail_response = client.get(f"/v1/models/{api_server._VIRTUAL_MODEL_ID}")

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert "models" not in health_response.json()
    assert "parser_version" not in health_response.json()
    assert "runtime" not in health_response.json()
    assert tiers_response.status_code == 200
    assert [tier["id"] for tier in tiers_response.json()["data"]] == ["flash", "basic", "standard", "advanced"]
    assert all(tier["current_model"] == api_server._VIRTUAL_MODEL_ID for tier in tiers_response.json()["data"])
    assert models_response.status_code == 200
    assert [model["id"] for model in models_response.json()["data"]] == [api_server._VIRTUAL_MODEL_ID]
    assert models_response.json()["data"][0]["description"] == api_server._VIRTUAL_MODEL_DESCRIPTION
    assert model_detail_response.status_code == 200
    assert model_detail_response.json()["id"] == api_server._VIRTUAL_MODEL_ID
    assert model_detail_response.json()["created"] == app.state.model_registered_at


def test_api_server_models_created_is_stable_across_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回归：created 是部署注册时间，两次请求之间不随墙钟跳动。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tiers=("flash", "basic", "standard", "advanced"))

    with TestClient(app) as client:
        first = client.get("/v1/models").json()["data"][0]["created"]
        second = client.get("/v1/models").json()["data"][0]["created"]

    assert first == second == app.state.model_registered_at


def test_api_server_model_detail_rejects_unknown_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """非虚拟模型 id 一律 404，包括历史上的引擎品牌名。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tiers=("flash", "basic", "standard", "advanced"))

    with TestClient(app) as client:
        response = client.get("/v1/models/MinerU2.5-Pro-2605-1.2B")

    assert response.status_code == 404
    assert response.json().get("error")["code"] == "model_not_found"


def test_api_server_no_advanced_http_metadata_keeps_shared_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证能力接口隐藏 Advanced；虚拟模型 id 不随档位开关变化。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(upload_dir=str(tmp_path), tiers=("flash", "basic", "standard"))

    with TestClient(app) as client:
        tiers_response = client.get("/v1/tiers")
        models_response = client.get("/v1/models")

    assert tiers_response.status_code == 200
    assert [tier["id"] for tier in tiers_response.json()["data"]] == ["flash", "basic", "standard"]
    assert models_response.status_code == 200
    assert [model["id"] for model in models_response.json()["data"]] == [api_server._VIRTUAL_MODEL_ID]


def test_api_server_model_preload_failure_keeps_health_diagnostics_and_rejects_capabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)

    def _fail_preload(startup_tier: DeploymentTier, *, vlm_config: VlmConfig | None = None) -> api_server._ModelPreloadResult:
        """模拟包含 VLM 初始化在内的服务预加载失败。"""
        raise ValueError("CUDA is not available.")

    monkeypatch.setattr(api_server, "_preload_server_models", _fail_preload)
    app = create_app(upload_dir=str(tmp_path), tiers=("flash", "basic", "standard", "advanced"), preload_models=True)

    with TestClient(app) as client:
        health_response = client.get("/v1/health")
        tiers_response = client.get("/v1/tiers")
        models_response = client.get("/v1/models")

    assert health_response.status_code == 503
    assert health_response.json().get("error")["code"] == "model_preload_device_unavailable"
    assert health_response.json().get("error")["message"] == "CUDA is not available."
    assert tiers_response.status_code == 503
    assert tiers_response.json().get("error")["code"] == "model_preload_device_unavailable"
    assert models_response.status_code == 503
    assert models_response.json().get("error")["code"] == "model_preload_device_unavailable"
    for response in (health_response, tiers_response, models_response):
        assert set(response.json()["error"]) == {"code", "message"}


def test_api_server_model_preload_is_opt_in_and_ignored_for_flash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    calls: list[str] = []

    def _preload(startup_tier: DeploymentTier, *, vlm_config: VlmConfig | None = None) -> api_server._ModelPreloadResult:
        """记录预加载调用，兼容显式 VLM 配置传入。"""
        calls.append(startup_tier)
        return api_server._ModelPreloadResult(tier=startup_tier, engine="test")

    monkeypatch.setattr(api_server, "_preload_server_models", _preload)

    with TestClient(create_app(upload_dir=str(tmp_path / "lazy"), tiers=("flash", "basic"))) as lazy_client:
        assert lazy_client.get("/v1/health").status_code == 200
    with TestClient(
        create_app(upload_dir=str(tmp_path / "preload"), tiers=("flash", "basic"), preload_models=True)
    ) as preload_client:
        assert preload_client.get("/v1/health").status_code == 200
    with TestClient(create_app(upload_dir=str(tmp_path / "flash"), tiers=("flash",), preload_models=True)) as flash_client:
        assert flash_client.get("/v1/health").status_code == 200

    assert calls == ["basic"]


def test_api_server_standard_jobs_use_requested_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Standard 服务按每个文件请求选择档位，并从实际解析结果记录执行信息。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    calls: list[dict[str, object]] = []

    async def fake_parse_async(*args: object, **kwargs: object) -> ParseResult:
        """记录生产解析调用，并返回相应档位的最小有效协议。"""
        calls.append(dict(kwargs))
        middle_json = _full_middle_json(PageInfo(page_idx=0))
        middle_json.extensions["mineru"] = {"tier": kwargs["tier"], "parse_mode": "txt"}
        return ParseResult(middle_json=middle_json)

    monkeypatch.setattr("mineru.parser.api_server.parse_async", fake_parse_async)
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = create_app(upload_dir=str(tmp_path / "api"), tiers=("flash", "basic", "standard", "advanced"))
    file_store = FileStore(tmp_path / "api-files")
    execution_store = api_server.ExecutionStore()

    async def run_request(tier: Tier | None = None) -> TaskExecution:
        """构造并运行一个真实内部执行请求，返回有类型快照。"""
        request = execution_request(file_store, stored_source(file_store, source), tier=tier or app.state.default_tier)
        rec = execution_store.create(request, file_store)
        await api_server._run_execution(
            rec,
            file_store,
            image_analysis=app.state.image_analysis,
            allow_http_source=app.state.allow_http_source,
        )
        return execution_store.snapshot(rec)

    default_response = asyncio.run(run_request())
    basic_response = asyncio.run(run_request("basic"))
    advanced_response = asyncio.run(run_request("advanced"))
    assert default_response.parse is not None and default_response.parse.tier == "standard"
    assert basic_response.parse is not None and basic_response.parse.tier == "basic"
    assert advanced_response.parse is not None and advanced_response.parse.tier == "advanced"
    assert [call["tier"] for call in calls] == ["standard", "basic", "advanced"]


def test_api_server_basic_rejects_advanced_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 Basic 能力上限不接受 Advanced 请求。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = create_app(upload_dir=str(tmp_path / "api"), tiers=("flash", "basic"))

    with TestClient(app) as client:
        response = client.post(
            "/v1/tasks",
            json={
                "files": [{"file_name": source.name, "bytes": source.stat().st_size}],
                "tier": "advanced",
            },
        )

    assert response.status_code == 400
    assert "Tier 'advanced' is not available" in response.text


def test_api_server_no_advanced_rejects_advanced_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Standard server 禁用 Advanced 后不会只隐藏能力而仍允许执行。"""
    _stub_api_server_dependency_preflight(monkeypatch)
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = create_app(
        upload_dir=str(tmp_path / "api"),
        tiers=("flash", "basic", "standard"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/tasks",
            json={
                "files": [{"file_name": source.name, "bytes": source.stat().st_size}],
                "tier": "advanced",
            },
        )

    assert response.status_code == 400
    assert "Tier 'advanced' is not available" in response.text


def test_api_server_no_flash_rejects_flash_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = create_app(
        upload_dir=str(tmp_path / "api"),
        tiers=("basic", "standard", "advanced"),
    )

    response = TestClient(app).post(
        "/v1/tasks",
        json={
            "files": [{"file_name": source.name, "bytes": source.stat().st_size}],
            "tier": "flash",
        },
    )

    assert response.status_code == 400
    assert "Tier 'flash' is not available" in response.text


def test_api_server_rejects_invalid_startup_tier_configuration(tmp_path: Path) -> None:
    """拒绝非法集合与已移除的旧启动参数。"""
    with pytest.raises(ValueError, match="Unsupported tier"):
        create_app(upload_dir=str(tmp_path / "invalid"), tiers=("ultra",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty collection"):
        create_app(upload_dir=str(tmp_path / "string"), tiers="standard")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty collection"):
        create_app(upload_dir=str(tmp_path / "empty"), tiers=())
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        create_app(tier="standard")  # type: ignore[call-arg]


def test_api_server_skips_dependency_preflight_for_flash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_import_module(module_name: str) -> None:
        pytest.fail(f"Flash startup unexpectedly imported dependency module: {module_name}")

    monkeypatch.setattr(importlib, "import_module", fail_import_module)

    create_app(upload_dir=str(tmp_path), tiers=("flash",))


def test_api_server_preflights_basic_tier_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imported_modules: list[str] = []

    def fake_import_module(module_name: str) -> object:
        imported_modules.append(module_name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(api_server.get_config().model, "small_backend", "torch")
    monkeypatch.setattr(api_server.get_config().model.vlm, "engine", "llama-cpp")

    create_app(upload_dir=str(tmp_path), tiers=("flash", "basic"))

    assert imported_modules == [
        "onnxruntime",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
    ]


def test_api_server_preflights_standard_tier_dependencies_for_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imported_modules: list[str] = []

    def fake_import_module(module_name: str) -> object:
        imported_modules.append(module_name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(api_server.get_config().model, "small_backend", "torch")
    monkeypatch.setattr(api_server.get_config().model.vlm, "engine", "llama-cpp")

    create_app(upload_dir=str(tmp_path), tiers=("flash", "basic", "standard", "advanced"))

    assert imported_modules == [
        "onnxruntime",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
        "mineru_llama_cpp",
    ]


def test_api_server_preflights_standard_tier_dependencies_skip_mlx_on_intel_macos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    imported_modules: list[str] = []

    def fake_import_module(module_name: str) -> object:
        imported_modules.append(module_name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(api_server.get_config().model, "small_backend", "torch")
    monkeypatch.setattr(api_server.get_config().model.vlm, "engine", "llama-cpp")

    create_app(upload_dir=str(tmp_path), tiers=("flash", "basic", "standard", "advanced"))

    assert imported_modules == [
        "onnxruntime",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
        "mineru_llama_cpp",
    ]


def test_api_server_preflight_rejects_missing_tier_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_import_module(module_name: str) -> object:
        if module_name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(api_server.get_config().model, "small_backend", "torch")
    monkeypatch.setattr(parser_tier.importlib_metadata, "packages_distributions", lambda: {"mineru": ["mineru"]})

    with pytest.raises(api_server.ParseServerStartupError, match="tier 'basic'.*torch.*mineru\\[torch\\]"):
        create_app(upload_dir=str(tmp_path), tiers=("flash", "basic"))


def test_api_server_create_app_rejects_backend_and_effort_parameters(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="backend"):
        create_app(upload_dir=str(tmp_path / "backend"), backend="pipeline")  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="effort"):
        create_app(upload_dir=str(tmp_path / "effort"), effort="high")  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="language"):
        create_app(upload_dir=str(tmp_path / "language"), language="ch")  # type: ignore[call-arg]


def test_api_server_stores_parser_runtime_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_server_dependency_preflight(monkeypatch)
    app = create_app(
        upload_dir=str(tmp_path),
        tiers=("flash", "basic"),
        image_analysis=False,
    )

    assert not hasattr(app.state, "ocr_mode")
    assert not hasattr(app.state, "language")
    assert app.state.tier_runtime_options["basic"].effort == "medium"
    assert app.state.image_analysis is False
    assert not hasattr(app.state, _REMOVED_TABLE_ENABLE_PARAM)
    assert not hasattr(app.state, _REMOVED_FORMULA_ENABLE_PARAM)


def test_api_server_cli_exposes_parser_runtime_options() -> None:
    option_names = {name for param in main.params for name in param.opts}

    assert "--tier" in option_names
    assert "--tiers" not in option_names
    assert "--backend" not in option_names
    assert "--language" not in option_names
    assert "--ocr-mode" not in option_names
    assert "--allow-local-source" not in option_names
    assert "--max-inline-bytes" not in option_names
    assert "--allow-http-source" in option_names
    assert "--effort" not in option_names
    assert "--disable-image-analysis" in option_names
    assert "--preload-models" in option_names
    assert _REMOVED_DISABLE_TABLE_OPTION not in option_names
    assert _REMOVED_DISABLE_FORMULA_OPTION not in option_names


def test_api_server_cli_rejects_backend_and_effort_options() -> None:
    backend_result = runner.invoke(main, ["--backend", "hybrid-engine"])
    effort_result = runner.invoke(main, ["--effort", "high"])

    assert backend_result.exit_code != 0
    assert "No such option" in backend_result.output
    assert "--backend" in backend_result.output
    assert effort_result.exit_code != 0
    assert "No such option" in effort_result.output
    assert "--effort" in effort_result.output


def test_api_server_cli_defaults_to_hardware_recommendation(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 省略集合时使用推荐结果，默认请求仍选择 standard。"""
    seen: dict[str, object] = {}

    def _fake_run(server: Any) -> None:
        """记录无 --tier 启动后的自动推荐能力，不启动真实服务。"""
        seen["tiers"] = [tier["id"] for tier in server.config.app.state.tiers]
        seen["default_tier"] = server.config.app.state.default_tier
        seen["effort_by_tier"] = {
            tier: runtime.effort for tier, runtime in server.config.app.state.tier_runtime_options.items()
        }

    monkeypatch.setattr("uvicorn.Server.run", _fake_run)

    result = runner.invoke(main, ["--host", "0.0.0.0", "--port", "15984"])

    assert result.exit_code == 0
    assert seen == {
        "tiers": ["flash", "basic", "standard", "advanced"],
        "default_tier": "standard",
        "effort_by_tier": {"flash": "flash", "basic": "medium", "standard": "high", "advanced": "xhigh"},
    }


def test_api_server_cli_accepts_advanced_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advanced 上限包含全部 PDF/图片档位。"""
    seen: dict[str, object] = {}

    def _fake_run(server: Any) -> None:
        """记录显式质量档位集合及其默认请求档位。"""
        seen["tiers"] = [tier["id"] for tier in server.config.app.state.tiers]
        seen["default_tier"] = server.config.app.state.default_tier
        seen["effort_by_tier"] = {
            tier: runtime.effort for tier, runtime in server.config.app.state.tier_runtime_options.items()
        }

    monkeypatch.setattr("uvicorn.Server.run", _fake_run)

    result = runner.invoke(main, ["--tier", "advanced", "--host", "0.0.0.0", "--port", "15984"])

    assert result.exit_code == 0
    assert seen == {
        "tiers": ["flash", "basic", "standard", "advanced"],
        "default_tier": "standard",
        "effort_by_tier": {"flash": "flash", "basic": "medium", "standard": "high", "advanced": "xhigh"},
    }


def test_api_server_cli_accepts_list_without_advanced(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证精确列表同时约束能力发布与执行 runtime。"""
    seen: dict[str, object] = {}

    def _fake_run(server: Any) -> None:
        """记录禁用 Advanced 后的 API server 能力。"""
        seen["tiers"] = [tier["id"] for tier in server.config.app.state.tiers]
        seen["default_tier"] = server.config.app.state.default_tier
        seen["effort_by_tier"] = {
            tier: runtime.effort for tier, runtime in server.config.app.state.tier_runtime_options.items()
        }

    monkeypatch.setattr("uvicorn.Server.run", _fake_run)

    result = runner.invoke(main, ["--tier", "standard", "--host", "0.0.0.0", "--port", "15984"])

    assert result.exit_code == 0
    assert seen == {
        "tiers": ["flash", "basic", "standard"],
        "default_tier": "standard",
        "effort_by_tier": {"flash": "flash", "basic": "medium", "standard": "high"},
    }


def test_api_server_cli_enables_model_preload(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def _fake_run(server: Any) -> None:
        seen["preload_models"] = server.config.app.state.preload_models

    monkeypatch.setattr("uvicorn.Server.run", _fake_run)

    result = runner.invoke(main, ["--tier", "basic", "--preload-models"])

    assert result.exit_code == 0
    assert seen == {"preload_models": True}


def test_api_server_cli_rejects_invalid_tier_names() -> None:
    result = runner.invoke(main, ["--tier", "ultra"])

    assert result.exit_code != 0
    assert "Unsupported tier 'ultra'" in result.output


def test_api_server_cli_rejects_old_options() -> None:
    """旧单值与禁用参数均被 CLI 拒绝。"""
    result = runner.invoke(main, ["--tier", "flash", "--no-flash"])

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_api_server_cli_rejects_removed_language_option() -> None:
    result = runner.invoke(main, ["--language", "en"])

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--language" in result.output


def test_api_server_cli_help_exposes_tier_only_runtime_selection() -> None:
    result = runner.invoke(main, ["--help"])
    normalized_output = result.output.replace("-\n                                  ", "-")

    assert result.exit_code == 0
    assert "--tier" in normalized_output
    assert "--tiers" not in normalized_output
    assert "flash" in normalized_output
    assert "basic" in normalized_output
    assert "standard" in normalized_output
    assert "advanced" in normalized_output
    assert "--no-advanced" not in normalized_output
    assert "--no-flash" not in normalized_output
    assert "--backend" not in normalized_output
    assert "--effort" not in normalized_output
    assert "hybrid-engine" not in normalized_output
    assert "pipeline" not in normalized_output
    assert "vlm-engine" not in normalized_output
    assert "vlm-http-client" not in normalized_output
    assert "hybrid-auto-engine" not in normalized_output
    assert "vlm-auto-engine" not in normalized_output


def test_api_server_public_exports_are_runtime_entrypoints_only() -> None:
    assert set(api_server.__all__) == {"create_app", "main"}
