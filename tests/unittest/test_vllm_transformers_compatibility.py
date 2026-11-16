"""验证 vLLM 启动前的版本检查及既有错误传播边界。"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from loguru import logger

from mineru.model.vlm import engine_utils
from mineru.model.vlm.errors import is_fatal_engine_error


@pytest.fixture
def dependency_versions(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """隔离发行包元数据，不要求测试环境安装真实 vLLM。"""
    versions = {"vllm": "0.28.0", "transformers": "5.23.0"}

    def read_version(package: str) -> str:
        """保留真实元数据查询在发行包不存在时的异常类型。"""
        if package not in versions:
            raise PackageNotFoundError(package)
        return versions[package]

    monkeypatch.setattr(
        engine_utils, "metadata", SimpleNamespace(version=read_version, PackageNotFoundError=PackageNotFoundError)
    )
    return versions


@pytest.fixture
def error_messages() -> Iterator[list[str]]:
    """仅收集检查器的 ERROR 日志，并在用例结束后移除测试日志接收器。"""
    messages: list[str] = []
    handler = logger.add(messages.append, level="ERROR", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler)


@pytest.fixture
def standalone_server(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """隔离服务模块及父包属性，避免真实导入干扰其他 CLI 用例的模块替身。"""
    from mineru.kit import vlm_server as server_package

    module_name = "mineru.kit.vlm_server.vllm_server"
    monkeypatch.setattr(server_package, "vllm_server", None, raising=False)
    # 先登记模块缓存的恢复点，再允许导入创建真实服务模块。
    monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.delitem(sys.modules, module_name)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("vllm_version", "transformers_version", "blocked"),
    [
        ("0.19.1", "5.23.0", True),
        ("0.28.0", "5.23.0", True),
        ("0.28.0", "5.24.0", True),
        ("0.19.1", "5.22.2", False),
        ("0.28.0", "5.22.2", False),
        ("0.29.0", "5.23.0", False),
        ("0.30.0", "5.24.0", False),
        ("0.29.0rc1", "5.23.0", True),
        ("0.28.0", "5.23.0rc1", False),
        ("0.28.0+cu130", "5.23.0+local", True),
        ("0.29.0+cu130", "5.23.0", False),
    ],
)
def test_version_boundary(
    dependency_versions: dict[str, str],
    error_messages: list[str],
    vllm_version: str,
    transformers_version: str,
    blocked: bool,
) -> None:
    """按 PEP 440 比较正式、预发布及本地版本，仅拒绝指定的不兼容组合。"""
    dependency_versions.update(vllm=vllm_version, transformers=transformers_version)
    if not blocked:
        engine_utils.ensure_vllm_transformers_compatibility()
        assert error_messages == []
        return

    with pytest.raises(RuntimeError) as caught:
        engine_utils.ensure_vllm_transformers_compatibility()
    message = str(caught.value)
    assert f"vllm=={vllm_version}" in message
    assert f"transformers=={transformers_version}" in message
    assert "Upgrade vLLM to >=0.29.0" in message
    assert "transformers>=5.10.1,<5.23.0" in message
    assert [entry.strip() for entry in error_messages] == [message]
    assert not is_fatal_engine_error(caught.value)


@pytest.mark.parametrize("package", ["vllm", "transformers"])
@pytest.mark.parametrize("invalid_value", [None, "not-a-version"])
def test_unverifiable_metadata_is_rejected(
    dependency_versions: dict[str, str],
    error_messages: list[str],
    package: str,
    invalid_value: str | None,
) -> None:
    """缺失或非法元数据必须明确失败，不能默认为兼容版本。"""
    if invalid_value is None:
        del dependency_versions[package]
    else:
        dependency_versions[package] = invalid_value
    with pytest.raises(RuntimeError) as caught:
        engine_utils.ensure_vllm_transformers_compatibility()
    message = str(caught.value)
    assert package in message
    assert "version metadata" in message
    assert "Install or reinstall" in message
    assert [entry.strip() for entry in error_messages] == [message]
    assert not is_fatal_engine_error(caught.value)


def test_check_does_not_import_inference_packages(
    monkeypatch: pytest.MonkeyPatch,
    dependency_versions: dict[str, str],
) -> None:
    """元数据检查的成功和失败路径均不得导入重型推理依赖。"""
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        """拒绝通过导入推理库获取版本，同时保留其他真实导入行为。"""
        assert name.split(".")[0] not in {"vllm", "transformers", "torch"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="incompatible"):
        engine_utils.ensure_vllm_transformers_compatibility()
    dependency_versions["vllm"] = "0.29.0"
    engine_utils.ensure_vllm_transformers_compatibility()


@pytest.mark.parametrize("backend", ["vllm-engine", "vllm-async-engine"])
def test_embedded_engine_rejects_before_startup_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    dependency_versions: dict[str, str],
    backend: str,
) -> None:
    """同步别名和异步入口均在设备探测、下载及运行时构造前停止。"""
    from mineru.model.vlm import runtime

    device = MagicMock()
    download = MagicMock()
    initialize = MagicMock()
    monkeypatch.setattr(runtime, "get_device", device)
    monkeypatch.setattr(type(runtime.MINERU_2_5_PRO_2605_1_2B), "ensure", download)
    monkeypatch.setattr(runtime.ModelSingleton, "_get_or_initialize", initialize)
    with pytest.raises(RuntimeError, match="incompatible"):
        runtime.ModelSingleton().get_model(backend, None, None)
    device.assert_not_called()
    download.assert_not_called()
    initialize.assert_not_called()


@pytest.mark.parametrize("backend", ["http-client", "llama-cpp-engine", "lmdeploy-engine", "mlx-engine", "transformers"])
def test_other_backends_skip_vllm_check(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """非 vLLM 后端不读取本地 vLLM 版本，也不受其兼容性约束影响。"""
    from mineru.model.vlm import runtime

    check = MagicMock(side_effect=AssertionError("Unexpected vLLM compatibility check"))
    initialize = MagicMock()
    monkeypatch.setattr(runtime, "ensure_vllm_transformers_compatibility", check)
    monkeypatch.setattr(runtime.ModelSingleton, "_get_or_initialize", initialize)
    result = runtime.ModelSingleton().get_model(backend, None, "http://unused")
    assert result is initialize.return_value
    check.assert_not_called()
    initialize.assert_called_once()


def test_standalone_server_stops_before_vllm_import_and_download(
    monkeypatch: pytest.MonkeyPatch,
    dependency_versions: dict[str, str],
    error_messages: list[str],
    standalone_server: ModuleType,
) -> None:
    """独立服务记录兼容性错误并非零退出，不导入 vLLM CLI 或下载模型。"""
    vllm_server = standalone_server

    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.main", None)
    download = MagicMock()
    logits = MagicMock()
    monkeypatch.setattr(type(vllm_server.MINERU_2_5_PRO_2605_1_2B), "ensure", download)
    monkeypatch.setattr(vllm_server, "enable_custom_logits_processors", logits)
    monkeypatch.setattr(sys, "argv", ["mineru-openai-server"])
    with pytest.raises(SystemExit) as caught:
        vllm_server.main()
    assert caught.value.code == 1
    assert len(error_messages) == 1
    assert "incompatible" in error_messages[0]
    download.assert_not_called()
    logits.assert_not_called()


def test_standalone_server_preserves_valid_startup(
    monkeypatch: pytest.MonkeyPatch,
    dependency_versions: dict[str, str],
    tmp_path: Path,
    standalone_server: ModuleType,
) -> None:
    """兼容组合继续下载模型并向 vLLM 转发原有服务参数。"""
    vllm_server = standalone_server

    dependency_versions["transformers"] = "5.22.2"
    entrypoint = MagicMock()
    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.cli", "vllm.entrypoints.cli.main"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setattr(sys.modules["vllm.entrypoints.cli.main"], "main", entrypoint, raising=False)
    download = MagicMock(return_value=tmp_path)
    monkeypatch.setattr(type(vllm_server.MINERU_2_5_PRO_2605_1_2B), "ensure", download)
    monkeypatch.setattr(vllm_server, "enable_custom_logits_processors", lambda: False)
    monkeypatch.setattr(vllm_server, "set_default_gpu_memory_utilization", lambda: 0.8)
    monkeypatch.delenv("MINERU_VLLM_DEVICE", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setattr(sys, "argv", ["mineru-openai-server", "--host", "127.0.0.1"])
    vllm_server.main()
    download.assert_called_once()
    entrypoint.assert_called_once_with()
    assert sys.argv == [
        "mineru-openai-server",
        "serve",
        str(tmp_path),
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--gpu-memory-utilization",
        "0.8",
    ]


def test_api_preload_retains_health_failure_without_engine_death(
    monkeypatch: pytest.MonkeyPatch,
    dependency_versions: dict[str, str],
    tmp_path: Path,
) -> None:
    """真实预加载调用链保留原错误文本，以既有 503 表示失败而不标记引擎死亡。"""
    from fastapi.testclient import TestClient

    from mineru.config import VlmConfig
    from mineru.model.vlm import runtime
    from mineru.parser import api_server
    from mineru.types import TIERS

    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "mineru.model.runtime.tier_recommendation.recommend_tiers", lambda **kwargs: SimpleNamespace(pdf_image_tiers=TIERS)
    )
    local_models = MagicMock()
    initialize = MagicMock()
    monkeypatch.setattr(api_server, "_preload_local_models", local_models)
    monkeypatch.setattr(runtime.ModelSingleton, "_get_or_initialize", initialize)
    app = api_server.create_app(
        upload_dir=str(tmp_path),
        tiers=("standard",),
        preload_models=True,
        vlm_config=VlmConfig(engine="vllm", server_url=""),
    )
    with TestClient(app) as client:
        for endpoint in ("/v1/health", "/v1/tiers", "/v1/models"):
            response = client.get(endpoint)
            assert response.status_code == 503
            error = response.json()["error"]
            assert error["code"] == "model_preload_failed"
            assert "vllm==0.28.0" in error["message"]
            assert "transformers==5.23.0" in error["message"]
            assert "transformers>=5.10.1,<5.23.0" in error["message"]
        assert app.state.execution_store.runtime_owner.engine_error is None
    local_models.assert_not_called()
    initialize.assert_not_called()
