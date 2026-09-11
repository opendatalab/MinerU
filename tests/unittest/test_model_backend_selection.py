"""验证安装标记、独立后端选择、资源组合与服务预检的一致性。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from pydantic import ValidationError
from typer.testing import CliRunner

from mineru.config import Config, VlmConfig, _load_effective_config, config
from mineru.kit.commands.models import app
from mineru.model import registry
from mineru.model.runtime import device
from mineru.model.vlm import selector
from mineru.parser import tier as parser_tier
from scripts.check_transformers5_dependencies import check_backend_dependencies

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@pytest.mark.parametrize(
    ("platform", "accelerator", "packages", "small", "vlm"),
    [
        ("darwin", "mps", {"torch", "mlx_vlm"}, "torch", "llama-cpp"),
        ("darwin", "cpu", set(), "onnx", "llama-cpp"),
        ("linux", "cpu", set(), "onnx", "llama-cpp"),
        ("win32", "cpu", set(), "onnx", "llama-cpp"),
        ("linux", "cuda", {"torch", "vllm", "lmdeploy"}, "torch", "vllm"),
        ("linux", "cuda", {"torch", "lmdeploy"}, "torch", "lmdeploy"),
        ("win32", "cuda", {"torch", "lmdeploy"}, "torch", "lmdeploy"),
        ("linux", "cuda", {"torch"}, "torch", "llama-cpp"),
        ("win32", "cuda", {"torch"}, "torch", "llama-cpp"),
        ("linux", "cpu", {"torch", "vllm"}, "onnx", "llama-cpp"),
        ("win32", "cpu", {"torch", "lmdeploy"}, "onnx", "llama-cpp"),
        ("other", "npu", {"torch"}, "torch", "llama-cpp"),
    ],
)
def test_automatic_platform_combinations(
    monkeypatch: pytest.MonkeyPatch, platform: str, accelerator: str, packages: set[str], small: str, vlm: str
) -> None:
    """依赖存在与设备可用分别参与自动选择，不通过 extra 安装历史推断。"""
    installed = packages | (set(device.TORCH_REQUIRED_MODULES) if "torch" in packages else set())
    monkeypatch.setattr(device, "module_available", lambda name: name in installed)
    monkeypatch.setattr(selector, "module_available", lambda name: name in installed)
    monkeypatch.setattr(device, "get_device", lambda: accelerator)
    monkeypatch.setattr(selector, "get_device", lambda: accelerator)
    monkeypatch.setattr(selector, "is_linux_environment", lambda: platform == "linux")
    monkeypatch.setattr(selector, "is_windows_environment", lambda: platform == "win32")
    assert device.resolve_small_model_backend("auto") == small
    assert selector.resolve_vlm_engine("auto") == vlm


@pytest.mark.parametrize("small", ["onnx", "torch"])
@pytest.mark.parametrize("vlm", ["llama-cpp", "vllm", "lmdeploy", "mlx"])
def test_explicit_cross_backend_downloads_without_installed_engines(
    monkeypatch: pytest.MonkeyPatch, small: str, vlm: str
) -> None:
    """任意显式组合都能离线确定仓库，不导入引擎或探测设备。"""
    unavailable = Mock(side_effect=AssertionError("Unexpected runtime probe"))
    monkeypatch.setattr(device, "get_device", unavailable)
    monkeypatch.setattr(selector, "get_device", unavailable)
    repos = registry.model_repos_for_tier("standard", small_backend=small, vlm_engine=vlm)
    assert repos[0].name == f"MinerU-4_models_{small}"
    assert repos[1].name.endswith("-GGUF") == (vlm == "llama-cpp")
    assert registry.model_repos_for_tier("basic", small_backend=small, vlm_engine=vlm) == repos[:1]


def test_config_environment_precedence_and_legacy_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量分别覆盖 YAML 后端配置，旧 stack 明确提示迁移。"""
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  small_backend: onnx\n  vlm:\n    engine: llama-cpp\n", encoding="utf-8")
    monkeypatch.setenv("MINERU_CONFIG", str(path))
    monkeypatch.setenv("MINERU_MODEL_SMALL_BACKEND", "torch")
    monkeypatch.setenv("MINERU_MODEL_VLM_ENGINE", "mlx")
    loaded = _load_effective_config()
    assert loaded.config.model.small_backend == "torch"
    assert loaded.config.model.vlm.engine == "mlx"
    assert loaded.sources[("model", "small_backend")] == "env"
    assert loaded.sources[("model", "vlm", "engine")] == "env"
    with pytest.raises(ValidationError, match="model.stack was removed"):
        Config(model={"stack": "light"})
    path.write_text("model:\n  stack: full\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="model.stack was removed"):
        _load_effective_config()
    monkeypatch.setenv("MINERU_MODEL_STACK", "auto")
    with pytest.raises(ValueError, match="MINERU_MODEL_STACK was removed"):
        _load_effective_config()


def test_old_cli_stack_option_is_removed() -> None:
    """模型管理不再接受旧 stack 参数。"""
    result = CliRunner().invoke(app, ["show", "--stack", "light"])
    assert result.exit_code != 0
    assert "No such option" in result.output


@pytest.mark.parametrize("engine", ["llama-cpp", "vllm", "lmdeploy", "mlx"])
def test_preflight_uses_supplied_vlm_config(monkeypatch: pytest.MonkeyPatch, engine: str) -> None:
    """显式 VLM 配置同时决定预检依赖与资源集合，不被全局设置覆盖。"""
    monkeypatch.setattr(config.model, "small_backend", "torch")
    monkeypatch.setattr(config.model.vlm, "engine", "auto")
    settings = VlmConfig(engine=engine)
    modules = parser_tier.required_modules_for_tier("standard", vlm_config=settings)
    assert set(modules) == {"onnxruntime", *device.TORCH_REQUIRED_MODULES, *selector.VLM_REQUIRED_MODULES[engine]}
    assert registry.model_repos_for_tier("standard", vlm_config=settings)[1] is registry.vlm_model_repo(engine)
    from mineru.model.vlm import client, runtime

    predictor = object()
    factory = Mock(return_value=predictor)
    monkeypatch.setattr(runtime.ModelSingleton, "get_model", factory)
    actual_predictor, backend = client.get_vlm_predictor(settings)
    assert actual_predictor is predictor
    assert backend == selector.get_vlm_engine(engine, is_async=True)
    assert factory.call_args.kwargs["backend"] == backend
    assert selector.get_vlm_engine(engine) == ("llama-cpp-engine" if engine == "llama-cpp" else f"{engine}-engine")


def test_remote_vlm_needs_no_local_engine_or_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """远程地址优先，不因未安装本地 MLX 而阻断服务或下载无用 VLM 权重。"""
    monkeypatch.setattr(config.model, "small_backend", "onnx")
    settings = VlmConfig(server_url="http://localhost:9000", engine="mlx")
    assert parser_tier.required_modules_for_tier("standard", vlm_config=settings) == ["onnxruntime"]
    assert registry.model_repos_for_tier("standard", vlm_config=settings) == (registry.MINERU_4_MODELS_ONNX,)
    repos = registry.model_repos_for_tier("standard", vlm_engine="llama-cpp", vlm_config=settings)
    assert repos[-1] is registry.MINERU_2_5_PRO_2605_1_2B_GGUF


def test_explicit_missing_engine_reports_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式引擎缺失时报错，不改用可用的 llama 掩盖配置问题。"""
    monkeypatch.setattr(config.model, "small_backend", "onnx")

    def missing_engine(name: str) -> object:
        """只模拟选定引擎缺失，避免加载真实 GPU 模块。"""
        if name == "vllm":
            raise ModuleNotFoundError(name, name=name)
        return object()

    monkeypatch.setattr(parser_tier.importlib, "import_module", missing_engine)
    with pytest.raises(parser_tier.TierDependencyError, match="vllm"):
        parser_tier.ensure_tier_runtime_dependencies("standard", vlm_config=VlmConfig(engine="vllm"))


@pytest.mark.parametrize(
    ("small_backend", "engine", "missing_module"),
    [
        ("torch", "llama-cpp", "torch"),
        ("onnx", "vllm", "vllm"),
        ("onnx", "lmdeploy", "lmdeploy"),
        ("onnx", "mlx", "mlx"),
        ("onnx", "mlx", "mlx_vlm"),
        ("onnx", "llama-cpp", "onnxruntime"),
        ("onnx", "llama-cpp", "mineru_llama_cpp"),
    ],
)
def test_standard_dependency_errors_survive_startup_wrappers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, small_backend: str, engine: str, missing_module: str
) -> None:
    """模拟实际后端缺失，验证 API Server 和 Doclib 原样保留安装提示与错误分类。"""
    from mineru.doclib import server as doclib_server
    from mineru.errors import InvalidRequestError
    from mineru.parser import api_server

    settings = VlmConfig(engine=engine)
    monkeypatch.setattr(config.model, "small_backend", small_backend)
    monkeypatch.setattr(config.model, "vlm", settings)
    monkeypatch.setattr(parser_tier, "installed_distribution_name", lambda: "mineru")

    def import_without_dependency(name: str) -> object:
        """只令目标依赖缺失，避免加载模型和真实设备运行时。"""
        if name == missing_module:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return object()

    monkeypatch.setattr(parser_tier.importlib, "import_module", import_without_dependency)
    with pytest.raises(parser_tier.TierDependencyError) as dependency_error:
        parser_tier.ensure_tier_runtime_dependencies("standard", vlm_config=settings)
    message = str(dependency_error.value)
    assert dependency_error.value.missing_modules == [missing_module]
    assert "tier 'standard'" in message
    assert "mineru[standard]" not in message
    assert "mineru[torch]" in message
    assert "mineru[full]" in message

    with pytest.raises(api_server.ParseServerStartupError) as startup_error:
        api_server.create_app(upload_dir=str(tmp_path), tier="standard", vlm_config=settings)
    assert str(startup_error.value) == message

    with pytest.raises(InvalidRequestError) as doclib_error:
        doclib_server._ensure_managed_parse_server_tier_available("standard", "value")
    assert doclib_error.value.code == "parse_server_dependency_missing"
    assert doclib_error.value.message == message
    assert doclib_error.value.param == "value"


def _resolved_direct_dependencies(platform: str, machine: str, extra: str) -> set[str]:
    """按目标平台展开本项目的递归 extra，不使用宿主环境的标记值。"""
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())["project"]
    environment = {**default_environment(), "sys_platform": platform, "platform_machine": machine}
    pending, visited, names = [extra], set(), set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        requirements = project["dependencies"] + project["optional-dependencies"].get(current, [])
        for raw in requirements:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({**environment, "extra": current}):
                continue
            if requirement.name == "mineru":
                pending.extend(requirement.extras)
            else:
                names.add(requirement.name)
    return names


@pytest.mark.parametrize(
    "platform,machine", [("darwin", "arm64"), ("darwin", "x86_64"), ("linux", "x86_64"), ("win32", "AMD64")]
)
@pytest.mark.parametrize("extra", ["", "torch", "full", "all"])
def test_package_markers(platform: str, machine: str, extra: str) -> None:
    """ARM 基础包引入完整 Torch 依赖，其他平台基础包保持 ONNX 与 llama。"""
    names = _resolved_direct_dependencies(platform, machine, extra)
    assert {"docvortex", "gradio", "onnxruntime", "mineru-llama-cpp"} <= names
    # lxml 由 DocVortex 声明，不在 MinerU 运行时 extras 中重复维护。
    assert "lxml" not in names
    expected_torch = (platform, machine) == ("darwin", "arm64") or extra in {"torch", "full", "all"}
    assert ("torch" in names) == expected_torch
    assert ("transformers" in names) == expected_torch
    assert ("vllm" in names) == (platform == "linux" and extra in {"full", "all"})
    assert ("lmdeploy" in names) == (platform == "win32" and extra in {"full", "all"})
    assert "mlx-vlm" not in names
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())["project"]
    assert "gradio" not in project["optional-dependencies"]


def test_real_resolution_policy_keeps_explicit_utils_mlx_separate() -> None:
    """真实 wheel 检查允许 ARM 基础 Torch，但不把 utils 的显式 MLX 当成 MinerU 默认安装。"""
    names = _resolved_direct_dependencies("darwin", "arm64", "")
    assert check_backend_dependencies(names, "mineru", "base", "macos") == []
    # 完整求解结果允许包含 DocVortex 带入的传递依赖。
    assert check_backend_dependencies(names | {"lxml"}, "mineru", "base", "macos") == []
    assert check_backend_dependencies(names - {"torch"}, "mineru", "base", "macos")
    assert check_backend_dependencies(names | {"mlx-vlm"}, "mineru", "full", "macos")
    assert check_backend_dependencies({"torch", "mlx-vlm"}, "mineru-vl-utils", "mlx", "macos") == []


@pytest.mark.parametrize("tier", ["basic", "standard"])
def test_download_rejects_invalid_vlm_engine_for_every_tier(tier: str) -> None:
    """basic 虽不下载 VLM，也不能静默接受拼错的显式引擎参数。"""
    result = CliRunner().invoke(app, ["download", "--tier", tier, "--small-backend", "onnx", "--vlm-engine", "invalid"])
    assert result.exit_code == 1
    assert "Unsupported VLM engine" in result.output
