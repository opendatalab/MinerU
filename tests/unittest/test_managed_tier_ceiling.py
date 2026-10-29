"""托管上限及模型配置分离，覆盖 WebUI、Router 与 Doclib 的 Advanced 启动。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mineru.doclib import server
from mineru.doclib.background.parse_server_health import api_server_args_for_tier
from mineru.doclib.config_schema import validate_config_value
from mineru.errors import InvalidRequestError
from mineru.kit.gradio import app as gradio_app
from mineru.kit.gradio.client import ManagedLocalApiServer, V1ServerCapabilities
from mineru.kit.router.workers import ManagedLocalWorker, RouterSettings
from mineru.types import SERVER_STARTUP_TIERS, ManagedServerTier, ServerStartupTier
from pathlib import Path


@pytest.mark.parametrize("tier", ["basic", "standard", "advanced"])
def test_managed_ceiling_is_forwarded(tier: ManagedServerTier, tmp_path: Path) -> None:
    """所有托管方原样传递单档上限，不额外展开或补齐 Advanced。"""
    validate_config_value("parse_server.local.managed_tier", tier)
    assert server._validate_managed_parse_server_tier(tier, "tier") == tier
    commands = [
        api_server_args_for_tier(tier, host="127.0.0.1", port=8000),
        ManagedLocalApiServer(tier=tier)._command(8000, tmp_path),
        ManagedLocalWorker("worker", "127.0.0.1", None, RouterSettings(worker_tier=tier)).command(8000, tmp_path),
    ]
    for command in commands:
        assert command[command.index("--tier") + 1] == tier
        assert not {"--tiers", "--no-flash", "--no-advanced"}.intersection(command)


@pytest.mark.parametrize(("tier", "model_tier"), [("basic", "basic"), ("standard", "standard"), ("advanced", "standard")])
def test_managed_model_preparation_uses_shared_profile(
    monkeypatch: pytest.MonkeyPatch, tier: ManagedServerTier, model_tier: str
) -> None:
    """Advanced 的依赖和模型检查复用 Standard，产品上限仍保留 Advanced。"""
    dependencies, models = Mock(), Mock(return_value=[])
    monkeypatch.setattr(server, "ensure_tier_runtime_dependencies", dependencies)
    monkeypatch.setattr(server, "model_repos_for_tier", models)
    server._ensure_managed_parse_server_tier_available(tier, "tier")
    dependencies.assert_called_once_with(model_tier)
    models.assert_called_once_with(model_tier)


def test_advanced_missing_models_suggests_standard_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型下载入口只有模型配置档位，Advanced 缺文件时不能提示不存在的参数。"""
    monkeypatch.setattr(server, "model_repos_for_tier", Mock(return_value=[SimpleNamespace(name="shared-vlm")]))
    monkeypatch.setattr(server, "verify_model_repo", Mock(return_value=SimpleNamespace(ready=False, missing_paths=["weights"])))
    with pytest.raises(InvalidRequestError) as error:
        server._ensure_managed_parse_server_models_ready("advanced", "tier")
    assert "--tier standard" in str(error.value)
    assert "--tier advanced" not in str(error.value)


@pytest.mark.parametrize("tier", SERVER_STARTUP_TIERS)
def test_remote_standard_only_gradio_never_starts_local_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tier: ServerStartupTier
) -> None:
    """连接 SaaS 仅 Standard 能力时只使用远端，不补起本地服务或扩展档位。"""
    capabilities = V1ServerCapabilities("http://remote.test", ("standard",), ("default",), ("upload",))
    client = SimpleNamespace(capabilities=capabilities, discover=AsyncMock(return_value=capabilities))
    managed = Mock(side_effect=AssertionError("must not start local fallback"))
    build = Mock(return_value=object())
    monkeypatch.setattr(gradio_app, "V1ArtifactClient", Mock(return_value=client))
    monkeypatch.setattr(gradio_app, "ManagedLocalApiServer", managed)
    monkeypatch.setattr(gradio_app, "build_gradio_app", build)
    monkeypatch.setattr(gradio_app, "_launch_gradio_app", Mock())
    gradio_app.launch_gradio(
        api_url=capabilities.base_url,
        api_key=None,
        server_name="127.0.0.1",
        server_port=None,
        output_dir=str(tmp_path),
        enable_example=False,
        enable_api=True,
        tier=tier,
        api_server_concurrency=1,
        api_server_disable_image_analysis=False,
        api_server_preload_models=False,
    )
    managed.assert_not_called()
    assert build.call_args.args[1].tiers == ("standard",)
