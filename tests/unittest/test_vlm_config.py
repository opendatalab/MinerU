from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner as TyperRunner

from mineru.config import Config, VlmConfig, _load_effective_config
from mineru.kit.commands import api_server as kit_api
from mineru.kit.main import app as kit_app
from mineru.parser import MinerUParser, api_server


def test_vlm_config_defaults_and_environment_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 YAML 和标准环境变量的字段覆盖、配置来源以及无配置默认值。"""
    assert Config().model.vlm == VlmConfig(server_url="", api_key="", model="", http_timeout=600, max_concurrency=100)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "model:\n  vlm:\n    server_url: http://localhost:9000/proxy/v1\n"
        "    api_key: file-key\n    model: file-model\n    http_timeout: 30\n    max_concurrency: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MINERU_CONFIG", str(config_file))
    monkeypatch.setenv("MINERU_MODEL_VLM_API_KEY", "env-key")
    monkeypatch.setenv("MINERU_MODEL_VLM_HTTP_TIMEOUT", "45")
    loaded = _load_effective_config()
    assert loaded.config.model.vlm == VlmConfig(
        server_url="http://localhost:9000/proxy/", api_key="env-key", model="file-model", http_timeout=45, max_concurrency=2
    )
    assert loaded.sources[("model", "vlm", "server_url")] == "file"
    assert loaded.sources[("model", "vlm", "api_key")] == "env"
    assert "env-key" not in repr(loaded.config.model.vlm)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ""),
        ("  ", ""),
        ("http://localhost:9000", "http://localhost:9000/"),
        ("https://example.test/v1/", "https://example.test/"),
        ("https://example.test/proxy/v1", "https://example.test/proxy/"),
        ("https://example.test/proxy/", "https://example.test/proxy/"),
        ("http://[::1]:9000/v1", "http://[::1]:9000/"),
    ],
)
def test_vlm_url_normalization(value: str, expected: str) -> None:
    """验证服务根地址、v1、代理前缀和 IPv6 的稳定规范化。"""
    settings = VlmConfig(server_url=value)
    assert settings.server_url == expected
    assert VlmConfig.model_validate(settings.model_dump()) == settings


@pytest.mark.parametrize(
    "value",
    [
        "ftp://host",
        "host:9000",
        "http://",
        "http://host:bad",
        "http://user:pass@host",
        "http://host?q=1",
        "http://host/#x",
        "http://bad host",
    ],
)
def test_vlm_rejects_invalid_url(value: str) -> None:
    """非法服务地址在配置边界失败，避免错误地址进入 HTTP 客户端。"""
    with pytest.raises(ValidationError, match="HTTP"):
        VlmConfig(server_url=value)


@pytest.mark.parametrize("field", ["http_timeout", "max_concurrency"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_vlm_rejects_invalid_limits(field: str, value: object) -> None:
    """连接超时和推理并发只接受正整数。"""
    with pytest.raises(ValidationError):
        VlmConfig.model_validate({field: value})


@pytest.mark.parametrize(("env_name", "field"), [("MINERU_VL_API_KEY", "api_key"), ("MINERU_VL_MODEL_NAME", "model")])
def test_vlm_legacy_conflicts_are_explicit_and_redacted(monkeypatch: pytest.MonkeyPatch, env_name: str, field: str) -> None:
    """拒绝旧变量覆盖新配置，异常不包含任一凭据值，本地模式不受影响。"""
    monkeypatch.delenv("MINERU_VL_API_KEY", raising=False)
    monkeypatch.delenv("MINERU_VL_MODEL_NAME", raising=False)
    monkeypatch.setenv(env_name, "legacy-secret")
    with pytest.raises(ValueError, match=env_name) as caught:
        VlmConfig.model_validate({"server_url": "http://localhost:9000", field: "explicit-secret"}).validate_environment()
    assert "secret" not in str(caught.value)
    VlmConfig.model_validate({"server_url": "http://localhost:9000", field: "legacy-secret"}).validate_environment()
    VlmConfig().validate_environment()


@pytest.mark.parametrize("standalone", [False, True])
def test_vlm_options_forward_including_empty_values(monkeypatch: pytest.MonkeyPatch, standalone: bool) -> None:
    """新版命令及独立别名保留所有显式 VLM 值，包括用于清空配置的空字符串。"""
    calls: list[list[str]] = []

    def fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录命令桥接参数，不启动服务。"""
        calls.append(args)

    monkeypatch.setattr(api_server.main, "main", fake_main)
    options = [
        "--vlm-server-url",
        "",
        "--vlm-api-key",
        "",
        "--vlm-model",
        "alias",
        "--vlm-http-timeout",
        "12",
        "--vlm-max-concurrency",
        "3",
    ]
    if standalone:
        monkeypatch.setattr("sys.argv", ["mineru-api", *options])
        with pytest.raises(SystemExit) as caught:
            kit_api.main()
        assert caught.value.code == 0
    else:
        assert TyperRunner().invoke(kit_app, ["api-server", *options]).exit_code == 0
    for option, value in zip(options[::2], options[1::2]):
        assert calls[0][calls[0].index(option) + 1] == value


@pytest.mark.parametrize("url", ["http://cli.test/proxy/v1", ""])
def test_api_cli_merges_vlm_without_mutating_global(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """底层 CLI 按字段覆盖全局设置，并保持 API Key、全局配置和环境变量隔离。"""
    settings = VlmConfig(server_url="http://global.test", api_key="global-key", model="global-model", max_concurrency=7)
    monkeypatch.setattr(api_server.mineru_config.model, "vlm", settings)
    seen: dict[str, Any] = {}

    def capture_app(**kwargs: Any) -> object:
        """记录应用构造参数，用于核验 CLI 覆盖而不创建文件。"""
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(api_server, "create_app", capture_app)
    monkeypatch.setattr(api_server.uvicorn.Server, "run", lambda self: None)
    previous_env = dict(os.environ)
    result = CliRunner().invoke(
        api_server.main, ["--vlm-server-url", url, "--vlm-api-key", "", "--vlm-http-timeout", "9", "--api-key", "inbound-key"]
    )
    assert result.exit_code == 0, result.output
    assert seen["api_key"] == "inbound-key"
    assert seen["vlm_config"] == VlmConfig(server_url=url, api_key="", model="global-model", http_timeout=9, max_concurrency=7)
    assert settings.server_url == "http://global.test/"
    assert settings.api_key == "global-key"
    assert dict(os.environ) == previous_env


def test_vlm_app_and_parser_snapshot_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """应用及解析器捕获独立副本，显式空配置完整覆盖全局远程设置。"""
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", lambda tier: None)
    settings = VlmConfig(server_url="http://remote.test", model="alias")
    monkeypatch.setattr(api_server.mineru_config.model, "vlm", settings)
    app = api_server.create_app(upload_dir=str(tmp_path))
    parser = MinerUParser()
    local_parser = MinerUParser(vlm_config=VlmConfig())
    settings.model = "changed"
    assert parser.vlm_config.model == app.state.vlm_config.model == "alias"
    assert parser.vlm_config is not app.state.vlm_config
    assert local_parser.vlm_config.server_url == ""
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200
        assert app.state.vlm_config.model == "alias"
