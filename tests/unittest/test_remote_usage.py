from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from mineru.cli import guidance
from mineru.cli.commands import usage
from mineru.cli.main import app
from mineru.doclib.remote_api import ResolvedRemoteApiKey, resolve_remote_api_key
from mineru.doclib.server import DoclibServer
from mineru.doclib.types import ConfigResponse, ConfigValueResponse, RemoteUsageResponse
from mineru.errors import MineruError
from mineru.parser.api_client import MinerUApiParser

runner = CliRunner()


def _usage_response(access_level: str = "anonymous") -> RemoteUsageResponse:
    return RemoteUsageResponse.model_validate(
        {
            "object": "usage",
            "access_level": access_level,
            "billing_period": {"start": "2026-07-14T00:00:00Z", "end": "2026-07-15T00:00:00Z"},
            "current": {"pages_processed": 26, "files_processed": 2, "jobs_created": 8},
            "limits": {
                "max_pages_per_file": 200,
                "max_file_size_bytes": 209715200,
                "max_files_per_job": 200,
                "max_concurrent_jobs": 500,
                "max_file_retention_days": 30,
            },
        }
    )


class _UsageClient:
    remote_url = guidance.OFFICIAL_REMOTE_API_URL
    response = _usage_response()

    def __init__(self, *, timeout: int) -> None:
        assert timeout == 30

    def get_config_key(self, key: str) -> ConfigValueResponse:
        assert key == guidance.REMOTE_API_URL_CONFIG
        return ConfigValueResponse(key=key, value=self.remote_url, source="default")

    def get_remote_usage(self) -> RemoteUsageResponse:
        return self.response


class _GuidanceConfigClient:
    remote_url = guidance.OFFICIAL_REMOTE_API_URL
    api_key = ""

    def __init__(self, *, timeout: int) -> None:
        assert timeout == 3

    def get_config(self) -> ConfigResponse:
        return ConfigResponse(
            config={
                guidance.REMOTE_API_URL_CONFIG: self.remote_url,
                guidance.REMOTE_API_KEY_CONFIG: self.api_key,
            },
            sources={},
        )


def test_remote_api_key_resolver_prefers_override(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ConfigService:
        async def get(self, key: str) -> str:
            assert key == "parse_server.remote.api_key"
            return "config-key"

    monkeypatch.setenv("MINERU_API_KEY", "env-key")

    result = asyncio.run(resolve_remote_api_key(_ConfigService()))  # type: ignore[arg-type]

    assert result == ResolvedRemoteApiKey(value="config-key", source="override")


def test_remote_api_key_resolver_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ConfigService:
        async def get(self, key: str) -> str:
            assert key == "parse_server.remote.api_key"
            return ""

    monkeypatch.setenv("MINERU_API_KEY", "env-key")

    result = asyncio.run(resolve_remote_api_key(_ConfigService()))  # type: ignore[arg-type]

    assert result == ResolvedRemoteApiKey(value="env-key", source="environment")


def test_remote_api_key_resolver_returns_anonymous(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ConfigService:
        async def get(self, key: str) -> str:
            assert key == "parse_server.remote.api_key"
            return ""

    monkeypatch.delenv("MINERU_API_KEY", raising=False)

    result = asyncio.run(resolve_remote_api_key(_ConfigService()))  # type: ignore[arg-type]

    assert result == ResolvedRemoteApiKey(value=None, source="anonymous")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://mineru.net/api", True),
        ("https://mineru.net/api/", True),
        ("https://mineru.net:443/api", True),
        ("http://mineru.net/api", False),
        ("https://staging.mineru.net/api", False),
        ("https://mineru.net/api/proxy", False),
        ("https://mineru.net/api?target=custom", False),
    ],
)
def test_official_remote_url_detection_is_strict(url: str, expected: bool) -> None:
    assert guidance.is_official_remote_url(url) is expected


def test_api_client_get_usage_async_uses_usage_endpoint_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    class _Response:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return _usage_response().model_dump()

    class _AsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["trust_env"] is True

        async def __aenter__(self) -> "_AsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str]) -> _Response:
            calls.append((url, headers))
            return _Response()

    monkeypatch.setattr("mineru.parser.api_client.httpx.AsyncClient", _AsyncClient)
    parser = MinerUApiParser(api_url=guidance.OFFICIAL_REMOTE_API_URL, api_key="api-key")

    result = asyncio.run(parser.get_usage_async())

    assert result["object"] == "usage"
    assert calls == [("https://mineru.net/api/v1/usage", {"Authorization": "Bearer api-key"})]


def test_doclib_remote_usage_maps_remote_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ConfigService:
        async def get(self, key: str) -> str:
            return {
                "parse_server.remote.url": guidance.OFFICIAL_REMOTE_API_URL,
                "parse_server.remote.api_key": "bad-key",
            }[key]

    async def fail_usage(_self: MinerUApiParser) -> dict[str, Any]:
        from mineru.parser.api_client import _V1APIError

        raise _V1APIError("invalid_api_key", "Invalid API Key.", "parse_server.remote.api_key")

    monkeypatch.setattr(MinerUApiParser, "get_usage_async", fail_usage)
    server = DoclibServer(SimpleNamespace(config_svc=_ConfigService()))

    with pytest.raises(MineruError) as exc_info:
        asyncio.run(server.get_remote_usage())

    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.param == "parse_server.remote.api_key"


def test_doclib_remote_usage_uses_resolved_environment_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_api_keys: list[str | None] = []

    class _ConfigService:
        async def get(self, key: str) -> str:
            return {
                "parse_server.remote.url": guidance.OFFICIAL_REMOTE_API_URL,
                "parse_server.remote.api_key": "",
            }[key]

    async def get_usage(parser: MinerUApiParser) -> dict[str, Any]:
        seen_api_keys.append(parser._api_key)
        return _usage_response("registered").model_dump()

    monkeypatch.setenv("MINERU_API_KEY", "environment-key")
    monkeypatch.setattr(MinerUApiParser, "get_usage_async", get_usage)
    server = DoclibServer(SimpleNamespace(config_svc=_ConfigService()))

    response = asyncio.run(server.get_remote_usage())

    assert response.access_level == "registered"
    assert seen_api_keys == ["environment-key"]


def test_doclib_config_key_reports_environment_api_key_without_exposing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ConfigService:
        async def get(self, key: str) -> str:
            assert key == "parse_server.remote.api_key"
            return ""

        async def get_source(self, key: str) -> str:
            assert key == "parse_server.remote.api_key"
            return "default"

    monkeypatch.setenv("MINERU_API_KEY", "environment-secret")
    server = DoclibServer(SimpleNamespace(config_svc=_ConfigService()))

    response = asyncio.run(server.get_config_key("parse_server.remote.api_key"))

    assert response.source == "environment"
    assert response.value != "environment-secret"
    assert "******" in response.value


def test_usage_json_wraps_remote_fields_and_adds_optional_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usage, "DoclibClient", _UsageClient)

    result = runner.invoke(app, ["usage", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["remote_url"] == "https://mineru.net/api"
    assert payload["usage"]["object"] == "usage"
    assert payload["usage"]["access_level"] == "anonymous"
    assert payload["usage"]["current"]["pages_processed"] == 26
    assert payload["guidance"]["type"] == "configure_official_api_key"
    assert payload["guidance"]["required"] is False
    assert "guidance_text" not in payload


def test_usage_json_returns_null_guidance_for_registered_access(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RegisteredClient(_UsageClient):
        response = _usage_response("registered")

    monkeypatch.setattr(usage, "DoclibClient", _RegisteredClient)

    result = runner.invoke(app, ["usage", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)["guidance"] is None


def test_usage_json_returns_null_official_guidance_for_custom_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CustomClient(_UsageClient):
        remote_url = "https://example.com/api"

    monkeypatch.setattr(usage, "DoclibClient", _CustomClient)

    result = runner.invoke(app, ["usage", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)["guidance"] is None


def test_usage_human_output_shows_complete_remote_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usage, "DoclibClient", _UsageClient)

    result = runner.invoke(app, ["usage"])

    assert result.exit_code == 0
    assert "Remote API Usage" in result.stdout
    assert "Remote URL: https://mineru.net/api" in result.stdout
    assert "Billing period: 2026-07-14 00:00 UTC - 2026-07-15 00:00 UTC" in result.stdout
    assert "Max file size: 200 MiB" in result.stdout
    assert "File retention: 30 days" in result.stdout
    assert guidance.OFFICIAL_API_KEY_URL in result.stdout


def test_usage_invalid_api_key_keeps_error_and_adds_required_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    class _InvalidClient(_UsageClient):
        def get_remote_usage(self) -> RemoteUsageResponse:
            raise MineruError("invalid_api_key", "Invalid API Key.", "parse_server.remote.api_key")

    monkeypatch.setattr(usage, "DoclibClient", _InvalidClient)
    monkeypatch.setattr(guidance, "DoclibClient", _GuidanceConfigClient)

    result = runner.invoke(app, ["usage", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == {
        "type": "authentication_error",
        "code": "invalid_api_key",
        "message": "Invalid API Key.",
        "param": "parse_server.remote.api_key",
    }
    assert payload["guidance"]["required"] is True


def test_usage_invalid_api_key_human_output_separates_error_and_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    class _InvalidClient(_UsageClient):
        def get_remote_usage(self) -> RemoteUsageResponse:
            raise MineruError("invalid_api_key", "Invalid API Key.", "parse_server.remote.api_key")

    monkeypatch.setattr(usage, "DoclibClient", _InvalidClient)
    monkeypatch.setattr(guidance, "DoclibClient", _GuidanceConfigClient)

    result = runner.invoke(app, ["usage"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Error: Invalid API Key." in result.stderr
    assert "Manage or create an API Key:" in result.stderr
    assert guidance.OFFICIAL_API_KEY_URL in result.stderr
    assert guidance.SET_API_KEY_COMMAND in result.stderr


@pytest.mark.parametrize("error_code", ["invalid_api_key", "feature_requires_api_key"])
def test_auth_and_feature_errors_require_api_key_guidance(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
) -> None:
    monkeypatch.setattr(guidance, "DoclibClient", _GuidanceConfigClient)

    result = guidance.api_key_guidance_for_error(MineruError(error_code, "API Key required."))

    assert result is not None
    assert result.data["required"] is True


@pytest.mark.parametrize("error_code", ["rate_limit_exceeded", "quota_exceeded"])
def test_anonymous_limit_errors_offer_optional_api_key_guidance(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
) -> None:
    monkeypatch.setattr(guidance, "DoclibClient", _GuidanceConfigClient)

    result = guidance.api_key_guidance_for_error(MineruError(error_code, "Limit reached."))

    assert result is not None
    assert result.data["required"] is False


def test_registered_rate_limit_does_not_offer_api_key_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RegisteredConfigClient(_GuidanceConfigClient):
        api_key = "******"

    monkeypatch.setattr(guidance, "DoclibClient", _RegisteredConfigClient)

    assert guidance.api_key_guidance_for_error(MineruError("rate_limit_exceeded", "Retry later.")) is None


def test_custom_remote_error_does_not_offer_official_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CustomConfigClient(_GuidanceConfigClient):
        remote_url = "https://example.com/api"

    monkeypatch.setattr(guidance, "DoclibClient", _CustomConfigClient)

    assert guidance.api_key_guidance_for_error(MineruError("invalid_api_key", "Invalid API Key.")) is None
