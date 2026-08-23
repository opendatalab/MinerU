"""Structured CLI guidance for Official Remote API access."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from ..doclib.client import DoclibClient
from ..errors import MineruError
from .contracts import CliGuidance

OFFICIAL_REMOTE_API_URL = "https://mineru.net/api"
OFFICIAL_API_KEY_URL = "https://mineru.net/apiManage/token"
REMOTE_API_KEY_CONFIG = "parse_server.remote.api_key"
REMOTE_API_URL_CONFIG = "parse_server.remote.url"
SET_API_KEY_COMMAND = "mineru config set parse_server.remote.api_key '<API_KEY>'"


@dataclass(frozen=True)
class RemoteApiContext:
    remote_url: str
    api_key_configured: bool


def is_official_remote_url(url: str) -> bool:
    """Return whether ``url`` is the canonical Official API base URL."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() == "https"
        and parsed.hostname == "mineru.net"
        and port in (None, 443)
        and parsed.path.rstrip("/") == "/api"
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )


def load_remote_api_context() -> RemoteApiContext:
    """Read effective Remote API configuration through doclib."""
    config = DoclibClient(timeout=3).get_config()
    return RemoteApiContext(
        remote_url=config.config.get(REMOTE_API_URL_CONFIG, ""),
        api_key_configured=bool(config.config.get(REMOTE_API_KEY_CONFIG, "")),
    )


def api_key_guidance_for_error(error: MineruError) -> CliGuidance | None:
    """Return Official API Key guidance for one eligible Remote error."""
    if error.code not in {"invalid_api_key", "feature_requires_api_key", "rate_limit_exceeded", "quota_exceeded"}:
        return None

    context = load_remote_api_context()
    if not is_official_remote_url(context.remote_url):
        return None

    if error.code == "invalid_api_key":
        return _api_key_guidance(required=True, message="Configure a valid Official API Key to continue.")
    if error.code == "feature_requires_api_key":
        return _api_key_guidance(required=True, message="This Remote API feature requires an Official API Key.")
    if context.api_key_configured:
        return None
    if error.code == "rate_limit_exceeded":
        return _api_key_guidance(
            required=False,
            message="An Official API Key is optional and may provide registered rate limits.",
        )
    return _api_key_guidance(
        required=False,
        message="An Official API Key is optional and enables registered access.",
    )


def api_key_guidance_for_anonymous_usage(remote_url: str) -> CliGuidance | None:
    if not is_official_remote_url(remote_url):
        return None
    return _api_key_guidance(
        required=False,
        message="An Official API Key is optional and enables registered access.",
    )


def _api_key_guidance(*, required: bool, message: str) -> CliGuidance:
    return CliGuidance(
        data={
            "type": "configure_official_api_key",
            "required": required,
            "message": message,
            "url": OFFICIAL_API_KEY_URL,
            "command": SET_API_KEY_COMMAND,
        },
        text=(f"{message}\n\nManage or create an API Key:\n{OFFICIAL_API_KEY_URL}\n\nSet the API Key:\n{SET_API_KEY_COMMAND}"),
    )


__all__ = [
    "OFFICIAL_API_KEY_URL",
    "OFFICIAL_REMOTE_API_URL",
    "REMOTE_API_KEY_CONFIG",
    "REMOTE_API_URL_CONFIG",
    "RemoteApiContext",
    "SET_API_KEY_COMMAND",
    "api_key_guidance_for_anonymous_usage",
    "api_key_guidance_for_error",
    "is_official_remote_url",
    "load_remote_api_context",
]
