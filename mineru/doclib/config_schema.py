"""Validation for doclib runtime config keys and values."""

from __future__ import annotations

from urllib.parse import urlparse

from ..errors import InvalidRequestError
from ..types import DEPLOYMENT_TIERS
from .config_defaults import CONFIG_DEFAULTS

LOCAL_PARSE_SERVER_MODES = ("disabled", "managed", "self_hosted")
URL_CONFIG_KEYS = ("parse_server.remote.url", "parse_server.local.self_hosted_url")


def validate_config_key(key: str) -> None:
    if key not in CONFIG_DEFAULTS:
        raise InvalidRequestError("invalid_config_key", f"Unknown config key: {key}", "key")


def validate_config_value(key: str, value: str) -> None:
    validate_config_key(key)
    if key == "parse_server.local.mode":
        _validate_enum_value(key, value, LOCAL_PARSE_SERVER_MODES)
        return
    if key == "parse_server.local.managed_tier":
        _validate_enum_value(key, value, DEPLOYMENT_TIERS)
        return
    if key in URL_CONFIG_KEYS:
        allow_empty = key == "parse_server.local.self_hosted_url"
        _validate_url_value(key, value, allow_empty=allow_empty)


def _validate_enum_value(key: str, value: str, allowed: tuple[str, ...]) -> None:
    if value in allowed:
        return
    expected = ", ".join(allowed)
    raise InvalidRequestError("invalid_config_value", f"{key} must be one of: {expected}.", "value")


def _validate_url_value(key: str, value: str, *, allow_empty: bool) -> None:
    if allow_empty and value == "":
        return
    parsed = urlparse(value)
    if parsed.scheme in ("http", "https") and parsed.hostname:
        return
    raise InvalidRequestError("invalid_config_value", f"{key} must be an http or https URL.", "value")
