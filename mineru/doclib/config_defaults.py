"""Code-backed default values for doclib runtime config."""

from __future__ import annotations

from typing import Literal

ConfigSource = Literal["default", "override"]

CONFIG_DEFAULTS: dict[str, str] = {
    "parse_server.local.mode": "disabled",
    "parse_server.local.managed_tier": "high",
    "parse_server.local.self_hosted_url": "",
    "parse_server.local.self_hosted_api_key": "",
    "parse_server.remote.url": "https://mineru.net/api",
    "parse_server.remote.api_key": "",
}

CONFIG_SOURCE_DEFAULT: ConfigSource = "default"
CONFIG_SOURCE_OVERRIDE: ConfigSource = "override"
