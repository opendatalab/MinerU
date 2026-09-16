"""Remote API runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .services.config_svc import ConfigService

REMOTE_API_KEY_CONFIG = "parse_server.remote.api_key"
REMOTE_API_KEY_ENV = "MINERU_API_KEY"

RemoteApiKeySource = Literal["override", "environment", "anonymous"]


@dataclass(frozen=True)
class ResolvedRemoteApiKey:
    value: str | None
    source: RemoteApiKeySource


async def resolve_remote_api_key(config_svc: ConfigService) -> ResolvedRemoteApiKey:
    """Resolve one effective Remote API Key for all doclib Remote callers."""
    configured = (await config_svc.get(REMOTE_API_KEY_CONFIG)) or ""
    if configured:
        return ResolvedRemoteApiKey(value=configured, source="override")

    environment = os.environ.get(REMOTE_API_KEY_ENV, "")
    if environment:
        return ResolvedRemoteApiKey(value=environment, source="environment")

    return ResolvedRemoteApiKey(value=None, source="anonymous")


__all__ = [
    "REMOTE_API_KEY_CONFIG",
    "REMOTE_API_KEY_ENV",
    "RemoteApiKeySource",
    "ResolvedRemoteApiKey",
    "resolve_remote_api_key",
]
