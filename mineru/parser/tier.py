# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from ..types import Tier
from ..utils.backend_options import SUPPORTED_BACKENDS, is_hybrid_backend, normalize_backend

__all__ = [
    "PARSER_BACKENDS",
    "backend_for_tier",
    "resolve_tier_and_backend",
    "tier_for_backend",
]

PARSER_BACKENDS = SUPPORTED_BACKENDS


def backend_for_tier(tier: Tier) -> str:
    """Return the local parser backend used for a parser-layer tier fallback."""
    mapping = {
        "flash": "flash",
        "standard": "pipeline",
        "pro": "hybrid-engine",
    }
    return mapping.get(tier, "pipeline")


def tier_for_backend(backend: str) -> Tier:
    normalized_backend = normalize_backend(backend)
    if normalized_backend == "flash":
        return "flash"
    if normalized_backend == "pipeline":
        return "standard"
    if is_hybrid_backend(normalized_backend):
        return "pro"
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(PARSER_BACKENDS)}")


def _backend_supports_tier(backend: str, tier: Tier) -> bool:
    normalized_backend = normalize_backend(backend)
    if tier == "flash":
        return normalized_backend == "flash"
    if tier == "standard":
        return normalized_backend == "pipeline"
    if tier == "pro":
        return is_hybrid_backend(normalized_backend)
    return False


def resolve_tier_and_backend(tier: Tier | None = None, backend: str | None = None) -> tuple[Tier, str]:
    """Resolve public tier and optional expert backend into an executable parser backend."""
    resolved_tier: Tier = tier or "pro"
    if backend:
        normalized_backend = normalize_backend(backend)
        if tier is None:
            return tier_for_backend(normalized_backend), normalized_backend
        if not _backend_supports_tier(normalized_backend, resolved_tier):
            raise ValueError(f"tier '{resolved_tier}' is incompatible with backend '{backend}'")
        return tier_for_backend(normalized_backend), normalized_backend
    return resolved_tier, backend_for_tier(resolved_tier)
