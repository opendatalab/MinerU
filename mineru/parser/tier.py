# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from ..types import Tier

__all__ = [
    "PARSER_BACKENDS",
    "backend_for_tier",
    "resolve_tier_and_backend",
    "tier_for_backend",
]

PARSER_BACKENDS = (
    "pipeline",
    "flash",
    "vlm-auto-engine",
    "vlm-http-client",
    "vlm-transformers",
    "vlm-vllm-engine",
    "vlm-vllm-async-engine",
    "vlm-lmdeploy-engine",
    "vlm-mlx-engine",
    "hybrid-auto-engine",
    "hybrid-http-client",
    "hybrid-transformers",
    "hybrid-vllm-engine",
    "hybrid-vllm-async-engine",
    "hybrid-lmdeploy-engine",
    "hybrid-mlx-engine",
)


def backend_for_tier(tier: Tier) -> str:
    """Return the local parser backend used for a parser-layer tier fallback."""
    mapping = {
        "flash": "flash",
        "standard": "pipeline",
        "pro": "hybrid-auto-engine",
    }
    return mapping.get(tier, "pipeline")


def tier_for_backend(backend: str) -> Tier:
    if backend == "flash":
        return "flash"
    if backend == "pipeline":
        return "standard"
    if backend.startswith("vlm-") or backend.startswith("hybrid-"):
        return "pro"
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(PARSER_BACKENDS)}")


def _backend_supports_tier(backend: str, tier: Tier) -> bool:
    if backend not in PARSER_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(PARSER_BACKENDS)}")
    if tier == "flash":
        return backend == "flash"
    if tier == "standard":
        return backend == "pipeline"
    if tier == "pro":
        return backend.startswith("vlm-") or backend.startswith("hybrid-")
    return False


def resolve_tier_and_backend(tier: Tier | None = None, backend: str | None = None) -> tuple[Tier, str]:
    """Resolve public tier and optional expert backend into an executable parser backend."""
    resolved_tier: Tier = tier or "pro"
    if backend:
        if tier is None:
            return tier_for_backend(backend), backend
        if not _backend_supports_tier(backend, resolved_tier):
            raise ValueError(f"tier '{resolved_tier}' is incompatible with backend '{backend}'")
        return tier_for_backend(backend), backend
    return resolved_tier, backend_for_tier(resolved_tier)
