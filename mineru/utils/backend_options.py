# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from typing import Final

CANONICAL_VLM_ENGINE: Final = "vlm-engine"
CANONICAL_HYBRID_ENGINE: Final = "hybrid-engine"
DEFAULT_BACKEND: Final = CANONICAL_HYBRID_ENGINE
DEFAULT_HYBRID_EFFORT: Final = "medium"
DEFAULT_EFFORT: Final = DEFAULT_HYBRID_EFFORT

LOCAL_BACKEND_CHOICES: Final[tuple[str, ...]] = (
    "pipeline",
    CANONICAL_VLM_ENGINE,
    CANONICAL_HYBRID_ENGINE,
)
HTTP_CLIENT_BACKEND_CHOICES: Final[tuple[str, ...]] = (
    "vlm-http-client",
    "hybrid-http-client",
)
HYBRID_EFFORT_CHOICES: Final[tuple[str, ...]] = ("medium", "high")

BACKEND_ALIASES: Final[dict[str, str]] = {
    "vlm-auto-engine": CANONICAL_VLM_ENGINE,
    "hybrid-auto-engine": CANONICAL_HYBRID_ENGINE,
}

SUPPORTED_BACKENDS: Final[tuple[str, ...]] = LOCAL_BACKEND_CHOICES + HTTP_CLIENT_BACKEND_CHOICES + ("flash",)

SUPPORTED_EFFORTS: Final[tuple[str, ...]] = HYBRID_EFFORT_CHOICES


def normalize_backend(backend: str | None) -> str:
    """规范化公开 backend 名称，并保留旧 auto-engine 名称作为兼容别名。"""
    normalized = (backend or "").strip()
    if normalized in BACKEND_ALIASES:
        return BACKEND_ALIASES[normalized]
    if normalized in SUPPORTED_BACKENDS:
        return normalized
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}")


def validate_effort(effort: str | None) -> str:
    """校验 Hybrid effort 级别，入口层统一使用 medium 作为默认值。"""
    normalized = (effort or DEFAULT_EFFORT).strip().lower()
    if normalized in SUPPORTED_EFFORTS:
        return normalized
    raise ValueError(f"Unsupported effort '{effort}'. Supported efforts: {', '.join(SUPPORTED_EFFORTS)}")


def is_vlm_backend(backend: str) -> bool:
    """判断规范化后的 backend 是否属于 VLM 后端族。"""
    return normalize_backend(backend).startswith("vlm-")


def is_hybrid_backend(backend: str) -> bool:
    """判断规范化后的 backend 是否属于 Hybrid 后端族。"""
    return normalize_backend(backend).startswith("hybrid-")
