# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from typing import Final

CANONICAL_HYBRID_ENGINE: Final = "hybrid-engine"
DEFAULT_BACKEND: Final = CANONICAL_HYBRID_ENGINE
DEFAULT_HYBRID_EFFORT: Final = "medium"
DEFAULT_EFFORT: Final = DEFAULT_HYBRID_EFFORT
HYBRID_EFFORT_HELP: Final = "Low uses local Hybrid processing. Medium is faster. High is more accurate and may take longer."

LOCAL_BACKEND_CHOICES: Final[tuple[str, ...]] = (
    CANONICAL_HYBRID_ENGINE,
)
HTTP_CLIENT_BACKEND_CHOICES: Final[tuple[str, ...]] = (
    "hybrid-http-client",
)
PUBLIC_BACKEND_CHOICES: Final[tuple[str, ...]] = LOCAL_BACKEND_CHOICES + HTTP_CLIENT_BACKEND_CHOICES
HYBRID_EFFORT_CHOICES: Final[tuple[str, ...]] = ("low", "medium", "high")
BACKEND_SCHEMA_EXTRA: Final[dict[str, list[str]]] = {"enum": list(PUBLIC_BACKEND_CHOICES)}
HYBRID_EFFORT_SCHEMA_EXTRA: Final[dict[str, list[str]]] = {"enum": list(HYBRID_EFFORT_CHOICES)}

BACKEND_ALIASES: Final[dict[str, str]] = {
    "hybrid-auto-engine": CANONICAL_HYBRID_ENGINE,
    "pipeline": CANONICAL_HYBRID_ENGINE,
    "vlm-engine": CANONICAL_HYBRID_ENGINE,
    "vlm-auto-engine": CANONICAL_HYBRID_ENGINE,
    "vlm-http-client": "hybrid-http-client",
}
LEGACY_PIPELINE_BACKEND_ALIASES: Final[frozenset[str]] = frozenset({"pipeline"})
LEGACY_VLM_BACKEND_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "vlm-engine",
        "vlm-auto-engine",
        "vlm-http-client",
    }
)

SUPPORTED_BACKENDS: Final[tuple[str, ...]] = PUBLIC_BACKEND_CHOICES + ("flash",)

SUPPORTED_EFFORTS: Final[tuple[str, ...]] = HYBRID_EFFORT_CHOICES


def normalize_backend(backend: str | None) -> str:
    """规范化 backend 名称，并将旧 VLM backend 兼容映射到 Hybrid backend。"""
    normalized = (backend or "").strip()
    if normalized in BACKEND_ALIASES:
        return BACKEND_ALIASES[normalized]
    if normalized in SUPPORTED_BACKENDS:
        return normalized
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}")


def normalize_public_backend(backend: str | None) -> str:
    """规范化公开 CLI/API backend 名称，但不暴露 flash 和旧 VLM 选项。"""
    normalized = (backend or "").strip()
    if normalized in BACKEND_ALIASES:
        return BACKEND_ALIASES[normalized]
    if normalized in PUBLIC_BACKEND_CHOICES:
        return normalized
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(PUBLIC_BACKEND_CHOICES)}")


def validate_effort(effort: str | None) -> str:
    """校验 Hybrid effort 级别，入口层统一使用 medium 作为默认值。"""
    normalized = (effort or DEFAULT_EFFORT).strip().lower()
    if normalized in SUPPORTED_EFFORTS:
        return normalized
    raise ValueError(f"Unsupported effort '{effort}'. Supported efforts: {', '.join(SUPPORTED_EFFORTS)}")


def resolve_backend_and_effort(backend: str | None, effort: str | None = None) -> tuple[str, str]:
    """同时解析 backend 与 effort，旧 pipeline/VLM backend 分别统一转为 Hybrid low/high。"""
    raw_backend = (backend or "").strip()
    resolved_backend = normalize_backend(raw_backend)
    resolved_effort = validate_effort(effort)
    if raw_backend in LEGACY_PIPELINE_BACKEND_ALIASES:
        resolved_effort = "low"
    elif raw_backend in LEGACY_VLM_BACKEND_ALIASES:
        resolved_effort = "high"
    return resolved_backend, resolved_effort


def is_hybrid_backend(backend: str) -> bool:
    """判断规范化后的 backend 是否属于 Hybrid 后端族。"""
    return normalize_backend(backend).startswith("hybrid-")
