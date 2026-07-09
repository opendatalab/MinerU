# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata

from ..types import Tier, validate_tier
from ..utils.backend_options import (
    CANONICAL_HYBRID_ENGINE,
    LEGACY_PIPELINE_BACKEND_ALIASES,
    LEGACY_VLM_BACKEND_ALIASES,
    LOCAL_HYBRID_EFFORT,
    SUPPORTED_BACKENDS,
    DEFAULT_HYBRID_EFFORT,
    effort_for_tier,
    is_hybrid_backend,
    normalize_backend,
    resolve_backend_and_effort,
)

__all__ = [
    "PARSER_BACKENDS",
    "ParserRuntimeOptions",
    "TierDependencyError",
    "backend_for_tier",
    "ensure_tier_runtime_dependencies",
    "installed_distribution_name",
    "missing_modules_for_tier",
    "required_modules_for_tier",
    "resolve_runtime_options",
    "resolve_tier_and_backend",
    "runtime_options_for_tier",
    "tier_for_backend",
]

PARSER_BACKENDS = SUPPORTED_BACKENDS

_STANDARD_REQUIRED_MODULES = [
    "ftfy",
    "shapely",
    "pyclipper",
    "six",
    "torch",
    "torchvision",
    "transformers",
]
_PRO_REQUIRED_MODULES_COMMON = [
    *_STANDARD_REQUIRED_MODULES,
    "accelerate",
]
_PRO_REQUIRED_MODULES_BY_PLATFORM = {
    "linux": ["vllm"],
    "win32": ["lmdeploy", "qwen_vl_utils"],
    "darwin": ["mlx", "mlx_vlm"],
}


class TierDependencyError(RuntimeError):
    def __init__(self, tier: Tier, missing_modules: list[str]) -> None:
        self.tier = tier
        self.missing_modules = missing_modules
        missing = ", ".join(missing_modules)
        package_name = installed_distribution_name()
        super().__init__(
            f"Parse server cannot start for tier '{tier}'; missing runtime dependencies: {missing}. "
            f"Install optional dependencies for this tier in the same Python environment as MinerU, "
            f"for example: pip install '{package_name}[{tier}]'."
        )


@dataclass(frozen=True)
class ParserRuntimeOptions:
    """记录某个 tier 实际执行时使用的 backend 与 effort。"""

    tier: Tier
    backend: str
    effort: str

    def as_kwargs(self) -> dict[str, str]:
        """转换为 parser 调用可直接展开的关键参数，方便测试和调用端复用。"""
        return {
            "tier": self.tier,
            "backend": self.backend,
            "effort": self.effort,
        }


def backend_for_tier(tier: Tier) -> str:
    """返回指定 tier 使用的 parser backend，tier 自身决定质量档位。"""
    tier = validate_tier(tier)
    mapping = {
        "flash": "flash",
        "medium": CANONICAL_HYBRID_ENGINE,
        "high": CANONICAL_HYBRID_ENGINE,
        "xhigh": CANONICAL_HYBRID_ENGINE,
    }
    return mapping[tier]


def tier_for_backend(backend: str) -> Tier:
    """根据旧 backend 专家输入推断等价 tier，仅服务本地 parser 兼容入口。"""
    raw_backend = (backend or "").strip()
    if raw_backend in LEGACY_PIPELINE_BACKEND_ALIASES:
        return "medium"
    if raw_backend in LEGACY_VLM_BACKEND_ALIASES:
        return "xhigh"
    normalized_backend = normalize_backend(backend)
    if normalized_backend == "flash":
        return "flash"
    if is_hybrid_backend(normalized_backend):
        return DEFAULT_HYBRID_EFFORT  # type: ignore[return-value]
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(PARSER_BACKENDS)}")


def _backend_supports_tier(backend: str, tier: Tier) -> bool:
    raw_backend = (backend or "").strip()
    normalized_backend = normalize_backend(backend)
    if tier == "flash":
        return normalized_backend == "flash"
    if tier == "medium":
        return raw_backend in LEGACY_PIPELINE_BACKEND_ALIASES or (
            is_hybrid_backend(normalized_backend) and raw_backend not in LEGACY_VLM_BACKEND_ALIASES
        )
    if tier == "high":
        return (
            is_hybrid_backend(normalized_backend)
            and raw_backend not in LEGACY_PIPELINE_BACKEND_ALIASES
            and raw_backend not in LEGACY_VLM_BACKEND_ALIASES
        )
    if tier == "xhigh":
        return is_hybrid_backend(normalized_backend) and raw_backend not in LEGACY_PIPELINE_BACKEND_ALIASES
    return False


def _tier_for_effort(effort: str) -> Tier:
    return validate_tier(effort)


def resolve_tier_and_backend(tier: Tier | None = None, backend: str | None = None) -> tuple[Tier, str]:
    """将公开 tier 和本地专家 backend 解析为可执行 parser backend。"""
    resolved_tier: Tier = validate_tier(tier) if tier is not None else "high"
    if backend:
        normalized_backend = normalize_backend(backend)
        if tier is None:
            return tier_for_backend(backend), normalized_backend
        if not _backend_supports_tier(backend, resolved_tier):
            raise ValueError(f"tier '{resolved_tier}' is incompatible with backend '{backend}'")
        return resolved_tier, normalized_backend
    return resolved_tier, backend_for_tier(resolved_tier)


def _effort_for_runtime(
    *,
    tier: Tier,
    backend: str,
    requested_effort: str | None,
    raw_backend: str | None,
    explicit_tier: bool,
) -> str:
    """按 tier 优先解析 Hybrid effort；无 tier 时保留旧 backend/effort 兼容输入。"""
    if backend == "flash":
        return LOCAL_HYBRID_EFFORT
    if explicit_tier:
        return effort_for_tier(tier)
    _resolved_backend, resolved_effort = resolve_backend_and_effort(
        raw_backend or backend,
        requested_effort or DEFAULT_HYBRID_EFFORT,
    )
    return resolved_effort


def resolve_runtime_options(
    tier: Tier | None = None,
    backend: str | None = None,
    effort: str | None = None,
) -> ParserRuntimeOptions:
    """统一解析 parser 运行所需的 tier/backend/effort，避免入口各自硬编码映射。"""
    explicit_tier = tier is not None
    resolved_tier, resolved_backend = resolve_tier_and_backend(tier=tier, backend=backend)
    if resolved_backend != "flash":
        resolved_backend, _ = resolve_backend_and_effort(backend or resolved_backend, effort)
    resolved_effort = _effort_for_runtime(
        tier=resolved_tier,
        backend=resolved_backend,
        requested_effort=effort,
        raw_backend=backend,
        explicit_tier=explicit_tier,
    )
    if not explicit_tier and resolved_backend != "flash":
        resolved_tier = _tier_for_effort(resolved_effort)
    return ParserRuntimeOptions(tier=resolved_tier, backend=resolved_backend, effort=resolved_effort)


def runtime_options_for_tier(
    tier: Tier,
    *,
    backend: str | None = None,
    effort: str | None = None,
) -> ParserRuntimeOptions:
    """解析指定 tier 的默认 runtime；调用端可传 backend 覆盖本地/远端执行形态。"""
    return resolve_runtime_options(tier=tier, backend=backend, effort=effort)


def required_modules_for_tier(tier: Tier) -> list[str]:
    tier = validate_tier(tier)
    if tier == "medium":
        return list(_STANDARD_REQUIRED_MODULES)
    if tier in {"high", "xhigh"}:
        return [
            *_PRO_REQUIRED_MODULES_COMMON,
            *_PRO_REQUIRED_MODULES_BY_PLATFORM.get(sys.platform, []),
        ]
    return []


def missing_modules_for_tier(tier: Tier) -> list[str]:
    missing_modules = []
    for module_name in required_modules_for_tier(tier):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name not in (None, module_name):
                raise
            missing_modules.append(module_name)
    return missing_modules


def installed_distribution_name(import_package: str = "mineru") -> str:
    try:
        distributions = importlib_metadata.packages_distributions().get(import_package, [])
    except Exception:
        return import_package
    return distributions[0] if distributions else import_package


def ensure_tier_runtime_dependencies(tier: Tier) -> None:
    missing_modules = missing_modules_for_tier(tier)
    if missing_modules:
        raise TierDependencyError(tier, missing_modules)
