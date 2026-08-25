# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Final

from ..types import DEPLOYMENT_TIERS, DeploymentTier, Tier, validate_tier

CANONICAL_HYBRID_ENGINE: Final = "hybrid-engine"
LOCAL_HYBRID_EFFORT: Final = "medium"
LAYOUT_HYBRID_EFFORT: Final = "high"
MAX_HYBRID_EFFORT: Final = "xhigh"
DEFAULT_BACKEND: Final = CANONICAL_HYBRID_ENGINE
DEFAULT_HYBRID_EFFORT: Final = LAYOUT_HYBRID_EFFORT
DEFAULT_EFFORT: Final = DEFAULT_HYBRID_EFFORT
HYBRID_EFFORT_HELP: Final = "Higher effort improves parsing quality but may be slower; medium is the fastest local Hybrid mode."
LOCAL_BACKEND_CHOICES: Final[tuple[str, ...]] = (CANONICAL_HYBRID_ENGINE,)
HTTP_CLIENT_BACKEND_CHOICES: Final[tuple[str, ...]] = ("hybrid-http-client",)
PUBLIC_BACKEND_CHOICES: Final[tuple[str, ...]] = LOCAL_BACKEND_CHOICES + HTTP_CLIENT_BACKEND_CHOICES
HYBRID_EFFORT_CHOICES: Final[tuple[str, ...]] = (LOCAL_HYBRID_EFFORT, LAYOUT_HYBRID_EFFORT, MAX_HYBRID_EFFORT)
HYBRID_EFFORT_BY_TIER: Final[dict[str, str]] = {
    "flash": "flash",
    "basic": LOCAL_HYBRID_EFFORT,
    "standard": LAYOUT_HYBRID_EFFORT,
    "advanced": MAX_HYBRID_EFFORT,
}
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
LEGACY_VLM_BACKEND_ALIASES: Final[frozenset[str]] = frozenset({"vlm-engine", "vlm-auto-engine", "vlm-http-client"})
SUPPORTED_BACKENDS: Final[tuple[str, ...]] = PUBLIC_BACKEND_CHOICES + ("flash",)
SUPPORTED_EFFORTS: Final[tuple[str, ...]] = HYBRID_EFFORT_CHOICES


def normalize_backend(backend: str | None) -> str:
    """规范化 backend 名称，并将旧 VLM backend 映射到 Hybrid backend。"""
    normalized = (backend or "").strip()
    if normalized in BACKEND_ALIASES:
        return BACKEND_ALIASES[normalized]
    if normalized in SUPPORTED_BACKENDS:
        return normalized
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}")


def normalize_public_backend(backend: str | None) -> str:
    """规范化公开 backend，同时拒绝 flash 和隐藏兼容选项。"""
    normalized = (backend or "").strip()
    if normalized in BACKEND_ALIASES:
        return BACKEND_ALIASES[normalized]
    if normalized in PUBLIC_BACKEND_CHOICES:
        return normalized
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(PUBLIC_BACKEND_CHOICES)}")


def validate_effort(effort: str | None) -> str:
    """校验 Hybrid effort 只使用 medium、high 或 xhigh。"""
    normalized = (effort or DEFAULT_EFFORT).strip().lower()
    if normalized in HYBRID_EFFORT_CHOICES:
        return normalized
    raise ValueError(f"Unsupported effort '{effort}'. Supported efforts: {', '.join(HYBRID_EFFORT_CHOICES)}")


def effort_for_tier(tier: str | None) -> str:
    """将公开 tier 映射为对应的 Hybrid effort。"""
    normalized = (tier or "").strip().lower()
    if normalized in HYBRID_EFFORT_BY_TIER:
        return HYBRID_EFFORT_BY_TIER[normalized]
    supported_tiers = ", ".join(HYBRID_EFFORT_BY_TIER)
    raise ValueError(f"Unsupported tier '{tier}'. Supported hybrid tiers: {supported_tiers}")


def resolve_backend_and_effort(backend: str | None, effort: str | None = None) -> tuple[str, str]:
    """同时解析 backend 与 effort，并保留旧 backend 的质量档位语义。"""
    raw_backend = (backend or "").strip()
    resolved_backend = normalize_backend(raw_backend)
    resolved_effort = validate_effort(effort)
    if raw_backend in LEGACY_PIPELINE_BACKEND_ALIASES:
        resolved_effort = LOCAL_HYBRID_EFFORT
    elif raw_backend in LEGACY_VLM_BACKEND_ALIASES:
        resolved_effort = MAX_HYBRID_EFFORT
    return resolved_backend, resolved_effort


def is_hybrid_backend(backend: str) -> bool:
    """判断规范化后的 backend 是否属于 Hybrid 后端族。"""
    return normalize_backend(backend).startswith("hybrid-")


__all__ = [
    "BACKEND_SCHEMA_EXTRA",
    "CANONICAL_HYBRID_ENGINE",
    "DEFAULT_BACKEND",
    "DEFAULT_EFFORT",
    "DEFAULT_HYBRID_EFFORT",
    "HYBRID_EFFORT_CHOICES",
    "HYBRID_EFFORT_HELP",
    "HYBRID_EFFORT_SCHEMA_EXTRA",
    "LEGACY_PIPELINE_BACKEND_ALIASES",
    "LEGACY_VLM_BACKEND_ALIASES",
    "PARSER_BACKENDS",
    "PUBLIC_BACKEND_CHOICES",
    "ParserRuntimeOptions",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_EFFORTS",
    "TierDependencyError",
    "backend_for_tier",
    "ensure_tier_runtime_dependencies",
    "installed_distribution_name",
    "missing_modules_for_tier",
    "effort_for_tier",
    "is_hybrid_backend",
    "normalize_backend",
    "normalize_public_backend",
    "required_modules_for_tier",
    "resolve_runtime_options",
    "resolve_tier_and_backend",
    "runtime_options_for_tier",
    "tier_for_backend",
    "resolve_backend_and_effort",
    "validate_effort",
]

PARSER_BACKENDS = SUPPORTED_BACKENDS

_BASIC_REQUIRED_MODULES = [
    "six",
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
]
_STANDARD_REQUIRED_MODULES_BY_PLATFORM = {
    "linux": ["vllm"],
    "win32": ["lmdeploy", "qwen_vl_utils"],
}
_APPLE_SILICON_STANDARD_REQUIRED_MODULES = [
    "mlx",
    "mlx_vlm",
]


class TierDependencyError(RuntimeError):
    def __init__(self, tier: DeploymentTier, missing_modules: list[str]) -> None:
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
        "basic": CANONICAL_HYBRID_ENGINE,
        "standard": CANONICAL_HYBRID_ENGINE,
        "advanced": CANONICAL_HYBRID_ENGINE,
    }
    return mapping[tier]


def tier_for_backend(backend: str) -> Tier:
    """根据旧 backend 专家输入推断等价 tier，仅服务本地 parser 兼容入口。"""
    raw_backend = (backend or "").strip()
    if raw_backend in LEGACY_PIPELINE_BACKEND_ALIASES:
        return "basic"
    if raw_backend in LEGACY_VLM_BACKEND_ALIASES:
        return "advanced"
    normalized_backend = normalize_backend(backend)
    if normalized_backend == "flash":
        return "flash"
    if is_hybrid_backend(normalized_backend):
        return "standard"
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(PARSER_BACKENDS)}")


def _backend_supports_tier(backend: str, tier: Tier) -> bool:
    raw_backend = (backend or "").strip()
    normalized_backend = normalize_backend(backend)
    if tier == "flash":
        return normalized_backend == "flash"
    if tier == "basic":
        return raw_backend in LEGACY_PIPELINE_BACKEND_ALIASES or (
            is_hybrid_backend(normalized_backend) and raw_backend not in LEGACY_VLM_BACKEND_ALIASES
        )
    if tier == "standard":
        return (
            is_hybrid_backend(normalized_backend)
            and raw_backend not in LEGACY_PIPELINE_BACKEND_ALIASES
            and raw_backend not in LEGACY_VLM_BACKEND_ALIASES
        )
    if tier == "advanced":
        return is_hybrid_backend(normalized_backend) and raw_backend not in LEGACY_PIPELINE_BACKEND_ALIASES
    return False


def resolve_tier_and_backend(tier: Tier | None = None, backend: str | None = None) -> tuple[Tier, str]:
    """将公开 tier 和本地专家 backend 解析为可执行 parser backend。"""
    resolved_tier: Tier = validate_tier(tier) if tier is not None else "standard"
    if backend:
        normalized_backend = normalize_backend(backend)
        if tier is None:
            return tier_for_backend(backend), normalized_backend
        if not _backend_supports_tier(backend, resolved_tier):
            raise ValueError(f"tier '{resolved_tier}' is incompatible with backend '{backend}'")
        return resolved_tier, normalized_backend
    return resolved_tier, backend_for_tier(resolved_tier)


def _effort_for_runtime(*, tier: Tier, backend: str) -> str:
    """按 tier 派生 Hybrid effort；flash tier 直接返回 flash effort。"""
    if backend == "flash":
        return "flash"
    return effort_for_tier(tier)


def resolve_runtime_options(
    tier: Tier | None = None,
    backend: str | None = None,
) -> ParserRuntimeOptions:
    """统一解析 parser 运行所需的 tier/backend，effort 作为派生值不暴露为入参。"""
    resolved_tier, resolved_backend = resolve_tier_and_backend(tier=tier, backend=backend)
    resolved_effort = _effort_for_runtime(tier=resolved_tier, backend=resolved_backend)
    return ParserRuntimeOptions(tier=resolved_tier, backend=resolved_backend, effort=resolved_effort)


def runtime_options_for_tier(
    tier: Tier,
    *,
    backend: str | None = None,
) -> ParserRuntimeOptions:
    """解析指定 tier 的默认 runtime；调用端可传 backend 覆盖本地/远端执行形态。"""
    return resolve_runtime_options(tier=tier, backend=backend)


def required_modules_for_tier(tier: DeploymentTier) -> list[str]:
    if tier not in DEPLOYMENT_TIERS:
        raise ValueError(f"Unsupported deployment tier '{tier}'. Supported tiers: {', '.join(DEPLOYMENT_TIERS)}")
    from ..model.runtime.device import get_model_stack

    stack = get_model_stack()
    if stack == "light":
        return []
    if tier == "basic":
        return list(_BASIC_REQUIRED_MODULES)
    platform_modules = list(_STANDARD_REQUIRED_MODULES_BY_PLATFORM.get(sys.platform, []))
    if sys.platform == "darwin" and platform.machine() == "arm64":
        platform_modules.extend(_APPLE_SILICON_STANDARD_REQUIRED_MODULES)
    return [*_BASIC_REQUIRED_MODULES, *platform_modules]


def missing_modules_for_tier(tier: DeploymentTier) -> list[str]:
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


def ensure_tier_runtime_dependencies(tier: DeploymentTier) -> None:
    missing_modules = missing_modules_for_tier(tier)
    if missing_modules:
        raise TierDependencyError(tier, missing_modules)
