# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import importlib
import sys
from importlib import metadata as importlib_metadata

from ..types import Tier
from ..utils.backend_options import (
    CANONICAL_HYBRID_ENGINE,
    LEGACY_PIPELINE_BACKEND_ALIASES,
    SUPPORTED_BACKENDS,
    is_hybrid_backend,
    normalize_backend,
)

__all__ = [
    "PARSER_BACKENDS",
    "TierDependencyError",
    "backend_for_tier",
    "ensure_tier_runtime_dependencies",
    "installed_distribution_name",
    "missing_modules_for_tier",
    "required_modules_for_tier",
    "resolve_tier_and_backend",
    "tier_for_backend",
]

PARSER_BACKENDS = SUPPORTED_BACKENDS

_STANDARD_REQUIRED_MODULES = [
    "ftfy",
    "shapely",
    "pyclipper",
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


def backend_for_tier(tier: Tier) -> str:
    """Return the local parser backend used for a parser-layer tier fallback."""
    mapping = {
        "flash": "flash",
        "standard": CANONICAL_HYBRID_ENGINE,
        "pro": CANONICAL_HYBRID_ENGINE,
    }
    return mapping.get(tier, CANONICAL_HYBRID_ENGINE)


def tier_for_backend(backend: str) -> Tier:
    raw_backend = (backend or "").strip()
    if raw_backend in LEGACY_PIPELINE_BACKEND_ALIASES:
        return "standard"
    normalized_backend = normalize_backend(backend)
    if normalized_backend == "flash":
        return "flash"
    if is_hybrid_backend(normalized_backend):
        return "pro"
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {', '.join(PARSER_BACKENDS)}")


def _backend_supports_tier(backend: str, tier: Tier) -> bool:
    raw_backend = (backend or "").strip()
    normalized_backend = normalize_backend(backend)
    if tier == "flash":
        return normalized_backend == "flash"
    if tier == "standard":
        return raw_backend in LEGACY_PIPELINE_BACKEND_ALIASES or normalized_backend == CANONICAL_HYBRID_ENGINE
    if tier == "pro":
        return is_hybrid_backend(normalized_backend) and raw_backend not in LEGACY_PIPELINE_BACKEND_ALIASES
    return False


def resolve_tier_and_backend(tier: Tier | None = None, backend: str | None = None) -> tuple[Tier, str]:
    """Resolve public tier and optional expert backend into an executable parser backend."""
    resolved_tier: Tier = tier or "pro"
    if backend:
        normalized_backend = normalize_backend(backend)
        if tier is None:
            return tier_for_backend(backend), normalized_backend
        if not _backend_supports_tier(backend, resolved_tier):
            raise ValueError(f"tier '{resolved_tier}' is incompatible with backend '{backend}'")
        return resolved_tier, normalized_backend
    return resolved_tier, backend_for_tier(resolved_tier)


def required_modules_for_tier(tier: Tier) -> list[str]:
    if tier == "standard":
        return list(_STANDARD_REQUIRED_MODULES)
    if tier == "pro":
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
