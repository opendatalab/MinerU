# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Final

from ..config import VlmConfig
from ..types import DEPLOYMENT_TIERS, DeploymentTier, Tier, validate_tier

_EFFORT_BY_TIER: Final[dict[str, str]] = {
    "flash": "flash",
    "basic": "medium",
    "standard": "high",
    "advanced": "xhigh",
}


def effort_for_tier(tier: str | None) -> str:
    """将公开 tier 映射为对应的 Hybrid effort。"""
    normalized = (tier or "").strip().lower()
    if normalized in _EFFORT_BY_TIER:
        return _EFFORT_BY_TIER[normalized]
    supported_tiers = ", ".join(_EFFORT_BY_TIER)
    raise ValueError(f"Unsupported tier '{tier}'. Supported hybrid tiers: {supported_tiers}")


class TierDependencyError(RuntimeError):
    def __init__(self, tier: DeploymentTier, missing_modules: list[str]) -> None:
        self.tier = tier
        self.missing_modules = missing_modules
        missing = ", ".join(missing_modules)
        package_name = installed_distribution_name()
        super().__init__(
            f"Parse server cannot start for tier '{tier}'; missing runtime dependencies: {missing}. "
            f"Install the dependencies for the selected backend in the same Python environment: "
            f"{package_name}[torch] for Torch; {package_name}[full] for vLLM/LMDeploy; "
            f"mlx-vlm>=0.7.0,<0.8.0 for explicit MLX; {package_name} for ONNX/llama.cpp."
        )


@dataclass(frozen=True)
class ParserRuntimeOptions:
    """记录某个 tier 实际执行时使用的 effort。"""

    tier: Tier
    effort: str


def resolve_runtime_options(tier: Tier | None = None) -> ParserRuntimeOptions:
    """统一解析 parser 运行所需的 runtime；effort 由 tier 派生。"""
    resolved_tier: Tier = validate_tier(tier) if tier is not None else "standard"
    resolved_effort = "flash" if resolved_tier == "flash" else effort_for_tier(resolved_tier)
    return ParserRuntimeOptions(tier=resolved_tier, effort=resolved_effort)


def runtime_options_for_tier(tier: Tier) -> ParserRuntimeOptions:
    """解析指定 tier 的默认 runtime。"""
    return resolve_runtime_options(tier=tier)


def required_modules_for_tier(tier: DeploymentTier, *, vlm_config: VlmConfig | None = None) -> list[str]:
    """按独立后端组合预检依赖，远程 VLM 不要求本地引擎。"""
    from ..config import config
    from ..model.runtime.device import TORCH_REQUIRED_MODULES, resolve_small_model_backend
    from ..model.vlm.selector import VLM_REQUIRED_MODULES, resolve_vlm_engine

    if tier not in DEPLOYMENT_TIERS:
        raise ValueError(f"Unsupported deployment tier '{tier}'. Supported tiers: {', '.join(DEPLOYMENT_TIERS)}")
    modules = ["onnxruntime"]
    if resolve_small_model_backend() == "torch":
        modules.extend(TORCH_REQUIRED_MODULES)
    settings = vlm_config if vlm_config is not None else config.model.vlm
    if tier == "standard" and not settings.server_url:
        modules.extend(VLM_REQUIRED_MODULES[resolve_vlm_engine(settings.engine)])
    return list(dict.fromkeys(modules))


def missing_modules_for_tier(tier: DeploymentTier, *, vlm_config: VlmConfig | None = None) -> list[str]:
    """导入当前后端真正需要的模块，保留依赖内部损坏的原始异常。"""
    missing_modules = []
    for module_name in required_modules_for_tier(tier, vlm_config=vlm_config):
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


def ensure_tier_runtime_dependencies(tier: DeploymentTier, *, vlm_config: VlmConfig | None = None) -> None:
    """使用与解析器一致的 VLM 配置预检当前档位。"""
    missing_modules = missing_modules_for_tier(tier, vlm_config=vlm_config)
    if missing_modules:
        raise TierDependencyError(tier, missing_modules)


__all__ = [
    "ParserRuntimeOptions",
    "TierDependencyError",
    "effort_for_tier",
    "ensure_tier_runtime_dependencies",
    "installed_distribution_name",
    "missing_modules_for_tier",
    "required_modules_for_tier",
    "resolve_runtime_options",
    "runtime_options_for_tier",
]
