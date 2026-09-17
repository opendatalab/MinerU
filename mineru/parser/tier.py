# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Final

from ..config import VlmConfig
from ..types import DEPLOYMENT_TIERS, DeploymentTier, Tier, validate_tier

CANONICAL_HYBRID_ENGINE: Final = "hybrid-engine"
LOCAL_HYBRID_EFFORT: Final = "medium"
LAYOUT_HYBRID_EFFORT: Final = "high"
MAX_HYBRID_EFFORT: Final = "xhigh"
DEFAULT_HYBRID_EFFORT: Final = LAYOUT_HYBRID_EFFORT
DEFAULT_EFFORT: Final = DEFAULT_HYBRID_EFFORT
HYBRID_EFFORT_HELP: Final = "Higher effort improves parsing quality but may be slower; medium is the fastest local Hybrid mode."
HYBRID_EFFORT_CHOICES: Final[tuple[str, ...]] = (LOCAL_HYBRID_EFFORT, LAYOUT_HYBRID_EFFORT, MAX_HYBRID_EFFORT)
HYBRID_EFFORT_BY_TIER: Final[dict[str, str]] = {
    "flash": "flash",
    "basic": LOCAL_HYBRID_EFFORT,
    "standard": LAYOUT_HYBRID_EFFORT,
    "advanced": MAX_HYBRID_EFFORT,
}
HYBRID_EFFORT_SCHEMA_EXTRA: Final[dict[str, list[str]]] = {"enum": list(HYBRID_EFFORT_CHOICES)}
SUPPORTED_EFFORTS: Final[tuple[str, ...]] = HYBRID_EFFORT_CHOICES


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


__all__ = [
    "CANONICAL_HYBRID_ENGINE",
    "DEFAULT_EFFORT",
    "DEFAULT_HYBRID_EFFORT",
    "HYBRID_EFFORT_CHOICES",
    "HYBRID_EFFORT_HELP",
    "HYBRID_EFFORT_SCHEMA_EXTRA",
    "ParserRuntimeOptions",
    "SUPPORTED_EFFORTS",
    "TierDependencyError",
    "backend_for_tier",
    "effort_for_tier",
    "ensure_tier_runtime_dependencies",
    "installed_distribution_name",
    "missing_modules_for_tier",
    "required_modules_for_tier",
    "resolve_runtime_options",
    "runtime_options_for_tier",
    "validate_effort",
]


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


def resolve_runtime_options(tier: Tier | None = None) -> ParserRuntimeOptions:
    """统一解析 parser 运行所需的 runtime；backend 与 effort 均由 tier 派生。"""
    resolved_tier: Tier = validate_tier(tier) if tier is not None else "standard"
    resolved_backend = backend_for_tier(resolved_tier)
    resolved_effort = "flash" if resolved_tier == "flash" else effort_for_tier(resolved_tier)
    return ParserRuntimeOptions(tier=resolved_tier, backend=resolved_backend, effort=resolved_effort)


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
