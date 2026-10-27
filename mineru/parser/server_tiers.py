"""API 服务的精确档位集合、CLI 参数解析与共享模型配置。"""

from __future__ import annotations

from collections.abc import Collection

from ..types import TIERS, DeploymentTier, Tier, validate_tier


def normalize_server_tiers(tiers: Collection[Tier]) -> tuple[Tier, ...]:
    """校验非空显式集合，去重并按公共档位顺序稳定输出。"""
    if not isinstance(tiers, Collection) or isinstance(tiers, (str, bytes)) or not tiers:
        raise ValueError("Server tiers must be a non-empty collection of flash, basic, standard, advanced")
    if any(not isinstance(tier, str) for tier in tiers):
        raise ValueError("Server tiers must contain tier names")
    selected = {validate_tier(tier) for tier in tiers}
    return tuple(tier for tier in TIERS if tier in selected)


def parse_server_tiers(value: str) -> tuple[Tier, ...] | None:
    """解析 auto 或逗号分隔列表；空项及 auto 混用均明确报错。"""
    if value.strip().lower() == "auto":
        return None
    return normalize_server_tiers(value.split(","))  # type: ignore[arg-type]


def resolve_server_tiers(tiers: Collection[Tier] | None) -> tuple[Tier, ...]:
    """显式列表完全跳过硬件推荐，自动模式只在调用时惰性探测。"""
    if tiers is None:
        from ..model.runtime.tier_recommendation import recommend_tiers

        tiers = recommend_tiers()
    return normalize_server_tiers(tiers)


def model_tier_for_server_tiers(tiers: Collection[Tier]) -> DeploymentTier | None:
    """合并共享模型需求；advanced 复用 Standard 配置但不增加发布档位。"""
    if "standard" in tiers or "advanced" in tiers:
        return "standard"
    if "basic" in tiers:
        return "basic"
    return None


__all__ = ["model_tier_for_server_tiers", "normalize_server_tiers", "parse_server_tiers", "resolve_server_tiers"]
