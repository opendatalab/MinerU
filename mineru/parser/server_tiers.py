"""PDF/图片的启动上限、内嵌精确集合与共享模型配置。"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from ..types import TIERS, TIERS_BY_SERVER_TIER, DeploymentTier, Tier, validate_tier

if TYPE_CHECKING:
    from ..config import VlmConfig


def normalize_server_tiers(tiers: Collection[Tier]) -> tuple[Tier, ...]:
    """校验非空显式集合，去重并按公共档位顺序稳定输出。"""
    if not isinstance(tiers, Collection) or isinstance(tiers, (str, bytes)) or not tiers:
        raise ValueError("Server tiers must be a non-empty collection of flash, basic, standard, advanced")
    if any(not isinstance(tier, str) for tier in tiers):
        raise ValueError("Server tiers must contain tier names")
    selected = {validate_tier(tier) for tier in tiers}
    return tuple(tier for tier in TIERS if tier in selected)


def parse_server_tier(value: str) -> tuple[Tier, ...] | None:
    """解析单档上限或 auto；上限包含自身及全部较低 PDF/图片档位。"""
    if value.strip().lower() == "auto":
        return None
    return TIERS_BY_SERVER_TIER[validate_tier(value)]


def resolve_server_tiers(tiers: Collection[Tier] | None, *, vlm_config: VlmConfig | None = None) -> tuple[Tier, ...]:
    """显式列表跳过推荐与启动校准；自动模式按硬件初选并完成适用的 OCR 选型。"""
    if tiers is None:
        from ..model.runtime.tier_recommendation import recommend_tiers

        recommended = normalize_server_tiers(recommend_tiers(vlm_config=vlm_config).pdf_image_tiers)
        tiers = TIERS_BY_SERVER_TIER[recommended[-1]]
    return normalize_server_tiers(tiers)


def model_tier_for_server_tiers(tiers: Collection[Tier]) -> DeploymentTier | None:
    """合并共享模型需求；advanced 复用 Standard 配置但不增加发布档位。"""
    if "standard" in tiers or "advanced" in tiers:
        return "standard"
    if "basic" in tiers:
        return "basic"
    return None


__all__ = ["model_tier_for_server_tiers", "normalize_server_tiers", "parse_server_tier", "resolve_server_tiers"]
