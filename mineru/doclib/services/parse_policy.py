"""Doclib 入库与主动解析共用的默认页范围和后台档位策略。"""

from __future__ import annotations

from ...parser.page_range import expand_page_range
from ...types import Tier, select_parsing_rule_tier


def default_parse_range(page_count: int | None) -> str:
    """主动读取默认选择前十页，未知页数时保留第一页请求。"""
    if page_count and page_count > 0:
        return expand_page_range("1-10", page_count)
    return "1"


def resolve_parsing_rule_default_tier(remote: bool = False) -> Tier:
    """后台入库规则沿用质量档位降级到 Flash 的既有策略。"""
    from ..background.parse_server_health import get_health

    health = get_health()
    supported = health.remote.probe.tiers if remote else health.local.probe.tiers
    return select_parsing_rule_tier(supported)


__all__ = ["default_parse_range", "resolve_parsing_rule_default_tier"]
