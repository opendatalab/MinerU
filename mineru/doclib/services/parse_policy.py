"""Doclib 入库与主动解析共用的默认页范围和后台档位策略。"""

from __future__ import annotations

from ...parser.page_range import expand_page_range, parse_page_range_set
from ...types import Tier, select_default_quality_tier, select_parsing_rule_tier
from ..rows import ParseRow
from .parse_errors import ParseFailure
from ..parse_server_state import ParseServerHealth


def default_parse_range(page_count: int | None) -> str:
    """未知页数请求第一页，已知页数时选择前十页。"""
    if page_count and page_count > 0:
        return expand_page_range("1-10", page_count)
    return "1"


def resolve_parsing_rule_default_tier(*, health: ParseServerHealth, remote: bool = False) -> Tier:
    """后台规则沿用可用质量档位降级到 Flash 的既有策略。"""
    supported = health.remote.probe.tiers if remote else health.local.probe.tiers
    return select_parsing_rule_tier(supported)


def page_range_covered(request_page_range: str, done_batches: list[ParseRow]) -> bool:
    """按较新完成批次优先的页范围集合判断完整覆盖。"""
    needed_page_numbers = parse_page_range_set(request_page_range)
    covered_page_numbers: set[int] = set()
    for b in sorted(done_batches, key=lambda r: r.get("done_at") or 0, reverse=True):
        covered_page_numbers |= parse_page_range_set(b["page_range"])
        if needed_page_numbers <= covered_page_numbers:
            return True
    return False


def page_range_uncovered(request_page_range: str, done_batches: list[ParseRow]) -> set[int]:
    """返回请求中尚未被完成批次覆盖的页号。"""
    needed_page_numbers = parse_page_range_set(request_page_range)
    covered_page_numbers: set[int] = set()
    for b in done_batches:
        covered_page_numbers |= parse_page_range_set(b["page_range"])
    return needed_page_numbers - covered_page_numbers


def resolve_default_tier(*, health: ParseServerHealth, remote: bool = False) -> Tier:
    """按健康信息选择默认质量档位，并保留认证及无引擎错误。"""
    probe = health.remote.probe if remote else health.local.probe
    supported = probe.tiers
    selected = select_default_quality_tier(supported)
    if selected is not None:
        return selected
    if remote and probe.error_code == "invalid_api_key":
        raise ParseFailure(
            "invalid_api_key",
            probe_error_message("Remote parse-server authentication failed", probe.error_msg),
            "parse_server.remote.api_key",
        )
    if not remote and health.local_mode == "self_hosted" and probe.error_code == "invalid_api_key":
        raise ParseFailure(
            "invalid_api_key",
            probe_error_message("Local self-hosted parse-server authentication failed", probe.error_msg),
            "parse_server.local.self_hosted_api_key",
        )
    actions = ["start a local parse-server"]
    if not remote and health.remote.probe.healthy and select_default_quality_tier(health.remote.probe.tiers) is not None:
        actions.append("use --remote")
    actions.append("explicitly pass --tier flash for text-only preview")
    raise ParseFailure(
        "quality_tier_unavailable",
        f"No basic, standard, or advanced engine available. You can {_format_action_list(actions)}.",
    )


def _format_action_list(actions: list[str]) -> str:
    """按原有语法连接错误提示中的操作选项。"""
    if len(actions) == 1:
        return actions[0]
    if len(actions) == 2:
        return f"{actions[0]} or {actions[1]}"
    return f"{', '.join(actions[:-1])}, or {actions[-1]}"


def probe_error_message(prefix: str, error_msg: str | None) -> str:
    """拼接探测错误详情，保持调用方原有前缀。"""
    if error_msg:
        return f"{prefix}: {error_msg}"
    return prefix


__all__ = [
    "default_parse_range",
    "resolve_parsing_rule_default_tier",
    "resolve_default_tier",
    "page_range_covered",
    "page_range_uncovered",
    "probe_error_message",
]
