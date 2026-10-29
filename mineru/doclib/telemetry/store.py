"""SQLite-backed telemetry state and aggregate storage."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any, cast

from .constants import (
    TELEMETRY_DISABLED,
    TELEMETRY_ENABLED,
    TELEMETRY_FLUSH_LOCK_TTL_SEC,
    TELEMETRY_MAX_FLUSH_PERIODS,
    TelemetryState,
    TelemetryStateSource,
)
from .payload import canonical_dimensions, dimensions_hash


@dataclass(frozen=True)
class TelemetryAggregate:
    id: int
    period_start: int
    period_end: int
    metric_name: str
    metric_value: int
    dimensions: dict[str, str]
    dimensions_hash: str


class TelemetryStore:
    def __init__(self, db: Any) -> None:
        self.db = db

    async def initialize(self) -> None:
        """原子迁移历史选择；新安装默认开启，已有关闭和异常状态不被升级重置。"""
        now = _now_ms()
        rows = await self.db.fetchall("SELECT key, value FROM telemetry_state")
        previous = {row["key"]: row["value"] for row in rows}
        if "telemetry_state" in previous:
            initial = previous["telemetry_state"]
            source = previous.get("state_source", "error")
        elif "consent_state" in previous:
            legacy = previous["consent_state"]
            initial = TELEMETRY_ENABLED if legacy in {"unset", "enabled"} else TELEMETRY_DISABLED
            source = "default" if legacy == "unset" else "user" if legacy in {"enabled", "disabled"} else "error"
        elif not previous:
            initial, source = TELEMETRY_ENABLED, "default"
        else:
            initial, source = TELEMETRY_DISABLED, "error"
        defaults = {
            "telemetry_state": initial,
            "state_source": source,
            "installation_id": _new_installation_id(),
            "last_flush_at": "0",
            "flush_locked_at": "0",
            "notice_shown": "0",
        }
        statements = [
            ("INSERT OR IGNORE INTO telemetry_state (key, value, updated_at) VALUES (?, ?, ?)", (key, value, now))
            for key, value in defaults.items()
        ]
        if "telemetry_state" not in previous:
            # 迁移或修复缺失开关时，旧来源也必须同步，不能沿用损坏前的 default 标记。
            statements.append(("UPDATE telemetry_state SET value=?, updated_at=? WHERE key='state_source'", (source, now)))
        statements.append(("DELETE FROM telemetry_state WHERE key='consent_state'", ()))
        if initial == TELEMETRY_DISABLED:
            statements.append(("DELETE FROM telemetry_aggregates", ()))
        await self.db.execute_atomic(statements)

    async def installation_id(self) -> str:
        value = await self.get_state("installation_id")
        if value:
            return value
        installation_id = _new_installation_id()
        await self.set_state("installation_id", installation_id)
        return installation_id

    async def preference(self) -> tuple[TelemetryState, TelemetryStateSource]:
        """在同一查询快照中读取状态与来源；异常值交给服务层暂停上报。"""
        rows = await self.db.fetchall("SELECT key, value FROM telemetry_state WHERE key IN ('telemetry_state', 'state_source')")
        values = {row["key"]: row["value"] for row in rows}
        state, source = values.get("telemetry_state"), values.get("state_source")
        if state not in {TELEMETRY_ENABLED, TELEMETRY_DISABLED} or source not in {"default", "user", "error"}:
            raise ValueError("Telemetry state is missing or invalid")
        return cast(TelemetryState, state), cast(TelemetryStateSource, source)

    async def telemetry_state(self) -> TelemetryState:
        """返回已迁移的持久化开关，不应用进程级环境覆盖。"""
        return (await self.preference())[0]

    async def set_telemetry_state(self, state: TelemetryState) -> None:
        """同一事务保存用户选择并在关闭时清空待发数据，避免部分成功。"""
        if state not in {TELEMETRY_ENABLED, TELEMETRY_DISABLED}:
            raise ValueError("Unsupported telemetry state")
        now = _now_ms()
        statements = [
            (
                "INSERT INTO telemetry_state (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, now),
            )
            for key, value in (("telemetry_state", state), ("state_source", "user"))
        ]
        if state == TELEMETRY_DISABLED:
            statements.append(("DELETE FROM telemetry_aggregates", ()))
        await self.db.execute_atomic(statements)

    async def get_state(self, key: str) -> str | None:
        row = await self.db.fetchone("SELECT value FROM telemetry_state WHERE key=?", (key,))
        if row is None:
            return None
        return str(row["value"])

    async def set_state(self, key: str, value: str) -> None:
        now = _now_ms()
        await self.db.execute(
            """
            INSERT INTO telemetry_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, now),
        )

    async def increment(
        self,
        *,
        metric_name: str,
        value: int,
        dimensions: dict[str, str],
        timestamp_ms: int | None = None,
    ) -> None:
        if value == 0:
            return
        period_start, period_end = period_bounds(timestamp_ms or _now_ms())
        canonical = canonical_dimensions(dimensions)
        dim_hash = dimensions_hash(canonical)
        now = _now_ms()
        await self.db.execute(
            """
            INSERT INTO telemetry_aggregates (
                period_start, period_end, metric_name, metric_value,
                dimensions, dimensions_hash, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(period_start, period_end, metric_name, dimensions_hash)
            DO UPDATE SET
                metric_value=telemetry_aggregates.metric_value + excluded.metric_value,
                updated_at=excluded.updated_at
            """,
            (
                period_start,
                period_end,
                metric_name,
                int(value),
                _encode_dimensions(canonical),
                dim_hash,
                now,
                now,
            ),
        )

    async def list_periods_for_flush(self, *, limit: int = TELEMETRY_MAX_FLUSH_PERIODS) -> list[tuple[int, int]]:
        rows = await self.db.fetchall(
            """
            SELECT period_start, period_end
            FROM telemetry_aggregates
            GROUP BY period_start, period_end
            ORDER BY period_start ASC, period_end ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [(int(row["period_start"]), int(row["period_end"])) for row in rows]

    async def list_aggregates(self, period_start: int, period_end: int) -> list[TelemetryAggregate]:
        rows = await self.db.fetchall(
            """
            SELECT id, period_start, period_end, metric_name, metric_value, dimensions, dimensions_hash
            FROM telemetry_aggregates
            WHERE period_start=? AND period_end=?
            ORDER BY metric_name ASC, dimensions_hash ASC
            """,
            (period_start, period_end),
        )
        return [_row_to_aggregate(row) for row in rows]

    async def pending_counts(self) -> tuple[int, int]:
        row = await self.db.fetchone(
            """
            SELECT
                COUNT(DISTINCT period_start || ':' || period_end) AS period_count,
                COUNT(*) AS aggregate_count
            FROM telemetry_aggregates
            """
        )
        if row is None:
            return 0, 0
        return int(row["period_count"] or 0), int(row["aggregate_count"] or 0)

    async def delete_period(self, period_start: int, period_end: int) -> None:
        await self.db.execute(
            "DELETE FROM telemetry_aggregates WHERE period_start=? AND period_end=?",
            (period_start, period_end),
        )

    async def clear_aggregates(self) -> None:
        await self.db.execute("DELETE FROM telemetry_aggregates")

    async def try_acquire_flush_lock(self) -> bool:
        now = _now_ms()
        locked_at_text = await self.get_state("flush_locked_at")
        try:
            locked_at = int(locked_at_text or "0")
        except ValueError:
            locked_at = 0
        if locked_at and now - locked_at < TELEMETRY_FLUSH_LOCK_TTL_SEC * 1000:
            return False
        await self.set_state("flush_locked_at", str(now))
        return True

    async def release_flush_lock(self) -> None:
        await self.set_state("flush_locked_at", "0")

    async def mark_flushed_now(self) -> None:
        await self.set_state("last_flush_at", str(_now_ms()))


def period_bounds(timestamp_ms: int) -> tuple[int, int]:
    hour_ms = 60 * 60 * 1000
    start = timestamp_ms - (timestamp_ms % hour_ms)
    return start, start + hour_ms


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_installation_id() -> str:
    return f"inst_{secrets.token_hex(16)}"


def _encode_dimensions(dimensions: dict[str, str]) -> str:
    import json

    return json.dumps(canonical_dimensions(dimensions), ensure_ascii=False, separators=(",", ":"))


def _decode_dimensions(raw: str) -> dict[str, str]:
    import json

    value = json.loads(raw)
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _row_to_aggregate(row: dict[str, Any]) -> TelemetryAggregate:
    return TelemetryAggregate(
        id=int(row["id"]),
        period_start=int(row["period_start"]),
        period_end=int(row["period_end"]),
        metric_name=str(row["metric_name"]),
        metric_value=int(row["metric_value"]),
        dimensions=_decode_dimensions(str(row["dimensions"])),
        dimensions_hash=str(row["dimensions_hash"]),
    )


__all__ = ["TelemetryAggregate", "TelemetryStore", "period_bounds"]
