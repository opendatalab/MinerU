"""SQLite-backed telemetry state and aggregate storage."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

from .constants import (
    CONSENT_DISABLED,
    CONSENT_ENABLED,
    CONSENT_UNSET,
    TELEMETRY_FLUSH_LOCK_TTL_SEC,
    TELEMETRY_MAX_FLUSH_PERIODS,
    TelemetryConsentState,
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
        now = _now_ms()
        await self._ensure_state("installation_id", _new_installation_id(), now)
        await self._ensure_state("consent_state", CONSENT_UNSET, now)
        await self._ensure_state("last_flush_at", "0", now)
        await self._ensure_state("flush_locked_at", "0", now)

    async def installation_id(self) -> str:
        value = await self.get_state("installation_id")
        if value:
            return value
        installation_id = _new_installation_id()
        await self.set_state("installation_id", installation_id)
        return installation_id

    async def consent_state(self) -> TelemetryConsentState:
        value = await self.get_state("consent_state")
        if value in {CONSENT_UNSET, CONSENT_ENABLED, CONSENT_DISABLED}:
            return value  # type: ignore[return-value]
        return CONSENT_UNSET

    async def set_consent_state(self, state: TelemetryConsentState) -> None:
        await self.set_state("consent_state", state)
        if state == CONSENT_DISABLED:
            await self.clear_aggregates()

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

    async def _ensure_state(self, key: str, value: str, now: int) -> None:
        await self.db.execute(
            """
            INSERT INTO telemetry_state (key, value, updated_at)
            SELECT ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM telemetry_state WHERE key=?)
            """,
            (key, value, now, key),
        )


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
