"""Telemetry recording facade."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .buckets import duration_bucket
from .constants import CONSENT_DISABLED, CONSENT_ENABLED, DIMENSION_VALUES, METRIC_SPECS, TELEMETRY_MAX_FLUSH_PERIODS, TelemetryConsentState
from .context import get_telemetry_context
from .payload import build_period_payload, collect_environment_context
from .store import TelemetryStore, period_bounds

logger = logging.getLogger("mineru.doclib.telemetry")


@dataclass(frozen=True)
class TelemetryFlushResult:
    status: str
    attempted: int = 0
    succeeded: int = 0
    discarded: int = 0


class TelemetryService:
    def __init__(self, store: TelemetryStore) -> None:
        self.store = store

    async def initialize(self) -> None:
        await self.store.initialize()

    async def record_count(
        self,
        metric_name: str,
        *,
        value: int = 1,
        dimensions: dict[str, str] | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        await self._record(metric_name, value=value, dimensions=dimensions or {}, timestamp_ms=timestamp_ms)

    async def record_duration_bucket(
        self,
        metric_name: str,
        *,
        duration_ms: int,
        dimensions: dict[str, str] | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        dims = dict(dimensions or {})
        dims["bucket"] = duration_bucket(duration_ms)
        await self._record(metric_name, value=1, dimensions=dims, timestamp_ms=timestamp_ms)

    async def status(self) -> dict[str, Any]:
        pending_periods, pending_metrics = await self.store.pending_counts()
        last_flush_at = await self.store.get_state("last_flush_at")
        return {
            "state": await self.store.consent_state(),
            "installation_id": await self.store.installation_id(),
            "pending_periods": pending_periods,
            "pending_metrics": pending_metrics,
            "last_flush_at": _int_or_none(last_flush_at),
        }

    async def set_consent(self, state: TelemetryConsentState) -> dict[str, Any]:
        await self.store.set_consent_state(state)
        return {
            "state": await self.store.consent_state(),
            "installation_id": await self.store.installation_id(),
        }

    async def preview_body(self) -> dict[str, Any]:
        periods = await self.store.list_periods_for_flush(limit=1)
        if periods:
            period_start, period_end = periods[0]
            aggregates = await self.store.list_aggregates(period_start, period_end)
        else:
            period_start, period_end = period_bounds(int(time.time() * 1000))
            aggregates = []
        metrics = [
            {
                "name": aggregate.metric_name,
                "value": aggregate.metric_value,
                "dimensions": aggregate.dimensions,
            }
            for aggregate in aggregates
        ]
        return build_period_payload(
            batch_id="tb_preview",
            installation_id=await self.store.installation_id(),
            period_start=period_start,
            period_end=period_end,
            context=collect_environment_context(),
            metrics=metrics,
        )

    async def flush_once(self) -> TelemetryFlushResult:
        if await self.store.consent_state() != CONSENT_ENABLED:
            return TelemetryFlushResult(status="disabled")
        if not await self.store.try_acquire_flush_lock():
            return TelemetryFlushResult(status="locked")
        attempted = 0
        succeeded = 0
        discarded = 0
        try:
            periods = await self.store.list_periods_for_flush(limit=TELEMETRY_MAX_FLUSH_PERIODS)
            if not periods:
                return TelemetryFlushResult(status="no_metrics")
            for period_start, period_end in periods:
                aggregates = await self.store.list_aggregates(period_start, period_end)
                if not aggregates:
                    await self.store.delete_period(period_start, period_end)
                    continue
                attempted += 1
                metrics = [
                    {
                        "name": aggregate.metric_name,
                        "value": aggregate.metric_value,
                        "dimensions": aggregate.dimensions,
                    }
                    for aggregate in aggregates
                ]
                payload = build_period_payload(
                    batch_id=_new_batch_id(),
                    installation_id=await self.store.installation_id(),
                    period_start=period_start,
                    period_end=period_end,
                    context=collect_environment_context(),
                    metrics=metrics,
                )
                send_result = await send_payload(payload)
                if send_result == "success":
                    succeeded += 1
                    await self.store.delete_period(period_start, period_end)
                elif send_result == "discard":
                    discarded += 1
                    await self.store.delete_period(period_start, period_end)
                else:
                    logger.debug("Telemetry flush deferred for period %s-%s", period_start, period_end)
            if succeeded or discarded:
                await self.store.mark_flushed_now()
            if attempted == succeeded + discarded:
                return TelemetryFlushResult(
                    status="success" if discarded == 0 else "partial_success",
                    attempted=attempted,
                    succeeded=succeeded,
                    discarded=discarded,
                )
            if succeeded or discarded:
                return TelemetryFlushResult(
                    status="partial_success",
                    attempted=attempted,
                    succeeded=succeeded,
                    discarded=discarded,
                )
            return TelemetryFlushResult(status="failed", attempted=attempted)
        finally:
            await self.store.release_flush_lock()

    async def _record(
        self,
        metric_name: str,
        *,
        value: int,
        dimensions: dict[str, str],
        timestamp_ms: int | None,
    ) -> None:
        try:
            if await self.store.consent_state() == CONSENT_DISABLED:
                return
            normalized = _normalize_dimensions(metric_name, dimensions)
            await self.store.increment(
                metric_name=metric_name,
                value=value,
                dimensions=normalized,
                timestamp_ms=timestamp_ms,
            )
        except Exception as exc:
            logger.debug("Telemetry record failed for %s: %s", metric_name, exc)


def _normalize_dimensions(metric_name: str, dimensions: dict[str, str]) -> dict[str, str]:
    spec = METRIC_SPECS.get(metric_name)
    if spec is None:
        raise ValueError(f"Unknown telemetry metric: {metric_name}")

    ctx = get_telemetry_context()
    raw = {str(key): str(value) for key, value in dimensions.items() if value is not None}
    if "source" in spec.allowed_dimensions and "source" not in raw:
        raw["source"] = ctx.source
    if "caller" in spec.allowed_dimensions and "caller" not in raw:
        raw["caller"] = ctx.caller

    normalized: dict[str, str] = {}
    for key in sorted(spec.allowed_dimensions):
        if key not in raw:
            continue
        value = _normalize_dimension_value(key, raw[key])
        if key == "error_code" and value == "":
            continue
        normalized[key] = value

    missing = sorted(spec.required_dimensions - normalized.keys())
    if missing:
        raise ValueError(f"Telemetry metric {metric_name} missing required dimensions: {', '.join(missing)}")
    return normalized


def _normalize_dimension_value(key: str, value: str) -> str:
    allowed = DIMENSION_VALUES.get(key)
    if allowed is None:
        return value
    if value in allowed:
        return value
    if key == "error_code":
        return "internal_error"
    if "unknown" in allowed:
        return "unknown"
    return value


def _int_or_none(value: str | None) -> int | None:
    try:
        parsed = int(value or "0")
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


async def send_payload(payload: dict[str, Any]) -> str:
    import httpx

    from .transport import signed_headers
    from .payload import compact_json_bytes
    from .constants import TELEMETRY_ENDPOINT, TELEMETRY_HTTP_TIMEOUT_SEC

    body = compact_json_bytes(payload)
    headers = signed_headers(body)
    try:
        async with httpx.AsyncClient(timeout=TELEMETRY_HTTP_TIMEOUT_SEC) as client:
            response = await client.post(TELEMETRY_ENDPOINT, content=body, headers=headers)
    except Exception as exc:
        logger.debug("Telemetry upload failed: %s", exc)
        return "retry"
    if 200 <= response.status_code < 300:
        return "success"
    if 400 <= response.status_code < 500:
        logger.debug("Telemetry upload discarded after HTTP %s", response.status_code)
        return "discard"
    logger.debug("Telemetry upload retry after HTTP %s", response.status_code)
    return "retry"


def _new_batch_id() -> str:
    import secrets

    return f"tb_{secrets.token_hex(16)}"


def service_from_state(state: Any) -> TelemetryService | None:
    telemetry = getattr(state, "telemetry_svc", None)
    if isinstance(telemetry, TelemetryService):
        return telemetry
    return None


__all__ = ["TelemetryFlushResult", "TelemetryService", "send_payload", "service_from_state"]
