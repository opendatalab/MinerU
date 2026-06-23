"""Telemetry observation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TelemetryObservation:
    metric_name: str
    value: int = 1
    duration_ms: int | None = None
    dimensions: dict[str, str] = field(default_factory=dict)


__all__ = ["TelemetryObservation"]
