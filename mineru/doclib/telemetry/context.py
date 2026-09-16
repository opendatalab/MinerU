"""Telemetry request context helpers."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from .caller import infer_caller_from_process_tree
from .constants import TelemetryCaller, TelemetrySource

_DEFAULT_CONTEXT = None
_TELEMETRY_CONTEXT: ContextVar["TelemetryContext | None"] = ContextVar("doclib_telemetry_context", default=_DEFAULT_CONTEXT)


@dataclass(frozen=True)
class TelemetryContext:
    source: TelemetrySource = "unknown"
    caller: TelemetryCaller = "unknown"

    def dimensions(self) -> dict[str, str]:
        return {"source": self.source, "caller": self.caller}


def get_telemetry_context() -> TelemetryContext:
    ctx = _TELEMETRY_CONTEXT.get()
    if ctx is None:
        return TelemetryContext()
    return ctx


def set_telemetry_context(context: TelemetryContext) -> Token[TelemetryContext | None]:
    return _TELEMETRY_CONTEXT.set(context)


def reset_telemetry_context(token: Token[TelemetryContext | None]) -> None:
    _TELEMETRY_CONTEXT.reset(token)


def infer_default_client_context(source: TelemetrySource = "sdk") -> TelemetryContext:
    return TelemetryContext(source=source, caller=infer_caller_from_process_tree())


__all__ = [
    "TelemetryContext",
    "get_telemetry_context",
    "infer_default_client_context",
    "reset_telemetry_context",
    "set_telemetry_context",
]
