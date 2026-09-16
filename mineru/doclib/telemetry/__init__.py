"""Doclib telemetry primitives."""

from .buckets import duration_bucket, file_size_bucket, pages_bucket, results_bucket
from .context import (
    TelemetryContext,
    get_telemetry_context,
    infer_default_client_context,
    reset_telemetry_context,
    set_telemetry_context,
)
from .service import TelemetryService
from .store import TelemetryStore

__all__ = [
    "TelemetryContext",
    "TelemetryService",
    "TelemetryStore",
    "duration_bucket",
    "file_size_bucket",
    "get_telemetry_context",
    "infer_default_client_context",
    "pages_bucket",
    "reset_telemetry_context",
    "results_bucket",
    "set_telemetry_context",
]
