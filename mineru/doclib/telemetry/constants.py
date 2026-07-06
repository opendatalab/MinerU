"""Telemetry constants and schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TelemetryConsentState = Literal["unset", "enabled", "disabled"]
TelemetrySource = Literal["cli", "sdk", "http_api", "watch", "background", "unknown"]
TelemetryCaller = Literal["agent", "user", "sdk", "http_client", "system", "unknown"]

CONSENT_UNSET: TelemetryConsentState = "unset"
CONSENT_ENABLED: TelemetryConsentState = "enabled"
CONSENT_DISABLED: TelemetryConsentState = "disabled"

TELEMETRY_ENDPOINT = "https://staging.mineru.org.cn/metrics/v2/metrics"
TELEMETRY_APP_KEY = "213a83db-44c8-4218-90c0-ded685cca86e"
TELEMETRY_APP_SECRET = "sec_5f8d9a2b"
TELEMETRY_API_VERSION = "v2"
TELEMETRY_SCHEMA_VERSION = "1"
TELEMETRY_HTTP_TIMEOUT_SEC = 10
TELEMETRY_FLUSH_INTERVAL_SEC = 7200
TELEMETRY_FLUSH_LOCK_TTL_SEC = 1800
TELEMETRY_MAX_FLUSH_PERIODS = 10

OFFICIAL_REMOTE_HOST_SUFFIXES = ("mineru.net", "mineru.org.cn")

DIM_SOURCE_VALUES = frozenset({"cli", "sdk", "http_api", "watch", "background", "unknown"})
DIM_CALLER_VALUES = frozenset({"agent", "user", "sdk", "http_client", "system", "unknown"})
DIM_STATUS_VALUES = frozenset(
    {
        "cached",
        "canceled",
        "changed",
        "deleted",
        "direct",
        "excluded",
        "failed",
        "hit",
        "miss",
        "new",
        "queued",
        "refreshed",
        "reused",
        "seen",
        "skipped",
        "succeeded",
        "timeout",
        "unreachable",
        "unsupported",
    }
)
DIM_TIER_VALUES = frozenset(
    {
        "default",
        "flash",
        "medium",
        "high",
        "extra_high",
        "medium(default)",
        "high(default)",
        "extra_high(default)",
        "unknown",
    }
)
DIM_SERVER_VALUES = frozenset(
    {"local(flash)", "local(managed)", "local(self-hosted)", "remote(official)", "remote(custom)", "none", "unknown"}
)
DIM_CONTENT_MODE_VALUES = frozenset({"read", "parse_output", "export", "unknown"})
DIM_OUTPUT_FORMAT_VALUES = frozenset({"markdown", "image", "other"})
DIM_TRIGGER_VALUES = frozenset({"parse", "scan", "watch", "show", "background", "unknown"})
DIM_RESULT_VALUES = frozenset(
    {"seen", "refreshed", "new", "changed", "deleted", "unreachable", "error", "unsupported", "excluded", "hit", "miss", "unknown"}
)
DIM_ERROR_CODE_VALUES = frozenset(
    {
        "internal_error",
        "invalid_request",
        "file_not_found",
        "file_permission_denied",
        "not_cached",
        "no_accessible_file",
        "quality_tier_unavailable",
        "no_engine",
        "engine_unavailable",
        "parse_server_unavailable",
        "tier_mismatch",
        "parse_failed",
        "parse_timeout",
        "metadata_failed",
        "open_failed",
        "parse_json_write_failed",
        "read_metadata_failed",
        "ingest_failed",
        "scan_failed",
    }
)
DIM_DURATION_BUCKET_VALUES = frozenset(
    {
        "lt_1s",
        "1_5s",
        "5_30s",
        "30_120s",
        "2_10m",
        "gt_10m",
    }
)
DIM_PAGES_BUCKET_VALUES = frozenset(
    {
        "1",
        "2_5",
        "6_20",
        "21_100",
        "101_500",
        "gt_500",
    }
)
DIM_FILE_SIZE_BUCKET_VALUES = frozenset(
    {
        "lt_1mb",
        "1_10mb",
        "10_50mb",
        "50_200mb",
        "gt_200mb",
    }
)
DIM_RESULTS_BUCKET_VALUES = frozenset(
    {
        "0",
        "1_5",
        "6_20",
        "21_100",
        "gt_100",
    }
)
DIM_BUCKET_VALUES = (
    DIM_DURATION_BUCKET_VALUES | DIM_PAGES_BUCKET_VALUES | DIM_FILE_SIZE_BUCKET_VALUES | DIM_RESULTS_BUCKET_VALUES
)

DIMENSION_VALUES: dict[str, frozenset[str]] = {
    "source": DIM_SOURCE_VALUES,
    "caller": DIM_CALLER_VALUES,
    "status": DIM_STATUS_VALUES,
    "tier": DIM_TIER_VALUES,
    "server": DIM_SERVER_VALUES,
    "error_code": DIM_ERROR_CODE_VALUES,
    "bucket": DIM_BUCKET_VALUES,
    "pages_bucket": DIM_PAGES_BUCKET_VALUES,
    "result": DIM_RESULT_VALUES,
    "content_mode": DIM_CONTENT_MODE_VALUES,
    "output_format": DIM_OUTPUT_FORMAT_VALUES,
    "trigger": DIM_TRIGGER_VALUES,
}


@dataclass(frozen=True)
class MetricSpec:
    allowed_dimensions: frozenset[str]
    required_dimensions: frozenset[str] = field(default_factory=frozenset)


REQ_CONTEXT = frozenset({"source", "caller"})
STATUS = frozenset({"status"})
TIER = frozenset({"tier"})
ERROR = frozenset({"error_code"})
BUCKET = frozenset({"bucket"})
SERVER = frozenset({"server"})

METRIC_SPECS: dict[str, MetricSpec] = {
    "parse.request.count": MetricSpec(REQ_CONTEXT),
    "parse.finished.count": MetricSpec(REQ_CONTEXT | STATUS | TIER | ERROR),
    "parse.duration_bucket.count": MetricSpec(REQ_CONTEXT | STATUS | TIER | BUCKET),
    "parse.wait.count": MetricSpec(REQ_CONTEXT | STATUS | TIER | frozenset({"pages_bucket"})),
    "parse.wait_duration_bucket.count": MetricSpec(REQ_CONTEXT | STATUS | TIER | frozenset({"pages_bucket", "bucket"})),
    "parse.files.count": MetricSpec(REQ_CONTEXT | TIER),
    "parse.pages.count": MetricSpec(REQ_CONTEXT | TIER),
    "parse.file_size_bucket.count": MetricSpec(REQ_CONTEXT | TIER | BUCKET),
    "parse.pages_bucket.count": MetricSpec(REQ_CONTEXT | TIER | BUCKET),
    "parse.invalidate.count": MetricSpec(STATUS),
    "parse_task.created.count": MetricSpec(TIER),
    "parse_task.started.count": MetricSpec(TIER),
    "parse_task.finished.count": MetricSpec(STATUS | TIER | ERROR),
    "parse_task.duration_bucket.count": MetricSpec(STATUS | TIER | BUCKET),
    "parse_task.execute.count": MetricSpec(STATUS | TIER | SERVER | ERROR),
    "parse_task.execute_duration_bucket.count": MetricSpec(STATUS | TIER | SERVER | BUCKET),
    "parse_task.write.count": MetricSpec(STATUS | TIER | ERROR),
    "parse_task.write_duration_bucket.count": MetricSpec(STATUS | TIER | BUCKET),
    "parse_task.files.count": MetricSpec(TIER),
    "parse_task.pages.count": MetricSpec(TIER),
    "ingest.finished.count": MetricSpec(STATUS | frozenset({"trigger"})),
    "ingest.duration_bucket.count": MetricSpec(STATUS | frozenset({"trigger", "bucket"})),
    "search.request.count": MetricSpec(REQ_CONTEXT),
    "search.finished.count": MetricSpec(REQ_CONTEXT | STATUS),
    "search.duration_bucket.count": MetricSpec(REQ_CONTEXT | STATUS | BUCKET),
    "search.results_bucket.count": MetricSpec(REQ_CONTEXT | BUCKET),
    "find.request.count": MetricSpec(REQ_CONTEXT),
    "find.finished.count": MetricSpec(REQ_CONTEXT | STATUS),
    "find.duration_bucket.count": MetricSpec(REQ_CONTEXT | STATUS | BUCKET),
    "find.results_bucket.count": MetricSpec(REQ_CONTEXT | BUCKET),
    "content.request.count": MetricSpec(REQ_CONTEXT | frozenset({"content_mode", "output_format"})),
    "content.finished.count": MetricSpec(REQ_CONTEXT | STATUS | TIER | frozenset({"content_mode", "output_format"})),
    "content.duration_bucket.count": MetricSpec(REQ_CONTEXT | STATUS | TIER | BUCKET | frozenset({"content_mode", "output_format"})),
    "scan.request.count": MetricSpec(REQ_CONTEXT),
    "scan.finished.count": MetricSpec(STATUS),
    "scan.duration_bucket.count": MetricSpec(STATUS | BUCKET),
    "scan.reuse.count": MetricSpec(STATUS),
    "scan.files.count": MetricSpec(frozenset({"result"})),
    "watch.add.count": MetricSpec(REQ_CONTEXT),
    "watch.remove.count": MetricSpec(REQ_CONTEXT),
    "watch.add.finished.count": MetricSpec(REQ_CONTEXT | STATUS),
    "watch.remove.finished.count": MetricSpec(REQ_CONTEXT | STATUS),
}
