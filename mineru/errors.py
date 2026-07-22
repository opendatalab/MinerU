"""Error definitions for MinerU server and CLI.  Aligned with docs/next/errors.md."""

from __future__ import annotations

from typing import Literal

ErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "rate_limit_error",
    "engine_error",
    "timeout_error",
    "api_error",
]

# ── error_code → type mapping ──────────────────────────────────────

_ERROR_TYPE_MAP: dict[str, ErrorType] = {
    # invalid_request_error
    "invalid_request": "invalid_request_error",
    "unsupported_output_format": "invalid_request_error",
    "unsupported_source": "invalid_request_error",
    "page_range_invalid": "invalid_request_error",
    "file_type_unsupported": "invalid_request_error",
    "parse_not_required": "invalid_request_error",
    "file_encrypted": "invalid_request_error",
    "file_corrupted": "invalid_request_error",
    "file_too_large": "invalid_request_error",
    "file_not_found": "invalid_request_error",
    "file_permission_denied": "invalid_request_error",
    "file_hash_mismatch": "invalid_request_error",
    "image_output_extension_unsupported": "invalid_request_error",
    "bytes_mismatch": "invalid_request_error",
    "upload_not_found": "invalid_request_error",
    "upload_not_ready": "invalid_request_error",
    "upload_expired": "invalid_request_error",
    "job_not_found": "invalid_request_error",
    "job_already_terminal": "invalid_request_error",
    "model_not_found": "invalid_request_error",
    "not_cached": "invalid_request_error",
    "no_accessible_file": "invalid_request_error",
    "asset_not_available": "invalid_request_error",
    "block_not_found": "invalid_request_error",
    "doc_not_found": "invalid_request_error",
    "page_not_cached": "invalid_request_error",
    "parse_server_dependency_missing": "invalid_request_error",
    "parse_server_model_not_ready": "invalid_request_error",
    "tier_not_cached": "invalid_request_error",
    "bbox_not_available": "invalid_request_error",
    "context_not_applicable": "invalid_request_error",
    "format_not_supported": "invalid_request_error",
    "invalid_config_key": "invalid_request_error",
    "invalid_config_value": "invalid_request_error",
    "invalid_locator": "invalid_request_error",
    "multi_page_image_not_supported": "invalid_request_error",
    "rule_not_found": "invalid_request_error",
    "scan_not_found": "invalid_request_error",
    "stat_failed": "invalid_request_error",
    "remote_unsupported_for_file_type": "invalid_request_error",
    "tier_mismatch": "invalid_request_error",
    "tier_unsupported_for_file_type": "invalid_request_error",
    "tier_unsupported_for_remote": "invalid_request_error",
    "watch_not_found": "invalid_request_error",
    # authentication_error
    "invalid_api_key": "authentication_error",
    # permission_error
    "feature_requires_api_key": "permission_error",
    "list_requires_api_key": "permission_error",
    "quota_exceeded": "permission_error",
    # rate_limit_error
    "rate_limit_exceeded": "rate_limit_error",
    # engine_error
    "no_engine": "engine_error",
    "engine_unavailable": "engine_error",
    "parse_empty": "engine_error",
    "parse_failed": "engine_error",
    "parse_page_remap_failed": "engine_error",
    "parse_timeout": "engine_error",
    "quality_tier_unavailable": "engine_error",
    "model_preload_dependency_missing": "engine_error",
    "model_preload_files_missing": "engine_error",
    "model_preload_device_unavailable": "engine_error",
    "model_preload_failed": "engine_error",
    # timeout_error
    "parse_wait_timeout": "timeout_error",
    # api_error
    "ingest_failed": "api_error",
    "internal_error": "api_error",
    "metadata_failed": "api_error",
    "open_failed": "api_error",
    "parse_json_write_failed": "api_error",
    "read_metadata_failed": "api_error",
    "scan_failed": "api_error",
    "service_unavailable": "api_error",
    "server_not_running": "api_error",
    "server_instance_mismatch": "api_error",
    "server_busy": "api_error",
    "remote_timeout": "api_error",
    "remote_unreachable": "api_error",
    "parse_server_unavailable": "api_error",
}

_ERROR_STATUS_MAP: dict[ErrorType, int] = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "rate_limit_error": 429,
    "engine_error": 503,
    "timeout_error": 408,
    "api_error": 500,
}

_ERROR_CODE_STATUS_MAP: dict[str, int] = {
    "file_not_found": 404,
    "asset_not_available": 404,
    "block_not_found": 404,
    "doc_not_found": 404,
    "job_not_found": 404,
    "model_not_found": 404,
    "not_cached": 404,
    "page_not_cached": 404,
    "rule_not_found": 404,
    "scan_not_found": 404,
    "tier_not_cached": 404,
    "upload_not_found": 404,
    "watch_not_found": 404,
    "remote_timeout": 503,
    "remote_unreachable": 503,
    "server_busy": 503,
}


def error_type_for(code: str) -> ErrorType:
    return _ERROR_TYPE_MAP.get(code, "api_error")


def registered_error_codes() -> frozenset[str]:
    return frozenset(_ERROR_TYPE_MAP)


def http_status_for(code: str, error_type: ErrorType | None = None) -> int:
    if code in _ERROR_CODE_STATUS_MAP:
        return _ERROR_CODE_STATUS_MAP[code]
    return _ERROR_STATUS_MAP.get(error_type or error_type_for(code), 500)


# ── exception hierarchy ────────────────────────────────────────────


class MineruError(Exception):
    """Base error with OpenAI-compatible fields."""

    def __init__(self, code: str, message: str = "", param: str | None = None) -> None:
        self.code = code
        self.message = message
        self.param = param
        self.type = error_type_for(code)

    def __str__(self) -> str:
        return self.message or self.code


class InvalidRequestError(MineruError):
    def __init__(self, code: str = "invalid_request", message: str = "", param: str | None = None) -> None:
        super().__init__(code, message, param)


class NotFoundError(MineruError):
    def __init__(self, code: str = "file_not_found", message: str = "", param: str | None = None) -> None:
        super().__init__(code, message, param)


class PermissionError_(MineruError):
    """Renamed to avoid shadowing builtin PermissionError."""

    def __init__(self, code: str = "file_permission_denied", message: str = "", param: str | None = None) -> None:
        super().__init__(code, message, param)


class ConflictError(MineruError):
    def __init__(self, code: str = "job_already_terminal", message: str = "", param: str | None = None) -> None:
        super().__init__(code, message, param)


class EngineError(MineruError):
    def __init__(self, code: str = "parse_failed", message: str = "", param: str | None = None) -> None:
        super().__init__(code, message, param)


class ServerNotRunningError(MineruError):
    def __init__(self) -> None:
        super().__init__("server_not_running", "Local mineru server is not running. Run 'mineru server start'.")


class ServerBusyError(MineruError):
    def __init__(self, message: str = "MinerU server is busy. Retry the request.") -> None:
        super().__init__("server_busy", message)


# ── FastAPI error response builder ─────────────────────────────────


def error_response(exc: MineruError) -> dict:
    return {
        "error": {
            "type": exc.type,
            "code": exc.code,
            "message": exc.message,
            "param": exc.param,
        }
    }
