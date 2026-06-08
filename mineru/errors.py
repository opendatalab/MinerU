"""Error definitions for MinerU server and CLI.  Aligned with NEXT-ERROR.md."""

from __future__ import annotations

# ── error_code → type mapping ──────────────────────────────────────

_ERROR_TYPE_MAP: dict[str, str] = {
    # invalid_request_error
    "invalid_request": "invalid_request_error",
    "page_range_invalid": "invalid_request_error",
    "file_type_unsupported": "invalid_request_error",
    "file_encrypted": "invalid_request_error",
    "file_corrupted": "invalid_request_error",
    "file_too_large": "invalid_request_error",
    "file_not_found": "invalid_request_error",
    "file_permission_denied": "invalid_request_error",
    "file_hash_mismatch": "invalid_request_error",
    "bytes_mismatch": "invalid_request_error",
    "upload_not_found": "invalid_request_error",
    "upload_not_ready": "invalid_request_error",
    "upload_expired": "invalid_request_error",
    "job_not_found": "invalid_request_error",
    "job_already_terminal": "invalid_request_error",
    "model_not_found": "invalid_request_error",
    "not_cached": "invalid_request_error",
    "tier_mismatch": "invalid_request_error",
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
    "parse_failed": "engine_error",
    "parse_timeout": "engine_error",
    # api_error
    "internal_error": "api_error",
    "service_unavailable": "api_error",
    "server_not_running": "api_error",
    "server_busy": "api_error",
    "remote_timeout": "api_error",
    "remote_unreachable": "api_error",
    "parse_server_unavailable": "api_error",
}


def error_type_for(code: str) -> str:
    return _ERROR_TYPE_MAP.get(code, "api_error")


# ── exception hierarchy ────────────────────────────────────────────


class MineruError(Exception):
    """Base error with OpenAI-compatible fields."""

    def __init__(self, code: str, message: str = "", param: str | None = None) -> None:
        self.code = code
        self.message = message
        self.param = param
        self.type = error_type_for(code)


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
