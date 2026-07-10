from __future__ import annotations

import httpx

from mineru.cli.runtime import to_mineru_error
from mineru.errors import MineruError, ServerNotRunningError, http_status_for, error_type_for, registered_error_codes


DOCLIB_PERSISTED_ERROR_CODES = {
    "file_permission_denied",
    "ingest_failed",
    "no_accessible_file",
    "open_failed",
    "parse_empty",
    "parse_failed",
    "parse_json_write_failed",
    "read_metadata_failed",
    "scan_failed",
    "stat_failed",
}

DOCLIB_CONTRACT_ERROR_CODES = {
    "asset_not_available",
    "bbox_not_available",
    "block_not_found",
    "context_not_applicable",
    "doc_not_found",
    "engine_unavailable",
    "format_not_supported",
    "image_output_extension_unsupported",
    "invalid_config_key",
    "invalid_locator",
    "multi_page_image_not_supported",
    "no_engine",
    "not_cached",
    "page_not_cached",
    "parse_page_remap_failed",
    "parse_server_unavailable",
    "quality_tier_unavailable",
    "rule_not_found",
    "scan_not_found",
    "tier_mismatch",
    "tier_not_cached",
    "watch_not_found",
}


def test_doclib_error_codes_are_registered() -> None:
    codes = registered_error_codes()

    assert DOCLIB_PERSISTED_ERROR_CODES <= codes
    assert DOCLIB_CONTRACT_ERROR_CODES <= codes


def test_doclib_error_code_types_are_stable() -> None:
    assert error_type_for("file_permission_denied") == "invalid_request_error"
    assert error_type_for("stat_failed") == "invalid_request_error"
    assert error_type_for("no_accessible_file") == "invalid_request_error"
    assert error_type_for("bbox_not_available") == "invalid_request_error"
    assert error_type_for("context_not_applicable") == "invalid_request_error"
    assert error_type_for("format_not_supported") == "invalid_request_error"
    assert error_type_for("image_output_extension_unsupported") == "invalid_request_error"
    assert error_type_for("invalid_config_key") == "invalid_request_error"
    assert error_type_for("invalid_locator") == "invalid_request_error"
    assert error_type_for("multi_page_image_not_supported") == "invalid_request_error"

    assert error_type_for("parse_empty") == "engine_error"
    assert error_type_for("parse_failed") == "engine_error"
    assert error_type_for("parse_page_remap_failed") == "engine_error"
    assert error_type_for("quality_tier_unavailable") == "engine_error"

    assert error_type_for("parse_wait_timeout") == "timeout_error"

    assert error_type_for("ingest_failed") == "api_error"
    assert error_type_for("open_failed") == "api_error"
    assert error_type_for("parse_json_write_failed") == "api_error"
    assert error_type_for("read_metadata_failed") == "api_error"
    assert error_type_for("scan_failed") == "api_error"


def test_http_status_for_error_codes_is_stable() -> None:
    assert http_status_for("invalid_request") == 400
    assert http_status_for("file_not_found") == 404
    assert http_status_for("asset_not_available") == 404
    assert http_status_for("block_not_found") == 404
    assert http_status_for("doc_not_found") == 404
    assert http_status_for("job_not_found") == 404
    assert http_status_for("not_cached") == 404
    assert http_status_for("page_not_cached") == 404
    assert http_status_for("rule_not_found") == 404
    assert http_status_for("scan_not_found") == 404
    assert http_status_for("tier_not_cached") == 404
    assert http_status_for("watch_not_found") == 404

    assert http_status_for("invalid_api_key") == 401
    assert http_status_for("feature_requires_api_key") == 403
    assert http_status_for("rate_limit_exceeded") == 429

    assert http_status_for("parse_failed") == 503
    assert http_status_for("engine_unavailable") == 503
    assert http_status_for("parse_wait_timeout") == 408
    assert http_status_for("parse_server_unavailable") == 500
    assert http_status_for("server_busy") == 503
    assert http_status_for("internal_error") == 500


def test_mineru_error_string_is_human_readable() -> None:
    assert str(MineruError("not_cached", "Requested parsed content is not cached.", "page_range")) == (
        "Requested parsed content is not cached."
    )
    assert str(MineruError("not_cached")) == "not_cached"


def test_cli_server_not_running_error_is_preserved() -> None:
    error = to_mineru_error(ServerNotRunningError())

    assert error.code == "server_not_running"
    assert error.type == "api_error"
    assert error.message == "Local mineru server is not running. Run 'mineru server start'."


def test_cli_connection_error_uses_server_not_running_code() -> None:
    error = to_mineru_error(httpx.ConnectError("connection refused"))

    assert error.code == "server_not_running"
    assert error.type == "api_error"
    assert error.message == "Local mineru server is not running. Run 'mineru server start'."


def test_cli_unknown_exception_uses_exception_message() -> None:
    error = to_mineru_error(RuntimeError("boom"))

    assert error.code == "api_error"
    assert error.type == "api_error"
    assert error.message == "boom"
