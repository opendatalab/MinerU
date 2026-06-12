from __future__ import annotations

from mineru.errors import http_status_for, error_type_for, registered_error_codes


DOCLIB_PERSISTED_ERROR_CODES = {
    "file_permission_denied",
    "ingest_failed",
    "metadata_failed",
    "no_accessible_file",
    "parse_empty",
    "parse_failed",
    "parse_json_write_failed",
    "scan_failed",
    "stat_failed",
}

DOCLIB_CONTRACT_ERROR_CODES = {
    "engine_unavailable",
    "no_engine",
    "not_cached",
    "parse_server_unavailable",
    "quality_tier_unavailable",
    "rule_not_found",
    "scan_not_found",
    "tier_mismatch",
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

    assert error_type_for("parse_empty") == "engine_error"
    assert error_type_for("parse_failed") == "engine_error"
    assert error_type_for("quality_tier_unavailable") == "engine_error"

    assert error_type_for("ingest_failed") == "api_error"
    assert error_type_for("metadata_failed") == "api_error"
    assert error_type_for("parse_json_write_failed") == "api_error"
    assert error_type_for("scan_failed") == "api_error"


def test_http_status_for_error_codes_is_stable() -> None:
    assert http_status_for("invalid_request") == 400
    assert http_status_for("file_not_found") == 404
    assert http_status_for("job_not_found") == 404
    assert http_status_for("not_cached") == 404
    assert http_status_for("rule_not_found") == 404
    assert http_status_for("scan_not_found") == 404
    assert http_status_for("watch_not_found") == 404

    assert http_status_for("invalid_api_key") == 401
    assert http_status_for("feature_requires_api_key") == 403
    assert http_status_for("rate_limit_exceeded") == 429

    assert http_status_for("parse_failed") == 503
    assert http_status_for("engine_unavailable") == 503
    assert http_status_for("parse_server_unavailable") == 500
    assert http_status_for("internal_error") == 500
