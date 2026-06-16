from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from mineru.cli_next.commands import config, list_resources, parse, show
from mineru.cli_next.main import app
from mineru.doclib.types import (
    ConfigSetResponse,
    DocInfo,
    FileInfo,
    FileInfoResponse,
    ListParsesResponse,
    ParseInfo,
    ParseResponse,
)


runner = CliRunner()


def test_cli_next_exposes_list_and_show_resource_trees() -> None:
    list_result = runner.invoke(app, ["list", "--help"])
    show_result = runner.invoke(app, ["show", "--help"])
    info_result = runner.invoke(app, ["info", "--help"])

    assert list_result.exit_code == 0
    assert show_result.exit_code == 0
    assert info_result.exit_code != 0


def test_config_rules_tree_uses_explicit_resource_names_and_remove() -> None:
    exclude_rules = runner.invoke(app, ["config", "exclude-rules", "--help"])
    exclude_legacy = runner.invoke(app, ["config", "exclude", "--help"])
    parse_server_legacy = runner.invoke(app, ["config", "parse-server", "--help"])
    remove = runner.invoke(app, ["config", "exclude-rules", "remove", "--help"])
    rm = runner.invoke(app, ["config", "exclude-rules", "rm", "--help"])

    assert exclude_rules.exit_code == 0
    assert exclude_legacy.exit_code != 0
    assert parse_server_legacy.exit_code != 0
    assert remove.exit_code == 0
    assert rm.exit_code != 0


def test_list_parses_passes_filters_to_doclib_client(monkeypatch: Any) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            calls.append(("init", {"timeout": timeout}))

        def list_parses(self, **kwargs: Any) -> ListParsesResponse:
            calls.append(("list_parses", kwargs))
            return ListParsesResponse(parses=[], total=0, limit=kwargs["limit"], offset=kwargs["offset"])

    monkeypatch.setattr(list_resources, "DoclibClient", _Client)

    result = runner.invoke(
        app,
        ["list", "parses", "--status", "pending", "--tier", "standard", "--limit", "10", "--offset", "5", "--json"],
    )

    assert result.exit_code == 0
    assert calls[-1] == (
        "list_parses",
        {"status": "pending", "tier": "standard", "limit": 10, "offset": 5},
    )


def test_parse_wait_ignores_failed_rows_outside_wait_ids(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    calls: list[dict[str, Any]] = []

    old_failed = ParseInfo(
        id=2,
        sha256="a" * 64,
        tier="pro",
        page_range="1~10",
        status="failed",
        privacy="remote",
        created_at=1,
        updated_at=2,
        error_code="parse_empty",
        error_msg="old failure",
    )
    new_done = ParseInfo(
        id=3,
        sha256="a" * 64,
        tier="pro",
        page_range="1~10",
        status="done",
        privacy="remote",
        via="remote",
        created_at=3,
        updated_at=4,
        done_at=4,
    )

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 31

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="a" * 64,
                tier="pro",
                page_range="1~10",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

        def list_parses(self, **kwargs: Any) -> ListParsesResponse:
            calls.append(kwargs)
            return ListParsesResponse(parses=[old_failed, new_done], total=2, limit=50, offset=0)

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--remote", "--wait", "1", "--json"])

    assert result.exit_code == 0
    assert calls == [{"ids": [3]}]
    assert "old failure" not in result.output


def test_show_file_uses_get_file_by_path(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[str] = []
    source = tmp_path / "demo.pdf"

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def get_file_by_path(self, path: str) -> FileInfoResponse:
            calls.append(path)
            return FileInfoResponse(
                file=FileInfo(
                    filename="demo.pdf",
                    path=path,
                    ext="pdf",
                    size_bytes=7,
                    mtime_ms=1,
                    status="active",
                    first_seen_at=1,
                    updated_at=1,
                )
            )

    monkeypatch.setattr(show, "DoclibClient", _Client)

    result = runner.invoke(app, ["show", "file", str(source), "--json"])

    assert result.exit_code == 0
    assert calls == [str(source.resolve())]


def test_show_doc_expands_files(monkeypatch: Any) -> None:
    calls: list[tuple[str, bool]] = []

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def get_doc(self, sha256: str, *, expand_files: bool = False) -> DocInfo:
            calls.append((sha256, expand_files))
            return DocInfo(sha256=sha256, short_id=sha256[:7], size_bytes=7, first_seen_at=1, updated_at=1, files=[])

    monkeypatch.setattr(show, "DoclibClient", _Client)

    result = runner.invoke(app, ["show", "doc", "a" * 64, "--json"])

    assert result.exit_code == 0
    assert calls == [("a" * 64, True)]


def test_config_set_uses_key_path_and_value_body(monkeypatch: Any) -> None:
    calls: list[tuple[str, str]] = []

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def set_config(self, key: str, request: Any) -> ConfigSetResponse:
            calls.append((key, request.value))
            return ConfigSetResponse(key=key, value=request.value, source="override")

    monkeypatch.setattr(config, "DoclibClient", _Client)

    result = runner.invoke(app, ["config", "set", "parse_server.local.mode", "managed"])

    assert result.exit_code == 0
    assert calls == [("parse_server.local.mode", "managed")]


def test_parse_content_read_failure_exits_nonzero(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.txt"
    source.write_text("hello", encoding="utf-8")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            assert request.path == str(source.resolve())
            return ParseResponse(sha256="a" * 64, tier="flash", page_range="all", status="done", cache_hit=True)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("content missing")

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--json"])

    assert result.exit_code == 1
    assert "Failed to read content: content missing" in result.output
