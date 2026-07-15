from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from mineru.cli.commands import config, invalidate, list_resources, parse, read, server, show, telemetry
from mineru.cli.commands import cleanup as cleanup_cmd
from mineru.cli.commands import scan as scan_mod
from mineru.cli.commands import search as search_mod
from mineru.cli.commands import watch as watch_cmd
from mineru.cli import output as output_mod
from mineru.cli import telemetry as cli_telemetry
from mineru.cli.main import app
from mineru.doclib.telemetry import TelemetryContext
from mineru.doclib.types import (
    ContentAsset,
    ContentNextRequest,
    ContentRange,
    ContentRequestScope,
    ConfigResponse,
    ConfigSetResponse,
    DocInfo,
    DocContentResponse,
    ExcludeRuleInfo,
    ExcludeRuleListResponse,
    FileInfo,
    FileInfoResponse,
    FindResponse,
    FindResult,
    InvalidateResponse,
    ListDocsResponse,
    ListFilesResponse,
    ListParsesResponse,
    ParsingRuleInfo,
    ParsingRuleListResponse,
    ParseInfo,
    ParseResponse,
    ScanListResponse,
    ScanInfo,
    SearchFile,
    SearchResponse,
    SearchResult,
    ServerStatusResponse,
    TelemetryActionResponse,
    TelemetryStatusResponse,
    WatchInfo,
    WatchListResponse,
)
from mineru.errors import ServerNotRunningError
from mineru.filetypes import FILE_TYPE_BY_EXTENSION
from mineru.version import __version__


runner = CliRunner()


class _FakeTable:
    def __init__(self, *, title: str) -> None:
        self.title = title
        self.rows: list[tuple[str, ...]] = []

    def add_column(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_row(self, *values: str) -> None:
        self.rows.append(tuple(values))


def test_cli_next_exposes_list_and_show_resource_trees() -> None:
    list_result = runner.invoke(app, ["list", "--help"])
    show_result = runner.invoke(app, ["show", "--help"])
    read_result = runner.invoke(app, ["read", "--help"])
    info_result = runner.invoke(app, ["info", "--help"])

    assert list_result.exit_code == 0
    assert show_result.exit_code == 0
    assert read_result.exit_code == 0
    assert info_result.exit_code != 0


def test_parse_and_read_help_document_output_parent_creation() -> None:
    parse_result = runner.invoke(app, ["parse", "--help"])
    read_result = runner.invoke(app, ["read", "--help"])

    assert parse_result.exit_code == 0
    assert read_result.exit_code == 0
    assert "creates parent" in parse_result.output
    assert "directories" in parse_result.output
    assert "creates parent" in read_result.output
    assert "directories" in read_result.output


def test_telemetry_command_tree_is_available() -> None:
    result = runner.invoke(app, ["telemetry", "--help"])

    assert result.exit_code == 0
    assert "status" in result.output
    assert "preview" in result.output
    assert "flush" in result.output


def test_version_command_prints_mineru_and_python_versions() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert f"MinerU version: {__version__}" in result.output
    assert f"Python version: {sys.version.split()[0]}" in result.output


def test_root_version_option_matches_version_command() -> None:
    option_result = runner.invoke(app, ["--version"])
    command_result = runner.invoke(app, ["version"])

    assert option_result.exit_code == 0
    assert command_result.exit_code == 0
    assert option_result.output == command_result.output


def test_version_json_writes_single_json_object() -> None:
    result = runner.invoke(app, ["version", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {"mineru_version": __version__, "python_version": sys.version.split()[0]}
    assert "MinerU version:" not in result.output


def test_root_commands_keep_product_order() -> None:
    command = get_command(app)

    assert command.list_commands(None) == [
        "parse",
        "read",
        "scan",
        "watch",
        "search",
        "find",
        "usage",
        "list",
        "show",
        "telemetry",
        "server",
        "config",
        "invalidate",
        "forget",
        "cleanup",
        "version",
    ]


def test_root_help_hides_typer_completion_options() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--version" in result.output
    assert "--install-completion" not in result.output
    assert "--show-completion" not in result.output


def test_root_show_completion_is_not_a_supported_option() -> None:
    result = runner.invoke(app, ["--show-completion"])

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_search_and_find_help_list_filter_values() -> None:
    search_result = runner.invoke(app, ["search", "--help"])
    find_result = runner.invoke(app, ["find", "--help"])

    assert search_result.exit_code == 0
    assert find_result.exit_code == 0
    assert "File type filter:" in search_result.output
    for token in dict.fromkeys(FILE_TYPE_BY_EXTENSION.values()):
        assert token in search_result.output
    assert "File extension filter:" in find_result.output
    for token in FILE_TYPE_BY_EXTENSION:
        assert token in find_result.output


def test_print_error_uses_single_rich_render(monkeypatch: Any) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _Console:
        def print(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

    monkeypatch.setattr(output_mod, "stderr_console", _Console())

    output_mod.print_error(
        "No medium, high, or xhigh engine available. You can start a local parse-server, use --remote, or explicitly pass "
        "--tier flash for text-only preview."
    )

    assert len(calls) == 1
    rendered = str(calls[0][0][0])
    assert rendered.startswith("Error: ")
    assert "--tier flash" in rendered


def test_find_results_render_as_list_without_empty_search_columns() -> None:
    rendered = search_mod._render_find_results(
        FindResponse(
            total=2,
            query="resume",
            results=[
                FindResult(filename="resume.pdf", paths=["/tmp/resume.pdf"]),
                FindResult(filename="cv.docx", paths=["/tmp/cv.docx"]),
            ],
        )
    )

    assert "Search results (2 total)" in rendered
    assert "1. resume.pdf (/tmp/resume.pdf)" in rendered
    assert "\n\n2. cv.docx (/tmp/cv.docx)" in rendered
    assert "Tier" not in rendered
    assert "Snippet" not in rendered
    assert "Paths" not in rendered


def test_search_results_render_as_list_without_table() -> None:
    rendered = search_mod._render_search_results(
        SearchResponse(
            total=1,
            query="python",
            results=[
                SearchResult(
                    sha256="a" * 64,
                    short_id="aaaaaaa",
                    tier="medium",
                    snippet="Python\nengineer",
                    files=[SearchFile(path="/tmp/resume.pdf", filename="resume.pdf", ext="pdf", status="active")],
                )
            ],
        )
    )

    assert "Search results (1 total)" in rendered
    assert "1. Document aaaaaaa Tier: medium" in rendered
    assert "   Files:\n   /tmp/resume.pdf" in rendered
    assert "   Python  engineer" in rendered
    assert "Snippet:" not in rendered


def test_search_result_snippet_preserves_fts_match_after_long_prefix() -> None:
    rendered = search_mod._render_search_results(
        SearchResponse(
            total=1,
            query="token dropping",
            results=[
                SearchResult(
                    sha256="a" * 64,
                    short_id="aaaaaaa",
                    tier="flash",
                    snippet=(
                        "...on each node. Under this constraint, our MoE training framework can nearly "
                        "achieve full computation-communication overlap. <mark>Token</mark> <mark>Dropping</mark>"
                    ),
                    files=[SearchFile(path="/tmp/deepseek.pdf", filename="deepseek.pdf", ext="pdf", status="active")],
                )
            ],
        )
    )

    assert "<mark>Token</mark>" in rendered
    assert "<mark>Dropping</mark>" in rendered


def test_search_results_render_active_files_then_inactive_fallback_and_orphan() -> None:
    rendered = search_mod._render_search_results(
        SearchResponse(
            total=3,
            query="needle",
            results=[
                SearchResult(
                    sha256="a" * 64,
                    short_id="aaaaaaa",
                    title="Active document",
                    tier="high",
                    snippet="active",
                    files=[
                        SearchFile(path="/tmp/deleted.pdf", filename="deleted.pdf", ext="pdf", status="deleted"),
                        SearchFile(path="/tmp/active.pdf", filename="active.pdf", ext="pdf", status="active"),
                    ],
                ),
                SearchResult(
                    sha256="b" * 64,
                    short_id="bbbbbbb",
                    tier="medium",
                    snippet="inactive",
                    files=[
                        SearchFile(
                            path="/volume/unreachable.pdf",
                            filename="unreachable.pdf",
                            ext="pdf",
                            status="unreachable",
                        ),
                        SearchFile(path="/tmp/old.pdf", filename="old.pdf", ext="pdf", status="deleted"),
                    ],
                ),
                SearchResult(
                    sha256="c" * 64,
                    short_id="ccccccc",
                    tier="flash",
                    snippet="orphan",
                    files=[],
                ),
            ],
        )
    )

    assert "/tmp/active.pdf" in rendered
    assert "/tmp/deleted.pdf" not in rendered
    assert "/volume/unreachable.pdf (unreachable)" in rendered
    assert "/tmp/old.pdf (deleted)" in rendered
    assert "3. Document ccccccc Tier: flash\n   File no longer exists." in rendered


def test_list_renderers_return_tables(monkeypatch: Any) -> None:
    monkeypatch.setattr(list_resources, "Table", _FakeTable)
    parse = ParseInfo(
        id=7,
        sha256="a" * 64,
        short_id="aaaaaaa",
        tier="medium",
        page_range="1~2",
        status="done",
        privacy="local",
        created_at=1,
        updated_at=2,
    )
    scan = ScanInfo(
        id=8,
        path="/tmp/docs",
        kind="manual",
        source="cli",
        status="done",
        files_seen=3,
        files_refreshed=2,
        files_error=1,
        created_at=1,
        updated_at=2,
    )
    file_info = FileInfo(
        filename="demo.pdf",
        path="/tmp/demo.pdf",
        ext="pdf",
        size_bytes=7,
        mtime_ms=1,
        sha256="b" * 64,
        short_id="bbbbbbb",
        status="active",
        first_seen_at=1,
        updated_at=2,
    )
    doc = DocInfo(
        sha256="c" * 64,
        short_id="ccccccc",
        size_bytes=7,
        file_type="pdf",
        title="Demo",
        page_count=5,
        first_seen_at=1,
        updated_at=2,
    )

    parses_table = list_resources._render_list_parses(ListParsesResponse(parses=[parse], total=1, limit=50))
    scans_table = list_resources._render_list_scans(ScanListResponse(scans=[scan], total=1, limit=50))
    files_table = list_resources._render_list_files(ListFilesResponse(files=[file_info], total=1, limit=50))
    docs_table = list_resources._render_list_docs(ListDocsResponse(docs=[doc], total=1, limit=50))

    assert parses_table.title == "Parses (1 total)"
    assert ("7", "done", "medium", "1~2", "aaaaaaa") in parses_table.rows
    assert scans_table.title == "Scans (1 total)"
    assert ("8", "done", "manual", "/tmp/docs", "3", "2", "1") in scans_table.rows
    assert files_table.title == "Files (1 total)"
    assert ("active", "/tmp/demo.pdf", "pdf", "bbbbbbb") in files_table.rows
    assert docs_table.title == "Docs (1 total)"
    assert ("ccccccc", "pdf", "5", "Demo") in docs_table.rows


def test_config_and_rule_renderers_return_tables(monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "Table", _FakeTable)

    config_table = config._render_config(
        ConfigResponse(config={"parse.default_tier": "medium"}, sources={"parse.default_tier": "override"})
    )
    exclude_table = config._render_exclude_rules(
        ExcludeRuleListResponse(rules=[ExcludeRuleInfo(id=3, pattern="*.tmp", priority=10)])
    )
    parsing_table = config._render_parsing_rules(
        ParsingRuleListResponse(
            rules=[
                ParsingRuleInfo(id=4, pattern="*.pdf", tier="medium", page_range="1~2", remote=True, name="pdfs")
            ]
        )
    )

    assert config_table.title == "Config"
    assert ("parse.default_tier", "medium", "override") in config_table.rows
    assert exclude_table.title == "Exclude Rules"
    assert ("3", "*.tmp", "10") in exclude_table.rows
    assert parsing_table.title == "Parsing Rules"
    assert ("4", "*.pdf", "medium", "1~2", "yes", "pdfs") in parsing_table.rows


def test_watch_list_renderer_returns_table(monkeypatch: Any) -> None:
    monkeypatch.setattr(watch_cmd, "Table", _FakeTable)

    table = watch_cmd._render_watch_list(
        WatchListResponse(watches=[WatchInfo(id=2, path="/tmp/docs", label="Docs", removable=True, status="active")])
    )

    assert table.title == "Watches"
    assert ("2", "/tmp/docs", "active", "yes", "Docs") in table.rows


def test_show_detail_renderers_return_tables(monkeypatch: Any) -> None:
    monkeypatch.setattr(show, "Table", _FakeTable)
    parse_info = ParseInfo(
        id=9,
        sha256="a" * 64,
        short_id="aaaaaaa",
        tier="high",
        page_range="1",
        status="done",
        privacy="local",
        created_at=1,
        updated_at=2,
    )
    scan_info = ScanInfo(
        id=10,
        path="/tmp/docs",
        kind="manual",
        source="cli",
        status="done",
        files_seen=4,
        files_refreshed=3,
        files_unsupported=1,
        created_at=1,
        updated_at=2,
    )
    doc = DocInfo(
        sha256="d" * 64,
        short_id="ddddddd",
        size_bytes=7,
        file_type="pdf",
        title="Doc",
        page_count=6,
        first_seen_at=1,
        updated_at=2,
    )

    parse_table = show._render_parse_info(parse_info)
    scan_table = show._render_scan(scan_info)
    doc_table = show._render_doc_info(doc)

    assert parse_table.title == "Parse 9: done"
    assert ("Tier", "high") in parse_table.rows
    assert scan_table.title == "Scan 10: done"
    assert ("Seen", "4") in scan_table.rows
    assert doc_table.title == "Doc ddddddd"
    assert ("Title", "Doc") in doc_table.rows


def test_server_status_renders_separate_recent_log_panels(monkeypatch: Any) -> None:
    panels: list[dict[str, Any]] = []

    class _Panel:
        def __init__(self, renderable: Any, *, title: str, border_style: str) -> None:
            self.renderable = renderable
            self.title = title
            self.border_style = border_style
            panels.append({"title": title, "renderable": renderable, "border_style": border_style})

    monkeypatch.setattr(server, "Panel", _Panel)

    rendered = list(
        server._render_server_status(
            ServerStatusResponse(
                running=True,
                pid=123,
                uptime_seconds=1,
                socket_path="/tmp/doclib.sock",
                data_dir="/tmp/mineru",
                sqlite_path="/tmp/doclib.db",
                log_path="/tmp/doclib.log",
                app_logs=["app\n"],
                access_logs=["access\n"],
                stderr_logs=["stderr\n"],
                stdout_logs=["stdout\n"],
                parse_server_stderr_logs=["parse stderr\n"],
                parse_server_stdout_logs=["parse stdout\n"],
                recent_logs=["legacy\n"],
            )
        )
    )

    assert [panel["title"] for panel in panels] == [
        "Recent App Logs",
        "Recent Access Logs",
        "Recent Stderr Logs",
        "Recent Stdout Logs",
        "Recent Parse Server Stderr Logs",
        "Recent Parse Server Stdout Logs",
    ]
    assert [panel["renderable"] for panel in panels] == [
        "app",
        "access",
        "stderr",
        "stdout",
        "parse stderr",
        "parse stdout",
    ]
    assert all(panel["border_style"] == "dim" for panel in panels)
    assert not any(getattr(item, "title", None) == "Recent Logs" for item in rendered)


def test_telemetry_status_command_prints_installation_id(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def get_telemetry_status(self) -> TelemetryStatusResponse:
            return TelemetryStatusResponse(
                state="unset",
                installation_id="inst_test",
                pending_periods=1,
                pending_metrics=2,
                last_flush_at=None,
            )

    monkeypatch.setattr(telemetry, "DoclibClient", _Client)

    result = runner.invoke(app, ["telemetry", "status"])

    assert result.exit_code == 0
    assert "state: unset" in result.output
    assert "installation_id: inst_test" in result.output


def test_interactive_unset_prompt_enables_telemetry(monkeypatch: Any) -> None:
    actions: list[str] = []
    echoed: list[str] = []
    prompts: list[tuple[str, bool]] = []

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 5

        def get_telemetry_status(self) -> TelemetryStatusResponse:
            return TelemetryStatusResponse(state="unset", installation_id="inst_test")

        def telemetry_action(self, action: str) -> TelemetryActionResponse:
            actions.append(action)
            return TelemetryActionResponse(action=action, state="enabled", installation_id="inst_test")

    monkeypatch.setattr(cli_telemetry, "DoclibClient", _Client)
    monkeypatch.setattr(cli_telemetry, "_is_interactive", lambda: True)
    monkeypatch.setattr("typer.echo", lambda msg="", *args, **kwargs: echoed.append(str(msg)))
    monkeypatch.setattr("typer.confirm", lambda prompt, default=True, *args, **kwargs: prompts.append((prompt, default)) or True)

    cli_telemetry.maybe_prompt_telemetry_consent()

    assert actions == ["enable"]
    prompt_text = "\n".join(echoed)
    assert "Help improve MinerU by sending anonymous, locally aggregated usage and diagnostic data." in prompt_text
    assert "Collected:" in prompt_text
    assert "command names, MinerU version, OS, architecture, Python version" in prompt_text
    assert "coarse CPU/GPU categories" in prompt_text
    assert "NOT collected:" in prompt_text
    assert "document contents, extracted text/images, file names, file paths" in prompt_text
    assert "search queries, prompts, snippets, tracebacks, exception messages" in prompt_text
    assert "API keys, or exact CPU/GPU models" in prompt_text
    assert "Press Enter or type Y to enable, or type N to disable." in prompt_text
    assert "`mineru telemetry enable` or `mineru telemetry disable`" in prompt_text
    assert "`mineru telemetry preview`" in prompt_text
    assert prompts == [("Enable telemetry?", True)]


@pytest.mark.parametrize(
    ("args", "reason"),
    [
        (["watch", "--help"], "help"),
        (["parse", "--help"], "help"),
        (["search", "needle", "--json"], "json"),
    ],
)
def test_prepare_cli_telemetry_skips_prompt_for_non_interactive_command_modes(
    monkeypatch: Any, args: list[str], reason: str
) -> None:
    del reason
    calls: list[str] = []

    monkeypatch.setattr(cli_telemetry, "_is_interactive", lambda: True)
    monkeypatch.setattr(cli_telemetry, "maybe_prompt_telemetry_consent", lambda: calls.append("prompt"))

    result = runner.invoke(app, args)

    assert "Enable telemetry?" not in result.output
    assert calls == []


def test_prepare_cli_telemetry_skips_prompt_in_ci(monkeypatch: Any) -> None:
    calls: list[str] = []

    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(cli_telemetry, "_is_interactive", lambda: True)
    monkeypatch.setattr(cli_telemetry, "maybe_prompt_telemetry_consent", lambda: calls.append("prompt"))

    result = runner.invoke(app, ["search", "needle", "--limit", "1"])

    assert "Enable telemetry?" not in result.output
    assert calls == []


def test_prepare_cli_telemetry_skips_prompt_for_agent_caller(monkeypatch: Any) -> None:
    calls: list[str] = []

    monkeypatch.setattr(cli_telemetry, "_is_interactive", lambda: True)
    monkeypatch.setattr(cli_telemetry, "maybe_prompt_telemetry_consent", lambda: calls.append("prompt"))
    monkeypatch.setattr(cli_telemetry, "infer_default_client_context", lambda source: TelemetryContext(source=source, caller="agent"))

    result = runner.invoke(app, ["search", "needle", "--limit", "1"])

    assert "Enable telemetry?" not in result.output
    assert calls == []


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
        ["list", "parses", "--status", "pending", "--tier", "medium", "--limit", "10", "--offset", "5", "--json"],
    )

    assert result.exit_code == 0
    assert calls[-1] == (
        "list_parses",
        {"status": "pending", "tier": "medium", "limit": 10, "offset": 5},
    )


def test_parse_wait_ignores_failed_rows_outside_wait_ids(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    calls: list[dict[str, Any]] = []

    old_failed = ParseInfo(
        id=2,
        sha256="a" * 64,
        short_id="aaaaaaa",
        tier="high",
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
        short_id="aaaaaaa",
        tier="high",
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
                tier="high",
                page_range="1~10",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

        def list_parses(self, **kwargs: Any) -> ListParsesResponse:
            calls.append(kwargs)
            return ListParsesResponse(parses=[old_failed, new_done], total=2, limit=50, offset=0)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="aaaaaaa",
                tier="high",
                format="markdown",
                content="parsed",
                request_scope=ContentRequestScope(page_range="1~10", after=None, limit=30000),
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--remote", "--wait", "1", "--json"])

    assert result.exit_code == 0
    assert calls == [{"ids": [3]}]
    assert "old failure" not in result.output
    assert "Parse status:" not in result.output
    payload = json.loads(result.output)
    assert payload["parse"]["status"] == "done"
    assert payload["parse"]["wait_parse_ids"] == [3]
    assert payload["content"]["content"] == "parsed"


def test_parse_wait_json_maps_invalid_api_key_param(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    failed = ParseInfo(
        id=3,
        sha256="a" * 64,
        short_id="aaaaaaa",
        tier="high",
        page_range="1~1",
        status="failed",
        privacy="remote",
        created_at=3,
        updated_at=4,
        error_code="invalid_api_key",
        error_msg="Remote authentication failed: user authenticate failed",
    )

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 31

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="a" * 64,
                tier="high",
                page_range="1~1",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

        def list_parses(self, **kwargs: Any) -> ListParsesResponse:
            return ListParsesResponse(parses=[failed], total=1, limit=50, offset=0)

    class _GuidanceClient:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 3

        def get_config(self) -> ConfigResponse:
            return ConfigResponse(
                config={
                    "parse_server.remote.url": "https://mineru.net/api",
                    "parse_server.remote.api_key": "******",
                },
                sources={},
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)
    monkeypatch.setattr("mineru.cli.guidance.DoclibClient", _GuidanceClient)
    monkeypatch.setattr(parse.time, "sleep", lambda seconds: None)

    result = runner.invoke(app, ["parse", str(source), "--remote", "--wait", "1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "authentication_error"
    assert payload["error"]["code"] == "invalid_api_key"
    assert payload["error"]["param"] == "parse_server.remote.api_key"
    assert payload["guidance"]["type"] == "configure_official_api_key"
    assert payload["guidance"]["required"] is True


def test_parse_wait_json_handles_failed_row_without_error_details(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.docx"
    source.write_bytes(b"office")
    failed = ParseInfo(
        id=3,
        sha256="a" * 64,
        short_id="aaaaaaa",
        tier="flash",
        page_range="1",
        status="failed",
        privacy="remote",
        created_at=3,
        updated_at=4,
    )

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 31

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="a" * 64,
                tier="flash",
                page_range="1",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

        def list_parses(self, **kwargs: Any) -> ListParsesResponse:
            return ListParsesResponse(parses=[failed], total=1, limit=50, offset=0)

    monkeypatch.setattr(parse, "DoclibClient", _Client)
    monkeypatch.setattr(parse.time, "sleep", lambda seconds: None)

    result = runner.invoke(app, ["parse", str(source), "--remote", "--wait", "1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "engine_error"
    assert payload["error"]["code"] == "parse_failed"
    assert payload["error"]["message"] == "Parse failed."


def test_parse_verbose_json_writes_cache_hit_notice_to_stderr(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(sha256="a" * 64, tier="flash", page_range="1~1", status="done", cache_hit=True)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="aaaaaaa",
                tier="flash",
                format="markdown",
                content="parsed",
                request_scope=ContentRequestScope(page_range="1~1", limit=30000),
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--verbose", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["parse"]["cache_hit"] is True
    assert payload["content"]["content"] == "parsed"
    assert "Cache hit" not in result.stdout
    assert "Cache hit" in result.stderr


def test_parse_verbose_json_writes_queue_notice_to_stderr(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    done = ParseInfo(
        id=3,
        sha256="a" * 64,
        short_id="aaaaaaa",
        tier="flash",
        page_range="1~1",
        status="done",
        privacy="local",
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
                tier="flash",
                page_range="1~1",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

        def list_parses(self, **kwargs: Any) -> ListParsesResponse:
            return ListParsesResponse(parses=[done], total=1, limit=50, offset=0)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="aaaaaaa",
                tier="flash",
                format="markdown",
                content="parsed",
                request_scope=ContentRequestScope(page_range="1~1", limit=30000),
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)
    monkeypatch.setattr(parse.time, "sleep", lambda seconds: None)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--verbose", "--wait", "1", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["parse"]["status"] == "done"
    assert payload["content"]["content"] == "parsed"
    assert "Parse queued" not in result.stdout
    assert "Parse queued" in result.stderr


def test_parse_json_connection_error_is_machine_readable(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            raise ServerNotRunningError()

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "server_not_running"
    assert "Local mineru server is not running" in payload["error"]["message"]
    assert "Error:" not in result.stdout


def test_cleanup_json_connection_error_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            raise ServerNotRunningError()

    monkeypatch.setattr(cleanup_cmd, "DoclibClient", _Client)

    result = runner.invoke(app, ["cleanup", "temp", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "server_not_running"
    assert "Local mineru server is not running" in payload["error"]["message"]
    assert "Error:" not in result.stdout


def test_config_parsing_rules_list_json_error_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def list_parsing_rules(self) -> Any:
            raise RuntimeError("('rule_not_found', 'No parsing rules backend.', 'rule_id')")

    monkeypatch.setattr(config, "DoclibClient", _Client)

    result = runner.invoke(app, ["config", "parsing-rules", "list", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "rule_not_found"
    assert payload["error"]["param"] == "rule_id"
    assert "Error:" not in result.stdout


def test_config_exclude_rules_remove_non_json_prints_message_not_error_tuple(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def remove_exclude_rule(self, rule_id: int) -> None:
            assert rule_id == 123
            raise RuntimeError("('rule_not_found', 'Exclude rule missing.', 'rule_id')")

    monkeypatch.setattr(config, "DoclibClient", _Client)

    result = runner.invoke(app, ["config", "exclude-rules", "remove", "123"])

    assert result.exit_code == 1
    assert "Exclude rule missing." in result.output
    assert "('rule_not_found'" not in result.output


def test_config_parsing_rules_remove_non_json_prints_message_not_error_tuple(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def remove_parsing_rule(self, rule_id: int) -> None:
            assert rule_id == 456
            raise RuntimeError("('rule_not_found', 'Parsing rule missing.', 'rule_id')")

    monkeypatch.setattr(config, "DoclibClient", _Client)

    result = runner.invoke(app, ["config", "parsing-rules", "remove", "456"])

    assert result.exit_code == 1
    assert "Parsing rule missing." in result.output
    assert "('rule_not_found'" not in result.output


def test_parse_ingest_error_json_is_machine_readable(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.unsupported"
    source.write_bytes(b"unsupported")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> None:
            raise RuntimeError("('ingest_failed', 'File could not be ingested.', 'path')")

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "api_error"
    assert payload["error"]["code"] == "ingest_failed"
    assert payload["error"]["message"] == "File could not be ingested."
    assert payload["error"]["param"] == "path"
    assert "parse" not in payload


def test_parse_next_marker_quotes_windows_path() -> None:
    path = r"C:\Users\jinzhenj\Downloads\2606.20787v1.pdf"
    marker = parse._parse_next_marker(path, ContentNextRequest(page_range="7~10"))

    assert marker is not None
    command = marker.removeprefix("<!-- Next: ").removesuffix(" -->")
    assert shlex.split(command) == ["mineru", "parse", path, "--pages", "7~10"]


def test_parse_next_marker_escapes_double_quotes_in_path() -> None:
    path = r'C:\Users\jinzhenj\Downloads\bad"name.pdf'
    marker = parse._parse_next_marker(path, ContentNextRequest(page_range="7~10"))

    assert marker is not None
    command = marker.removeprefix("<!-- Next: ").removesuffix(" -->")
    assert shlex.split(command) == ["mineru", "parse", path, "--pages", "7~10"]


def test_parse_next_marker_preserves_explicit_parse_context(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            assert request.tier == "flash"
            assert request.remote is True
            return ParseResponse(sha256="a" * 64, tier="flash", page_range="1~10", status="done", cache_hit=True)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            assert kwargs["limit"] == 30000
            return DocContentResponse(
                sha256="a" * 64,
                short_id="aaaaaaa",
                tier="flash",
                format="markdown",
                content="parsed",
                request_scope=ContentRequestScope(page_range="1~10", limit=30000),
                next_request=ContentNextRequest(page_range="11~20"),
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--remote", "--limit", "30000"])

    assert result.exit_code == 0
    marker = result.output.strip().splitlines()[-1]
    command = marker.removeprefix("<!-- Next: ").removesuffix(" -->")
    assert shlex.split(command) == [
        "mineru",
        "parse",
        str(source.resolve()),
        "--tier",
        "flash",
        "--remote",
        "--limit",
        "30000",
        "--pages",
        "11~20",
    ]


def test_parse_next_marker_omits_default_parse_context(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            assert request.tier is None
            assert request.remote is False
            return ParseResponse(sha256="a" * 64, tier="medium", page_range="1~10", status="done", cache_hit=True)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            assert kwargs["limit"] == 30000
            return DocContentResponse(
                sha256="a" * 64,
                short_id="aaaaaaa",
                tier="medium",
                format="markdown",
                content="parsed",
                request_scope=ContentRequestScope(page_range="1~10", limit=30000),
                next_request=ContentNextRequest(page_range="11~20"),
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source)])

    assert result.exit_code == 0
    marker = result.output.strip().splitlines()[-1]
    command = marker.removeprefix("<!-- Next: ").removesuffix(" -->")
    assert shlex.split(command) == ["mineru", "parse", str(source.resolve()), "--pages", "11~20"]


def test_parse_json_no_wait_wraps_parse_and_null_content(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="a" * 64,
                tier="flash",
                page_range="1~1",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--no-wait", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["parse"]["status"] == "pending"
    assert payload["parse"]["wait_parse_ids"] == [3]
    assert payload["content"] is None


def test_parse_no_wait_text_prints_parse_id_tracking_hint(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="a" * 64,
                tier="high",
                page_range="1~1",
                status="pending",
                wait_parse_ids=[4686],
                created_parse_ids=[4686],
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "high", "--remote", "--no-wait"])

    assert result.exit_code == 0
    assert result.output == (
        "Parse still in progress (tier=high).\n"
        "Parse ID: 4686\n"
        "Check status: mineru show parse 4686\n"
    )


def test_parse_no_wait_text_prints_multiple_parse_id_tracking_hints(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="a" * 64,
                tier="high",
                page_range="1~2",
                status="pending",
                wait_parse_ids=[4686, 4687],
                created_parse_ids=[4687],
                reused_parse_ids=[4686],
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "high", "--remote", "--no-wait"])

    assert result.exit_code == 0
    assert result.output == (
        "Parse still in progress (tier=high).\n"
        "Parse IDs: 4686, 4687\n"
        "Check status:\n"
        "  mineru show parse 4686\n"
        "  mineru show parse 4687\n"
    )


def test_parse_expands_user_home_in_input_path(monkeypatch: Any, tmp_path: Path) -> None:
    home = tmp_path / "home"
    source = home / "mineru-e2e-test" / "sample.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7\n")
    seen_paths: list[str] = []

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            seen_paths.append(request.path)
            return ParseResponse(
                sha256="a" * 64,
                tier="flash",
                page_range="1~1",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", "~/mineru-e2e-test/sample.pdf", "--tier", "flash", "--no-wait", "--json"])

    assert result.exit_code == 0
    assert seen_paths == [str(source.resolve())]


def test_watch_add_normalizes_user_path_before_request(monkeypatch: Any, tmp_path: Path) -> None:
    home = tmp_path / "home"
    watched = home / "Documents"
    watched.mkdir(parents=True)
    seen_paths: list[str] = []

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def add_watch(self, request: Any) -> WatchInfo:
            seen_paths.append(request.path)
            return WatchInfo(id=1, path=request.path, status="active")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(watch_cmd, "DoclibClient", _Client)

    result = runner.invoke(app, ["watch", "add", "~/Documents/../Documents", "--json"])

    assert result.exit_code == 0
    assert seen_paths == [os.path.normpath(os.path.abspath(os.path.expanduser("~/Documents/../Documents")))]


def test_watch_remove_missing_id_preserves_watch_not_found_json_error(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def list_watches(self) -> Any:
            return type("WatchList", (), {"watches": []})()

    monkeypatch.setattr(watch_cmd, "DoclibClient", _Client)

    result = runner.invoke(app, ["watch", "remove", "123", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "watch_not_found"
    assert payload["error"]["param"] == "watch_id"
    assert "Error:" not in result.output


def test_parse_json_output_writes_file_and_returns_output_object(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    output = tmp_path / "nested" / "out.md"

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(sha256="a" * 64, tier="flash", page_range="1~1", status="done", cache_hit=True)

        def export_doc_content(self, doc_ref: str, request: Any) -> Any:
            Path(request.output).write_text("exported", encoding="utf-8")
            return type("Exported", (), {"output": request.output})()

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--output", str(output), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["parse"]["status"] == "done"
    assert payload["content"] is None
    assert payload["output"] == {"status": "written", "path": str(output.resolve())}
    assert output.read_text(encoding="utf-8") == "exported"
    assert "Written to" not in result.output


def test_parse_wait_status_line_is_verbose_only(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    done = ParseInfo(
        id=3,
        sha256="a" * 64,
        short_id="aaaaaaa",
        tier="flash",
        page_range="1~1",
        status="done",
        privacy="local",
        via="local",
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
                tier="flash",
                page_range="1~1",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

        def list_parses(self, **kwargs: Any) -> ListParsesResponse:
            return ListParsesResponse(parses=[done], total=1, limit=50, offset=0)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="aaaaaaa",
                tier="flash",
                format="markdown",
                content="parsed",
                request_scope=ContentRequestScope(locator="doc:aaaaaaa/tier:flash", context=0, limit=30000),
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)
    monkeypatch.setattr(parse.time, "sleep", lambda seconds: None)

    default_result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--wait", "1"])
    verbose_result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--wait", "1", "--verbose"])

    assert default_result.exit_code == 0
    assert "Parse status:" not in default_result.output
    assert "parsed" in default_result.output
    assert verbose_result.exit_code == 0
    assert "Parse status: done" in verbose_result.output
    assert "parsed" in verbose_result.output


def test_parse_wait_timeout_json_returns_timeout_error(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    pending = ParseInfo(
        id=3,
        sha256="a" * 64,
        short_id="aaaaaaa",
        tier="flash",
        page_range="1~1",
        status="pending",
        privacy="local",
        via="local",
        created_at=3,
        updated_at=4,
    )

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 31

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="a" * 64,
                tier="flash",
                page_range="1~1",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

        def list_parses(self, **kwargs: Any) -> ListParsesResponse:
            return ListParsesResponse(parses=[pending], total=1, limit=50, offset=0)

        def record_observations(self, request: Any) -> Any:
            return None

    times = iter([0.0, 0.0, 0.2, 2.0, 2.0])
    monkeypatch.setattr(parse, "DoclibClient", _Client)
    monkeypatch.setattr(parse.time, "time", lambda: next(times, 2.0))

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--wait", "1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "timeout_error"
    assert payload["error"]["code"] == "parse_wait_timeout"
    assert payload["error"]["param"] == "wait"
    assert payload["parse"]["status"] == "pending"
    assert payload["parse"]["wait_parse_ids"] == [3]
    assert payload["content"] is None


def test_parse_wait_timeout_text_returns_nonzero_exit(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 31

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="a" * 64,
                tier="flash",
                page_range="1~1",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

        def record_observations(self, request: Any) -> Any:
            return None

    times = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(parse, "DoclibClient", _Client)
    monkeypatch.setattr(parse.time, "time", lambda: next(times, 2.0))

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--wait", "1"])

    assert result.exit_code == 1
    assert result.output == (
        "Parse still in progress after 1s (tier=flash).\n"
        "Parse ID: 3\n"
        "Check status: mineru show parse 3\n"
    )


def test_parse_wait_timeout_json_uses_latest_parsing_status(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    parsing = ParseInfo(
        id=3,
        sha256="a" * 64,
        short_id="aaaaaaa",
        tier="flash",
        page_range="1~1",
        status="parsing",
        privacy="local",
        via="local",
        created_at=3,
        updated_at=4,
    )

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 31

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="a" * 64,
                tier="flash",
                page_range="1~1",
                status="pending",
                wait_parse_ids=[3],
                created_parse_ids=[3],
            )

        def list_parses(self, **kwargs: Any) -> ListParsesResponse:
            return ListParsesResponse(parses=[parsing], total=1, limit=50, offset=0)

        def record_observations(self, request: Any) -> Any:
            return None

    times = iter([0.0, 0.0, 0.2, 2.0, 2.0])
    monkeypatch.setattr(parse, "DoclibClient", _Client)
    monkeypatch.setattr(parse.time, "time", lambda: next(times, 2.0))

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--wait", "1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "parse_wait_timeout"
    assert payload["parse"]["status"] == "parsing"


def test_parse_json_error_output_is_machine_readable(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 31

        def ensure_parse(self, request: Any) -> ParseResponse:
            raise RuntimeError(
                "('quality_tier_unavailable', 'No medium, high, or xhigh engine available. Use --tier flash.', None)"
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--wait", "1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "quality_tier_unavailable"
    assert "No medium, high, or xhigh engine available" in payload["error"]["message"]


def test_parse_invalid_after_cursor_json_error_is_validation_error(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 31

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(sha256="a" * 64, tier="medium", page_range="1", status="done")

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            raise RuntimeError(
                "('invalid_locator', 'Invalid doclib content cursor: invalid-cursor-for-param-test', 'after')"
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(
        app,
        ["parse", str(source), "--after", "invalid-cursor-for-param-test", "--wait", "1", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "invalid_locator"
    assert payload["error"]["param"] == "after"
    assert "Invalid doclib content cursor" in payload["error"]["message"]


def test_parse_missing_file_json_error_is_machine_readable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"

    result = runner.invoke(app, ["parse", str(missing), "--tier", "flash", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "file_not_found"
    assert payload["error"]["param"] == "path"
    assert "Error:" not in result.output


def test_parse_rejects_reverse_page_range_as_json_error(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            raise AssertionError("DoclibClient should not be constructed for invalid page ranges")

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--pages", "2~1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "page_range_invalid"
    assert payload["error"]["param"] == "pages"
    assert "Error:" not in result.output


def test_scan_missing_path_json_error_is_machine_readable(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    result = runner.invoke(app, ["scan", str(missing), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "file_not_found"
    assert payload["error"]["param"] == "path"
    assert "Error:" not in result.output


def test_scan_wait_failed_status_exits_nonzero_after_json_output(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    failed_scan = ScanInfo(
        id=7,
        path=str(source.resolve()),
        kind="manual",
        source="cli",
        status="failed",
        files_seen=1,
        files_error=1,
        error_code="scan_failed",
        error_msg="scan boom",
        created_at=1,
        updated_at=2,
    )

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def create_scan(self, request: Any) -> ScanInfo:
            assert request.path == str(source.resolve())
            return failed_scan.model_copy(update={"status": "pending"})

        def get_scan(self, scan_id: int) -> ScanInfo:
            assert scan_id == 7
            return failed_scan

    monkeypatch.setattr(scan_mod, "DoclibClient", _Client)

    result = runner.invoke(app, ["scan", str(source), "--wait", "1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["error_code"] == "scan_failed"


def test_watch_rescan_wait_failed_status_exits_nonzero_after_json_output(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "watched"
    source.mkdir()
    watch = WatchInfo(id=3, path=str(source.resolve()), status="active")
    failed_scan = ScanInfo(
        id=9,
        path=str(source.resolve()),
        kind="watch",
        source="cli",
        watch_id=3,
        status="failed",
        files_seen=1,
        files_error=1,
        error_code="scan_failed",
        error_msg="watch scan boom",
        created_at=1,
        updated_at=2,
    )

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def list_watches(self) -> WatchListResponse:
            return WatchListResponse(watches=[watch])

        def create_scan(self, request: Any) -> ScanInfo:
            assert request.watch_id == 3
            return failed_scan.model_copy(update={"status": "pending"})

        def get_scan(self, scan_id: int) -> ScanInfo:
            assert scan_id == 9
            return failed_scan

    monkeypatch.setattr(watch_cmd, "DoclibClient", _Client)

    result = runner.invoke(app, ["watch", "rescan", "3", "--wait", "1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["error_code"] == "scan_failed"


def test_show_scan_failed_status_remains_successful_query(monkeypatch: Any, tmp_path: Path) -> None:
    failed_scan = ScanInfo(
        id=9,
        path=str(tmp_path / "watched"),
        kind="watch",
        source="cli",
        status="failed",
        files_seen=1,
        files_error=1,
        error_code="scan_failed",
        error_msg="watch scan boom",
        created_at=1,
        updated_at=2,
    )

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def get_scan(self, scan_id: int) -> ScanInfo:
            assert scan_id == 9
            return failed_scan

    monkeypatch.setattr(show, "DoclibClient", _Client)

    result = runner.invoke(app, ["show", "scan", "9", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "failed"


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


def test_show_file_normalizes_user_path_before_request(monkeypatch: Any, tmp_path: Path) -> None:
    home = tmp_path / "home"
    source = home / "docs" / "demo.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7\n")
    calls: list[str] = []

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

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(show, "DoclibClient", _Client)

    result = runner.invoke(app, ["show", "file", "~/docs/../docs/demo.pdf", "--json"])

    assert result.exit_code == 0
    assert calls == [os.path.normpath(os.path.abspath(os.path.expanduser("~/docs/../docs/demo.pdf")))]


def test_show_file_renderer_uses_file_info_table(monkeypatch: Any) -> None:
    tables: list[Any] = []

    class _Table:
        def __init__(self, *, title: str) -> None:
            self.title = title
            self.rows: list[tuple[str, str]] = []
            tables.append(self)

        def add_column(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_row(self, key: str, value: str) -> None:
            self.rows.append((key, value))

    monkeypatch.setattr(show, "Table", _Table)

    rendered = show._render_file_info(
        FileInfoResponse(
            file=FileInfo(
                filename="demo.pdf",
                path="/tmp/demo.pdf",
                ext="pdf",
                size_bytes=2048,
                mtime_ms=1,
                sha256="a" * 64,
                short_id="abcdef1",
                status="active",
                first_seen_at=1,
                updated_at=1,
            ),
            doc=DocInfo(
                sha256="a" * 64,
                short_id="abcdef1",
                size_bytes=2048,
                title="Demo",
                author="Alice",
                page_count=3,
                first_seen_at=1,
                updated_at=1,
            ),
        )
    )

    assert rendered == tables[0]
    assert tables[0].title == "File Info: demo.pdf"
    assert ("Path", "/tmp/demo.pdf") in tables[0].rows
    assert ("Size", "2 KB") in tables[0].rows
    assert ("Doc ID", "abcdef1") in tables[0].rows
    assert ("Title", "Demo") in tables[0].rows
    assert ("Author", "Alice") in tables[0].rows


def test_invalidate_normalizes_user_path_before_request(monkeypatch: Any, tmp_path: Path) -> None:
    home = tmp_path / "home"
    source = home / "docs" / "demo.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7\n")
    calls: list[str] = []

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 10

        def invalidate(self, request: Any) -> InvalidateResponse:
            calls.append(request.path)
            return InvalidateResponse(
                target="parses",
                sha256="a" * 64,
                short_id="aaaaaaa",
                tier=request.tier,
                invalidated_count=1,
            )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(invalidate, "DoclibClient", _Client)

    result = runner.invoke(app, ["invalidate", "~/docs/../docs/demo.pdf", "--tier", "medium"])

    assert result.exit_code == 0
    assert calls == [os.path.normpath(os.path.abspath(os.path.expanduser("~/docs/../docs/demo.pdf")))]


def test_invalidate_non_json_prints_message_not_error_tuple(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 10

        def invalidate(self, request: Any) -> None:
            assert request.path == str(source.resolve())
            raise RuntimeError("('not_cached', 'Requested parsed content is not cached.', 'path')")

    monkeypatch.setattr(invalidate, "DoclibClient", _Client)

    result = runner.invoke(app, ["invalidate", str(source)])

    assert result.exit_code == 1
    assert "Requested parsed content is not cached." in result.output
    assert "('not_cached'" not in result.output


def test_show_doc_expands_files(monkeypatch: Any) -> None:
    calls: list[tuple[str, bool]] = []

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def get_doc(self, doc_ref: str, *, expand_files: bool = False) -> DocInfo:
            calls.append((doc_ref, expand_files))
            return DocInfo(sha256=doc_ref, short_id=doc_ref[:7], size_bytes=7, first_seen_at=1, updated_at=1, files=[])

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


def test_server_start_failure_points_to_log_and_does_not_discard_child_stderr(monkeypatch: Any, tmp_path: Path) -> None:
    popen_calls: list[dict[str, Any]] = []
    log_path = tmp_path / "doclib.log"
    stdout_log_path = tmp_path / "doclib.stdout.log"
    stderr_log_path = tmp_path / "doclib.stderr.log"

    class _Proc:
        pid = 12345

        def kill(self) -> None:
            popen_calls.append({"kill": True})

    def _popen(*args: Any, **kwargs: Any) -> _Proc:
        popen_calls.append({"args": args, "kwargs": kwargs})
        assert kwargs["stdout"] is not server.subprocess.DEVNULL
        assert kwargs["stderr"] is not server.subprocess.DEVNULL
        assert kwargs["stdout"] is not kwargs["stderr"]
        kwargs["stdout"].write("child stdout\n")
        kwargs["stdout"].flush()
        kwargs["stderr"].write("child failed\n")
        kwargs["stderr"].flush()
        return _Proc()

    monkeypatch.setattr(server, "_server_running", lambda: False)
    monkeypatch.setattr(server, "_wait_for_server", lambda: False)
    monkeypatch.setattr(server, "_socket_path", lambda: str(tmp_path / "doclib.sock"))
    monkeypatch.setattr(server, "_endpoint_path", lambda: str(tmp_path / "doclib.endpoint.json"))
    monkeypatch.setattr(server, "_server_log_path", lambda: str(log_path))
    monkeypatch.setattr(server, "_server_stdout_log_path", lambda: str(stdout_log_path))
    monkeypatch.setattr(server, "_server_stderr_log_path", lambda: str(stderr_log_path))
    monkeypatch.setattr(server.subprocess, "Popen", _popen)

    result = runner.invoke(app, ["server", "start"])

    assert result.exit_code == 1
    assert "See log:" in result.output
    assert "doclib.log" in result.output
    assert "doclib.stdout.log" in result.output
    assert "doclib.stderr.log" in result.output
    assert "child stdout\n" not in log_path.read_text(encoding="utf-8")
    assert "child failed\n" not in log_path.read_text(encoding="utf-8")
    assert "child stdout\n" in stdout_log_path.read_text(encoding="utf-8")
    assert "child failed\n" in stderr_log_path.read_text(encoding="utf-8")
    assert popen_calls[-1] == {"kill": True}


def test_server_start_lock_blocks_concurrent_start(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(server, "_server_running", lambda: False)
    lock_path = tmp_path / "doclib.start.lock"

    with server._ServerStartLock(str(lock_path), timeout=0.1, stale_after=60.0) as lock:
        assert lock.acquired is True
        assert lock_path.is_file()
        with pytest.raises(RuntimeError, match="already in progress"):
            with server._ServerStartLock(str(lock_path), timeout=0.1, stale_after=60.0):
                pass

    assert not lock_path.exists()


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
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "api_error"
    assert "content missing" in payload["error"]["message"]
    assert "Error:" not in result.output


def test_parse_content_read_failure_non_json_prints_message_not_error_tuple(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.txt"
    source.write_text("hello", encoding="utf-8")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            assert request.path == str(source.resolve())
            return ParseResponse(sha256="a" * 64, tier="flash", page_range="all", status="done", cache_hit=True)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("('not_cached', 'Requested parsed content is not cached.', 'page_range')")

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash"])

    assert result.exit_code == 1
    assert "Requested parsed content is not cached." in result.output
    assert "('not_cached'" not in result.output


def test_parse_empty_cached_content_json_reports_error_envelope(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            assert request.path == str(source.resolve())
            return ParseResponse(sha256="a" * 64, tier="flash", page_range="1", status="done", cache_hit=True)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="flash",
                format="markdown",
                content="",
                request_scope=ContentRequestScope(page_range="1", limit=30000),
                content_ranges=[],
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "parse_empty"
    assert payload["error"]["message"] == "No content returned from parse."
    assert "Error:" not in result.output


def test_parse_empty_cached_content_non_json_reports_no_renderable_content(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            assert request.path == str(source.resolve())
            return ParseResponse(sha256="a" * 64, tier="flash", page_range="1", status="done", cache_hit=True)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="flash",
                format="markdown",
                content="",
                request_scope=ContentRequestScope(page_range="1", limit=30000),
                content_ranges=[
                    ContentRange(page_range="1", start="doc:ab12cd3/tier:flash/page:1", end="doc:ab12cd3/tier:flash/page:1")
                ],
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--tier", "flash"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "No renderable content in requested pages." in result.stderr


def test_read_passes_locator_parameters_to_doclib_client(monkeypatch: Any) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            calls.append((locator, kwargs))
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="medium",
                format="markdown",
                content="hello",
                request_scope=ContentRequestScope(locator=locator, context=kwargs["context"], limit=kwargs["limit"]),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(
        app,
        ["read", "doc:ab12cd3/tier:medium/page:4", "--context", "2", "--limit", "123", "--format", "markdown"],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            "doc:ab12cd3/tier:medium/page:4",
            {"context": 2, "limit": 123, "format": "markdown", "no_marker": False},
        )
    ]
    assert "hello" in result.output


def test_read_json_output_preserves_locator_next_request(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="medium",
                format="markdown",
                content="hello",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                next_request=ContentNextRequest(locator="doc:ab12cd3/tier:medium/page:5"),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:medium/page:4", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["next_request"]["locator"] == "doc:ab12cd3/tier:medium/page:5"


def test_read_markdown_output_separates_next_marker_with_blank_line(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="medium",
                format="markdown",
                content="hello",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                next_request=ContentNextRequest(locator="doc:ab12cd3/tier:medium/page:5"),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:medium/page:4"])

    assert result.exit_code == 0
    assert "hello\n\n<!-- Next: mineru read doc:ab12cd3/tier:medium/page:5 -->" in result.output


def test_read_json_error_output_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            raise RuntimeError("('not_cached', 'Requested parsed content is not cached.', 'locator')")

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:medium/page:4", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "not_cached"
    assert "Error:" not in result.output


def test_read_json_output_writes_markdown_file_and_returns_output_object(monkeypatch: Any, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "read.md"

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="medium",
                format="markdown",
                content="hello",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:medium/page:4", "--output", str(output), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["content"] is None
    assert payload["output"] == {"status": "written", "path": str(output.resolve())}
    assert output.read_text(encoding="utf-8") == "hello"
    assert "Written to" not in result.output


def test_read_json_output_writes_image_file_and_returns_output_object(monkeypatch: Any, tmp_path: Path) -> None:
    asset = tmp_path / "server.png"
    asset.write_bytes(b"png-bytes")
    output = tmp_path / "nested" / "local.png"

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            assert kwargs["image_format"] == "png"
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="medium",
                format="image",
                content="",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                asset=ContentAsset(path=str(asset), mime_type="image/png"),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:medium/page:4", "--format", "image", "--output", str(output), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["content"] is None
    assert payload["output"] == {"status": "written", "path": str(output.resolve())}
    assert output.read_bytes() == b"png-bytes"
    assert "Written to" not in result.output


@pytest.mark.parametrize(
    ("suffix", "expected_image_format"),
    [
        (".png", "png"),
        (".jpg", "jpeg"),
        (".jpeg", "jpeg"),
        (".webp", "webp"),
    ],
)
def test_read_image_output_extension_selects_server_image_format(
    monkeypatch: Any, tmp_path: Path, suffix: str, expected_image_format: str
) -> None:
    asset = tmp_path / f"server{suffix}"
    asset.write_bytes(b"image-bytes")
    output = tmp_path / "nested" / f"local{suffix}"

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            assert kwargs["image_format"] == expected_image_format
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="medium",
                format="image",
                content="",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                asset=ContentAsset(path=str(asset), mime_type=f"image/{expected_image_format}"),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:medium/page:4", "--format", "image", "--output", str(output)])

    assert result.exit_code == 0
    assert output.read_bytes() == b"image-bytes"


@pytest.mark.parametrize("output_arg", ["-", "page", "page.gif", "page.md"])
def test_read_image_output_rejects_unsupported_output_extension_before_client_call(
    monkeypatch: Any, tmp_path: Path, output_arg: str
) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            raise AssertionError("DoclibClient should not be called for invalid image output paths")

    monkeypatch.setattr(read, "DoclibClient", _Client)
    output = output_arg if output_arg == "-" else str(tmp_path / output_arg)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:medium/page:4", "--format", "image", "--output", output, "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "image_output_extension_unsupported"
    assert payload["error"]["param"] == "output"
    if output != "-":
        assert not Path(output).exists()


def test_read_json_image_output_missing_asset_reports_error_envelope(monkeypatch: Any, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "local.png"

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            assert kwargs["image_format"] == "png"
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="medium",
                format="image",
                content="",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                asset=None,
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:medium/page:4", "--format", "image", "--output", str(output), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "asset_not_available"
    assert payload["error"]["message"] == "No image asset returned."
    assert not output.exists()
    assert "Error:" not in result.output


def test_read_image_output_copies_server_asset_locally(monkeypatch: Any, tmp_path: Path) -> None:
    asset = tmp_path / "server.png"
    asset.write_bytes(b"png-bytes")
    output = tmp_path / "local.png"

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            assert kwargs["image_format"] == "png"
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="medium",
                format="image",
                content="",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                asset=ContentAsset(path=str(asset), mime_type="image/png"),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:medium/page:4", "--format", "image", "--output", str(output)])

    assert result.exit_code == 0
    assert output.read_bytes() == b"png-bytes"
    assert "Written to" in result.output
    assert str(output) in result.output.replace("\n", "")


def test_search_json_error_output_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 10

        def search(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("('quality_tier_unavailable', 'No medium, high, or xhigh engine available.', 'tier')")

    monkeypatch.setattr(search_mod, "DoclibClient", _Client)

    result = runner.invoke(app, ["search", "needle", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "quality_tier_unavailable"
    assert "Error:" not in result.output


def test_watch_add_json_error_output_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def add_watch(self, request: Any) -> Any:
            raise RuntimeError("('invalid_request', 'Watch path does not exist.', 'path')")

    monkeypatch.setattr(watch_cmd, "DoclibClient", _Client)

    result = runner.invoke(app, ["watch", "add", "/not/exist", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["param"] == "path"
    assert "Error:" not in result.output


def test_config_get_json_error_output_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def get_config_key(self, key: str) -> Any:
            raise RuntimeError("('invalid_request', 'Unknown config key.', 'key')")

    monkeypatch.setattr(config, "DoclibClient", _Client)

    result = runner.invoke(app, ["config", "get", "not.a.real.key", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["param"] == "key"
    assert "Error:" not in result.output


def test_cleanup_temp_json_error_output_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 30

        def cleanup_temp_files(self, request: Any) -> Any:
            raise RuntimeError("('invalid_request', 'older_than must be >= 0', 'older_than')")

    monkeypatch.setattr(cleanup_cmd, "DoclibClient", _Client)

    result = runner.invoke(app, ["cleanup", "temp", "--older-than", "-1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["param"] == "older_than"
    assert "Error:" not in result.output


def test_server_status_json_not_running_returns_state_json(monkeypatch: Any) -> None:
    monkeypatch.setattr(server, "_server_running", lambda: False)
    monkeypatch.setattr(server, "_socket_path", lambda: "/tmp/doclib.sock")
    monkeypatch.setattr(server.config.doclib, "data_dir", "~/.mineru")
    monkeypatch.setattr(server.config.doclib.sqlite, "path", "~/.mineru/doclib.db")
    monkeypatch.setattr(server.config.doclib.log, "app_path", "~/.mineru/logs/doclib.log")
    monkeypatch.setattr(server.config.doclib.log, "access_path", "~/.mineru/logs/doclib.access.log")
    monkeypatch.setattr(server.config.doclib.log, "stdout_path", "~/.mineru/logs/doclib.stdout.log")
    monkeypatch.setattr(server.config.doclib.log, "stderr_path", "~/.mineru/logs/doclib.stderr.log")

    result = runner.invoke(app, ["server", "status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["running"] is False
    assert payload["socket_path"] == "/tmp/doclib.sock"
    assert payload["sqlite_path"] == os.path.expanduser("~/.mineru/doclib.db")
    assert payload["log_path"] == os.path.expanduser("~/.mineru/logs/doclib.log")
    assert payload["access_log_path"] == os.path.expanduser("~/.mineru/logs/doclib.access.log")
    assert payload["stdout_log_path"] == os.path.expanduser("~/.mineru/logs/doclib.stdout.log")
    assert payload["stderr_log_path"] == os.path.expanduser("~/.mineru/logs/doclib.stderr.log")
    assert payload["tcp"] == {"enabled": False, "host": None, "port": None}
    assert "Server is not running." not in result.output
