from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mineru.cli.commands import config, invalidate, list_resources, parse, read, server, show, telemetry
from mineru.cli.commands import cleanup as cleanup_cmd
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
    ConfigSetResponse,
    DocInfo,
    DocContentResponse,
    FileInfo,
    FileInfoResponse,
    InvalidateResponse,
    ListParsesResponse,
    ParseInfo,
    ParseResponse,
    TelemetryActionResponse,
    TelemetryStatusResponse,
    WatchInfo,
)
from mineru.version import __version__


runner = CliRunner()


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


def test_version_command_is_last_in_root_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert result.output.rfind("version") > result.output.rfind("cleanup")


def test_root_help_hides_typer_completion_options() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
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
    for token in ("pdf", "image", "docx", "pptx", "xlsx", "html", "markdown", "csv", "rst", "tex", "txt"):
        assert token in search_result.output
    assert "File extension filter:" in find_result.output
    for token in (
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "jp2",
        "webp",
        "gif",
        "bmp",
        "tiff",
        "docx",
        "pptx",
        "xlsx",
        "html",
        "htm",
        "md",
        "markdown",
        "csv",
        "rst",
        "tex",
        "txt",
    ):
        assert token in find_result.output


def test_print_error_uses_single_rich_render(monkeypatch: Any) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _Console:
        def print(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

    monkeypatch.setattr(output_mod, "stderr_console", _Console())

    output_mod.print_error(
        "No standard or pro engine available. You can start a local parse-server, use --remote, or explicitly pass "
        "--tier flash for text-only preview."
    )

    assert len(calls) == 1
    rendered = str(calls[0][0][0])
    assert rendered.startswith("Error: ")
    assert "--tier flash" in rendered


def test_find_results_render_as_list_without_empty_search_columns(monkeypatch: Any) -> None:
    printed: list[Any] = []

    class _Console:
        def print(self, item: Any, *args: Any, **kwargs: Any) -> None:
            printed.append(item)

    monkeypatch.setattr(output_mod, "console", _Console())

    output_mod.format_find_results(
        {
            "total": 2,
            "results": [
                {"filename": "resume.pdf", "paths": ["/tmp/resume.pdf"]},
                {"filename": "cv.docx", "paths": ["/tmp/cv.docx"]},
            ],
        }
    )

    assert len(printed) == 1
    rendered = str(printed[0])
    assert "Search results (2 total)" in rendered
    assert "1. resume.pdf (/tmp/resume.pdf)" in rendered
    assert "\n\n2. cv.docx (/tmp/cv.docx)" in rendered
    assert "Tier" not in rendered
    assert "Snippet" not in rendered
    assert "Paths" not in rendered


def test_search_results_render_as_list_without_table(monkeypatch: Any) -> None:
    printed: list[Any] = []

    class _Console:
        def print(self, item: Any, *args: Any, **kwargs: Any) -> None:
            printed.append(item)

    monkeypatch.setattr(output_mod, "console", _Console())

    output_mod.format_search_results(
        {
            "total": 1,
            "results": [
                {
                    "filename": "resume.pdf",
                    "tier": "standard",
                    "snippet": "Python\nengineer",
                    "paths": ["/tmp/resume.pdf"],
                }
            ],
        }
    )

    assert len(printed) == 1
    rendered = str(printed[0])
    assert "Search results (1 total)" in rendered
    assert "1. resume.pdf (/tmp/resume.pdf) Tier: standard" in rendered
    assert "   Python  engineer" in rendered
    assert "Snippet:" not in rendered
    assert not hasattr(printed[0], "columns")


def test_search_result_snippet_preserves_fts_match_after_long_prefix(monkeypatch: Any) -> None:
    printed: list[Any] = []

    class _Console:
        def print(self, item: Any, *args: Any, **kwargs: Any) -> None:
            printed.append(item)

    monkeypatch.setattr(output_mod, "console", _Console())

    output_mod.format_search_results(
        {
            "total": 1,
            "results": [
                {
                    "filename": "deepseek.pdf",
                    "tier": "flash",
                    "snippet": (
                        "...on each node. Under this constraint, our MoE training framework can nearly "
                        "achieve full computation-communication overlap. <mark>Token</mark> <mark>Dropping</mark>"
                    ),
                    "paths": ["/tmp/deepseek.pdf"],
                }
            ],
        }
    )

    rendered = str(printed[0])
    assert "<mark>Token</mark>" in rendered
    assert "<mark>Dropping</mark>" in rendered


def test_server_status_renders_separate_recent_log_panels(monkeypatch: Any) -> None:
    printed: list[Any] = []
    panels: list[dict[str, Any]] = []

    class _Console:
        def print(self, item: Any, *args: Any, **kwargs: Any) -> None:
            printed.append(item)

    class _Panel:
        def __init__(self, renderable: Any, *, title: str, border_style: str) -> None:
            self.renderable = renderable
            self.title = title
            self.border_style = border_style
            panels.append({"title": title, "renderable": renderable, "border_style": border_style})

    monkeypatch.setattr(output_mod, "console", _Console())
    monkeypatch.setattr(output_mod, "Panel", _Panel)

    output_mod.format_server_status(
        {
            "running": True,
            "pid": 123,
            "uptime_seconds": 1,
            "tcp": {"enabled": False},
            "workers": {},
            "app_logs": ["app\n"],
            "access_logs": ["access\n"],
            "stderr_logs": ["stderr\n"],
            "stdout_logs": ["stdout\n"],
            "recent_logs": ["legacy\n"],
        }
    )

    assert [panel["title"] for panel in panels] == [
        "Recent App Logs",
        "Recent Access Logs",
        "Recent Stderr Logs",
        "Recent Stdout Logs",
    ]
    assert [panel["renderable"] for panel in panels] == ["app", "access", "stderr", "stdout"]
    assert all(panel["border_style"] == "dim" for panel in panels)
    assert not any(getattr(item, "title", None) == "Recent Logs" for item in printed)


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
    assert "Collected: command names, MinerU version, OS, architecture, Python version" in prompt_text
    assert "coarse CPU/GPU categories" in prompt_text
    assert "Not collected: document contents, extracted text/images, file names, file paths" in prompt_text
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

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="aaaaaaa",
                tier="pro",
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
            raise RuntimeError("cannot connect")

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "api_error"
    assert "Cannot connect to mineru server" in payload["error"]["message"]
    assert "Error:" not in result.stdout


def test_cleanup_json_connection_error_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            raise RuntimeError("cannot connect")

    monkeypatch.setattr(cleanup_cmd, "DoclibClient", _Client)

    result = runner.invoke(app, ["cleanup", "temp", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "api_error"
    assert "Cannot connect to mineru server" in payload["error"]["message"]
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


def test_parse_failed_response_exits_nonzero(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.unsupported"
    source.write_bytes(b"unsupported")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(
                sha256="",
                tier="flash",
                page_range="",
                status="failed",
                tip="File could not be ingested.",
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["parse"]["status"] == "failed"
    assert payload["parse"]["tip"] == "File could not be ingested."


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
            return ParseResponse(sha256="a" * 64, tier="standard", page_range="1~10", status="done", cache_hit=True)

        def get_doc_content(self, *args: Any, **kwargs: Any) -> DocContentResponse:
            assert kwargs["limit"] == 30000
            return DocContentResponse(
                sha256="a" * 64,
                short_id="aaaaaaa",
                tier="standard",
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


def test_parse_json_output_writes_file_and_returns_output_object(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    output = tmp_path / "nested" / "out.md"

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 90

        def ensure_parse(self, request: Any) -> ParseResponse:
            return ParseResponse(sha256="a" * 64, tier="flash", page_range="1~1", status="done", cache_hit=True)

        def export_doc_content(self, sha256: str, request: Any) -> Any:
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


def test_parse_json_error_output_is_machine_readable(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 31

        def ensure_parse(self, request: Any) -> ParseResponse:
            raise RuntimeError(
                "('quality_tier_unavailable', 'No standard or pro engine available. Use --tier flash.', None)"
            )

    monkeypatch.setattr(parse, "DoclibClient", _Client)

    result = runner.invoke(app, ["parse", str(source), "--wait", "1", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "quality_tier_unavailable"
    assert "No standard or pro engine available" in payload["error"]["message"]


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
            return InvalidateResponse(target="parses", sha256="a" * 64, tier=request.tier, invalidated_count=1)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(invalidate, "DoclibClient", _Client)

    result = runner.invoke(app, ["invalidate", "~/docs/../docs/demo.pdf", "--tier", "standard"])

    assert result.exit_code == 0
    assert calls == [os.path.normpath(os.path.abspath(os.path.expanduser("~/docs/../docs/demo.pdf")))]


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
                tier="standard",
                format="markdown",
                content="hello",
                request_scope=ContentRequestScope(locator=locator, context=kwargs["context"], limit=kwargs["limit"]),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(
        app,
        ["read", "doc:ab12cd3/tier:standard/page:4", "--context", "2", "--limit", "123", "--format", "markdown"],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            "doc:ab12cd3/tier:standard/page:4",
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
                tier="standard",
                format="markdown",
                content="hello",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                next_request=ContentNextRequest(locator="doc:ab12cd3/tier:standard/page:5"),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:standard/page:4", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["next_request"]["locator"] == "doc:ab12cd3/tier:standard/page:5"


def test_read_markdown_output_separates_next_marker_with_blank_line(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="standard",
                format="markdown",
                content="hello",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                next_request=ContentNextRequest(locator="doc:ab12cd3/tier:standard/page:5"),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:standard/page:4"])

    assert result.exit_code == 0
    assert "hello\n\n<!-- Next: mineru read doc:ab12cd3/tier:standard/page:5 -->" in result.output


def test_read_json_error_output_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            raise RuntimeError("('not_cached', 'Requested parsed content is not cached.', 'locator')")

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:standard/page:4", "--json"])

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
                tier="standard",
                format="markdown",
                content="hello",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:standard/page:4", "--output", str(output), "--json"])

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
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="standard",
                format="image",
                content="",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                asset=ContentAsset(path=str(asset), mime_type="image/png"),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:standard/page:4", "--format", "image", "--output", str(output), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["content"] is None
    assert payload["output"] == {"status": "written", "path": str(output.resolve())}
    assert output.read_bytes() == b"png-bytes"
    assert "Written to" not in result.output


def test_read_image_output_copies_server_asset_locally(monkeypatch: Any, tmp_path: Path) -> None:
    asset = tmp_path / "server.png"
    asset.write_bytes(b"png-bytes")
    output = tmp_path / "local.png"

    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            return DocContentResponse(
                sha256="a" * 64,
                short_id="ab12cd3",
                tier="standard",
                format="image",
                content="",
                request_scope=ContentRequestScope(locator=locator, context=0, limit=30000),
                asset=ContentAsset(path=str(asset), mime_type="image/png"),
            )

    monkeypatch.setattr(read, "DoclibClient", _Client)

    result = runner.invoke(app, ["read", "doc:ab12cd3/tier:standard/page:4", "--format", "image", "--output", str(output)])

    assert result.exit_code == 0
    assert output.read_bytes() == b"png-bytes"
    assert "Written to" in result.output
    assert str(output) in result.output.replace("\n", "")


def test_search_json_error_output_is_machine_readable(monkeypatch: Any) -> None:
    class _Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 10

        def search(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("('quality_tier_unavailable', 'No standard or pro engine available.', 'tier')")

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
