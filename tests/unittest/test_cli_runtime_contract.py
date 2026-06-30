from __future__ import annotations

import json
import sys
import types

import pytest
import typer

from mineru.cli import output
from mineru.cli import contracts
from mineru.cli.contracts import CliContext
from mineru.cli.runtime import cli_ok, cli_task, emit_error, emit_result, run_cli
from mineru.errors import MineruError


@pytest.fixture(autouse=True)
def _plain_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output, "print_notice", lambda msg: print(msg, file=sys.stderr))
    monkeypatch.setattr(output, "print_error", lambda msg: print(f"Error: {msg}", file=sys.stderr))

    def print_rich(*objects: object) -> None:
        for item in objects:
            print(item)

    monkeypatch.setattr(output, "print_rich", print_rich)


def test_emit_result_json_mode_writes_single_json_object(capsys: pytest.CaptureFixture[str]) -> None:
    result = cli_ok({"status": "done", "count": 2})

    emit_result(CliContext(json_mode=True), result)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "done", "count": 2}
    assert captured.err == ""


def test_run_cli_accepts_plain_action_result(capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(
        CliContext(json_mode=False),
        lambda: {"name": "demo"},
        render=lambda data: f"name={data['name']}",
        warnings=lambda data: [f"warning for {data['name']}"],
    )

    captured = capsys.readouterr()
    assert captured.out == "name=demo\n"
    assert captured.err == "warning for demo\n"


def test_run_cli_accepts_renderable_sequence(capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(
        CliContext(json_mode=False),
        lambda: {"name": "demo"},
        render=lambda data: [f"name={data['name']}", "status=done"],
    )

    captured = capsys.readouterr()
    assert captured.out == "name=demo\nstatus=done\n"
    assert captured.err == ""


def test_run_cli_accepts_renderable_generator(capsys: pytest.CaptureFixture[str]) -> None:
    def render(data: dict[str, str]):
        yield f"name={data['name']}"
        yield "status=done"

    run_cli(CliContext(json_mode=False), lambda: {"name": "demo"}, render=render)

    captured = capsys.readouterr()
    assert captured.out == "name=demo\nstatus=done\n"
    assert captured.err == ""


def test_cli_renderer_type_does_not_allow_side_effect_only_renderer() -> None:
    assert not isinstance(contracts.CliRenderer, types.UnionType)
    assert "None" in str(contracts.CliRenderer)


def test_run_cli_treats_none_renderer_result_as_no_output(capsys: pytest.CaptureFixture[str]) -> None:
    def render(data: dict[str, str]) -> None:
        del data

    run_cli(CliContext(json_mode=False), lambda: {"name": "demo"}, render=render)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_run_cli_json_mode_does_not_call_renderer(capsys: pytest.CaptureFixture[str]) -> None:
    called = False

    def render(data: dict[str, str]) -> None:
        nonlocal called
        called = True
        print(f"side-effect={data['name']}")

    run_cli(CliContext(json_mode=True), lambda: {"name": "demo"}, render=render)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"name": "demo"}
    assert captured.err == ""
    assert called is False


def test_emit_error_json_mode_writes_error_envelope_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        emit_error(CliContext(json_mode=True), MineruError("file_not_found", "File missing.", "path"))

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 1
    assert json.loads(captured.out) == {
        "error": {
            "type": "invalid_request_error",
            "code": "file_not_found",
            "message": "File missing.",
            "param": "path",
        }
    }
    assert captured.err == ""


def test_emit_error_non_json_writes_human_error_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        emit_error(CliContext(json_mode=False), MineruError("file_not_found", "File missing.", "path"))

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 1
    assert captured.out == ""
    assert captured.err == "Error: File missing.\n"


def test_emit_result_writes_notices_and_warnings_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    result = cli_ok(
        {"status": "done"},
        notices=["started"],
        warnings=["be careful"],
    )

    emit_result(CliContext(json_mode=True), result)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "done"}
    assert captured.err == "started\nbe careful\n"


def test_task_result_failed_status_exits_one_after_output(capsys: pytest.CaptureFixture[str]) -> None:
    result = cli_task(
        {"id": 3, "status": "failed"},
        status="failed",
        fail_if_final_failed=True,
    )

    with pytest.raises(typer.Exit) as exc_info:
        emit_result(CliContext(json_mode=True), result)

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 1
    assert json.loads(captured.out) == {"id": 3, "status": "failed"}


def test_task_result_failed_status_can_be_query_success(capsys: pytest.CaptureFixture[str]) -> None:
    result = cli_task(
        {"id": 3, "status": "failed"},
        status="failed",
        fail_if_final_failed=False,
    )

    emit_result(CliContext(json_mode=True), result)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"id": 3, "status": "failed"}
