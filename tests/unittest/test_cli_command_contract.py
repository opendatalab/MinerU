from __future__ import annotations

import ast
from pathlib import Path


CLI_COMMANDS_DIR = Path(__file__).parents[2] / "mineru" / "cli" / "commands"
CLI_MAIN = Path(__file__).parents[2] / "mineru" / "cli" / "main.py"
CLI_RENDERERS = Path(__file__).parents[2] / "mineru" / "cli" / "renderers.py"

ALLOWED_PRINT_ERROR_CALLS: set[tuple[str, str]] = set()

ALLOWED_TYPER_EXIT_CALLS: set[tuple[str, str]] = set()

ALLOWED_EXIT_WITH_ERROR_CALLS: set[tuple[str, str]] = set()

ALLOWED_EMIT_RESULT_CALLS = {
    ("commands/parse.py", "_emit_notice"),
    ("commands/parse.py", "_parse"),
    ("commands/parse.py", "_output_parse_result"),
    ("commands/parse.py", "_emit_parse_json_response"),
}

MIGRATED_QUERY_COMMANDS = {
    "commands/config.py",
    "commands/cleanup.py",
    "commands/forget.py",
    "commands/invalidate.py",
    "commands/list_resources.py",
    "commands/parse.py",
    "commands/read.py",
    "commands/scan.py",
    "commands/search.py",
    "commands/server.py",
    "commands/show.py",
    "commands/telemetry.py",
    "commands/watch.py",
}

ALLOWED_DIRECT_OUTPUT_CALLS: set[tuple[str, str, str]] = set()

DIRECT_OUTPUT_NAMES = {
    "print",
    "print_info",
    "print_json",
    "print_success",
}


def test_cli_commands_do_not_add_unreviewed_direct_error_output_or_exit() -> None:
    print_error_calls: set[tuple[str, str]] = set()
    typer_exit_calls: set[tuple[str, str]] = set()
    exit_with_error_calls: set[tuple[str, str]] = set()
    emit_result_calls: set[tuple[str, str]] = set()

    for path in [*sorted(CLI_COMMANDS_DIR.glob("*.py")), CLI_MAIN]:
        module = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parent_map(module)
        rel_path = _relative_cli_path(path)
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            function_name = _enclosing_function_name(node, parents)
            if _is_name_call(node, "print_error"):
                print_error_calls.add((rel_path, function_name))
            if _is_typer_exit_call(node):
                typer_exit_calls.add((rel_path, function_name))
            if _is_name_call(node, "exit_with_error"):
                exit_with_error_calls.add((rel_path, function_name))
            if _is_name_call(node, "emit_result"):
                emit_result_calls.add((rel_path, function_name))

    assert print_error_calls <= ALLOWED_PRINT_ERROR_CALLS
    assert typer_exit_calls <= ALLOWED_TYPER_EXIT_CALLS
    assert exit_with_error_calls <= ALLOWED_EXIT_WITH_ERROR_CALLS
    assert emit_result_calls <= ALLOWED_EMIT_RESULT_CALLS


def test_migrated_query_commands_do_not_write_output_directly() -> None:
    direct_output_calls: set[tuple[str, str, str]] = set()

    for path in sorted(CLI_COMMANDS_DIR.glob("*.py")):
        rel_path = _relative_cli_path(path)
        if rel_path not in MIGRATED_QUERY_COMMANDS:
            continue
        module = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parent_map(module)
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            function_name = _enclosing_function_name(node, parents)
            output_name = _direct_output_call_name(node)
            if output_name is not None:
                direct_output_calls.add((rel_path, function_name, output_name))

    assert direct_output_calls <= ALLOWED_DIRECT_OUTPUT_CALLS


def test_cli_renderers_do_not_use_any_annotations() -> None:
    any_annotations: set[tuple[str, str]] = set()

    renderer_paths = [CLI_RENDERERS] if CLI_RENDERERS.exists() else []
    for path in [*renderer_paths, *sorted(CLI_COMMANDS_DIR.glob("*.py"))]:
        module = ast.parse(path.read_text(encoding="utf-8"))
        rel_path = _relative_cli_path(path) if path != CLI_RENDERERS else "renderers.py"
        for node in ast.walk(module):
            if not isinstance(node, ast.FunctionDef) or not _is_render_function(node):
                continue
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                if _annotation_uses_any(arg.annotation):
                    any_annotations.add((rel_path, f"{node.name}.{arg.arg}"))
            if _annotation_uses_any(node.returns):
                any_annotations.add((rel_path, f"{node.name}.return"))

    assert any_annotations == set()


def test_migrated_query_commands_do_not_use_cliresult_wrapper_helpers() -> None:
    wrapper_helpers: set[tuple[str, str]] = set()

    for path in sorted(CLI_COMMANDS_DIR.glob("*.py")):
        rel_path = _relative_cli_path(path)
        if rel_path not in MIGRATED_QUERY_COMMANDS:
            continue
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name.startswith("_")
                and _annotation_is_cliresult(node.returns)
            ):
                wrapper_helpers.add((rel_path, node.name))

    assert wrapper_helpers == set()


def _parent_map(module: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(module):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _relative_cli_path(path: Path) -> str:
    cli_dir = Path(__file__).parents[2] / "mineru" / "cli"
    return path.relative_to(cli_dir).as_posix()


def _enclosing_function_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.FunctionDef):
            return current.name
    return "<module>"


def _is_name_call(node: ast.Call, name: str) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == name


def _is_typer_exit_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "Exit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "typer"
    )


def _direct_output_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id in DIRECT_OUTPUT_NAMES:
        return node.func.id
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "echo"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "typer"
    ):
        return "typer.echo"
    return None


def _annotation_uses_any(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id == "Any":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "Any":
            return True
    return False


def _annotation_is_cliresult(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id == "CliResult"
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "CliResult"
    if isinstance(annotation, ast.Subscript):
        return _annotation_is_cliresult(annotation.value)
    return False


def _is_render_function(node: ast.FunctionDef) -> bool:
    return node.name.startswith("render_") or node.name.startswith("_render_")
