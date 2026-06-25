from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from mineru.kit.commands import api_server, models, parse, router, vlm_server
from mineru.kit.main import app

runner = CliRunner()


def test_kit_root_and_models_help() -> None:
    result = runner.invoke(app, ["--help"])
    models_result = runner.invoke(app, ["models", "--help"])

    assert result.exit_code == 0
    assert models_result.exit_code == 0
    assert "models" in result.output
    assert "api-server" in result.output
    assert "vlm-server" in result.output
    assert "router" in result.output


def test_kit_main_import_does_not_import_legacy_router() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys


class BlockLegacyRouterFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mineru.cli_old.router":
            raise ModuleNotFoundError("blocked legacy router import")
        return None


sys.meta_path.insert(0, BlockLegacyRouterFinder())
import mineru.kit.main
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_models_download_pipeline(monkeypatch: Any) -> None:
    monkeypatch.setattr(models, "_download_pipeline_models", lambda: "/tmp/pipeline")
    monkeypatch.setattr(models, "_download_vlm_models", lambda: "/tmp/vlm")
    monkeypatch.setattr(models, "_update_models_dir", lambda bundle, model_dir: Path(f"/tmp/{bundle}.json"))

    result = runner.invoke(app, ["models", "download", "pipeline"])

    assert result.exit_code == 0
    assert "Downloaded pipeline models" in result.output


def test_models_show_and_verify(tmp_path: Path, monkeypatch: Any) -> None:
    config = tmp_path / "mineru.json"
    pipeline_root = tmp_path / "pipeline"
    vlm_root = tmp_path / "vlm"
    for rel in models.PIPELINE_MODEL_PATHS:
        target = pipeline_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    for rel in models.VLM_MODEL_MARKERS:
        target = vlm_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    config.write_text(
        (
            "{\n"
            f'  "models-dir": {{"pipeline": "{pipeline_root}", "vlm": "{vlm_root}"}},\n'
            '  "config_version": "1.3.1"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MINERU_TOOLS_CONFIG_JSON", str(config))

    show_result = runner.invoke(app, ["models", "show"])
    verify_result = runner.invoke(app, ["models", "verify"])

    assert show_result.exit_code == 0
    assert "pipeline.exists: True" in show_result.output
    assert "vlm.exists: True" in show_result.output
    assert verify_result.exit_code == 0
    assert "pipeline: ok" in verify_result.output
    assert "vlm: ok" in verify_result.output


def test_api_server_omits_tier_when_backend_only(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(api_server.parser_api_server.main, "main", _fake_main)

    result = runner.invoke(app, ["api-server", "--backend", "hybrid-auto-engine"])

    assert result.exit_code == 0
    assert "--backend" in seen["args"]
    assert "hybrid-auto-engine" in seen["args"]
    assert "--tier" not in seen["args"]


def test_vlm_server_rejects_unimplemented_engine() -> None:
    result = runner.invoke(app, ["vlm-server", "--engine", "sglang"])

    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_vlm_server_forwards_extra_args(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(vlm_server.old_vlm_server.openai_server, "main", _fake_main)

    result = runner.invoke(app, ["vlm-server", "--host", "0.0.0.0", "--port", "30000"])

    assert result.exit_code == 0
    assert seen == {
        "args": ["--engine", "auto", "--host", "0.0.0.0", "--port", "30000"],
        "prog_name": "mineru-kit vlm-server",
        "standalone_mode": False,
    }


def test_router_forwards_known_and_extra_args(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(
        router,
        "_load_old_router",
        lambda: SimpleNamespace(main=SimpleNamespace(main=_fake_main)),
    )

    result = runner.invoke(
        app,
        [
            "router",
            "--host",
            "0.0.0.0",
            "--port",
            "8002",
            "--allow-public-http-client",
            "--upstream-url",
            "http://mineru-api:8000",
            "--local-gpus",
            "none",
            "--worker-host",
            "127.0.0.1",
            "--gpu-memory-utilization",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert seen["prog_name"] == "mineru-kit router"
    assert seen["standalone_mode"] is False
    assert seen["args"] == [
        "--host",
        "0.0.0.0",
        "--port",
        "8002",
        "--allow-public-http-client",
        "--upstream-url",
        "http://mineru-api:8000",
        "--local-gpus",
        "none",
        "--worker-host",
        "127.0.0.1",
        "--gpu-memory-utilization",
        "0.5",
    ]


def test_parse_rejects_file_output_for_directory_input(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "a.pdf").write_bytes(b"%PDF-1.7\n")

    result = runner.invoke(app, ["parse", str(source_dir), "-o", str(tmp_path / "out.md")])

    assert result.exit_code == 1
    assert "directory path" in result.output


def test_parse_single_file_markdown(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def markdown(self) -> str:
            return "# demo\n"

        def to_json(self) -> str:
            return '{"pages":[]}'

        def save(self, writer: Any) -> None:
            writer.write_string("markdown.md", self.markdown())
            writer.write_string("middle_json.json", self.to_json())

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(app, ["parse", str(source), "-o", str(output)])

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8") == "# demo\n"


def test_parse_output_replaces_surrogate_chars(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def markdown(self) -> str:
            return "before \ud83d after\n"

        def to_json(self) -> str:
            return '{"pages":[]}'

        def save(self, writer: Any) -> None:
            writer.write_string("markdown.md", self.markdown())
            writer.write_string("middle_json.json", self.to_json())

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(app, ["parse", str(source), "-o", str(output)])

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8") == "before ? after\n"
