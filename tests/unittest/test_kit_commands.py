from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import typer
from fastapi.testclient import TestClient
from typer.main import get_command
from typer.testing import CliRunner

import mineru.kit.main as kit_main
from mineru.cli.main import app as mineru_app
from mineru.cli.version_command import version_cmd
from mineru.kit.commands import api_server, gradio, models, parse, router, vlm_server
from mineru.kit.main import app
from mineru.kit.vlm_server import mlx_vlm_server
from mineru.parser.base import ParseResult
from mineru.model.registry import MODEL_COMPLETE_MARKER
from mineru.version import __version__

runner = CliRunner()

_REMOVED_DISABLE_TABLE_OPTION = "--disable-" + "table"
_REMOVED_DISABLE_FORMULA_OPTION = "--disable-" + "formula"


def test_kit_main_configures_standard_streams_before_running_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(kit_main, "configure_standard_streams", lambda: calls.append("configure"))
    monkeypatch.setattr(kit_main, "app", lambda: calls.append("app"))

    kit_main.main()

    assert calls == ["configure", "app"]


def test_gradio_compatibility_main_runs_modern_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验独立兼容入口复用新版 Gradio 命令且先配置标准流。"""
    calls: list[object] = []
    monkeypatch.setattr(gradio, "configure_standard_streams", lambda: calls.append("configure"))
    monkeypatch.setattr(gradio.typer, "run", lambda command: calls.append(command))

    gradio.main()

    assert calls == ["configure", gradio.gradio_cmd]


def test_gradio_console_script_targets_modern_command() -> None:
    """校验兼容命令不再通过已删除的旧 CLI 包启动。"""
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["mineru-gradio"] == "mineru.kit.commands.gradio:main"


def _invoke_standalone_command(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: Callable[[], None],
    script: str,
    args: list[str],
) -> int:
    """通过真实入口解析命令行参数，并在调用后恢复进程参数。"""
    with monkeypatch.context() as context:
        context.setattr(sys, "argv", [script, *args])
        with pytest.raises(SystemExit) as exc_info:
            entrypoint()
    assert isinstance(exc_info.value.code, int)
    return exc_info.value.code


@pytest.mark.parametrize(
    ("script", "target"),
    [
        ("mineru-openai-server", "mineru.kit.commands.vlm_server:main"),
        ("mineru-models-download", "mineru.kit.commands.models:download_main"),
        ("mineru-api", "mineru.kit.commands.api_server:main"),
    ],
)
def test_standalone_console_scripts_target_kit_commands(script: str, target: str) -> None:
    """校验三个独立命令都映射到新版 kit 实现。"""
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"][script] == target


@pytest.mark.parametrize(
    ("module", "entrypoint", "callback"),
    [(api_server, api_server.main, api_server.api_server_cmd), (models, models.download_main, models.download_cmd)],
)
def test_standalone_main_configures_streams_before_typer_run(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    entrypoint: Callable[[], None],
    callback: Callable[..., None],
) -> None:
    """校验 API 和下载入口先配置标准流，再直接复用现有回调。"""
    calls: list[object] = []
    monkeypatch.setattr(module, "configure_standard_streams", lambda: calls.append("configure"))
    monkeypatch.setattr(typer, "run", lambda command: calls.append(command))

    entrypoint()

    assert calls == ["configure", callback]


def test_standalone_vlm_main_configures_streams_and_forwards_extra_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 VLM 独立入口的标准流初始化顺序及共享参数透传配置。"""
    calls: list[object] = []

    def _record_command(ctx: typer.Context) -> None:
        """记录单命令应用收到的额外参数。"""
        calls.append(list(ctx.args))

    monkeypatch.setattr(vlm_server, "configure_standard_streams", lambda: calls.append("configure"))
    monkeypatch.setattr(vlm_server, "vlm_server_cmd", _record_command)
    code = _invoke_standalone_command(monkeypatch, vlm_server.main, "mineru-openai-server", ["--port", "30000"])
    kit_command = next(command for command in app.registered_commands if command.name == "vlm-server")

    assert code == 0
    assert calls == ["configure", ["--port", "30000"]]
    assert kit_command.context_settings is vlm_server.FORWARD_CONTEXT_SETTINGS


@pytest.mark.parametrize(
    ("script", "entrypoint", "expected_options"),
    [
        ("mineru-gradio", gradio.main, ("--api-url", "--api-server-tier")),
        ("mineru-openai-server", vlm_server.main, ("--engine",)),
        ("mineru-models-download", models.download_main, ("--tier", "--stack", "--source", "--verbose")),
        (
            "mineru-api",
            api_server.main,
            ("--host", "--port", "--tier", "--no-flash", "--no-advanced", "--preload-models"),
        ),
    ],
)
def test_standalone_command_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: str,
    entrypoint: Callable[[], None],
    expected_options: tuple[str, ...],
) -> None:
    """校验独立命令直接展示新版参数，且不暴露补全或额外子命令。"""
    code = _invoke_standalone_command(monkeypatch, entrypoint, script, ["--help"])
    output = capsys.readouterr().out

    assert code == 0
    assert script in output
    for option in expected_options:
        assert option in output
    assert "--install-completion" not in output
    assert "--show-completion" not in output
    assert "COMMAND [ARGS]" not in output


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize(
    "args",
    [
        [],
        [
            "--host",
            "0.0.0.0",
            "--port",
            "18000",
            "--tier",
            "basic",
            "--no-flash",
            "--no-advanced",
            "--preload-models",
        ],
    ],
)
def test_standalone_api_matches_kit_arguments_and_exit_code(
    monkeypatch: pytest.MonkeyPatch, args: list[str], exit_code: int
) -> None:
    """校验独立 API 入口与 kit 的默认值、参数转发和退出码一致。"""
    calls: list[tuple[list[str], str, bool]] = []

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录启动参数并模拟底层服务退出，不启动真实服务器。"""
        calls.append((args, prog_name, standalone_mode))
        raise SystemExit(exit_code)

    monkeypatch.setattr(api_server.parser_api_server.main, "main", _fake_main)
    code = _invoke_standalone_command(monkeypatch, api_server.main, "mineru-api", args)
    kit_result = runner.invoke(app, ["api-server", *args])

    assert code == kit_result.exit_code == exit_code
    assert len(calls) == 2
    assert calls[0] == calls[1]


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize(
    "args",
    [
        ["PDF-Extract-Kit-1.0", "--source", "modelscope", "--verbose"],
        ["--tier", "standard", "--stack", "full", "--source", "huggingface"],
    ],
)
def test_standalone_models_download_matches_kit(monkeypatch: pytest.MonkeyPatch, args: list[str], fails: bool) -> None:
    """校验下载入口无需子命令，并沿用模型选择、下载参数及失败退出码。"""
    calls: list[tuple[str, str | None, bool]] = []

    def _fake_download(repo: models.ModelRepo, *, source: str | None = None, local_as_auto: bool = False) -> Path:
        """记录下载请求或模拟失败，避免网络下载和配置写入。"""
        calls.append((repo.name, source, local_as_auto))
        if fails:
            raise RuntimeError("download unavailable")
        return Path("/tmp/models") / repo.local_name

    monkeypatch.setattr(models, "download_model_repo", _fake_download)
    code = _invoke_standalone_command(monkeypatch, models.download_main, "mineru-models-download", args)
    standalone_calls = calls[:]
    calls.clear()
    kit_result = runner.invoke(app, ["models", "download", *args])

    assert code == kit_result.exit_code == int(fails)
    assert standalone_calls == calls
    assert calls


@pytest.mark.parametrize("engine", ["vllm", "lmdeploy"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_standalone_vlm_matches_kit_arguments_and_exit_code(
    monkeypatch: pytest.MonkeyPatch, engine: str, exit_code: int
) -> None:
    """校验两种 argv 引擎的参数透传、退出码和进程参数恢复行为。"""
    calls: list[list[str]] = []

    def _fake_main() -> None:
        """记录引擎收到的原始参数并模拟退出，不加载真实推理引擎。"""
        calls.append(sys.argv[1:])
        raise SystemExit(exit_code)

    module_name = f"mineru.kit.vlm_server.{engine}_server"
    fake_server = ModuleType(module_name)
    fake_server.main = _fake_main
    monkeypatch.setitem(sys.modules, module_name, fake_server)
    monkeypatch.setattr(vlm_server, "_module_available", lambda module_name: module_name == engine)
    extra_args = ["--host", "127.0.0.1", "--port", "30000", "--model=test-model", "--trust-remote-code"]
    args = ["--engine", engine, *extra_args]
    original_argv = ["mineru-openai-server", *args]
    monkeypatch.setattr(sys, "argv", original_argv)

    with pytest.raises(SystemExit) as exc_info:
        vlm_server.main()
    assert sys.argv is original_argv
    kit_result = runner.invoke(app, ["vlm-server", *args])

    assert exc_info.value.code == kit_result.exit_code == exit_code
    assert calls == [extra_args, extra_args]
    assert sys.argv is original_argv


def test_standalone_vlm_mlx_matches_kit(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 MLX 的显式参数调用同样复用新版 kit 路径。"""
    calls: list[tuple[list[str], str, bool]] = []

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录 MLX 启动参数，不加载 MLX 模型或启动服务。"""
        calls.append((args, prog_name, standalone_mode))

    monkeypatch.setattr(vlm_server, "_mlx_server_available", lambda: True)
    monkeypatch.setattr(mlx_vlm_server, "main", _fake_main)
    args = ["--engine", "mlx", "--model", "test-model", "--port", "18080"]
    code = _invoke_standalone_command(monkeypatch, vlm_server.main, "mineru-openai-server", args)
    kit_result = runner.invoke(app, ["vlm-server", *args])

    assert code == kit_result.exit_code == 0
    assert calls == [(args[2:], "mineru-kit vlm-server", False)] * 2


@pytest.mark.parametrize(
    ("script", "entrypoint", "args", "prefix", "exit_code"),
    [
        ("mineru-api", api_server.main, ["--backend", "hybrid-engine"], ["api-server"], 2),
        ("mineru-api", api_server.main, ["--tier", "advanced"], ["api-server"], 1),
        ("mineru-models-download", models.download_main, [], ["models", "download"], 1),
        ("mineru-models-download", models.download_main, ["--tier", "flash"], ["models", "download"], 1),
        ("mineru-openai-server", vlm_server.main, ["--engine", "sglang"], ["vlm-server"], 1),
    ],
)
def test_standalone_invalid_arguments_match_kit(
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    entrypoint: Callable[[], None],
    args: list[str],
    prefix: list[str],
    exit_code: int,
) -> None:
    """校验独立入口沿用新版参数校验，不恢复旧参数或默认下载行为。"""
    code = _invoke_standalone_command(monkeypatch, entrypoint, script, args)
    kit_result = runner.invoke(app, [*prefix, *args])

    assert code == kit_result.exit_code == exit_code


def _fake_apply_chat_template(_processor: Any, _config: Any, _prompt: Any, *_args: Any, **_kwargs: Any) -> str:
    return "formatted"


def test_kit_root_and_models_help() -> None:
    result = runner.invoke(app, ["--help"])
    models_result = runner.invoke(app, ["models", "--help"])

    assert result.exit_code == 0
    assert models_result.exit_code == 0
    assert "models" in result.output
    assert "api-server" in result.output
    assert "vlm-server" in result.output
    assert "router" in result.output
    assert "version" in result.output


def test_top_level_commands_register_implementation_callbacks_directly() -> None:
    callbacks = {command.name: command.callback for command in app.registered_commands}

    assert callbacks["parse"] is parse.parse_cmd
    assert callbacks["gradio"] is gradio.gradio_cmd
    assert callbacks["api-server"] is api_server.api_server_cmd
    assert callbacks["vlm-server"] is vlm_server.vlm_server_cmd
    assert callbacks["router"] is router.router_cmd
    assert callbacks["version"] is version_cmd


@pytest.mark.parametrize(
    ("command", "expected_options"),
    [
        ("parse", ("--output", "--format", "--tier")),
        ("gradio", ("--api-url", "--server-name", "--api-server-tier")),
        ("api-server", ("--host", "--port", "--tier", "--no-flash", "--no-advanced", "--preload-models")),
        ("vlm-server", ("--engine",)),
        ("router", ("--host", "--upstream-url", "--local-gpus")),
        ("version", ("--json",)),
    ],
)
def test_directly_registered_command_help(command: str, expected_options: tuple[str, ...]) -> None:
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0
    for option in expected_options:
        assert option in result.output


def test_kit_root_help_hides_typer_completion_options() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--version" in result.output
    assert "--install-completion" not in result.output
    assert "--show-completion" not in result.output


def test_kit_root_commands_keep_product_order() -> None:
    command = get_command(app)

    assert command.list_commands(None) == [
        "parse",
        "gradio",
        "api-server",
        "vlm-server",
        "router",
        "models",
        "version",
    ]


def test_kit_version_command_matches_mineru() -> None:
    kit_result = runner.invoke(app, ["version"])
    mineru_result = runner.invoke(mineru_app, ["version"])

    assert kit_result.exit_code == 0
    assert mineru_result.exit_code == 0
    assert kit_result.output == mineru_result.output
    assert f"MinerU version: {__version__}" in kit_result.output
    assert f"Python version: {sys.version.split()[0]}" in kit_result.output


def test_kit_root_version_option_matches_version_command() -> None:
    option_result = runner.invoke(app, ["--version"])
    command_result = runner.invoke(app, ["version"])

    assert option_result.exit_code == 0
    assert command_result.exit_code == 0
    assert option_result.output == command_result.output


def test_kit_version_json_matches_mineru() -> None:
    kit_result = runner.invoke(app, ["version", "--json"])
    mineru_result = runner.invoke(mineru_app, ["version", "--json"])

    assert kit_result.exit_code == 0
    assert mineru_result.exit_code == 0
    assert kit_result.output == mineru_result.output
    assert json.loads(kit_result.output) == {
        "mineru_version": __version__,
        "python_version": sys.version.split()[0],
    }


def test_kit_root_show_completion_is_not_a_supported_option() -> None:
    result = runner.invoke(app, ["--show-completion"])

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_router_upstream_only_worker_pool_builds_remote_server() -> None:
    """校验新 WorkerPool 从 V1 upstream 发现能力且不创建本地 worker。"""
    import httpx

    from mineru.kit.router.workers import RouterSettings, WorkerPool

    def _handler(request: httpx.Request) -> httpx.Response:
        """返回 Router 能力发现所需的最小 V1 响应。"""
        if request.url.path == "/v1/health":
            return httpx.Response(200, json={"status": "ok", "features": {"sources": ["file_id"]}})
        if request.url.path == "/v1/tiers":
            return httpx.Response(200, json={"data": [{"id": "standard"}]})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model-a"}]})
        return httpx.Response(404)

    settings = RouterSettings(upstream_urls=("http://mineru-api:8000",), local_gpus="none")
    pool = WorkerPool(settings, transport=httpx.MockTransport(_handler))

    async def _run() -> list[tuple[str, str, str, object, set[str]]]:
        """启动、读取并关闭测试 WorkerPool。"""
        await pool.start()
        try:
            return [
                (worker.worker_id, worker.source, worker.base_url, worker.local_worker, set(worker.tiers))
                for worker in pool.workers
            ]
        finally:
            await pool.close()

    assert asyncio.run(_run()) == [
        ("remote-1", "remote", "http://mineru-api:8000", None, {"standard"}),
    ]


def test_router_startup_with_no_servers_fails_health_check() -> None:
    """校验未配置 worker 时应用可启动，但 V1 health 返回 503。"""
    from mineru.kit.router import RouterSettings, create_app

    router_app = create_app(RouterSettings(upstream_urls=(), local_gpus="none"))

    with TestClient(router_app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "upstream_unavailable"


def test_models_download_tier_basic(monkeypatch: Any) -> None:
    monkeypatch.setattr(models.config.model, "stack", "full")
    captured: list[str] = []

    def fake_download_model_repo(repo: Any, *, source: str | None = None, local_as_auto: bool = False) -> Path:
        captured.append(repo.name)
        return Path("/tmp/models") / repo.local_name

    monkeypatch.setattr(models, "download_model_repo", fake_download_model_repo)

    result = runner.invoke(app, ["models", "download", "--tier", "basic"])

    assert result.exit_code == 0
    assert captured == ["PDF-Extract-Kit-1.0"]
    assert "Downloaded models for tier basic" in result.output


def test_models_download_tier_standard(monkeypatch: Any) -> None:
    monkeypatch.setattr(models.config.model, "stack", "full")
    captured: list[str] = []

    def fake_download_model_repo(repo: Any, *, source: str | None = None, local_as_auto: bool = False) -> Path:
        captured.append(repo.name)
        return Path("/tmp/models") / repo.local_name

    monkeypatch.setattr(models, "download_model_repo", fake_download_model_repo)

    result = runner.invoke(app, ["models", "download", "--tier", "standard"])

    assert result.exit_code == 0
    assert captured == ["PDF-Extract-Kit-1.0", "MinerU2.5-Pro-2605-1.2B"]
    assert "Downloaded models for tier standard" in result.output


@pytest.mark.parametrize("tier", ["flash", "advanced"])
def test_model_registry_rejects_non_model_tiers(tier: str) -> None:
    with pytest.raises(ValueError, match="Supported model tiers: basic, standard"):
        models.model_repos_for_tier(tier, stack="full")


@pytest.mark.parametrize("command", ["download", "verify"])
@pytest.mark.parametrize("tier", ["flash", "advanced"])
def test_models_commands_reject_non_model_tiers(command: str, tier: str) -> None:
    result = runner.invoke(app, ["models", command, "--tier", tier])

    assert result.exit_code == 1
    assert "Supported model tiers: basic, standard" in " ".join(result.output.split())


def test_models_download_repo_uses_explicit_source(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_download_model_repo(repo: Any, *, source: str | None = None, local_as_auto: bool = False) -> Path:
        captured["repo"] = repo.name
        captured["source"] = source
        captured["local_as_auto"] = local_as_auto
        return Path("/tmp/models") / repo.local_name

    monkeypatch.setattr(models, "download_model_repo", fake_download_model_repo)

    result = runner.invoke(app, ["models", "download", "PDF-Extract-Kit-1.0", "--source", "auto"])

    assert result.exit_code == 0
    assert captured == {
        "repo": "PDF-Extract-Kit-1.0",
        "source": "auto",
        "local_as_auto": True,
    }
    assert "Downloaded models for PDF-Extract-Kit-1.0" in result.output


def test_models_show_and_verify(tmp_path: Path, monkeypatch: Any) -> None:
    base_dir = tmp_path / "models"
    monkeypatch.setattr(models.config.model, "base_dir", str(base_dir))
    monkeypatch.setattr(models.config.model, "stack", "full")
    for repo in models.MODEL_REPOS:
        if repo.download_mode == "full":
            repo.local_dir().mkdir(parents=True, exist_ok=True)
            (repo.local_dir() / MODEL_COMPLETE_MARKER).touch()
            continue
        for model_path in repo.required_paths():
            target = repo.local_dir() / model_path.relative_path
            if Path(model_path.relative_path).suffix:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x", encoding="utf-8")
                continue
            target.mkdir(parents=True, exist_ok=True)
            (target / MODEL_COMPLETE_MARKER).touch()

    show_result = runner.invoke(app, ["models", "show"])
    verify_result = runner.invoke(app, ["models", "verify"])

    assert show_result.exit_code == 0
    assert "Config exists:" in show_result.output
    assert "PDF-Extract-Kit-1.0: ready" in show_result.output
    assert "MinerU2.5-Pro-2605-1.2B: ready" in show_result.output
    assert "Model tiers:" in show_result.output
    assert "  basic:" in show_result.output
    assert "  standard:" in show_result.output
    assert "  flash:" not in show_result.output
    assert "  advanced:" not in show_result.output
    assert verify_result.exit_code == 0
    assert "PDF-Extract-Kit-1.0: ok" in verify_result.output
    assert "MinerU2.5-Pro-2605-1.2B: ok" in verify_result.output


def test_api_server_rejects_backend_and_effort_options() -> None:
    backend_result = runner.invoke(app, ["api-server", "--backend", "hybrid-engine"])
    effort_result = runner.invoke(app, ["api-server", "--effort", "high"])

    assert backend_result.exit_code != 0
    assert "--backend" in backend_result.output
    assert effort_result.exit_code != 0
    assert "--effort" in effort_result.output


@pytest.mark.parametrize(
    ("command", "option"),
    [("api-server", "--ocr-mode"), ("gradio", "--ocr-mode"), ("gradio", "--api-server-ocr-mode")],
)
def test_server_commands_reject_startup_ocr_configuration(command: str, option: str) -> None:
    """服务启动命令不再接受或展示 OCR 配置，单次解析命令继续提供该参数。"""
    result = runner.invoke(app, [command, option, "ocr"])
    assert result.exit_code != 0
    assert "No such option" in result.output
    help_result = runner.invoke(app, [command, "--help"])
    assert help_result.exit_code == 0
    assert option not in help_result.output
    assert "--ocr-mode" in runner.invoke(app, ["parse", "--help"]).output


def test_kit_commands_do_not_expose_formula_table_switches() -> None:
    """校验 mineru-kit 公开命令不再暴露公式/表格识别开关。"""
    parse_help = runner.invoke(app, ["parse", "--help"])
    api_server_help = runner.invoke(app, ["api-server", "--help"])

    assert parse_help.exit_code == 0
    assert api_server_help.exit_code == 0
    for output in (parse_help.output, api_server_help.output):
        assert _REMOVED_DISABLE_TABLE_OPTION not in output
        assert _REMOVED_DISABLE_FORMULA_OPTION not in output


def test_api_server_forwards_single_tier_and_disabled_tiers(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录 mineru-kit api-server 对启动能力参数的原样转发。"""
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(api_server.parser_api_server.main, "main", _fake_main)

    result = runner.invoke(
        app,
        ["api-server", "--tier", "standard", "--no-flash", "--no-advanced", "--preload-models"],
    )

    assert result.exit_code == 0
    assert seen["prog_name"] == "mineru-kit api-server"
    assert seen["standalone_mode"] is False
    assert [seen["args"][index + 1] for index, item in enumerate(seen["args"]) if item == "--tier"] == ["standard"]
    assert "--no-flash" in seen["args"]
    assert "--no-advanced" in seen["args"]
    assert "--preload-models" in seen["args"]
    assert "--ocr-mode" not in seen["args"]


def test_api_server_without_tier_lets_parser_api_apply_standard_default(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录 mineru-kit api-server 默认参数，由底层应用 Standard 能力默认值。"""
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(api_server.parser_api_server.main, "main", _fake_main)

    result = runner.invoke(app, ["api-server", "--host", "0.0.0.0", "--port", "15985"])

    assert result.exit_code == 0
    assert seen["prog_name"] == "mineru-kit api-server"
    assert seen["standalone_mode"] is False
    assert "--tier" not in seen["args"]
    assert "--no-flash" not in seen["args"]
    assert "--no-advanced" not in seen["args"]
    assert "--preload-models" not in seen["args"]


def test_api_server_rejects_advanced_startup_tier() -> None:
    result = runner.invoke(app, ["api-server", "--tier", "advanced"])

    assert result.exit_code == 1
    assert "Unsupported server tier 'advanced'" in result.output


def test_api_server_rejects_flash_no_flash_conflict() -> None:
    result = runner.invoke(app, ["api-server", "--tier", "flash", "--no-flash"])

    assert result.exit_code == 1
    assert "--tier flash cannot be combined with --no-flash" in result.output


def test_api_server_normalizes_hidden_language_alias(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录 api-server 转发参数，确认隐藏语言别名不会继续下传。"""
        seen["args"] = args

    monkeypatch.setattr(api_server.parser_api_server.main, "main", _fake_main)

    result = runner.invoke(app, ["api-server", "--language", "en"])

    assert result.exit_code == 0
    language_index = seen["args"].index("--language")
    assert seen["args"][language_index + 1] == "ch"


def test_api_server_rejects_removed_ch_lite_language() -> None:
    result = runner.invoke(app, ["api-server", "--language", "ch_lite"])

    assert result.exit_code == 1
    assert "Language ch_lite not supported" in result.output


def test_vlm_server_rejects_removed_sglang_engine() -> None:
    result = runner.invoke(app, ["vlm-server", "--engine", "sglang"])

    assert result.exit_code == 1
    assert "Unsupported engine 'sglang'" in result.output


def test_vlm_server_forwards_mlx_to_mlx_server(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(vlm_server, "_mlx_server_available", lambda: True)
    monkeypatch.setattr(mlx_vlm_server, "main", _fake_main)

    result = runner.invoke(
        app,
        ["vlm-server", "--engine", "mlx", "--model", "test-model", "--host", "127.0.0.1", "--port", "18080"],
    )

    assert result.exit_code == 0
    assert seen == {
        "args": ["--model", "test-model", "--host", "127.0.0.1", "--port", "18080"],
        "prog_name": "mineru-kit vlm-server",
        "standalone_mode": False,
    }


def test_vlm_server_auto_falls_back_to_mlx_when_non_mlx_engines_are_unavailable(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(vlm_server, "_mlx_server_available", lambda: True)
    monkeypatch.setattr(vlm_server, "_module_available", lambda _module_name: False)
    monkeypatch.setattr(mlx_vlm_server, "main", _fake_main)

    result = runner.invoke(app, ["vlm-server", "--host", "127.0.0.1", "--port", "18080"])

    assert result.exit_code == 0
    assert seen == {
        "args": ["--host", "127.0.0.1", "--port", "18080"],
        "prog_name": "mineru-kit vlm-server",
        "standalone_mode": False,
    }


def test_vlm_server_auto_treats_missing_mlx_as_unavailable(monkeypatch: Any) -> None:
    def _missing_spec(_module_name: str) -> None:
        raise ModuleNotFoundError("No module named 'mlx_vlm'")

    monkeypatch.setattr(vlm_server, "is_mac_os_version_supported", lambda: True)
    monkeypatch.setattr(vlm_server.importlib.util, "find_spec", _missing_spec)

    assert vlm_server._mlx_server_available() is False


def test_vlm_server_forwards_extra_args(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_vllm_main() -> None:
        seen["args"] = sys.argv[1:]

    fake_vllm_server = ModuleType("mineru.kit.vlm_server.vllm_server")
    fake_vllm_server.main = _fake_vllm_main
    monkeypatch.setattr(vlm_server, "_mlx_server_available", lambda: False)
    monkeypatch.setattr(vlm_server, "_module_available", lambda module_name: module_name == "vllm")
    monkeypatch.setitem(sys.modules, "mineru.kit.vlm_server.vllm_server", fake_vllm_server)

    result = runner.invoke(app, ["vlm-server", "--host", "0.0.0.0", "--port", "30000"])

    assert result.exit_code == 0
    assert seen == {"args": ["--host", "0.0.0.0", "--port", "30000"]}


def test_mlx_vlm_server_adapter_exposes_v1_chat_completions(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    class FakeChatRequest:
        def __init__(self, **payload: Any) -> None:
            self.payload = payload
            self.model = payload.get("model")

        @classmethod
        def model_validate(cls, payload: dict[str, Any]) -> "FakeChatRequest":
            return cls(**payload)

    async def _fake_chat_completions_endpoint(request: FakeChatRequest) -> dict[str, Any]:
        seen["model"] = request.model
        return {"model": request.model}

    fake_mlx_server = SimpleNamespace(
        ChatRequest=FakeChatRequest,
        apply_chat_template=_fake_apply_chat_template,
        chat_completions_endpoint=_fake_chat_completions_endpoint,
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(server=fake_mlx_server))
    monkeypatch.setitem(sys.modules, "mlx_vlm.server", fake_mlx_server)

    client = TestClient(mlx_vlm_server.create_app(default_model="test-model"))
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 200
    assert response.json() == {"model": "test-model"}
    assert seen == {"model": "test-model"}


def test_mlx_vlm_server_adapter_exposes_single_v1_model(monkeypatch: Any) -> None:
    fake_mlx_server = SimpleNamespace(apply_chat_template=_fake_apply_chat_template)
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(server=fake_mlx_server))
    monkeypatch.setitem(sys.modules, "mlx_vlm.server", fake_mlx_server)

    client = TestClient(mlx_vlm_server.create_app(default_model="test-model"))
    response = client.get("/v1/models")
    body = response.json()

    assert response.status_code == 200
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "test-model"
    assert body["data"][0]["object"] == "model"
    assert isinstance(body["data"][0]["created"], int)


def test_mlx_vlm_server_adapter_defaults_to_mineru_vlm_model(monkeypatch: Any) -> None:
    fake_mlx_server = SimpleNamespace(
        DEFAULT_MODEL_PATH="mlx-community/nanoLLaVA-1.5-8bit",
        apply_chat_template=_fake_apply_chat_template,
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(server=fake_mlx_server))
    monkeypatch.setitem(sys.modules, "mlx_vlm.server", fake_mlx_server)
    monkeypatch.setattr(mlx_vlm_server, "_default_model_id", lambda configured_model: "/models/mineru-vlm")

    client = TestClient(mlx_vlm_server.create_app())
    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "/models/mineru-vlm"


def test_mlx_vlm_server_adapter_uses_mineru_mlx_compat_loader(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _raw_load() -> None:
        raise AssertionError("raw mlx-vlm load should be replaced")

    def _compat_load(path_or_hf_repo: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        seen["path_or_hf_repo"] = path_or_hf_repo
        seen["kwargs"] = kwargs
        return path_or_hf_repo, kwargs

    fake_mlx_server = SimpleNamespace(load=_raw_load, apply_chat_template=_fake_apply_chat_template)
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(server=fake_mlx_server))
    monkeypatch.setitem(sys.modules, "mlx_vlm.server", fake_mlx_server)
    monkeypatch.setattr(mlx_vlm_server, "load_mlx_model", _compat_load)

    mlx_vlm_server.create_app(default_model="test-model")
    result = fake_mlx_server.load("model-path", "adapter-path", trust_remote_code=True)

    assert result == ("model-path", {"adapter_path": "adapter-path", "trust_remote_code": True})
    assert seen == {
        "path_or_hf_repo": "model-path",
        "kwargs": {"adapter_path": "adapter-path", "trust_remote_code": True},
    }


def test_mlx_vlm_server_adapter_strips_data_urls_before_chat_template() -> None:
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "Read this page."},
            ],
        }
    ]

    sanitized = mlx_vlm_server._sanitize_chat_template_prompt(prompt)

    assert sanitized == [{"role": "user", "content": "Read this page."}]


def test_mlx_vlm_server_adapter_patches_chat_template_sanitizer(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _apply_chat_template(_processor: Any, _config: Any, prompt: Any, *_args: Any, **_kwargs: Any) -> Any:
        seen["prompt"] = prompt
        return "formatted"

    fake_mlx_server = SimpleNamespace(
        apply_chat_template=_apply_chat_template,
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(server=fake_mlx_server))
    monkeypatch.setitem(sys.modules, "mlx_vlm.server", fake_mlx_server)

    mlx_vlm_server.create_app(default_model="test-model")
    result = fake_mlx_server.apply_chat_template(
        object(),
        {"model_type": "qwen2_5_vl"},
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "Read this page."},
                ],
            }
        ],
    )

    assert result == "formatted"
    assert seen["prompt"] == [{"role": "user", "content": "Read this page."}]


def test_router_uses_explicit_v1_worker_options_and_rejects_legacy_args(monkeypatch: Any) -> None:
    """校验 Router 直接创建新应用，并拒绝旧参数和未知参数透传。"""
    from mineru.kit.router import cli as router_cli

    seen: dict[str, Any] = {}

    def _fake_run(application: Any, *, host: str, port: int, reload: bool) -> None:
        """记录 Router 交给 uvicorn 的新应用与显式配置。"""
        seen["settings"] = application.state.settings
        seen["host"] = host
        seen["port"] = port
        seen["reload"] = reload

    monkeypatch.setattr(router_cli.uvicorn, "run", _fake_run)

    result = runner.invoke(
        app,
        [
            "router",
            "--host",
            "0.0.0.0",
            "--port",
            "8002",
            "--upstream-url",
            "http://mineru-api:8000",
            "--local-gpus",
            "none",
            "--worker-host",
            "127.0.0.1",
            "--worker-tier",
            "basic",
            "--worker-concurrency",
            "2",
            "--preload-models",
        ],
    )

    assert result.exit_code == 0
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 8002
    assert seen["reload"] is False
    assert seen["settings"].upstream_urls == ("http://mineru-api:8000",)
    assert seen["settings"].local_gpus == "none"
    assert seen["settings"].worker_tier == "basic"
    assert seen["settings"].worker_concurrency == 2
    assert seen["settings"].preload_models is True

    legacy_result = runner.invoke(app, ["router", "--allow-public-http-client"])
    unknown_result = runner.invoke(app, ["router", "--gpu-memory-utilization", "0.5"])
    assert legacy_result.exit_code != 0
    assert unknown_result.exit_code != 0


def test_router_reload_uses_environment_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验 reload 模式通过 factory 重建应用，不在 import 阶段创建连接。"""
    from mineru.kit.router import cli as router_cli

    seen: dict[str, Any] = {}

    def _fake_run(target: str, **kwargs: Any) -> None:
        """记录 Uvicorn reload 使用的 factory import string。"""
        seen["target"] = target
        seen.update(kwargs)

    monkeypatch.setattr(router_cli.uvicorn, "run", _fake_run)

    result = runner.invoke(app, ["router", "--reload", "--local-gpus", "none"])

    assert result.exit_code == 0
    assert seen["target"] == "mineru.kit.router.app:create_app_from_env"
    assert seen["reload"] is True
    assert seen["factory"] is True


def test_parse_rejects_file_output_for_directory_input(tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "a.pdf").write_bytes(b"%PDF-1.7\n")

    result = runner.invoke(app, ["parse", str(source_dir), "-o", str(tmp_path / "out.md")])

    assert result.exit_code == 1
    assert "directory path" in " ".join(result.output.split())


def test_parse_single_file_markdown(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def markdown(self) -> str:
            """返回用于单文件 markdown 输出的测试内容。"""
            return "# demo\n"

        def to_json(self) -> str:
            """保留旧 fake 接口，避免无关测试关注 JSON 输出细节。"""
            return '{"pages":[]}'

        def images(self) -> dict[str, bytes]:
            """当前 markdown 无图片 sidecar 时返回空图片集合。"""
            return {}

        def save(self, writer: Any) -> None:
            """模拟 zip 输出所需的完整保存接口。"""
            writer.write_string("markdown.md", self.markdown())
            writer.write_string("middle_json.json", self.to_json())

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(app, ["parse", str(source), "-o", str(output)])

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8") == "# demo\n"


def test_parse_uses_backend_alias_only_to_resolve_tier(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    source.write_bytes(b"%PDF-1.7\n")
    seen: dict[str, Any] = {}

    class _Result:
        def markdown(self) -> str:
            """返回用于验证 parse 参数透传的测试内容。"""
            return "# demo\n"

        def to_json(self) -> str:
            """保留 zip 输出所需接口。"""
            return '{"pages":[]}'

        def images(self) -> dict[str, bytes]:
            """本用例不关注图片 sidecar。"""
            return {}

        def save(self, writer: Any) -> None:
            """模拟 zip 输出所需的完整保存接口。"""
            writer.write_string("markdown.md", self.markdown())
            writer.write_string("middle_json.json", self.to_json())

    def _fake_local_parse(*args: Any, **kwargs: Any) -> _Result:
        """记录 mineru-kit parse 透传给 parser 的运行参数。"""
        seen.update(kwargs)
        return _Result()

    monkeypatch.setattr(parse, "local_parse", _fake_local_parse)

    result = runner.invoke(
        app,
        ["parse", str(source), "-o", str(output), "--backend", "hybrid-auto-engine", "--tier", "standard"],
    )

    assert result.exit_code == 0
    assert seen["tier"] == "standard"
    assert "backend" not in seen
    assert "effort" not in seen


def test_parse_rejects_single_office_input_with_quality_tier(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.docx"
    output = tmp_path / "out.md"
    source.write_bytes(b"docx")

    def _fake_local_parse(*args: Any, **kwargs: Any) -> ParseResult:
        pytest.fail("single lightweight input with quality tier should fail before parsing")

    monkeypatch.setattr(parse, "local_parse", _fake_local_parse)

    result = runner.invoke(app, ["parse", str(source), "-o", str(output), "--tier", "standard"])

    assert result.exit_code == 1
    assert "Tier 'standard' is only supported for PDF and image files" in " ".join(result.output.split())


def test_parse_batch_normalizes_office_quality_tier_to_flash(monkeypatch: Any, tmp_path: Path) -> None:
    pdf = tmp_path / "demo.pdf"
    html = tmp_path / "page.html"
    output = tmp_path / "out"
    pdf.write_bytes(b"%PDF-1.7\n")
    html.write_text("<p>content</p>", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    class _Result:
        def markdown(self) -> str:
            return "# demo\n"

        def images(self) -> dict[str, bytes]:
            return {}

        def to_json(self) -> str:
            return '{"pages":[]}'

        def save(self, writer: Any) -> None:
            writer.write_string("markdown.md", self.markdown())
            writer.write_string("middle_json.json", self.to_json())

    def _fake_local_parse(path: Path, **kwargs: Any) -> _Result:
        calls.append({"path": path, **kwargs})
        return _Result()

    monkeypatch.setattr(parse, "local_parse", _fake_local_parse)

    result = runner.invoke(app, ["parse", str(pdf), str(html), "-o", str(output), "--tier", "standard"])

    assert result.exit_code == 0
    assert [(call["path"].name, call["tier"]) for call in calls] == [
        ("demo.pdf", "standard"),
        ("page.html", "flash"),
    ]
    assert all("backend" not in call for call in calls)


def test_parse_remote_requests_image_cache(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    source.write_bytes(b"%PDF-1.7\n")
    seen: dict[str, Any] = {}

    class _Result:
        def markdown(self) -> str:
            return "# demo\n"

        def images(self) -> dict[str, bytes]:
            return {}

    class _FakeApiParser:
        def __init__(
            self,
            *,
            api_url: str,
            api_key: str | None,
            tier: str | None,
            include_images: bool,
        ) -> None:
            seen["api_url"] = api_url
            seen["api_key"] = api_key
            seen["tier"] = tier
            seen["include_images"] = include_images

        def parse(self, path: Path, *, page_range: str) -> _Result:
            seen["path"] = path
            seen["page_range"] = page_range
            return _Result()

    monkeypatch.setattr(parse, "MinerUApiParser", _FakeApiParser)

    result = runner.invoke(
        app,
        [
            "parse",
            str(source),
            "-o",
            str(output),
            "--remote-url",
            "http://localhost:8000/api",
            "--api-key",
            "test-key",
            "--tier",
            "standard",
            "--pages",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8") == "# demo\n"
    assert seen == {
        "api_url": "http://localhost:8000/api",
        "api_key": "test-key",
        "tier": "standard",
        "include_images": True,
        "path": source,
        "page_range": "1",
    }


@pytest.mark.parametrize("language", ["japan", "ch_lite"])
def test_parse_rejects_removed_language_option(language: str, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    result = runner.invoke(app, ["parse", str(source), "-o", str(tmp_path / "out.md"), "--language", language])

    assert result.exit_code == 2
    assert "No such option: --language" in " ".join(result.output.split())


def test_parse_uses_flash_backend_to_resolve_flash_tier(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    source.write_bytes(b"%PDF-1.7\n")
    seen: dict[str, Any] = {}

    class _Result:
        def markdown(self) -> str:
            """返回用于验证 flash backend 透传的测试内容。"""
            return "# demo\n"

        def to_json(self) -> str:
            """保留 zip 输出所需接口。"""
            return '{"pages":[]}'

        def images(self) -> dict[str, bytes]:
            """本用例不关注图片 sidecar。"""
            return {}

        def save(self, writer: Any) -> None:
            """模拟 zip 输出所需的完整保存接口。"""
            writer.write_string("markdown.md", self.markdown())
            writer.write_string("middle_json.json", self.to_json())

    def _fake_local_parse(*args: Any, **kwargs: Any) -> _Result:
        """记录 mineru-kit parse 透传给 parser 的 flash backend 参数。"""
        seen.update(kwargs)
        return _Result()

    monkeypatch.setattr(parse, "local_parse", _fake_local_parse)

    result = runner.invoke(app, ["parse", str(source), "-o", str(output), "--backend", "flash"])

    assert result.exit_code == 0
    assert seen["tier"] == "flash"
    assert "backend" not in seen
    assert output.read_text(encoding="utf-8") == "# demo\n"


def test_parse_output_replaces_surrogate_chars(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def markdown(self) -> str:
            """返回包含孤立 surrogate 的 markdown 内容。"""
            return "before \ud83d after\n"

        def to_json(self) -> str:
            """保留旧 fake 接口，避免无关测试关注 JSON 输出细节。"""
            return '{"pages":[]}'

        def images(self) -> dict[str, bytes]:
            """当前 markdown 无图片 sidecar 时返回空图片集合。"""
            return {}

        def save(self, writer: Any) -> None:
            """模拟 zip 输出所需的完整保存接口。"""
            writer.write_string("markdown.md", self.markdown())
            writer.write_string("middle_json.json", self.to_json())

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(app, ["parse", str(source), "-o", str(output)])

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8") == "before ? after\n"


def test_models_download_tier_basic_light(monkeypatch: Any) -> None:
    captured: list[str] = []

    def fake_download_model_repo(repo: Any, *, source: str | None = None, local_as_auto: bool = False) -> Path:
        captured.append(repo.name)
        return Path("/tmp/models") / repo.local_name

    monkeypatch.setattr(models, "download_model_repo", fake_download_model_repo)

    result = runner.invoke(app, ["models", "download", "--tier", "basic", "--stack", "light"])

    assert result.exit_code == 0
    assert captured == [
        "PP-DocLayoutV2_onnx",
        "PP-OCRv6_small_det_onnx",
        "PP-OCRv6_small_rec_onnx",
        "PP-FormulaNet_plus-M_onnx",
    ]
    assert "Downloaded models for tier basic" in result.output


def test_models_download_tier_standard_light(monkeypatch: Any) -> None:
    captured: list[str] = []

    def fake_download_model_repo(repo: Any, *, source: str | None = None, local_as_auto: bool = False) -> Path:
        captured.append(repo.name)
        return Path("/tmp/models") / repo.local_name

    monkeypatch.setattr(models, "download_model_repo", fake_download_model_repo)

    result = runner.invoke(app, ["models", "download", "--tier", "standard", "--stack", "light"])

    assert result.exit_code == 0
    assert captured == [
        "PP-DocLayoutV2_onnx",
        "PP-OCRv6_small_det_onnx",
        "PP-OCRv6_small_rec_onnx",
        "PP-FormulaNet_plus-M_onnx",
        "MinerU2.5-Pro-2605-1.2B-GGUF",
    ]


def test_models_download_rejects_invalid_stack() -> None:
    result = runner.invoke(app, ["models", "download", "--tier", "basic", "--stack", "torch"])

    assert result.exit_code == 1
    assert "Unsupported stack 'torch'" in " ".join(result.output.split())


def test_models_download_repo_ignores_stack(monkeypatch: Any) -> None:
    """传具体 repo 名时 --stack 应被忽略，repo 自身的 stack 字段决定行为。"""
    captured: dict[str, Any] = {}

    def fake_download_model_repo(repo: Any, *, source: str | None = None, local_as_auto: bool = False) -> Path:
        captured["repo"] = repo.name
        captured["stack"] = repo.stack
        return Path("/tmp/models") / repo.local_name

    monkeypatch.setattr(models, "download_model_repo", fake_download_model_repo)

    result = runner.invoke(app, ["models", "download", "PP-DocLayoutV2_onnx", "--stack", "full"])

    assert result.exit_code == 0
    assert captured["repo"] == "PP-DocLayoutV2_onnx"
    assert captured["stack"] == "light"
    assert "Downloaded models for PP-DocLayoutV2_onnx" in result.output


def test_models_show_displays_stack_fields(tmp_path: Path, monkeypatch: Any) -> None:
    base_dir = tmp_path / "models"
    monkeypatch.setattr(models.config.model, "base_dir", str(base_dir))
    monkeypatch.setattr(models.config.model, "stack", "full")

    result = runner.invoke(app, ["models", "show"])

    assert result.exit_code == 0
    assert "model.stack: full" in result.output
    assert "Effective stack: full" in result.output
    assert "[stack=full]" in result.output
    assert "[stack=light]" in result.output


def test_models_show_with_light_stack_filter(tmp_path: Path, monkeypatch: Any) -> None:
    """--stack light 时 effective stack 解析为 light，tiers 部分按 light 显示。"""
    base_dir = tmp_path / "models"
    monkeypatch.setattr(models.config.model, "base_dir", str(base_dir))
    monkeypatch.setattr(models.config.model, "stack", "full")

    result = runner.invoke(app, ["models", "show", "--stack", "light"])

    assert result.exit_code == 0
    assert "Effective stack: light" in result.output
    assert "PP-DocLayoutV2_onnx: " in result.output
    assert "PDF-Extract-Kit-1.0: " in result.output


def test_models_show_rejects_invalid_stack() -> None:
    result = runner.invoke(app, ["models", "show", "--stack", "torch"])

    assert result.exit_code == 1
    assert "Unsupported stack 'torch'" in " ".join(result.output.split())


def test_models_verify_filters_by_effective_stack_full(tmp_path: Path, monkeypatch: Any) -> None:
    """config.model.stack=full 时，verify 默认只验证 full repos。"""
    base_dir = tmp_path / "models"
    monkeypatch.setattr(models.config.model, "base_dir", str(base_dir))
    monkeypatch.setattr(models.config.model, "stack", "full")

    # 仅把 full repos 设为 ready，light repos 不创建
    for repo in models.MODEL_REPOS:
        if repo.stack == "full":
            if repo.download_mode == "full":
                repo.local_dir().mkdir(parents=True, exist_ok=True)
                (repo.local_dir() / MODEL_COMPLETE_MARKER).touch()
                continue
            for model_path in repo.required_paths():
                target = repo.local_dir() / model_path.relative_path
                if Path(model_path.relative_path).suffix:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("x", encoding="utf-8")
                else:
                    target.mkdir(parents=True, exist_ok=True)
                    (target / MODEL_COMPLETE_MARKER).touch()

    result = runner.invoke(app, ["models", "verify"])

    assert result.exit_code == 0
    assert "PDF-Extract-Kit-1.0: ok" in result.output
    assert "MinerU2.5-Pro-2605-1.2B: ok" in result.output
    assert "PP-DocLayoutV2_onnx" not in result.output


def test_models_verify_with_light_stack(tmp_path: Path, monkeypatch: Any) -> None:
    """--stack light 时，verify 只验证 light repos。"""
    base_dir = tmp_path / "models"
    monkeypatch.setattr(models.config.model, "base_dir", str(base_dir))

    result = runner.invoke(app, ["models", "verify", "--stack", "light"])

    assert result.exit_code == 1  # light repos 未准备，应失败
    assert "PP-DocLayoutV2_onnx: missing key paths" in " ".join(result.output.split())
    assert "PDF-Extract-Kit-1.0" not in result.output


def test_models_verify_repo_ignores_stack(tmp_path: Path, monkeypatch: Any) -> None:
    """传具体 repo 名时 --stack 被忽略。"""
    base_dir = tmp_path / "models"
    monkeypatch.setattr(models.config.model, "base_dir", str(base_dir))

    # PP-DocLayoutV2_onnx 是 download_mode=full，只需创建 marker 文件
    repo = next(r for r in models.MODEL_REPOS if r.name == "PP-DocLayoutV2_onnx")
    repo.local_dir().mkdir(parents=True, exist_ok=True)
    (repo.local_dir() / MODEL_COMPLETE_MARKER).touch()

    result = runner.invoke(app, ["models", "verify", "PP-DocLayoutV2_onnx", "--stack", "full"])

    assert result.exit_code == 0
    assert "PP-DocLayoutV2_onnx: ok" in result.output


def test_models_verify_rejects_invalid_stack() -> None:
    result = runner.invoke(app, ["models", "verify", "--stack", "torch"])

    assert result.exit_code == 1
    assert "Unsupported stack 'torch'" in " ".join(result.output.split())
