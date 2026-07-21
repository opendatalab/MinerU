from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import click
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from mineru.kit.commands import api_server, models, parse, router, vlm_server
from mineru.kit.main import app
from mineru.kit.vlm_server import mlx_vlm_server
from mineru.parser.base import ParseResult
from mineru.types import Block, BlockType, ContentType, Line, PageInfo, Span
from mineru.utils.image_payload import ImagePayloadCache
from mineru.utils.model_registry import MODEL_COMPLETE_MARKER

runner = CliRunner()

_REMOVED_DISABLE_TABLE_OPTION = "--disable-" + "table"
_REMOVED_DISABLE_FORMULA_OPTION = "--disable-" + "formula"
_REMOVED_TABLE_ENABLE_PARAM = "table" + "_enable"
_REMOVED_FORMULA_ENABLE_PARAM = "formula" + "_enable"
_REMOVED_INLINE_FORMULA_PARAM = "inline_" + _REMOVED_FORMULA_ENABLE_PARAM


def _assert_unsafe_sidecar_error(output: str) -> None:
    """归一化 Typer/Click 自动换行后的错误输出，再匹配 sidecar 安全错误。"""
    assert "Unsafe image sidecar path" in " ".join(output.split())


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


def test_kit_root_help_hides_typer_completion_options() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--install-completion" not in result.output
    assert "--show-completion" not in result.output


def test_kit_root_show_completion_is_not_a_supported_option() -> None:
    result = runner.invoke(app, ["--show-completion"])

    assert result.exit_code != 0
    assert "No such option" in result.output


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


def test_kit_vlm_server_import_does_not_import_legacy_vlm_server() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys


class BlockLegacyVlmServerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mineru.cli_old.vlm_server":
            raise ModuleNotFoundError("blocked legacy vlm server import")
        return None


sys.meta_path.insert(0, BlockLegacyVlmServerFinder())
import mineru.kit.commands.vlm_server
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


def test_cli_old_router_import_supports_upstream_only_without_local_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc


class BlockLocalPipelineFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"mineru.cli_old.common", "mineru.cli_old.vlm_preload"}:
            raise ModuleNotFoundError(f"blocked local dependency import: {fullname}")
        if fullname.startswith("mineru.backend.office."):
            raise ModuleNotFoundError(f"blocked local dependency import: {fullname}")
        return None


import sys
sys.meta_path.insert(0, BlockLocalPipelineFinder())

from mineru.cli_old.router import RouterSettings, WorkerPool, create_app

settings = RouterSettings(upstream_urls=("http://127.0.0.1:8000",), local_gpus="none")
create_app(settings)
pool = WorkerPool(settings, object())
servers = pool.servers
assert len(servers) == 1
assert servers[0].source == "remote"
assert servers[0].local_server is None
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


def test_router_upstream_only_worker_pool_builds_remote_server() -> None:
    from mineru.cli_old.router import RouterSettings, WorkerPool

    settings = RouterSettings(upstream_urls=("http://mineru-api:8000",), local_gpus="none")
    pool = WorkerPool(settings, object())

    assert [(server.server_id, server.source, server.base_url, server.local_server) for server in pool.servers] == [
        ("remote-1", "remote", "http://mineru-api:8000", None),
    ]


def test_router_startup_with_no_servers_fails_health_check() -> None:
    from mineru.cli_old.router import RouterSettings, create_app, startup_router_state

    app = create_app(RouterSettings(upstream_urls=(), local_gpus="none"))

    async def _startup() -> None:
        await startup_router_state(app, RouterSettings(upstream_urls=(), local_gpus="none"))

    with pytest.raises(RuntimeError, match="No healthy upstream MinerU API servers are available"):
        asyncio.run(_startup())


def test_upload_filename_helper_import_boundary_is_explicit() -> None:
    """校验上传文件名 helper 只由需要的入口直接导入，避免 common.py 继续承担兼容转发。"""
    repo_root = Path(__file__).resolve().parents[2]
    common_tree = ast.parse((repo_root / "mineru/cli_old/common.py").read_text(encoding="utf-8"))
    fast_api_tree = ast.parse((repo_root / "mineru/cli_old/fast_api.py").read_text(encoding="utf-8"))

    common_imports = {
        alias.name
        for node in ast.walk(common_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "upload_utils"
        for alias in node.names
    }
    fast_api_common_imports = {
        alias.name
        for node in ast.walk(fast_api_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "mineru.cli_old.common"
        for alias in node.names
    }
    fast_api_upload_imports = {
        alias.name
        for node in ast.walk(fast_api_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "mineru.cli_old.upload_utils"
        for alias in node.names
    }

    assert "normalize_upload_filename" not in common_imports
    assert "normalize_upload_filename" not in fast_api_common_imports
    assert "normalize_upload_filename" in fast_api_upload_imports


def test_models_download_tier_basic(monkeypatch: Any) -> None:
    captured: list[str] = []

    def fake_download_model_repo(repo: Any, *, source: str | None = None, local_as_auto: bool = False) -> Path:
        captured.append(repo.name)
        return Path("/tmp/models") / repo.local_name

    monkeypatch.setattr(models, "download_model_repo", fake_download_model_repo)

    result = runner.invoke(app, ["models", "download", "--tier", "basic"])

    assert result.exit_code == 0
    assert captured == ["PDF-Extract-Kit-1.0"]
    assert "Downloaded models for tier basic" in result.output


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


def test_kit_commands_do_not_expose_formula_table_switches() -> None:
    """校验 mineru-kit 公开命令不再暴露公式/表格识别开关。"""
    parse_help = runner.invoke(app, ["parse", "--help"])
    api_server_help = runner.invoke(app, ["api-server", "--help"])

    assert parse_help.exit_code == 0
    assert api_server_help.exit_code == 0
    for output in (parse_help.output, api_server_help.output):
        assert _REMOVED_DISABLE_TABLE_OPTION not in output
        assert _REMOVED_DISABLE_FORMULA_OPTION not in output


def test_cli_old_api_form_builders_remove_formula_table_fields() -> None:
    """校验旧 API client 表单构造不再声明或发送公式/表格开关。"""
    from mineru.cli_old import api_client as old_api_client
    from mineru.cli_old import client as old_client

    for target in (old_api_client.build_parse_request_form_data, old_client.build_request_form_data):
        parameters = inspect.signature(target).parameters
        assert _REMOVED_FORMULA_ENABLE_PARAM not in parameters
        assert _REMOVED_TABLE_ENABLE_PARAM not in parameters

    data = old_api_client.build_parse_request_form_data(
        lang_list=["ch"],
        backend="hybrid-engine",
        parse_method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None,
        image_analysis=True,
        effort="medium",
        return_md=True,
        return_middle_json=True,
        return_model_output=True,
        return_content_list=True,
        return_images=True,
        response_format_zip=False,
        return_original_file=False,
    )

    assert _REMOVED_FORMULA_ENABLE_PARAM not in data
    assert _REMOVED_TABLE_ENABLE_PARAM not in data


def test_cli_old_api_request_models_remove_formula_table_fields() -> None:
    """校验旧 FastAPI 表单参数对象不再保存公式/表格开关。"""
    from mineru.cli_old import api_request as old_api_request
    from mineru.cli_old import fast_api as old_fast_api

    for model in (old_api_request.ParseRequestOptions, old_fast_api.ParseRequestOptions, old_fast_api.AsyncParseTask):
        annotations = getattr(model, "__annotations__", {})
        assert _REMOVED_FORMULA_ENABLE_PARAM not in annotations
        assert _REMOVED_TABLE_ENABLE_PARAM not in annotations


def test_api_server_forwards_repeated_tiers(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录 mineru-kit api-server 对多 tier 启动参数的原样转发。"""
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(api_server.parser_api_server.main, "main", _fake_main)

    result = runner.invoke(app, ["api-server", "--tier", "basic", "--tier", "advanced"])

    assert result.exit_code == 0
    assert seen["prog_name"] == "mineru-kit api-server"
    assert seen["standalone_mode"] is False
    assert [seen["args"][index + 1] for index, item in enumerate(seen["args"]) if item == "--tier"] == [
        "basic",
        "advanced",
    ]


def test_api_server_without_tier_lets_parser_api_apply_all_tier_default(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录 mineru-kit api-server 默认参数，确认不再强制单 standard tier。"""
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(api_server.parser_api_server.main, "main", _fake_main)

    result = runner.invoke(app, ["api-server", "--host", "0.0.0.0", "--port", "15985"])

    assert result.exit_code == 0
    assert seen["prog_name"] == "mineru-kit api-server"
    assert seen["standalone_mode"] is False
    assert "--tier" not in seen["args"]


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


def test_parse_forwards_backend_alias_and_effort(monkeypatch: Any, tmp_path: Path) -> None:
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
        ["parse", str(source), "-o", str(output), "--backend", "hybrid-auto-engine", "--effort", "high"],
    )

    assert result.exit_code == 0
    assert seen["backend"] == "hybrid-engine"
    assert seen["effort"] == "high"


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
    assert [(call["path"].name, call["tier"], call["backend"]) for call in calls] == [
        ("demo.pdf", "standard", "hybrid-engine"),
        ("page.html", "flash", "flash"),
    ]


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


def test_parse_normalizes_hidden_language_alias(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    source.write_bytes(b"%PDF-1.7\n")
    seen: dict[str, Any] = {}

    class _Result:
        def markdown(self) -> str:
            """返回用于验证 language 归一化的测试内容。"""
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
        """记录 mineru-kit parse 透传给 parser 的语言参数。"""
        seen.update(kwargs)
        return _Result()

    monkeypatch.setattr(parse, "local_parse", _fake_local_parse)

    result = runner.invoke(app, ["parse", str(source), "-o", str(output), "--language", "japan"])

    assert result.exit_code == 0
    assert seen["language"] == "ch"


def test_parse_rejects_removed_ch_lite_language(tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    result = runner.invoke(app, ["parse", str(source), "-o", str(tmp_path / "out.md"), "--language", "ch_lite"])

    assert result.exit_code == 1
    assert "Language ch_lite not supported" in result.output


def test_gradio_tier_selection_derives_v1_runtime() -> None:
    from mineru.cli_old import gradio_app

    assert gradio_app.resolve_gradio_runtime_options("basic").as_kwargs() == {
        "tier": "basic",
        "backend": "hybrid-engine",
        "effort": "medium",
    }
    assert gradio_app.resolve_gradio_runtime_options("standard").as_kwargs() == {
        "tier": "standard",
        "backend": "hybrid-engine",
        "effort": "high",
    }
    assert gradio_app.resolve_gradio_runtime_options("advanced").as_kwargs() == {
        "tier": "advanced",
        "backend": "hybrid-engine",
        "effort": "xhigh",
    }


def test_gradio_extracts_supported_tiers_from_v1_tiers_payload() -> None:
    from mineru.cli_old import gradio_app

    payload = {
        "data": [
            {"id": "flash"},
            {"id": "advanced"},
            {"id": "experimental"},
            {"id": "basic"},
            {"id": "standard"},
            {"id": "advanced"},
        ]
    }

    assert gradio_app.extract_v1_tier_choices(payload) == ("flash", "advanced", "basic", "standard")
    assert gradio_app.default_v1_gradio_tier(("flash", "advanced", "basic", "standard")) == "standard"


def test_gradio_rejects_v1_tiers_payload_without_supported_tiers() -> None:
    from mineru.cli_old import gradio_app

    with pytest.raises(click.ClickException, match="did not advertise any supported tier"):
        gradio_app.extract_v1_tier_choices({"data": [{"id": "experimental"}]})


def test_gradio_persists_v1_parse_result_for_preview_and_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mineru.cli_old import gradio_app

    image_cache = ImagePayloadCache()
    image_path = image_cache.register_bytes(b"figure-bytes", "png", image_path="figures/figure.png")
    image_body = Block(
        index=1,
        type=BlockType.IMAGE_BODY,
        bbox=(0, 0, 20, 20),
        lines=[Line(bbox=(0, 0, 20, 20), spans=[Span(type=ContentType.IMAGE, bbox=(0, 0, 20, 20), image_path=image_path)])],
    )
    image_block = Block(index=0, type=BlockType.IMAGE, bbox=(0, 0, 20, 20), blocks=[image_body])
    parse_result = ParseResult(
        pages=[PageInfo(page_idx=0, page_size=(100, 100), para_blocks=[image_block], _backend="hybrid")],
        _image_cache=image_cache,
        _model_output=[[{"raw": "model"}]],
    )
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    extract_root = tmp_path / "extract"
    archive_zip_path = tmp_path / "archive.zip"
    visualization_calls: list[Any] = []

    def fake_run_visualization_job(job: Any) -> Any:
        """模拟 layout 可视化生成，避免本用例依赖真实 PDF 绘制。"""
        visualization_calls.append(job)
        layout_path = job.parse_dir / f"{job.document_stem}_layout.pdf"
        layout_path.write_bytes(b"%PDF-1.7\n%layout\n")
        return SimpleNamespace(status="finished", message="generated", generated_files=(layout_path.name,))

    monkeypatch.setattr(gradio_app, "run_visualization_job", fake_run_visualization_job)

    output = gradio_app.persist_v1_gradio_result(
        parse_result=parse_result,
        file_path=str(source),
        extract_root=extract_root,
        archive_zip_path=archive_zip_path,
        backend="hybrid-engine",
        effort="medium",
        page_range="",
    )

    local_md_dir = extract_root / "demo"
    assert output.file_name == "demo"
    assert output.local_md_dir == local_md_dir
    assert output.preview_pdf_path == local_md_dir / "demo_layout.pdf"
    assert visualization_calls[0].document_stem == "demo"
    assert visualization_calls[0].parse_dir == local_md_dir
    assert visualization_calls[0].draw_span is False
    assert (local_md_dir / "demo.md").read_text(encoding="utf-8") == "![](images/figures/figure.png)"
    content_list = json.loads((local_md_dir / "demo_content_list.json").read_text(encoding="utf-8"))
    assert content_list[0]["img_path"] == "images/figures/figure.png"
    middle_json = json.loads((local_md_dir / "demo_middle.json").read_text(encoding="utf-8"))
    assert middle_json["_backend"] == "hybrid"
    assert middle_json["_version_name"]
    assert json.loads((local_md_dir / "demo_model_output.json").read_text(encoding="utf-8")) == [[{"raw": "model"}]]
    assert (local_md_dir / "images" / "figures" / "figure.png").read_bytes() == b"figure-bytes"
    assert (local_md_dir / "demo_origin.pdf").read_bytes() == b"%PDF-1.7\n"
    assert (local_md_dir / "demo_layout.pdf").read_bytes() == b"%PDF-1.7\n%layout\n"
    assert archive_zip_path.is_file()
    with zipfile.ZipFile(archive_zip_path) as archive:
        assert "demo_layout.pdf" in archive.namelist()
        assert "demo_model_output.json" in archive.namelist()
        assert json.loads(archive.read("demo_model_output.json").decode("utf-8")) == [[{"raw": "model"}]]


def test_gradio_persists_page_ranged_origin_pdf_for_v1_preview(tmp_path: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    from mineru.cli_old import gradio_app

    pdf_writer = PdfWriter()
    pdf_writer.add_blank_page(width=100, height=100)
    pdf_writer.add_blank_page(width=200, height=200)
    pdf_writer.add_blank_page(width=300, height=300)
    source_pdf = BytesIO()
    pdf_writer.write(source_pdf)

    parse_result = ParseResult(
        pages=[
            PageInfo(page_idx=0, page_size=(100, 100), _backend="hybrid"),
            PageInfo(page_idx=1, page_size=(200, 200), _backend="hybrid"),
        ],
    )
    source = tmp_path / "demo.pdf"
    source.write_bytes(source_pdf.getvalue())
    extract_root = tmp_path / "extract"
    archive_zip_path = tmp_path / "archive.zip"

    output = gradio_app.persist_v1_gradio_result(
        parse_result=parse_result,
        file_path=str(source),
        extract_root=extract_root,
        archive_zip_path=archive_zip_path,
        backend="hybrid-engine",
        effort="medium",
        page_range="1~2",
    )

    origin_pdf_path = output.local_md_dir / "demo_origin.pdf"
    origin_reader = PdfReader(str(origin_pdf_path))
    assert len(origin_reader.pages) == 2
    assert [float(page.cropbox[2]) for page in origin_reader.pages] == [100.0, 200.0]

    layout_pdf_path = output.local_md_dir / "demo_layout.pdf"
    layout_reader = PdfReader(str(layout_pdf_path))
    assert output.preview_pdf_path == layout_pdf_path
    assert len(layout_reader.pages) == 2

    with zipfile.ZipFile(archive_zip_path) as archive:
        assert "demo_layout.pdf" in archive.namelist()
        assert "demo_model_output.json" not in archive.namelist()
        zipped_origin_reader = PdfReader(BytesIO(archive.read("demo_origin.pdf")))
        zipped_layout_reader = PdfReader(BytesIO(archive.read("demo_layout.pdf")))
    assert len(zipped_origin_reader.pages) == 2
    assert len(zipped_layout_reader.pages) == 2
    assert [float(page.cropbox[2]) for page in zipped_origin_reader.pages] == [100.0, 200.0]


def test_gradio_persists_image_origin_as_pdf_for_v1_preview(tmp_path: Path) -> None:
    from PIL import Image
    from pypdf import PdfReader

    from mineru.cli_old import gradio_app

    source = tmp_path / "ref-merge.png"
    Image.new("RGB", (64, 48), "white").save(source)
    parse_result = ParseResult(
        pages=[PageInfo(page_idx=0, page_size=(64, 48), _backend="hybrid")],
    )
    extract_root = tmp_path / "extract"
    archive_zip_path = tmp_path / "archive.zip"

    output = gradio_app.persist_v1_gradio_result(
        parse_result=parse_result,
        file_path=str(source),
        extract_root=extract_root,
        archive_zip_path=archive_zip_path,
        backend="hybrid-engine",
        effort="medium",
        page_range="",
    )

    origin_pdf_path = output.local_md_dir / "ref-merge_origin.pdf"
    layout_pdf_path = output.local_md_dir / "ref-merge_layout.pdf"
    assert origin_pdf_path.read_bytes().startswith(b"%PDF")
    assert len(PdfReader(str(origin_pdf_path)).pages) == 1
    assert layout_pdf_path.is_file()
    assert output.preview_pdf_path == layout_pdf_path
    assert len(PdfReader(str(layout_pdf_path)).pages) == 1

    with zipfile.ZipFile(archive_zip_path) as archive:
        assert archive.read("ref-merge_origin.pdf").startswith(b"%PDF")
        assert "ref-merge_layout.pdf" in archive.namelist()
        assert len(PdfReader(BytesIO(archive.read("ref-merge_origin.pdf"))).pages) == 1
        assert len(PdfReader(BytesIO(archive.read("ref-merge_layout.pdf"))).pages) == 1


def test_gradio_v1_job_reuses_page_range_for_api_and_origin_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mineru.cli_old import gradio_app

    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    calls: dict[str, Any] = {}

    class _FakeParser:
        def __init__(self, *, api_url: str, tier: str, include_images: bool, include_model_output: bool) -> None:
            calls["parser_api_url"] = api_url
            calls["parser_tier"] = tier
            calls["include_images"] = include_images
            calls["include_model_output"] = include_model_output

        async def parse_async(self, file_path: str, *, page_range: str) -> ParseResult:
            calls["api_page_range"] = page_range
            return ParseResult(pages=[PageInfo(page_idx=0, page_size=(100, 100), _backend="hybrid")])

    async def _fake_server_health(_http_client: Any, api_url: str | None) -> Any:
        """模拟 v1 server 健康检查，只保留 Gradio 任务需要的字段。"""
        return SimpleNamespace(base_url=api_url or "http://127.0.0.1:30000/api", max_concurrent_requests=1)

    def _fake_persist(**kwargs: Any) -> Any:
        """记录本地持久化收到的 page_range，避免联动测试访问真实 PDFium。"""
        calls["persist_page_range"] = kwargs["page_range"]
        local_md_dir = tmp_path / "persisted"
        local_md_dir.mkdir()
        (local_md_dir / "demo.md").write_text("markdown", encoding="utf-8")
        (local_md_dir / "demo_content_list.json").write_text("[]", encoding="utf-8")
        archive_zip_path = tmp_path / "demo.zip"
        archive_zip_path.write_bytes(b"zip")
        return SimpleNamespace(
            file_name="demo",
            local_md_dir=local_md_dir,
            archive_zip_path=archive_zip_path,
            preview_pdf_path=local_md_dir / "demo_origin.pdf",
        )

    monkeypatch.setattr(gradio_app, "MinerUApiParser", _FakeParser)
    monkeypatch.setattr(gradio_app, "resolve_v1_server_health", _fake_server_health)
    monkeypatch.setattr(gradio_app, "persist_v1_gradio_result", _fake_persist)

    asyncio.run(gradio_app._run_to_markdown_job(str(source), end_pages=2, api_url="http://example.test/api"))

    assert calls["api_page_range"] == "1~2"
    assert calls["include_model_output"] is True
    assert calls["include_images"] is True
    assert calls["persist_page_range"] == "1~2"


def test_gradio_run_paths_are_absolute(tmp_path: Path) -> None:
    from mineru.cli_old import gradio_app

    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    default_run_root, default_extract_root, default_archive_zip_path = gradio_app.create_gradio_run_paths(str(source))
    assert default_run_root.is_absolute()
    assert default_extract_root.is_absolute()
    assert default_archive_zip_path.is_absolute()

    run_root, extract_root, archive_zip_path = gradio_app.create_gradio_run_paths(
        str(source),
        output_root=str(tmp_path / "output"),
    )

    assert run_root.is_absolute()
    assert extract_root.is_absolute()
    assert archive_zip_path.is_absolute()
    assert extract_root.parent == run_root
    assert archive_zip_path.parent == run_root


def test_gradio_frontend_uses_v1_only_tier_visibility() -> None:
    js_text = Path("mineru/resources/gradio_app.js").read_text(encoding="utf-8")

    assert ".mineru-remote-server-toggle" not in js_text
    assert ".mineru-advanced-popover" not in js_text
    assert "getUseRemoteServer" not in js_text
    assert "getBackendValue" not in js_text
    assert "getEffortValue" not in js_text
    assert ".mineru-backend-select" not in js_text
    assert ".mineru-hybrid-effort" not in js_text
    assert 'input[type="radio"]' not in js_text
    assert "mineru-show-client-options" not in js_text
    assert "mineru-show-image-analysis" not in js_text
    assert "mineru-show-ocr-language" not in js_text
    assert "mineru-hide-force-ocr" not in js_text


def test_gradio_header_model_and_paper_links_use_popovers() -> None:
    """校验 Gradio header 的模型和论文入口使用可悬浮展开的多链接菜单。"""
    header_text = Path("mineru/resources/gradio_header.html").read_text(encoding="utf-8")
    gradio_text = Path("mineru/cli_old/gradio_app.py").read_text(encoding="utf-8")

    assert 'class="link-block mineru-header-menu mineru-model-menu"' in header_text
    assert 'class="link-block mineru-header-menu mineru-paper-menu"' in header_text
    assert "mineru-header-popover mineru-model-popover" in header_text
    assert "mineru-header-popover mineru-paper-popover" in header_text
    assert "https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B" in header_text
    assert "https://modelscope.cn/models/OpenDataLab/MinerU2.5-Pro-2605-1.2B" in header_text
    assert "https://arxiv.org/abs/2409.18839" in header_text
    assert "https://arxiv.org/abs/2509.22186" in header_text
    assert "https://arxiv.org/abs/2604.04771" in header_text
    assert ".block.mineru-header-html" in header_text
    assert ".mineru-demo-header .external-link {\n    font-size: 14px !important;" in header_text
    assert 'gr.HTML(render_header_html(i18n), elem_classes=["mineru-header-html"])' in gradio_text

    for placeholder, translation_key in {
        "{{HEADER_MODEL_HUGGINGFACE_LINK}}": "header_model_huggingface_link",
        "{{HEADER_MODEL_MODELSCOPE_LINK}}": "header_model_modelscope_link",
        "{{HEADER_PAPER_MINERU_REPORT}}": "header_paper_mineru_report",
        "{{HEADER_PAPER_MINERU25_REPORT}}": "header_paper_mineru25_report",
        "{{HEADER_PAPER_MINERU25PRO_REPORT}}": "header_paper_mineru25pro_report",
    }.items():
        assert placeholder in header_text
        assert f'"{placeholder}": "{translation_key}"' in gradio_text
        assert f'"{translation_key}"' in gradio_text


def test_gradio_submit_inputs_are_v1_only() -> None:
    """校验 Gradio 公开控制面只保留 v1 API 支持的单次任务输入。"""
    gradio_text = Path("mineru/cli_old/gradio_app.py").read_text(encoding="utf-8")

    backend_block_idx = gradio_text.index('elem_classes=["mineru-backend-options-block"]')
    tier_idx = gradio_text.index("tier = gr.Dropdown(")
    max_pages_idx = gradio_text.index("max_pages = gr.Slider(")

    assert "backend = gr.Dropdown(" not in gradio_text
    assert "effort = gr.Dropdown(" not in gradio_text
    assert "effort = gr.Radio(" not in gradio_text
    assert "Hybrid effort" not in gradio_text
    assert "解析强度" not in gradio_text
    assert "use_remote_server = gr.Checkbox(" not in gradio_text
    assert "url = gr.Textbox(" not in gradio_text
    assert 'elem_classes=["mineru-client-options"]' not in gradio_text
    assert 'elem_classes=["mineru-advanced-popover"]' not in gradio_text
    assert "is_ocr = gr.Checkbox(" not in gradio_text
    assert _REMOVED_FORMULA_ENABLE_PARAM + " = gr.Checkbox(" not in gradio_text
    assert _REMOVED_TABLE_ENABLE_PARAM + " = gr.Checkbox(" not in gradio_text
    assert "image_analysis = gr.Checkbox(" not in gradio_text
    assert "language = gr.Dropdown(" not in gradio_text
    assert "use_remote_server=use_remote_server" not in gradio_text
    assert "tier=tier" in gradio_text
    assert "inputs=[input_file, max_pages, tier]" in gradio_text
    assert backend_block_idx < tier_idx < max_pages_idx


def test_parse_forwards_flash_backend(monkeypatch: Any, tmp_path: Path) -> None:
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
    assert seen["backend"] == "flash"
    assert output.read_text(encoding="utf-8") == "# demo\n"


def test_cli_old_legacy_vlm_branch_maps_to_hybrid_xhigh(monkeypatch: Any, tmp_path: Path) -> None:
    from mineru.cli_old import common

    seen: dict[str, Any] = {}

    monkeypatch.setattr(common, "_process_office_doc", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        common,
        "_prepare_pdf_inputs",
        lambda pdfs, start, end: [
            SimpleNamespace(pdf_bytes=pdf, retained_page_indices=None, broken_page_indices=None) for pdf in pdfs
        ],
    )
    monkeypatch.setattr(common, "ensure_backend_dependencies", lambda backend: None)
    monkeypatch.setattr(common, "get_vlm_engine", lambda inference_engine="auto", is_async=False: "vllm-engine")

    def _fake_process_hybrid(*args: Any, **kwargs: Any) -> None:
        """记录 legacy VLM 输入最终进入 Hybrid advanced 分支。"""
        seen["backend"] = args[3]
        seen["hybrid_backend"] = args[5]
        seen["kwargs"] = kwargs

    monkeypatch.setattr(common, "_process_hybrid", _fake_process_hybrid)

    common.do_parse(
        output_dir=str(tmp_path),
        pdf_file_names=["demo.pdf"],
        pdf_bytes_list=[b"%PDF-1.7\n"],
        p_lang_list=["ch"],
        backend="vlm-engine",
        effort="high",
    )

    assert seen["hybrid_backend"] == "vllm-engine"
    assert seen["kwargs"]["effort"] == "xhigh"
    assert seen["kwargs"]["image_analysis"] is True
    assert not hasattr(common, "_process_vlm")


def test_cli_old_hybrid_branch_keeps_effort(monkeypatch: Any, tmp_path: Path) -> None:
    from mineru.cli_old import common

    seen: dict[str, Any] = {}

    monkeypatch.setattr(common, "_process_office_doc", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        common,
        "_prepare_pdf_inputs",
        lambda pdfs, start, end: [
            SimpleNamespace(pdf_bytes=pdf, retained_page_indices=None, broken_page_indices=None) for pdf in pdfs
        ],
    )
    monkeypatch.setattr(common, "ensure_backend_dependencies", lambda backend: None)
    monkeypatch.setattr(common, "get_vlm_engine", lambda inference_engine="auto", is_async=False: "vllm-engine")

    def _fake_process_hybrid(*args: Any, **kwargs: Any) -> None:
        """记录 Hybrid 分支收到的 kwargs，确认 effort 仍传给 hybrid analyzer。"""
        seen["backend"] = args[5]
        seen["kwargs"] = kwargs

    monkeypatch.setattr(common, "_process_hybrid", _fake_process_hybrid)
    monkeypatch.setenv("MINERU_VLM_FORMULA_ENABLE", "sentinel-formula")
    monkeypatch.setenv("MINERU_VLM_TABLE_ENABLE", "sentinel-table")

    common.do_parse(
        output_dir=str(tmp_path),
        pdf_file_names=["demo.pdf"],
        pdf_bytes_list=[b"%PDF-1.7\n"],
        p_lang_list=["ch"],
        backend="hybrid-engine",
        effort="high",
    )

    assert seen["backend"] == "vllm-engine"
    assert seen["kwargs"]["effort"] == "high"
    assert os.environ["MINERU_VLM_FORMULA_ENABLE"] == "sentinel-formula"
    assert os.environ["MINERU_VLM_TABLE_ENABLE"] == "sentinel-table"


def test_cli_old_hybrid_medium_skips_vlm_engine_resolution(monkeypatch: Any, tmp_path: Path) -> None:
    from mineru.cli_old import common

    seen: dict[str, Any] = {}

    monkeypatch.setattr(common, "_process_office_doc", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        common,
        "_prepare_pdf_inputs",
        lambda pdfs, start, end: [
            SimpleNamespace(pdf_bytes=pdf, retained_page_indices=None, broken_page_indices=None) for pdf in pdfs
        ],
    )
    monkeypatch.setattr(common, "ensure_backend_dependencies", lambda backend: None)

    def fail_get_vlm_engine(*_args: Any, **_kwargs: Any) -> str:
        """Hybrid basic 不应触发 VLM engine 解析。"""
        raise AssertionError("medium effort should not resolve VLM engine")

    def _fake_process_hybrid(*args: Any, **kwargs: Any) -> None:
        """记录 Hybrid basic 仍进入 Hybrid 处理分支。"""
        seen["backend"] = args[5]
        seen["langs"] = list(args[3])
        seen["pdf_count"] = len(args[2])
        seen["kwargs"] = kwargs

    monkeypatch.setattr(common, "get_vlm_engine", fail_get_vlm_engine)
    monkeypatch.setattr(common, "_process_hybrid", _fake_process_hybrid)

    common.do_parse(
        output_dir=str(tmp_path),
        pdf_file_names=["a.pdf", "b.pdf"],
        pdf_bytes_list=[b"%PDF-1.7\n", b"%PDF-1.7\n"],
        p_lang_list=["en", "en"],
        backend="hybrid-engine",
        effort="medium",
    )

    assert seen["backend"] == "engine"
    assert seen["langs"] == ["en", "en"]
    assert seen["pdf_count"] == 2
    assert seen["kwargs"]["effort"] == "medium"


def test_process_hybrid_medium_calls_analyzer_per_file(monkeypatch: Any, tmp_path: Path) -> None:
    from mineru.cli_old import common

    calls: list[bytes] = []
    languages: list[str] = []
    analyzer_kwargs: list[dict[str, Any]] = []
    outputs: list[tuple[str, str, str]] = []

    def fake_doc_analyze(pdf_bytes: bytes, **kwargs: Any) -> tuple[list[PageInfo], list[object], bool]:
        """记录每个文件独立进入 Hybrid basic analyzer。"""
        calls.append(pdf_bytes)
        languages.append(kwargs["language"])
        analyzer_kwargs.append(kwargs)
        return [PageInfo(page_idx=0, _backend="hybrid")], [], False

    monkeypatch.setattr(common, "_load_hybrid_analyze_entrypoint", lambda *_args, **_kwargs: fake_doc_analyze)
    monkeypatch.setattr(
        common,
        "prepare_env",
        lambda output_dir, pdf_file_name, method: (str(tmp_path / pdf_file_name / "images"), str(tmp_path / pdf_file_name)),
    )
    monkeypatch.setattr(common, "FileBasedDataWriter", lambda _path: object())
    monkeypatch.setattr(
        common,
        "_process_output",
        lambda middle_json, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir, *_args, **_kwargs: outputs.append(
            (pdf_file_name, local_md_dir, local_image_dir)
        ),
    )

    common._process_hybrid(
        output_dir=str(tmp_path),
        pdf_file_names=["a.pdf", "b.pdf"],
        pdf_bytes_list=[b"a", b"b"],
        h_lang_list=["en", "en"],
        parse_method="auto",
        backend="engine",
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode="mm_markdown",
        effort="medium",
    )

    assert calls == [b"a", b"b"]
    assert languages == ["en", "en"]
    assert all(_REMOVED_INLINE_FORMULA_PARAM not in kwargs for kwargs in analyzer_kwargs)
    assert [item[0] for item in outputs] == ["a.pdf", "b.pdf"]


def test_cli_old_async_legacy_vlm_branch_maps_to_hybrid_xhigh(monkeypatch: Any, tmp_path: Path) -> None:
    from mineru.cli_old import common

    seen: dict[str, Any] = {}

    monkeypatch.setattr(common, "_process_office_doc", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        common,
        "_prepare_pdf_inputs",
        lambda pdfs, start, end: [
            SimpleNamespace(pdf_bytes=pdf, retained_page_indices=None, broken_page_indices=None) for pdf in pdfs
        ],
    )
    monkeypatch.setattr(common, "ensure_backend_dependencies", lambda backend: None)
    monkeypatch.setattr(common, "get_vlm_engine", lambda inference_engine="auto", is_async=True: "vllm-async-engine")

    async def _fake_async_process_hybrid(*args: Any, **kwargs: Any) -> None:
        """记录异步 legacy VLM 输入最终进入 Hybrid advanced 分支。"""
        seen["backend"] = args[3]
        seen["hybrid_backend"] = args[5]
        seen["kwargs"] = kwargs

    monkeypatch.setattr(common, "_async_process_hybrid", _fake_async_process_hybrid)

    asyncio.run(
        common.aio_do_parse(
            output_dir=str(tmp_path),
            pdf_file_names=["demo.pdf"],
            pdf_bytes_list=[b"%PDF-1.7\n"],
            p_lang_list=["ch"],
            backend="vlm-engine",
            effort="high",
        )
    )

    assert seen["hybrid_backend"] == "vllm-async-engine"
    assert seen["kwargs"]["effort"] == "xhigh"
    assert seen["kwargs"]["image_analysis"] is True
    assert not hasattr(common, "_async_process_vlm")


def test_cli_old_async_hybrid_branch_keeps_effort(monkeypatch: Any, tmp_path: Path) -> None:
    from mineru.cli_old import common

    seen: dict[str, Any] = {}

    monkeypatch.setattr(common, "_process_office_doc", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        common,
        "_prepare_pdf_inputs",
        lambda pdfs, start, end: [
            SimpleNamespace(pdf_bytes=pdf, retained_page_indices=None, broken_page_indices=None) for pdf in pdfs
        ],
    )
    monkeypatch.setattr(common, "ensure_backend_dependencies", lambda backend: None)
    monkeypatch.setattr(common, "get_vlm_engine", lambda inference_engine="auto", is_async=True: "vllm-async-engine")

    async def _fake_async_process_hybrid(*args: Any, **kwargs: Any) -> None:
        """记录异步 Hybrid 分支收到的 kwargs，确认 effort 仍传给 hybrid analyzer。"""
        seen["backend"] = args[5]
        seen["kwargs"] = kwargs

    monkeypatch.setattr(common, "_async_process_hybrid", _fake_async_process_hybrid)
    monkeypatch.setenv("MINERU_VLM_FORMULA_ENABLE", "sentinel-formula")
    monkeypatch.setenv("MINERU_VLM_TABLE_ENABLE", "sentinel-table")

    asyncio.run(
        common.aio_do_parse(
            output_dir=str(tmp_path),
            pdf_file_names=["demo.pdf"],
            pdf_bytes_list=[b"%PDF-1.7\n"],
            p_lang_list=["ch"],
            backend="hybrid-engine",
            effort="high",
        )
    )

    assert seen["backend"] == "vllm-async-engine"
    assert seen["kwargs"]["effort"] == "high"
    assert os.environ["MINERU_VLM_FORMULA_ENABLE"] == "sentinel-formula"
    assert os.environ["MINERU_VLM_TABLE_ENABLE"] == "sentinel-table"


def test_parse_single_file_middle_json_writes_image_sidecars(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.json"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def to_json(self) -> str:
            """返回带图片引用的 public middle_json。"""
            return '{"pages":[{"para_blocks":[{"lines":[{"spans":[{"image_path":"figure.png"}]}]}]}]}'

        def images(self) -> dict[str, bytes]:
            """返回需要随 public middle_json 一起落盘的图片 sidecar。"""
            return {"figure.png": b"figure-bytes"}

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(
        app,
        ["parse", str(source), "-o", str(output), "--format", "middle_json"],
    )

    assert result.exit_code == 0
    payload = output.read_text(encoding="utf-8")
    assert '"image_path":"figure.png"' in payload
    assert "image_base64" not in payload
    assert (tmp_path / "figure.png").read_bytes() == b"figure-bytes"


def test_parse_single_file_middle_json_rejects_parent_sidecar_paths(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """拒绝 API 或解析结果提供的父目录逃逸 sidecar 路径。"""
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.json"
    escaped = tmp_path / "escape.png"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def to_json(self) -> str:
            """返回引用逃逸路径的 public middle_json。"""
            return '{"pages":[]}'

        def images(self) -> dict[str, bytes]:
            """模拟远端返回包含 .. 的图片 sidecar 路径。"""
            return {"../escape.png": b"escape-bytes"}

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(
        app,
        ["parse", str(source), "-o", str(output), "--format", "middle_json"],
    )

    assert result.exit_code == 1
    _assert_unsafe_sidecar_error(result.output)
    assert not escaped.exists()


def test_parse_single_file_middle_json_rejects_absolute_sidecar_paths(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """拒绝 API 或解析结果提供的绝对 sidecar 路径。"""
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.json"
    absolute = tmp_path / "absolute.png"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def to_json(self) -> str:
            """返回普通 public middle_json 内容，重点验证 sidecar 路径。"""
            return '{"pages":[]}'

        def images(self) -> dict[str, bytes]:
            """模拟远端返回绝对图片 sidecar 路径。"""
            return {str(absolute): b"absolute-bytes"}

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(
        app,
        ["parse", str(source), "-o", str(output), "--format", "middle_json"],
    )

    assert result.exit_code == 1
    _assert_unsafe_sidecar_error(result.output)
    assert not absolute.exists()


def test_parse_single_file_middle_json_rejects_windows_rooted_sidecar_paths(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """拒绝 Windows rooted 形式的 sidecar 路径，避免跨平台逃逸。"""
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.json"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def to_json(self) -> str:
            """返回普通 public middle_json 内容，重点验证 Windows 路径。"""
            return '{"pages":[]}'

        def images(self) -> dict[str, bytes]:
            """模拟远端返回 Windows rooted 图片 sidecar 路径。"""
            return {"\\escape.png": b"escape-bytes"}

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(
        app,
        ["parse", str(source), "-o", str(output), "--format", "middle_json"],
    )

    assert result.exit_code == 1
    _assert_unsafe_sidecar_error(result.output)
    assert not (tmp_path / "\\escape.png").exists()


def test_parse_single_file_markdown_writes_image_sidecars(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """markdown 单文件输出也要写出渲染结果引用的图片 sidecar。"""
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def markdown(self) -> str:
            """返回带相对图片引用的 markdown 内容。"""
            return "![](figure.png)\n"

        def images(self) -> dict[str, bytes]:
            """返回 markdown 引用的图片 sidecar。"""
            return {"figure.png": b"figure-bytes"}

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(app, ["parse", str(source), "-o", str(output)])

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8") == "![](figure.png)\n"
    assert (tmp_path / "figure.png").read_bytes() == b"figure-bytes"


def test_parse_single_file_markdown_rejects_unsafe_sidecar_paths(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """markdown 补写 sidecar 时同样不能允许路径逃逸输出目录。"""
    source = tmp_path / "demo.pdf"
    output = tmp_path / "out.md"
    escaped = tmp_path / "escape.png"
    source.write_bytes(b"%PDF-1.7\n")

    class _Result:
        def markdown(self) -> str:
            """返回引用逃逸路径的 markdown 内容。"""
            return "![](../escape.png)\n"

        def images(self) -> dict[str, bytes]:
            """模拟远端或解析结果返回逃逸图片路径。"""
            return {"../escape.png": b"escape-bytes"}

    monkeypatch.setattr(parse, "local_parse", lambda *args, **kwargs: _Result())

    result = runner.invoke(app, ["parse", str(source), "-o", str(output)])

    assert result.exit_code == 1
    _assert_unsafe_sidecar_error(result.output)
    assert not escaped.exists()


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
