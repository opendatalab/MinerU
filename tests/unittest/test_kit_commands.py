from __future__ import annotations

import ast
import asyncio
import json
import zipfile
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import click
import pytest
from typer.testing import CliRunner

from mineru.kit.commands import api_server, models, parse, router, vlm_server
from mineru.kit.main import app
from mineru.parser.base import ParseResult
from mineru.types import Block, BlockType, ContentType, Line, PageInfo, Span
from mineru.utils.image_payload import ImagePayloadCache

runner = CliRunner()


def _assert_unsafe_sidecar_error(output: str) -> None:
    """归一化 Typer/Click 自动换行后的错误输出，再匹配 sidecar 安全错误。"""
    assert "Unsafe image sidecar path" in " ".join(output.split())


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


def test_models_download_pipeline(monkeypatch: Any) -> None:
    monkeypatch.setattr(models, "_download_pipeline_models", lambda: "/tmp/pipeline")
    monkeypatch.setattr(models, "_download_vlm_models", lambda: "/tmp/vlm")
    monkeypatch.setattr(models, "_update_models_dir", lambda bundle, model_dir: Path(f"/tmp/{bundle}.json"))

    result = runner.invoke(app, ["models", "download", "pipeline"])

    assert result.exit_code == 0
    assert "Downloaded pipeline models" in result.output


def test_models_download_auto_source_resolves_before_download(monkeypatch: Any) -> None:
    captured: dict[str, str] = {}

    def fake_resolve_model_source(model_source: str | None = None, allow_auto: bool = False) -> str:
        assert model_source == "auto"
        assert allow_auto is True
        return "modelscope"

    def fake_update_models_dir(bundle: str, model_dir: str) -> Path:
        captured["bundle"] = bundle
        captured["model_dir"] = model_dir
        captured["effective_source"] = models.os.getenv(models.MODEL_SOURCE_ENV_VAR, "")
        return Path(f"/tmp/{bundle}.json")

    monkeypatch.delenv(models.MODEL_SOURCE_ENV_VAR, raising=False)
    monkeypatch.setattr(models, "resolve_model_source", fake_resolve_model_source, raising=False)
    monkeypatch.setattr(models, "_download_pipeline_models", lambda: "/tmp/pipeline")
    monkeypatch.setattr(models, "_update_models_dir", fake_update_models_dir)

    result = runner.invoke(app, ["models", "download", "pipeline", "--source", "auto"])

    assert result.exit_code == 0
    assert captured == {
        "bundle": "pipeline",
        "model_dir": "/tmp/pipeline",
        "effective_source": "modelscope",
    }
    assert "Downloaded pipeline models from modelscope" in result.output


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


def test_api_server_rejects_backend_and_effort_options() -> None:
    backend_result = runner.invoke(app, ["api-server", "--backend", "hybrid-engine"])
    effort_result = runner.invoke(app, ["api-server", "--effort", "high"])

    assert backend_result.exit_code != 0
    assert "--backend" in backend_result.output
    assert effort_result.exit_code != 0
    assert "--effort" in effort_result.output


def test_api_server_forwards_repeated_tiers(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录 mineru-kit api-server 对多 tier 启动参数的原样转发。"""
        seen["args"] = args
        seen["prog_name"] = prog_name
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(api_server.parser_api_server.main, "main", _fake_main)

    result = runner.invoke(app, ["api-server", "--tier", "medium", "--tier", "extra_high"])

    assert result.exit_code == 0
    assert seen["prog_name"] == "mineru-kit api-server"
    assert seen["standalone_mode"] is False
    assert [seen["args"][index + 1] for index, item in enumerate(seen["args"]) if item == "--tier"] == [
        "medium",
        "extra_high",
    ]


def test_api_server_without_tier_lets_parser_api_apply_all_tier_default(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def _fake_main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
        """记录 mineru-kit api-server 默认参数，确认不再强制单 high tier。"""
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

    assert gradio_app.resolve_gradio_runtime_options("medium").as_kwargs() == {
        "tier": "medium",
        "backend": "hybrid-engine",
        "effort": "medium",
    }
    assert gradio_app.resolve_gradio_runtime_options("high").as_kwargs() == {
        "tier": "high",
        "backend": "hybrid-engine",
        "effort": "high",
    }
    assert gradio_app.resolve_gradio_runtime_options("extra_high").as_kwargs() == {
        "tier": "extra_high",
        "backend": "hybrid-engine",
        "effort": "extra_high",
    }


def test_gradio_extracts_supported_tiers_from_v1_tiers_payload() -> None:
    from mineru.cli_old import gradio_app

    payload = {
        "data": [
            {"id": "flash"},
            {"id": "extra_high"},
            {"id": "experimental"},
            {"id": "medium"},
            {"id": "high"},
            {"id": "extra_high"},
        ]
    }

    assert gradio_app.extract_v1_tier_choices(payload) == ("flash", "extra_high", "medium", "high")
    assert gradio_app.default_v1_gradio_tier(("flash", "extra_high", "medium", "high")) == "high"


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
    assert (local_md_dir / "images" / "figures" / "figure.png").read_bytes() == b"figure-bytes"
    assert (local_md_dir / "demo_origin.pdf").read_bytes() == b"%PDF-1.7\n"
    assert (local_md_dir / "demo_layout.pdf").read_bytes() == b"%PDF-1.7\n%layout\n"
    assert archive_zip_path.is_file()
    with zipfile.ZipFile(archive_zip_path) as archive:
        assert "demo_layout.pdf" in archive.namelist()


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
        zipped_origin_reader = PdfReader(BytesIO(archive.read("demo_origin.pdf")))
        zipped_layout_reader = PdfReader(BytesIO(archive.read("demo_layout.pdf")))
    assert len(zipped_origin_reader.pages) == 2
    assert len(zipped_layout_reader.pages) == 2
    assert [float(page.cropbox[2]) for page in zipped_origin_reader.pages] == [100.0, 200.0]


def test_gradio_v1_job_reuses_page_range_for_api_and_origin_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mineru.cli_old import gradio_app

    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    calls: dict[str, Any] = {}

    class _FakeParser:
        def __init__(self, *, api_url: str, tier: str) -> None:
            calls["parser_api_url"] = api_url
            calls["parser_tier"] = tier

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
    assert "formula_enable = gr.Checkbox(" not in gradio_text
    assert "table_enable = gr.Checkbox(" not in gradio_text
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


def test_cli_old_legacy_vlm_branch_maps_to_hybrid_extra_high(monkeypatch: Any, tmp_path: Path) -> None:
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
        """记录 legacy VLM 输入最终进入 Hybrid extra_high 分支。"""
        seen["backend"] = args[3]
        seen["hybrid_backend"] = args[6]
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
    assert seen["kwargs"]["effort"] == "extra_high"
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
        seen["backend"] = args[6]
        seen["kwargs"] = kwargs

    monkeypatch.setattr(common, "_process_hybrid", _fake_process_hybrid)

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
        """Hybrid medium 不应触发 VLM engine 解析。"""
        raise AssertionError("medium effort should not resolve VLM engine")

    def _fake_process_hybrid(*args: Any, **kwargs: Any) -> None:
        """记录 Hybrid medium 仍进入 Hybrid 处理分支。"""
        seen["backend"] = args[6]
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
    outputs: list[tuple[str, str, str]] = []

    def fake_doc_analyze(pdf_bytes: bytes, **kwargs: Any) -> tuple[list[PageInfo], list[object], bool]:
        """记录每个文件独立进入 Hybrid medium analyzer。"""
        calls.append(pdf_bytes)
        languages.append(kwargs["language"])
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
        inline_formula_enable=True,
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
    assert [item[0] for item in outputs] == ["a.pdf", "b.pdf"]


def test_cli_old_async_legacy_vlm_branch_maps_to_hybrid_extra_high(monkeypatch: Any, tmp_path: Path) -> None:
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
        """记录异步 legacy VLM 输入最终进入 Hybrid extra_high 分支。"""
        seen["backend"] = args[3]
        seen["hybrid_backend"] = args[6]
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
    assert seen["kwargs"]["effort"] == "extra_high"
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
        seen["backend"] = args[6]
        seen["kwargs"] = kwargs

    monkeypatch.setattr(common, "_async_process_hybrid", _fake_async_process_hybrid)

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
