"""验证本地 PDF.js 的资源边界、原生文件契约和前端生命周期。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import quote

import gradio as gr
import pytest
import typer
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gradio.data_classes import FileData

from mineru.kit.commands import gradio as gradio_command
from mineru.kit.gradio.app import build_gradio_app
from mineru.kit.gradio.client import V1ServerCapabilities
from mineru.kit.gradio.pdf_preview import register_pdf_preview_resources


def test_pdf_preview_transport_and_success_events(tmp_path: Path) -> None:
    """公开转换保留原生 PDF 文件输出，预览仅在成功事件通过前端更新。"""
    capability = V1ServerCapabilities("http://unused.test", ("flash",), ("zip",), ("file_id",))
    demo = build_gradio_app(Mock(), capability, output_root=tmp_path, enable_example=False)
    conversion = next(fn for fn in demo.fns.values() if fn.name == "convert_handler")
    source = next(fn for fn in demo.fns.values() if fn.name == "update_file_preview")
    transport = conversion.outputs[2]
    assert isinstance(transport, gr.File) and transport.visible is False
    assert source.outputs[0] is transport
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    assert isinstance(transport.postprocess(str(pdf)), FileData)
    assert not any(type(block).__module__.startswith("gradio_pdf") for block in demo.blocks.values())
    viewer = next(block for block in demo.blocks.values() if "mineru-kit-pdf-preview" in (block.elem_classes or []))
    assert isinstance(viewer, gr.HTML) and viewer.visible is True
    assert ':not(:has(.mineru-pdf-frame, [role="alert"]))' in demo._mineru_kit_css
    for event in (source, conversion):
        previews = [item for item in demo.config["dependencies"] if item.get("trigger_after") == event._id]
        assert len(previews) == 1
        assert previews[0]["trigger_only_on_success"] is True
        assert previews[0]["outputs"] == [viewer._id]
        assert previews[0]["backend_fn"] is False and previews[0]["queue"] is False
    assert len(conversion.inputs) == 4


@pytest.mark.parametrize("mount", ["", "/mineru"])
def test_pdf_preview_static_routes_and_range(tmp_path: Path, mount: str) -> None:
    """真实文件路由保留挂载前缀、字符编码、MIME、Range 和访问边界。"""
    entry = register_pdf_preview_resources()
    pdf = tmp_path / "中文 # % 空格.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"fixture" * 100)
    denied = tmp_path / "private.txt"
    denied.write_text("private", encoding="utf-8")
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.HTML("")
    app = gr.mount_gradio_app(FastAPI(), demo, path=mount or "/", allowed_paths=[str(pdf)])
    with TestClient(app) as client:
        for path, mime in (
            (entry, "text/html"),
            (entry.with_name("viewer.mjs"), "text/javascript"),
            (entry.parent / "vendor/pdfjs/legacy/build/pdf.worker.min.mjs", "text/javascript"),
            (entry.parent / "vendor/pdfjs/wasm/openjpeg.wasm", "application/wasm"),
            (pdf, "application/pdf"),
        ):
            response = client.get(f"{mount}/gradio_api/file={quote(str(path), safe='/')}")
            assert response.status_code == 200
            assert response.headers["content-type"].split(";")[0] == mime
            assert response.headers["content-disposition"].startswith("inline")
        response = client.get(f"{mount}/gradio_api/file={quote(str(pdf), safe='/')}", headers={"Range": "bytes=0-7"})
        assert response.status_code == 206 and response.content == b"%PDF-1.7"
        assert client.get(f"{mount}/gradio_api/file={quote(str(denied), safe='/')}").status_code == 403


def test_pdf_preview_static_routes_require_login(tmp_path: Path) -> None:
    """预览资源与 PDF 均复用 Gradio 的登录检查，不新增绕过认证的文件入口。"""
    entry = register_pdf_preview_resources()
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.HTML("")
    app = gr.mount_gradio_app(FastAPI(), demo, path="/mineru", auth=("user", "password"), allowed_paths=[str(pdf)])
    with TestClient(app) as client:
        for path in (entry, pdf):
            response = client.get(f"/mineru/gradio_api/file={quote(str(path), safe='/')}")
            assert response.status_code == 401
        assert client.post("/mineru/login", data={"username": "user", "password": "password"}).status_code == 200
        assert client.get(f"/mineru/gradio_api/file={quote(str(pdf), safe='/')}").content == pdf.read_bytes()


def test_pdfjs_manifest_matches_bundled_files() -> None:
    """每个打包的发行文件都与校验清单一致，避免 worker 和核心版本漂移。"""
    root = register_pdf_preview_resources().parent / "vendor/pdfjs"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "6.3.289"
    assert manifest["source"].endswith("pdfjs-dist-6.3.289.tgz")
    # Finder 可能在浏览资源目录时写入本机元数据，校验仍只针对发行文件。
    assert {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != ".DS_Store"} == {
        "manifest.json",
        *manifest["sha256"],
    }
    for relative, digest in manifest["sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    assert "Apache License" in (root / "LICENSE").read_text(encoding="utf-8")


@pytest.mark.parametrize("version", ["5.49.1", "5.50.0", "6.7.0", "7.0.0"])
def test_gradio_command_rejects_unsupported_version(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    """即使用户跳过安装器检查，启动时仍明确拒绝旧版或未支持的主版本。"""
    monkeypatch.setattr(gradio_command, "version", Mock(return_value=version))
    with pytest.raises(typer.Exit) as error:
        gradio_command._require_gradio_dependencies()
    assert getattr(error.value, "exit_code", None) == 1


def test_pdf_preview_frontend_lifecycle() -> None:
    """用 Node 执行实际适配器，验证乱序、编码和重复上传行为。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend lifecycle checks")
    result = subprocess.run([node, str(Path(__file__).with_suffix(".cjs"))], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
