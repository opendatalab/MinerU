"""自有上传入口只通过公共传输接口工作，并隔离迟到回执。"""

from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import gradio as gr
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mineru.kit.gradio.app import build_gradio_app
from mineru.kit.gradio.client import V1ServerCapabilities


def test_upload_frontend_lifecycle() -> None:
    """运行真实脚本验证传输进度、同名替换、重试、粘贴和主动中止。"""
    node = shutil.which("node")
    assert node is not None
    subprocess.run([node, str(Path(__file__).with_suffix(".cjs"))], check=True)


@pytest.mark.parametrize("mount", ["", "/mineru"])
def test_upload_context_keeps_mount_authentication_and_size_limit(tmp_path: Path, mount: str) -> None:
    """从真实组件服务获取上传地址，并验证原生路由仍要求登录、限制大小且可读取上传文件。"""
    demo = build_gradio_app(
        SimpleNamespace(),
        V1ServerCapabilities("http://localhost:1", ("flash",), ("zip",), ("file_id",)),
        output_root=tmp_path,
        enable_example=False,
    )
    uploader = next(block for block in demo.blocks.values() if "mineru-upload-file" in (block.elem_classes or []))
    app = gr.mount_gradio_app(FastAPI(), demo, path=mount or "/", auth=("user", "password"), max_file_size=1024)
    body = {"component_id": uploader._id, "fn_name": "upload_context", "session_hash": "upload-test", "data": []}
    with TestClient(app) as client:
        assert client.post(f"{mount}/gradio_api/component_server", json=body).status_code == 401
        assert client.post(f"{mount}/gradio_api/upload", files={"files": ("a.pdf", b"pdf")}).status_code == 401
        assert client.post(f"{mount}/login", data={"username": "user", "password": "password"}).status_code == 200
        response = client.post(f"{mount}/gradio_api/component_server", json=body)
        assert response.status_code == 200, response.text
        context = response.json()
        assert context == {
            "upload_url": f"http://testserver{mount}/gradio_api/upload",
            "file_url_prefix": f"http://testserver{mount}/gradio_api/file=",
        }
        uploaded = client.post(context["upload_url"], files={"files": ("test.pdf", b"%PDF-test")})
        assert uploaded.status_code == 200, uploaded.text
        assert client.get(context["file_url_prefix"] + uploaded.json()[0]).content == b"%PDF-test"
        too_large = client.post(context["upload_url"], files={"files": ("large.pdf", b"x" * 1025)})
        assert too_large.status_code == 413
