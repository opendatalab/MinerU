"""下载请求与前端首次点击、缓存、错误和过期响应的回归。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from mineru.kit.gradio import app as gradio_app
from mineru.kit.gradio.artifacts import create_run_artifacts
from mineru.kit.gradio.client import V1ServerCapabilities


def test_download_receipt_keeps_request_on_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实结果标识校验覆盖正常生成、缓存复用及渲染异常，并在错误回执中保留请求标识。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(b"test")
    artifacts = create_run_artifacts(source, tmp_path / "output")
    token = json.dumps({"run_id": artifacts.root.name, "sequence": 1})
    target = artifacts.downloads_dir / "source.html"
    render = Mock(return_value=str(target))
    monkeypatch.setattr(gradio_app, "render_download", render)
    handler = gradio_app._download_handler("html", tmp_path / "output")
    for sequence in (1, 2):
        token = json.dumps({"run_id": artifacts.root.name, "sequence": sequence})
        path, receipt = handler(artifacts.as_state(), token)
        assert path == str(target)
        assert json.loads(receipt) == {"request": token, "error": ""}
    render.side_effect = RuntimeError("render failed")
    path, receipt = handler(artifacts.as_state(), token)
    assert path is None
    assert json.loads(receipt) == {"request": token, "error": "render failed"}
    render.reset_mock()
    path, receipt = handler(artifacts.as_state(), json.dumps({"run_id": "old-run", "sequence": 3}))
    assert path is None and "已变更" in json.loads(receipt)["error"]
    render.assert_not_called()


def test_download_event_chain_and_pdf_mount(tmp_path: Path) -> None:
    """每种格式使用独立文件与成功事件，所有空 PDF 预览状态保持挂载。"""
    cap = V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("zip",), ("file_id",))
    app = gradio_app.build_gradio_app(Mock(), cap, output_root=tmp_path, enable_example=False)
    pdf = next(component for component in app.blocks.values() if component.__class__.__name__ == "PDF")
    assert pdf.visible == "hidden"
    preview = next(fn.fn for fn in app.fns.values() if fn.name == "update_file_preview")
    for source in (None, "photo.png", "book.docx", "book.epub"):
        assert preview(source)[0]["value"] is None
        assert preview(source)[0]["visible"] == "hidden"
        assert "mineru-kit-pdf-empty" in preview(source)[0]["elem_classes"]
    reset = next(fn.fn for fn in app.fns.values() if fn.name == "reset_ui")
    assert reset()[4]["visible"] == "hidden"
    assert reset()[9] == ""
    conversion = next(fn for fn in app.fns.values() if fn.name == "convert_handler")
    dependency = next(dep for dep in app.config["dependencies"] if dep["id"] == conversion._id)
    # 转换必须等待真正的重置完成事件，纯 JS 事件的 then 在支持版本中不会可靠触发。
    reset_event = app.fns[dependency["trigger_after"]]
    assert reset_event.name == "reset_download_ui"
    assert len(reset_event.fn()) == len(reset_event.outputs)
    handlers = [fn for fn in app.fns.values() if fn.name == "handler"]
    assert len(handlers) == 6
    files = []
    for handler in handlers:
        file, receipt = handler.outputs
        assert file.__class__.__name__ == "File" and file.visible is False
        files.append(file._id)
        success = next(dep for dep in app.config["dependencies"] if dep.get("trigger_after") == handler._id)
        assert success["trigger_only_on_success"] is True
        assert success["backend_fn"] is False and success["queue"] is False
        assert success["inputs"][:2] == [file._id, receipt._id]
    assert len(set(files)) == 6


def test_frontend_download_lifecycle() -> None:
    """运行真实前端脚本，验证恰好一次下载、失败重试和跨文档失效。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend state tests")
    result = subprocess.run([node, str(Path(__file__).with_suffix(".cjs"))], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
