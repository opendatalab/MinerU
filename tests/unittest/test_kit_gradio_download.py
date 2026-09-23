"""下载请求与前端首次点击、缓存、错误和过期响应的回归。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from mineru.kit.gradio import app as gradio_app
from mineru.kit.gradio.i18n import MESSAGES
from mineru.kit.gradio.artifacts import create_run_artifacts
from mineru.kit.gradio.client import V1ServerCapabilities


def test_pdf_download_renders_cjk_inline_formula_without_error(tmp_path: Path) -> None:
    """真实 PDF 下载回调读取持久化协议，长行内公式成功导出且再次点击可复用文件。"""
    from pypdf import PdfReader

    from mineru.types import MiddleJson, PageInfo

    source = tmp_path / "中文公式.pdf"
    source.write_bytes(b"test")
    artifacts = create_run_artifacts(source, tmp_path / "output")
    middle = MiddleJson(
        is_full_document=False,
        pages=[
            PageInfo.model_validate(
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "text",
                            "index": 0,
                            "bbox": [0.1, 0.1, 0.9, 0.8],
                            "content": [
                                {"type": "text", "content": "中文前文"},
                                {"type": "equation_inline", "content": "+".join(["x_i"] * 80)},
                                {"type": "text", "content": "中文后文"},
                            ],
                        }
                    ],
                }
            )
        ],
        metadata={"file_suffix": "pdf", "producer": {"name": "test", "version": "1"}},
    )
    artifacts.middle_json_path.write_text(json.dumps(middle.to_dict(), ensure_ascii=False), encoding="utf-8")
    handler = gradio_app._download_handler("pdf", tmp_path / "output")
    previous = None
    for sequence in (1, 2):
        token = json.dumps({"run_id": artifacts.root.name, "sequence": sequence})
        path, receipt = handler(artifacts.as_state(), token)
        assert json.loads(receipt) == {"request": token, "error": ""}
        assert path is not None
        content = Path(path).read_bytes()
        assert content.startswith(b"%PDF-")
        text = PdfReader(path).pages[0].extract_text()
        assert "中文前文" in text and "中文后文" in text
        if previous is not None:
            assert content == previous
        previous = content


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


def test_download_event_chain_and_pdf_transport(tmp_path: Path) -> None:
    """下载事件独立，PDF 仍以隐藏的原生文件组件输出并由 HTML 显示。"""
    cap = V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("zip",), ("file_id",))
    app = gradio_app.build_gradio_app(Mock(), cap, output_root=tmp_path, enable_example=False)
    conversion = next(fn for fn in app.fns.values() if fn.name == "convert_handler")
    assert (
        "mineru-convert-button"
        in next(
            block for block in app.blocks.values() if block.__class__.__name__ == "Button" and "mineru.convert" in block.value
        ).elem_classes
    )
    pdf = conversion.outputs[2]
    assert pdf.__class__.__name__ == "File" and pdf.visible is False
    preview = next(fn.fn for fn in app.fns.values() if fn.name == "update_file_preview")
    for source in (None, "photo.png", "book.docx", "book.epub"):
        assert preview(source)[0]["value"] is None
        assert preview(source)[0]["visible"] is False
    reset = next(fn.fn for fn in app.fns.values() if fn.name == "reset_ui")
    assert reset()[2]["visible"] is False
    assert reset()[7] == ""
    conversion = next(fn for fn in app.fns.values() if fn.name == "convert_handler")
    dependency = next(dep for dep in app.config["dependencies"] if dep["id"] == conversion._id)
    # 转换必须等待真正的重置完成事件，纯 JS 事件的 then 在支持版本中不会可靠触发。
    reset_event = app.fns[dependency["trigger_after"]]
    assert reset_event.name == "reset_download_ui"
    assert len(reset_event.fn()) == len(reset_event.outputs)
    assert reset_event.outputs[0].__class__.__name__ == "HTML"
    assert "Preparing request..." in reset_event.fn()[0]
    reset_dependency = next(dep for dep in app.config["dependencies"] if dep["id"] == reset_event._id)
    assert "Preparing request..." in reset_dependency["js"]
    handlers = [fn for fn in app.fns.values() if fn.name == "handler"]
    assert len(handlers) == 7
    files = []
    for handler in handlers:
        file, receipt = handler.outputs
        assert file.__class__.__name__ == "File" and file.visible is False
        files.append(file._id)
        success = next(dep for dep in app.config["dependencies"] if dep.get("trigger_after") == handler._id)
        assert success["trigger_only_on_success"] is True
        assert success["backend_fn"] is False and success["queue"] is False
        assert success["inputs"][:2] == [file._id, receipt._id]
    assert len(set(files)) == 7


def test_frontend_download_lifecycle() -> None:
    """运行真实前端脚本，验证恰好一次下载、失败重试和跨文档失效。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend state tests")
    result = subprocess.run(
        [node, str(Path(__file__).with_suffix(".cjs"))], input=json.dumps(MESSAGES), capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
