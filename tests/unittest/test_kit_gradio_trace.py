"""溯源预览的真实产物、公共门面和前端协议回归。"""

from pathlib import Path
import json
import shutil
import subprocess

from bs4 import BeautifulSoup

from mineru.kit.gradio.artifacts import persist_parse_result, render_html_preview, render_download
from mineru.kit.gradio.conversion_service import _prepare_preview
from mineru.parser.base import ParseResult
from test_kit_gradio import _middle_json, _pdf_bytes


def test_pdf_trace_preview_and_public_render(tmp_path: Path) -> None:
    """抽页预览携带原始页映射与来源属性，下载走普通渲染入口。"""
    source = tmp_path / "source.pdf"
    source.write_bytes(_pdf_bytes())
    middle = _middle_json(with_image=False)
    artifacts = persist_parse_result(ParseResult(middle_json=middle), source, output_root=tmp_path, page_range="")
    assert _prepare_preview("pdf", artifacts).pdf_path == artifacts.origin_pdf_path
    assert not (artifacts.root / "layout.pdf").exists()
    preview = render_html_preview(artifacts, public_base_url="http://testserver")
    frame = BeautifulSoup(preview, "html.parser").iframe
    assert "allow-same-origin" not in frame["sandbox"]
    document = BeautifulSoup(frame["srcdoc"], "html.parser")
    assert document.select_one("[data-docvortex-trace-id]")
    assert artifacts.root.name in document.get_text() or artifacts.root.name in str(document)
    assert "pageIndices" in str(document)
    trace_style = next(style.string for style in document.select("style") if ".mineru-trace-pages {" in (style.string or ""))
    assert "max-height: calc(100vh - var(--mineru-preview-offset, 8px) - 8px)" in trace_style
    assert "overflow-y: auto" in trace_style
    assert "box-sizing: border-box" in trace_style
    script = next(script.string for script in document.select("script") if script.string and '"layoutRegions"' in script.string)
    config = json.loads(script.rsplit(")(", 1)[1].removesuffix(");"))
    assert config["layoutRegions"]
    assert config["layoutRegions"][0]["type"] == str(middle.pages[0].blocks[0].type)
    assert config["layoutRegions"][0]["index"] == middle.pages[0].blocks[0].index
    assert config["layoutRegions"][0]["child_path"] == []
    download = render_download(artifacts.as_state(), "html", public_base_url="http://testserver")
    assert "data-docvortex-sources" not in Path(download).read_text()
    assert "layoutRegions" not in Path(download).read_text()


def test_trace_frontend_coordination() -> None:
    """用 Node 执行真实父窗口脚本的双向消息与失效回归。"""
    node = shutil.which("node")
    assert node is not None
    subprocess.run([node, str(Path(__file__).with_suffix(".cjs"))], check=True)
