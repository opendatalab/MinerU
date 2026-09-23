"""源文档预览（OFD/HTML）的真实转换、隔离和失败回执回归。"""

import asyncio
import codecs
import io
import json
import zipfile
from pathlib import Path

import socket
import pytest
from lxml import html
from PIL import Image

from mineru.kit.gradio.ofd_preview import build_ofd_preview
from mineru.kit.gradio.source_preview import build_html_preview, prepare_source_preview


def ofd_bytes(text: str = "中文发票 &amp; &lt;script&gt;", *, entity: str = "") -> bytes:
    """构造包含两页、模板、路径及内嵌图片的有效 OFD 测试文档。"""
    ns = 'xmlns:ofd="http://www.ofdspec.org/2016"'
    image = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(image, format="PNG")
    files = {
        "OFD.xml": (
            f'<ofd:OFD {ns} Version="1.0" DocType="OFD">'
            "<ofd:DocBody><ofd:DocRoot>Doc/Document.xml</ofd:DocRoot></ofd:DocBody></ofd:OFD>"
        ),
        "Doc/Document.xml": f"""<ofd:Document {ns}><ofd:CommonData>
            <ofd:PageArea><ofd:PhysicalBox>0 0 210 297</ofd:PhysicalBox></ofd:PageArea>
            <ofd:DocumentRes>Res.xml</ofd:DocumentRes>
            <ofd:TemplatePage ID="9" BaseLoc="Template.xml" ZOrder="Background"/>
            </ofd:CommonData><ofd:Pages><ofd:Page ID="1" BaseLoc="Page.xml"/>
            <ofd:Page ID="2" BaseLoc="Page.xml"/></ofd:Pages></ofd:Document>""",
        "Doc/Res.xml": f"""<ofd:Res {ns}><ofd:MultiMedias><ofd:MultiMedia ID="4" Type="Image">
            <ofd:MediaFile>red.png</ofd:MediaFile></ofd:MultiMedia></ofd:MultiMedias></ofd:Res>""",
        "Doc/Template.xml": f"""<ofd:Page {ns}><ofd:Content><ofd:Layer ID="10">
            <ofd:PathObject ID="11" Boundary="10 10 190 277" Stroke="true" LineWidth="0.5">
            <ofd:AbbreviatedData>M 0 0 L 190 0 L 190 277 L 0 277 C</ofd:AbbreviatedData>
            </ofd:PathObject></ofd:Layer></ofd:Content></ofd:Page>""",
        "Doc/Page.xml": f"""{entity}<ofd:Page {ns}><ofd:Template TemplateID="9"/>
            <ofd:Content><ofd:Layer ID="1"><ofd:TextObject ID="2" Boundary="20 30 160 20" Size="5">
            <ofd:TextCode X="0" Y="8" DeltaX="g 40 5">{text}</ofd:TextCode></ofd:TextObject>
            <ofd:ImageObject ID="3" ResourceID="4" Boundary="20 60 30 30" CTM="30 0 0 30 0 0"/>
            </ofd:Layer></ofd:Content></ofd:Page>""",
        "Doc/red.png": image.getvalue(),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return output.getvalue()


def _frame_element(frame_html: str) -> html.HtmlElement:
    """从可能带缩放外壳的预览标记中取出唯一 iframe。"""
    root = html.fromstring(frame_html)
    if root.tag == "iframe":
        return root
    frames = root.xpath(".//iframe")
    assert len(frames) == 1
    return frames[0]


def _frame_document(frame_html: str) -> str:
    """取出沙箱 iframe 的 srcdoc 文档，属性转义在往返解析中被验证。"""
    return _frame_element(frame_html).get("srcdoc")


def test_real_conversion_embeds_pages_and_escapes_text() -> None:
    """验证两页 SVG、模板路径、图片和文字转义，并禁止 iframe 脚本。"""
    frame = html.fromstring(build_ofd_preview(ofd_bytes()))
    assert frame.get("sandbox") == ""
    assert frame.get("class") == "mineru-source-frame"
    document = html.fromstring(frame.get("srcdoc"))
    assert len(document.xpath("//svg")) == 2
    assert len(document.xpath("//svg/path")) == 2
    assert "中文发票 & <script>" in document.text_content()
    assert not document.xpath("//script")
    assert "data:image/png;base64," in frame.get("srcdoc")
    assert "max-width:100%" in frame.get("srcdoc")


@pytest.mark.parametrize("payload", [b"", b"garbage", b"PK\x03\x04"])
def test_invalid_payload(payload: bytes) -> None:
    """空文件和损坏压缩包必须明确失败。"""
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        build_ofd_preview(payload)


def test_missing_ofd_xml() -> None:
    """合法 ZIP 缺少 OFD 入口时不得显示空白成功预览。"""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("unrelated.xml", "<x/>")
    with pytest.raises(ValueError, match="missing OFD.xml"):
        build_ofd_preview(output.getvalue())


def test_external_entity_is_not_read(tmp_path: Path) -> None:
    """外部实体不得读取本地文件或进入最终 HTML。"""
    secret = tmp_path / "secret.txt"
    secret.write_text("OFD_EXTERNAL_ENTITY_SECRET")
    entity = f'<!DOCTYPE ofd:Page [<!ENTITY leak SYSTEM "{secret.as_uri()}">]>'
    frame = build_ofd_preview(ofd_bytes("&leak;", entity=entity))
    assert "OFD_EXTERNAL_ENTITY_SECRET" not in frame


def test_html_preview_injects_csp_into_head_and_keeps_content() -> None:
    """源 HTML 允许脚本和联网资源，但 sandbox 仍保持 opaque origin 隔离。"""
    payload = (
        '<!DOCTYPE html><html><head><meta name="referrer" content="always"><title>报告</title></head>'
        '<body><p>中文 &amp; 引号 "quoted"</p><img src="a.png"></body></html>'
    ).encode("utf-8")
    markup = build_html_preview(payload)
    root = html.fromstring(markup)
    assert root.get("class") == "mineru-source-viewport"
    assert root.xpath("./div[@class='mineru-source-stage']/iframe[@class='mineru-source-frame']")
    frame = _frame_element(markup)
    assert frame.get("sandbox") == "allow-scripts"
    assert frame.get("referrerpolicy") == "no-referrer"
    assert frame.get("class") == "mineru-source-frame"
    document = html.fromstring(frame.get("srcdoc"))
    csp = document.xpath("//head/meta[@http-equiv='Content-Security-Policy']")
    assert len(csp) == 1
    referrer = document.xpath("//head/meta[@name='referrer']")
    assert len(referrer) == 1 and referrer[0].get("content") == "no-referrer"
    policy = csp[0].get("content")
    assert "default-src 'none'" in policy
    assert "script-src 'unsafe-inline' 'unsafe-eval' https: http:" in policy
    assert "connect-src https: http:" in policy
    assert "img-src data: blob: https: http:" in policy
    assert "frame-src 'none'" in policy and "form-action 'none'" in policy
    assert '中文 & 引号 "quoted"' in document.text_content()


def test_html_preview_replaces_source_referrer_policy_before_remote_resources() -> None:
    """预览固定使用 no-referrer，避免沿用来源文件的强制 Referer 策略。"""
    payload = (
        '<html><head><meta name="referrer" content="always">'
        '<link rel="stylesheet" href="https://cdn.example.test/site.css"></head><body></body></html>'
    ).encode()
    document = html.fromstring(_frame_document(build_html_preview(payload)))
    head_children = list(document.xpath("//head")[0])
    referrer_index = next(i for i, node in enumerate(head_children) if node.tag == "meta" and node.get("name") == "referrer")
    stylesheet_index = next(i for i, node in enumerate(head_children) if node.tag == "link" and node.get("rel") == "stylesheet")
    assert head_children[referrer_index].get("content") == "no-referrer"
    assert referrer_index < stylesheet_index
    assert not document.xpath("//meta[@name='referrer' and @content='always']")


def test_html_preview_normalizes_explicit_resource_referrer_policies() -> None:
    """标签覆盖策略和重复 meta 不得覆盖预览的无 Referer 策略。"""
    payload = (
        '<!DOCTYPE html><html><head><META NAME="Referrer" content="always">'
        '<meta name=" referrer " content="origin">'
        '<link rel="preload" href="/a.png" as="image" referrerpolicy="unsafe-url">'
        '<link rel="stylesheet" href="/a.css" referrerpolicy="origin">'
        '<script src="/a.js" referrerpolicy="same-origin"></script></head><body>'
        '<img src="/a.png" referrerpolicy="unsafe-url">'
        '<img src="/b.png" REFERRERPOLICY=""><meta name="referrer" content="always">'
        "</body></html>"
    ).encode()
    document = html.fromstring(_frame_document(build_html_preview(payload)))
    metas = document.xpath("//meta[@name='referrer']")
    assert len(metas) == 1 and metas[0].get("content") == "no-referrer"
    assert document.xpath("//head/*")[0] == metas[0]
    resources = document.xpath("//link | //script[@src] | //img")
    assert len(resources) == 5
    assert all(tag.get("referrerpolicy") == "no-referrer" for tag in resources)


@pytest.mark.parametrize("suffix", ["html", "htm", "shtml"])
def test_html_preview_preserves_resources_without_server_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str
) -> None:
    """真实上传构建路径不联网，图片候选、查询参数与外部 CSS 均交由浏览器解析。"""

    def reject_connection(*args: object, **kwargs: object) -> None:
        """禁止服务端连接网络，使任何资源下载回归立即失败。"""
        pytest.fail("HTML preview must not fetch resources on the server")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    payload = (
        '<html><head><link rel="canonical" href="https://blog.csdn.net/article/1">'
        '<link rel="stylesheet" href="https://csdnimg.cn/base.css"></head><body>'
        '<picture><source media="(min-width:800px)" srcset="https://i-blog.csdnimg.cn/picture.webp 2x">'
        '<img id="image" src="https://i-blog.csdnimg.cn/main.png" '
        'srcset="https://i-blog.csdnimg.cn/a.png?resize=m_fixed,h_64,w_64 1x, /b.png 2x" '
        'data-src="/lazy.png" width="1080" height="688"></picture>'
        '<img src="https://profile-avatar.csdnimg.cn/avatar.jpg!1">'
        '<img src="https://csdnimg.cn/identity/blog5.png"></body></html>'
    ).encode()
    source = tmp_path / f"page.{suffix}"
    source.write_bytes(payload)
    receipt = json.loads(asyncio.run(prepare_source_preview(str(source), json.dumps({"id": 1, "path": str(source)}))))
    document = html.fromstring(_frame_document(receipt["html"]))
    original = html.fromstring(payload)
    for selector in ("//picture/source", "//img", "//link[@rel='stylesheet']"):
        assert [dict(tag.attrib) for tag in document.xpath(selector)] == [dict(tag.attrib) for tag in original.xpath(selector)]
    assert "mineru-source-broken-image" not in _frame_document(receipt["html"])


def test_html_preview_preserves_stylesheet_positions_and_cascade() -> None:
    """混合外部与内联样式保留原始位置、相对 URL 和级联次序。"""
    payload = (
        '<html><head><link rel="stylesheet" href="https://example.test/base.css">'
        "<style>p {color:red}</style></head><body>"
        "<style>p {color:green;background:url(../bg.png)}</style>"
        '<link rel="stylesheet" href="article.css" media="screen" title="article">'
        '<div><link rel="stylesheet" href="code.css"><style>p {color:blue}</style></div>'
        '<link rel="alternate stylesheet" href="alternate.css" title="alternate">'
        '<link rel="stylesheet" href="disabled.css" disabled>'
        '<link rel="preload" href="font.woff2">'
        '<noscript><link rel="stylesheet" href="no-script.css"></noscript>'
        "<p>正文</p></body></html>"
    ).encode()
    document = html.fromstring(_frame_document(build_html_preview(payload)))
    original = html.fromstring(payload)
    for selector in ("//head/link | //head/style", "//body/link | //body/style", "//body/div/*", "//noscript/*"):
        actual = document.xpath(selector)
        expected = original.xpath(selector)
        assert len(actual) == len(expected)
        for actual_tag, expected_tag in zip(actual, expected, strict=True):
            assert actual_tag.tag == expected_tag.tag
            assert actual_tag.text == expected_tag.text
            assert ("disabled" in actual_tag.attrib) == ("disabled" in expected_tag.attrib)
            assert {key: value for key, value in actual_tag.attrib.items() if key != "disabled"} == {
                key: value for key, value in expected_tag.attrib.items() if key != "disabled"
            }


def test_html_preview_bridge_uses_stable_layout_width_not_document_scroll_width() -> None:
    """Safari 会把横向轮播离屏项计入 scrollWidth，尺寸桥必须改用桌面声明和顶层布局盒。"""
    payload = (
        '<html><head><meta name="applicable-device" content="pc"></head>'
        '<body><div style="width:1200px"><div style="width:12000px"></div></div></body></html>'
    ).encode()
    frame = _frame_element(build_html_preview(payload))
    assert frame.get("data-mineru-source-content-width") == "1200"
    document = html.fromstring(frame.get("srcdoc"))
    hint = document.xpath("//head/meta[@name='mineru-source-preview-width']")
    assert len(hint) == 1 and hint[0].get("content") == "1200"
    bridge = document.xpath("//script[@id='mineru-source-preview-bridge']")[0].text or ""
    assert 'meta[name="mineru-source-preview-width"]' in bridge
    assert 'meta[name="applicable-device"]' in bridge
    assert "return 1200" in bridge
    assert "parent.postMessage" in bridge
    assert "body.children" in bridge
    assert "body.scrollWidth" not in bridge
    assert "root.scrollWidth" not in bridge
    assert "ResizeObserver" not in bridge


@pytest.mark.parametrize(
    ("viewport", "expected"),
    [
        ("width=1024, initial-scale=1", "1024"),
        ("initial-scale=1, width=1440", "1440"),
    ],
)
def test_html_preview_injects_numeric_viewport_width_hint(viewport: str, expected: str) -> None:
    """数值 viewport 声明优先于通用桌面宽度，保持源页面原始设计视口。"""
    payload = f'<html><head><meta name="viewport" content="{viewport}"></head><body></body></html>'.encode()
    frame = _frame_element(build_html_preview(payload))
    assert frame.get("data-mineru-source-content-width") == expected
    document = html.fromstring(frame.get("srcdoc"))
    hint = document.xpath("//head/meta[@name='mineru-source-preview-width']")
    assert len(hint) == 1 and hint[0].get("content") == expected


def test_html_preview_keeps_formula_scripts_for_typesetting() -> None:
    """MathJax/KaTeX 源码和 TeX 文本完整保留，CSP 允许其执行并加载远程依赖。"""
    payload = (
        "<!DOCTYPE html><html><head>"
        '<script>window.MathJax = {tex: {inlineMath: [["$", "$"]]}};</script>'
        '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>'
        "</head><body><p>质能方程 \\(E=mc^2\\)</p></body></html>"
    ).encode("utf-8")
    frame = _frame_element(build_html_preview(payload))
    assert frame.get("sandbox") == "allow-scripts"
    document = html.fromstring(frame.get("srcdoc"))
    scripts = [script for script in document.xpath("//script") if script.get("id") != "mineru-source-preview-bridge"]
    assert len(scripts) == 2
    assert scripts[0].text is not None and "inlineMath" in scripts[0].text
    assert scripts[1].get("src").endswith("tex-chtml.js")
    assert document.xpath("//script[@id='mineru-source-preview-bridge']")
    assert "E=mc^2" in document.text_content()
    csp = document.xpath("//head/meta[@http-equiv='Content-Security-Policy']")[0].get("content")
    script_policy = next(item.strip() for item in csp.split(";") if item.strip().startswith("script-src"))
    assert script_policy == "script-src 'unsafe-inline' 'unsafe-eval' https: http:"
    assert "connect-src https: http:" in csp


def test_html_preview_removes_auto_navigation_but_keeps_normal_scripts_and_events() -> None:
    """只清理自动导航入口，普通页面初始化、交互脚本、外链和文内锚点保持可用。"""
    payload = (
        "<!DOCTYPE html><html><head>"
        '<meta http-equiv="refresh" content="3;url=https://www.csdn.net">'
        "</head>"
        "<body onload=\"document.body.dataset.ready='1'\">"
        '<img src="" onerror="setTimeout(function(){window.location.href=\'https://www.csdn.net\'},3000)">'
        "<button onclick=\"this.classList.toggle('open')\">toggle</button>"
        "<script>window.location.href='https://example.test/redirect'</script>"
        '<a id="external" href="https://example.test/page" rel="nofollow">external</a>'
        '<a id="internal" href="#section">internal</a>'
        '<p id="section">正文仍需展示</p></body></html>'
    ).encode("utf-8")
    frame = _frame_element(build_html_preview(payload))
    assert frame.get("sandbox") == "allow-scripts"
    document = html.fromstring(frame.get("srcdoc"))
    assert not document.xpath("//meta[translate(@http-equiv, 'REFSH', 'refsh')='refresh']")
    assert document.xpath("//img")[0].get("onerror") is None
    assert document.xpath("//body")[0].get("onload") == "document.body.dataset.ready='1'"
    assert document.xpath("//button")[0].get("onclick") == "this.classList.toggle('open')"
    source_scripts = [script for script in document.xpath("//script") if script.get("id") != "mineru-source-preview-bridge"]
    assert any(script.text and "window.location.href" in script.text for script in source_scripts)
    external = document.get_element_by_id("external")
    assert external.get("target") == "_blank"
    assert set(external.get("rel").split()) == {"nofollow", "noopener", "noreferrer"}
    internal = document.get_element_by_id("internal")
    assert internal.get("href") == "#section" and internal.get("target") is None
    assert "正文仍需展示" in document.text_content()


@pytest.mark.parametrize(
    "handler",
    [
        "window.location.href='https://example.test'",
        "location.assign('https://example.test')",
        "location.replace('https://example.test')",
        "top.location='https://example.test'",
        "parent.location.href='https://example.test'",
        "window.open('https://example.test')",
    ],
)
def test_html_preview_removes_navigation_event_handler_variants(handler: str) -> None:
    """常见自动导航写法在事件属性中被移除，避免来源页把预览 iframe 替换掉。"""
    payload = f'<html><body><img src="" onerror="{handler}"></body></html>'.encode()
    document = html.fromstring(_frame_document(build_html_preview(payload)))
    assert document.xpath("//img")[0].get("onerror") is None


def test_html_preview_restores_source_base_from_canonical_or_existing_base() -> None:
    """canonical 用于恢复根路径资源基址，已有绝对 base 时不重复注入。"""
    canonical_payload = (
        '<html><head><link rel="canonical" href="https://example.test/articles/one">'
        '<script src="/assets/app.js"></script></head><body></body></html>'
    ).encode()
    canonical = html.fromstring(_frame_document(build_html_preview(canonical_payload)))
    assert [base.get("href") for base in canonical.xpath("//head/base")] == ["https://example.test/articles/one"]
    assert canonical.xpath("//head/base")[0].getnext().get("http-equiv") == "Content-Security-Policy"

    existing_payload = (
        '<html><head><base href="https://cdn.example.test/root/">'
        '<link rel="canonical" href="https://example.test/ignored"></head><body></body></html>'
    ).encode()
    existing = html.fromstring(_frame_document(build_html_preview(existing_payload)))
    assert [base.get("href") for base in existing.xpath("//head/base")] == ["https://cdn.example.test/root/"]


def test_html_preview_decodes_declared_charset() -> None:
    """按头部 charset 声明解码 GBK 等存量文档，未知编码回退 UTF-8 替换。"""
    payload = '<html><head><meta charset="gbk"></head><body><p>中文内容</p></body></html>'.encode("gbk")
    assert "中文内容" in _frame_document(build_html_preview(payload))
    broken = b"<html><head><meta charset='utf-8'></head><body>\xff\xfe</body></html>"
    assert "\ufffd" in _frame_document(build_html_preview(broken))


@pytest.mark.parametrize(
    ("prefix", "encoding"),
    [
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF8, "utf-8"),
    ],
)
def test_html_preview_decodes_bom_prefixed_documents(prefix: bytes, encoding: str) -> None:
    """带 BOM 的 UTF-16/32 及 UTF-8 文档按 BOM 解码，正文不出现替换字符。"""
    payload = prefix + "<html><body><p>中文标题</p></body></html>".encode(encoding)
    document = _frame_document(build_html_preview(payload))
    assert "中文标题" in document
    assert "\ufffd" not in document


@pytest.mark.parametrize(
    "payload",
    [
        b"<!DOCTYPE html><html><body><p>no head</p></body></html>",
        b"<p>fragment only</p>",
        b"<html><body><header>site nav</header><p>body</p></body></html>",
    ],
)
def test_html_preview_builds_missing_head_after_doctype(payload: bytes) -> None:
    """缺失 head 时补建注入点，<header> 标签不得被误当作 head。"""
    document = _frame_document(build_html_preview(payload))
    assert "Content-Security-Policy" in document
    if payload.startswith(b"<!DOCTYPE"):
        assert document.lstrip().lower().startswith("<!doctype")
    if b"<header>" in payload:
        assert document.index("Content-Security-Policy") < document.index("<header>")


def test_preview_receipts_and_failure(tmp_path: Path) -> None:
    """成功和失败均回传请求标识，预览错误不向解析流程抛出。"""
    source = tmp_path / "中文.OFD"
    source.write_bytes(ofd_bytes())
    ticket = json.dumps({"id": "request-1", "path": str(source)})
    receipt = json.loads(asyncio.run(prepare_source_preview(str(source), ticket)))
    assert receipt["id"] == "request-1" and "iframe" in receipt["html"]
    source.write_bytes(b"bad")
    receipt = json.loads(asyncio.run(prepare_source_preview(str(source), ticket)))
    assert "ofd_preview_failed" in receipt["html"]
    assert json.loads(asyncio.run(prepare_source_preview(None, ticket)))["html"] == ""
    assert json.loads(asyncio.run(prepare_source_preview(str(source) + "x", ticket)))["html"] == ""


@pytest.mark.parametrize("name", ["page.html", "page.htm", "page.shtml"])
def test_html_preview_receipts(tmp_path: Path, name: str) -> None:
    """三种 HTML 后缀都生成预览回执；文件消失与未知后缀走失败或空回执。"""
    source = tmp_path / name
    source.write_text("<!DOCTYPE html><html><body><p>hello</p></body></html>", encoding="utf-8")
    ticket = json.dumps({"id": "request-2", "path": str(source)})
    receipt = json.loads(asyncio.run(prepare_source_preview(str(source), ticket)))
    assert receipt["id"] == "request-2"
    assert "mineru-source-frame" in receipt["html"] and "Content-Security-Policy" in receipt["html"]
    source.unlink()
    receipt = json.loads(asyncio.run(prepare_source_preview(str(source), ticket)))
    assert "html_preview_failed" in receipt["html"]
    other = tmp_path / "notes.txt"
    other.write_text("plain", encoding="utf-8")
    assert json.loads(asyncio.run(prepare_source_preview(str(other), json.dumps({"id": 1, "path": str(other)}))))["html"] == ""


def test_epub_preview_receipt_delegates_to_browser_without_reading_payload(tmp_path: Path) -> None:
    """EPUB 源预览只返回浏览器 viewer 标识，不在 Gradio 后端重复解析 EPUB。"""
    source = tmp_path / "source.epub"
    source.write_bytes(b"not an epub package")
    ticket = json.dumps({"id": "request-epub", "path": str(source)})
    receipt = json.loads(asyncio.run(prepare_source_preview(str(source), ticket)))
    assert receipt == {"id": "request-epub", "html": "", "kind": "epub"}


@pytest.mark.parametrize("name", ["source.ofd", "source.html"])
@pytest.mark.parametrize("fail", [False, True])
def test_parse_preserves_source_preview(tmp_path: Path, name: str, fail: bool) -> None:
    """解析成功或失败只更新结果，OFD/HTML 源预览始终由独立事件维护。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from test_kit_gradio import _middle_json

    from mineru.kit.gradio.app import build_gradio_app
    from mineru.kit.gradio.client import V1ServerCapabilities
    from mineru.parser.base import ParseResult

    source = tmp_path / name
    source.write_bytes(ofd_bytes() if name.endswith(".ofd") else b"<html><body><p>hi</p></body></html>")
    parse = AsyncMock(return_value=ParseResult(middle_json=_middle_json(with_image=False)))
    if fail:
        parse.side_effect = ValueError("expected parse failure")
    demo = build_gradio_app(
        SimpleNamespace(parse_file=parse),
        V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("zip",), ("file_id",)),
        output_root=tmp_path / "results",
        enable_example=False,
    )
    convert = next(fn for fn in demo.fns.values() if fn.name == "convert_handler")
    preview_block = next(block for block in demo.blocks.values() if "mineru-kit-source-preview" in (block.elem_classes or []))
    assert preview_block not in convert.outputs
    upload = next(fn for fn in demo.fns.values() if fn.name == "update_file_preview")
    updates = upload.fn(str(source))
    assert all(update.get("visible") is False for update in updates[:4])

    async def collect() -> list:
        """收集真实解析事件的完整输出序列。"""
        return [value async for value in convert.fn(str(source), 0, "")]

    final = asyncio.run(collect())[-1]
    assert final[2:6] == ({"__type__": "update"},) * 4
    if fail:
        assert "expected parse failure" in final[0]
    else:
        assert final[6] is not None
        assert final[-1]


def test_browser_request_lifecycle() -> None:
    """通过真实前端脚本验证乱序完成、清除和同文件重传不会恢复旧预览。"""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for preview lifecycle checks")
    subprocess.run([node, str(Path(__file__).with_suffix(".cjs"))], check=True, timeout=15)
