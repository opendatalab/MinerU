"""High-level OFD reader: parses OFD.xml + Document.xml + per-page Content.xml.

Designed for the HTML export pipeline only -- enough structure to render the
page contents to SVG. We keep the parsed forms as light dataclasses since
rendering is the only consumer here.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Optional

from lxml import etree

from ..gv import NSMAP
from ..pkg.container import OFDContainer
from .resource_locator import ResourceLocator


# --------------------------------------------------------------------------- #
# Light value objects.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Box:
    """OFD ``ST_Box`` value: x, y, w, h in millimetres."""

    x: float
    y: float
    w: float
    h: float

    @classmethod
    # 将坐标文本转换为矩形。
    def parse(cls, raw: Optional[str]) -> Optional["Box"]:
        if not raw:
            return None
        parts = raw.replace(",", " ").split()
        if len(parts) != 4:
            return None
        return cls(*(float(p) for p in parts))


@dataclass
class PageRef:
    """A pointer to one page's Content.xml inside the container."""

    page_id: str
    base_loc: str  # absolute virtual path


@dataclass
class MediaRes:
    res_id: str
    media_type: str  # "Image" / "Sound" / ...
    file_path: str  # absolute virtual path resolved against res base


@dataclass
class DrawParam:
    """Subset of OFD ``CT_DrawParam`` needed for HTML rendering.

    OFD allows ``Relative`` to chain a base DrawParam; we keep the raw id
    here and resolve transitively via :meth:`Document.resolve_draw_param`
    so callers always see the merged effective values.
    """

    res_id: str
    relative: Optional[str] = None
    line_width: Optional[float] = None
    fill_color: Optional[str] = None  # raw ``Value`` channel string, e.g. "255 0 0"
    stroke_color: Optional[str] = None


@dataclass
class TemplatePage:
    """Reusable page-content shared by multiple pages.

    A page references it via ``<ofd:Template TemplateID="..." ZOrder="..."/>``
    and the renderer composites the template content under or over the page's
    own content according to ``ZOrder`` (Background / Body / Foreground).
    """

    template_id: str
    base_loc: str  # absolute virtual path to the template's Content.xml
    z_order: str = "Background"


@dataclass
class Document:
    physical_box: Box
    pages: list[PageRef] = field(default_factory=list)
    medias: dict[str, MediaRes] = field(default_factory=dict)
    draw_params: dict[str, DrawParam] = field(default_factory=dict)
    templates: dict[str, TemplatePage] = field(default_factory=dict)

    # 合并继承的绘制参数。
    def resolve_draw_param(self, res_id: Optional[str]) -> Optional[DrawParam]:
        """Return a flattened ``DrawParam`` with ``Relative`` chain merged in.

        The OFD spec lets a DrawParam inherit from a base via ``Relative`` and
        override individual properties. We walk the chain (base-first), then
        let the leaf override -- mirroring the reference renderer behaviour.
        Returns ``None`` for unknown ids so callers can do a single guarded
        lookup.
        """
        if not res_id:
            return None
        leaf = self.draw_params.get(res_id)
        if leaf is None:
            return None
        chain: list[DrawParam] = [leaf]
        seen = {res_id}
        cur = leaf
        while cur.relative and cur.relative not in seen:
            base = self.draw_params.get(cur.relative)
            if base is None:
                break
            seen.add(cur.relative)
            chain.append(base)
            cur = base
        # Walk base-first, then leaf, so leaf overrides win.
        merged = DrawParam(res_id=res_id)
        for dp in reversed(chain):
            if dp.line_width is not None:
                merged.line_width = dp.line_width
            if dp.fill_color is not None:
                merged.fill_color = dp.fill_color
            if dp.stroke_color is not None:
                merged.stroke_color = dp.stroke_color
        return merged


# --------------------------------------------------------------------------- #
# Reader.
# --------------------------------------------------------------------------- #


class OFDReader:
    """Read an OFD container and expose per-page content trees."""

    # 初始化当前转换对象及资源。
    def __init__(self, ofd_bytes: bytes) -> None:
        self._container = OFDContainer(ofd_bytes)
        self._locator = ResourceLocator(self._container, cwd="/")
        self._documents: list[Document] = self._load_documents()

    # ----- public API -------------------------------------------------------

    @property
    # 返回已加载的 OFD 文档列表。
    def documents(self) -> list[Document]:
        return self._documents

    # 读取并解析页面 XML。
    def page_content(self, doc: Document, page: PageRef) -> etree._Element:
        """Return the parsed ``<ofd:Page>`` root for one page."""
        del doc  # currently unused, but kept for future multi-doc routing
        xml = self._container.read(page.base_loc.lstrip("/"))
        return etree.fromstring(xml, parser=_xml_parser())

    # 读取并解析模板页 XML。
    def template_content(
        self, doc: Document, template_id: str
    ) -> Optional[etree._Element]:
        """Return the parsed root element of the template page identified by
        ``template_id`` (or ``None`` if unknown / missing)."""
        del doc
        tpl = self._documents[0].templates.get(template_id) if self._documents else None
        # Search across all documents because ``doc`` is currently unused above.
        for d in self._documents:
            tpl = d.templates.get(template_id)
            if tpl is not None:
                break
        if tpl is None:
            return None
        path = tpl.base_loc.lstrip("/")
        if not self._container.has(path):
            return None
        return etree.fromstring(self._container.read(path), parser=_xml_parser())

    # 读取媒体资源字节。
    def read_media(self, doc: Document, res_id: str) -> Optional[bytes]:
        media = doc.medias.get(res_id)
        if media is None:
            return None
        return self._container.read(media.file_path.lstrip("/"))

    # 按资源编号查找媒体信息。
    def media(self, doc: Document, res_id: str) -> Optional[MediaRes]:
        return doc.medias.get(res_id)

    # 关闭持有的 OFD 压缩包。
    def close(self) -> None:
        self._container.close()

    # 进入资源管理上下文。
    def __enter__(self) -> "OFDReader":
        return self

    # 退出上下文并释放资源。
    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- loading ----------------------------------------------------------

    # 从入口 XML 加载文档列表。
    def _load_documents(self) -> list[Document]:
        if not self._container.has("OFD.xml"):
            raise ValueError("invalid OFD: missing OFD.xml")
        ofd_root = etree.fromstring(self._container.read("OFD.xml"), parser=_xml_parser())
        documents: list[Document] = []
        for doc_body in ofd_root.findall("ofd:DocBody", NSMAP):
            doc_root_el = doc_body.find("ofd:DocRoot", NSMAP)
            if doc_root_el is None or not doc_root_el.text:
                continue
            doc_root_path = self._locator.resolve(doc_root_el.text.strip())
            documents.append(self._load_document(doc_root_path))
        if not documents:
            raise ValueError("invalid OFD: no DocBody/DocRoot found")
        return documents

    # 加载文档页序、页面尺寸、模板和资源。
    def _load_document(self, doc_root_path: str) -> Document:
        doc_xml = self._container.read(doc_root_path)
        doc_el = etree.fromstring(doc_xml, parser=_xml_parser())
        doc_dir = posixpath.dirname("/" + doc_root_path)

        # Page area.
        physical_box = Box.parse(
            _text(doc_el, ".//ofd:CommonData/ofd:PageArea/ofd:PhysicalBox")
        ) or Box(0, 0, 210, 297)

        # Pages list.
        pages: list[PageRef] = []
        for page_el in doc_el.findall(".//ofd:Pages/ofd:Page", NSMAP):
            base_loc = page_el.get("BaseLoc") or ""
            if not base_loc:
                continue
            pages.append(
                PageRef(
                    page_id=page_el.get("ID") or "",
                    base_loc=_join_doc(doc_dir, base_loc),
                )
            )

        # Resources (PublicRes + DocumentRes). Each entry can declare
        # ``BaseLoc`` to relocate sub-files relative to the resource file.
        medias: dict[str, MediaRes] = {}
        draw_params: dict[str, DrawParam] = {}
        for tag in ("ofd:PublicRes", "ofd:DocumentRes"):
            for res_el in doc_el.findall(f".//ofd:CommonData/{tag}", NSMAP):
                if not res_el.text:
                    continue
                res_path = _join_doc(doc_dir, res_el.text.strip())
                self._collect_resources(res_path, medias, draw_params)

        # Template pages are referenced by individual pages via TemplateID;
        # their content lives in a separate Content.xml that we resolve once
        # here so the renderer can simply look it up.
        templates: dict[str, TemplatePage] = {}
        for tpl_el in doc_el.findall(
            ".//ofd:CommonData/ofd:TemplatePage", NSMAP
        ):
            tpl_id = tpl_el.get("ID")
            base_loc = tpl_el.get("BaseLoc") or ""
            if not tpl_id or not base_loc:
                continue
            templates[tpl_id] = TemplatePage(
                template_id=tpl_id,
                base_loc=_join_doc(doc_dir, base_loc),
                z_order=tpl_el.get("ZOrder") or "Background",
            )

        return Document(
            physical_box=physical_box,
            pages=pages,
            medias=medias,
            draw_params=draw_params,
            templates=templates,
        )

    # 收集媒体资源和绘制参数。
    def _collect_resources(
        self,
        res_path: str,
        media_sink: dict[str, MediaRes],
        draw_param_sink: dict[str, DrawParam],
    ) -> None:
        if not self._container.has(res_path.lstrip("/")):
            return
        res_el = etree.fromstring(self._container.read(res_path.lstrip("/")), parser=_xml_parser())
        res_dir = posixpath.dirname(res_path)
        # ``BaseLoc`` on <ofd:Res> further roots subsequent file refs.
        base_loc = res_el.get("BaseLoc") or ""
        media_dir = (
            posixpath.normpath(posixpath.join(res_dir, base_loc))
            if base_loc
            else res_dir
        )
        if not media_dir.startswith("/"):
            media_dir = "/" + media_dir

        for mm in res_el.findall(".//ofd:MultiMedias/ofd:MultiMedia", NSMAP):
            res_id = mm.get("ID")
            if not res_id:
                continue
            mfile_el = mm.find("ofd:MediaFile", NSMAP)
            if mfile_el is None or not mfile_el.text:
                continue
            file_path = posixpath.normpath(
                posixpath.join(media_dir, mfile_el.text.strip())
            )
            if not file_path.startswith("/"):
                file_path = "/" + file_path
            media_sink[res_id] = MediaRes(
                res_id=res_id,
                media_type=mm.get("Type") or "Image",
                file_path=file_path,
            )

        for dp_el in res_el.findall(".//ofd:DrawParams/ofd:DrawParam", NSMAP):
            res_id = dp_el.get("ID")
            if not res_id:
                continue
            line_width: Optional[float] = None
            lw_raw = dp_el.get("LineWidth")
            if lw_raw:
                try:
                    line_width = float(lw_raw)
                except ValueError:
                    line_width = None
            fill_el = dp_el.find("ofd:FillColor", NSMAP)
            stroke_el = dp_el.find("ofd:StrokeColor", NSMAP)
            draw_param_sink[res_id] = DrawParam(
                res_id=res_id,
                relative=dp_el.get("Relative"),
                line_width=line_width,
                fill_color=(fill_el.get("Value") if fill_el is not None else None),
                stroke_color=(stroke_el.get("Value") if stroke_el is not None else None),
            )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


# 按 XPath 读取可选文字内容。
def _text(node: etree._Element, xpath: str) -> Optional[str]:
    el = node.find(xpath, NSMAP)
    if el is None:
        return None
    return (el.text or "").strip() or None


# 解析相对文档路径。
def _join_doc(doc_dir: str, rel: str) -> str:
    """Resolve a path that may be absolute (``/Doc_0/...``) or relative
    to the directory of Document.xml."""
    if rel.startswith("/"):
        return posixpath.normpath(rel)
    return posixpath.normpath(posixpath.join(doc_dir, rel))


def _xml_parser() -> etree.XMLParser:
    """每次读取使用独立解析器，禁止外部实体、DTD 加载和网络访问。"""
    return etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
