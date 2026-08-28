from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile


def text_object(
    object_id: int,
    text: str,
    *,
    boundary: str,
    size: float = 5,
    x: float = 0,
    y: float = 5,
    delta_x: str | None = None,
    ctm: str | None = None,
) -> str:
    """构造测试用 TextObject XML。"""
    delta = f' DeltaX="{delta_x}"' if delta_x else ""
    transform = f' CTM="{ctm}"' if ctm else ""
    return (
        f'<ofd:TextObject ID="{object_id}" Boundary="{boundary}" Font="1" Size="{size}"{transform}>'
        f'<ofd:TextCode X="{x}" Y="{y}"{delta}>{text}</ofd:TextCode>'
        "</ofd:TextObject>"
    )


def path_object(object_id: int, *, boundary: str, data: str) -> str:
    """构造测试用 PathObject XML。"""
    return (
        f'<ofd:PathObject ID="{object_id}" Boundary="{boundary}" LineWidth="0.2">'
        f"<ofd:AbbreviatedData>{data}</ofd:AbbreviatedData>"
        "</ofd:PathObject>"
    )


def page_xml(
    content: str,
    *,
    namespace: str = "http://www.ofdspec.org/2016",
    physical_box: str = "0 0 100 100",
    template_id: int | None = None,
    page_res: str | None = None,
) -> str:
    """构造测试用 OFD 页面 XML。"""
    template = f'<ofd:Template TemplateID="{template_id}" ZOrder="Background"/>' if template_id is not None else ""
    page_resource = f"<ofd:PageRes>{page_res}</ofd:PageRes>" if page_res else ""
    return (
        f'<ofd:Page xmlns:ofd="{namespace}">'
        f"<ofd:Area><ofd:PhysicalBox>{physical_box}</ofd:PhysicalBox></ofd:Area>"
        f'{page_resource}{template}<ofd:Content><ofd:Layer ID="2">{content}</ofd:Layer></ofd:Content>'
        "</ofd:Page>"
    )


def build_ofd_package(
    pages: list[tuple[str, str]],
    *,
    namespace: str = "http://www.ofdspec.org/2016",
    version: str = "1.0",
    templates: dict[int, tuple[str, str]] | None = None,
    extra_parts: dict[str, bytes | str] | None = None,
) -> bytes:
    """构造包含指定页树、模板和附加成员的最小 OFD 包。"""
    templates = templates or {}
    page_refs = "".join(f'<ofd:Page ID="{index + 10}" BaseLoc="{path}"/>' for index, (path, _content) in enumerate(pages))
    template_refs = "".join(
        f'<ofd:TemplatePage ID="{template_id}" BaseLoc="{path}"/>' for template_id, (path, _content) in templates.items()
    )
    ofd_xml = (
        f'<ofd:OFD xmlns:ofd="{namespace}" Version="{version}" DocType="OFD">'
        "<ofd:DocBody><ofd:DocInfo><ofd:Title>Fixture Title</ofd:Title>"
        "<ofd:Author>Fixture Author</ofd:Author></ofd:DocInfo>"
        "<ofd:DocRoot>Doc_0/Document.xml</ofd:DocRoot></ofd:DocBody></ofd:OFD>"
    )
    document_xml = (
        f'<ofd:Document xmlns:ofd="{namespace}"><ofd:CommonData><ofd:MaxUnitID>999</ofd:MaxUnitID>'
        "<ofd:PageArea><ofd:PhysicalBox>0 0 100 100</ofd:PhysicalBox>"
        "<ofd:ContentBox>5 5 90 90</ofd:ContentBox></ofd:PageArea>"
        "<ofd:PublicRes>PublicRes.xml</ofd:PublicRes>"
        f"{template_refs}</ofd:CommonData><ofd:Pages>{page_refs}</ofd:Pages></ofd:Document>"
    )
    public_res = (
        f'<ofd:Res xmlns:ofd="{namespace}" BaseLoc="Res"><ofd:Fonts>'
        '<ofd:Font ID="1" FontName="Fixture Sans" FamilyName="Fixture Sans"/>'
        "</ofd:Fonts></ofd:Res>"
    )
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as package:
        package.writestr("OFD.xml", ofd_xml)
        package.writestr("Doc_0/Document.xml", document_xml)
        package.writestr("Doc_0/PublicRes.xml", public_res)
        for path, content in pages:
            package.writestr(f"Doc_0/{path}", content)
        for _template_id, (path, content) in templates.items():
            package.writestr(f"Doc_0/{path}", content)
        for path, content in (extra_parts or {}).items():
            package.writestr(path, content)
    return output.getvalue()


def build_multi_document_ofd() -> bytes:
    """构造包含两个 DocBody 的测试 OFD 包。"""
    namespace = "http://www.ofdspec.org/2016"
    ofd_xml = (
        f'<ofd:OFD xmlns:ofd="{namespace}" Version="1.0" DocType="OFD">'
        "<ofd:DocBody><ofd:DocInfo><ofd:Title>First</ofd:Title></ofd:DocInfo>"
        "<ofd:DocRoot>Doc_0/Document.xml</ofd:DocRoot></ofd:DocBody>"
        "<ofd:DocBody><ofd:DocInfo><ofd:Title>Second</ofd:Title></ofd:DocInfo>"
        "<ofd:DocRoot>Doc_1/Document.xml</ofd:DocRoot></ofd:DocBody></ofd:OFD>"
    )
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as package:
        package.writestr("OFD.xml", ofd_xml)
        for document_index, label in enumerate(("doc-zero", "doc-one")):
            document_xml = (
                f'<ofd:Document xmlns:ofd="{namespace}"><ofd:CommonData><ofd:MaxUnitID>9</ofd:MaxUnitID>'
                "<ofd:PageArea><ofd:PhysicalBox>0 0 100 100</ofd:PhysicalBox></ofd:PageArea>"
                "<ofd:PublicRes>PublicRes.xml</ofd:PublicRes></ofd:CommonData>"
                '<ofd:Pages><ofd:Page ID="1" BaseLoc="Pages/Page_0/Content.xml"/></ofd:Pages></ofd:Document>'
            )
            public_res = (
                f'<ofd:Res xmlns:ofd="{namespace}" BaseLoc="Res"><ofd:Fonts>'
                '<ofd:Font ID="1" FontName="Fixture"/></ofd:Fonts></ofd:Res>'
            )
            package.writestr(f"Doc_{document_index}/Document.xml", document_xml)
            package.writestr(f"Doc_{document_index}/PublicRes.xml", public_res)
            package.writestr(
                f"Doc_{document_index}/Pages/Page_0/Content.xml",
                page_xml(text_object(2, label, boundary="10 10 40 10")),
            )
    return output.getvalue()


__all__ = ["build_multi_document_ofd", "build_ofd_package", "page_xml", "path_object", "text_object"]
