from __future__ import annotations

import base64
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def build_epub_fixture(
    *,
    corrupt_second_chapter: bool = False,
    dtd_first_chapter: bool = False,
    encrypted_paths: tuple[str, ...] = (),
    omit_mimetype: bool = False,
    unsafe_member: str | None = None,
    use_foreign_fallback: bool = False,
    include_nav: bool = True,
    include_ncx: bool = True,
    corrupt_nav: bool = False,
    nav_in_spine: bool = False,
    strip_headings: bool = False,
) -> bytes:
    """构造覆盖 XHTML、SVG、CSS、公式、表格、列表和图片的最小 EPUB 3。"""
    container = """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="EPUB/package.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    opf = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:mineru-epub-test</dc:identifier>
    <dc:title>EPUB Fixture</dc:title><dc:creator>Alice</dc:creator>
    <dc:subject>Testing</dc:subject><meta property="keywords">epub, mineru</meta>
  </metadata>
  <manifest>
    <item id="c1" href="text/ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>
    <item id="c3" href="fixed/page.svg" media-type="image/svg+xml"/>
    <item id="css" href="styles/book.css" media-type="text/css"/>
    <item id="img" href="images/dot.png" media-type="image/png"/>
  </manifest>
  <spine><itemref idref="c1"/><itemref idref="c2" linear="no"/><itemref idref="c3"/></spine>
</package>"""
    manifest_additions = []
    if include_nav:
        manifest_additions.append('<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>')
    if include_ncx:
        manifest_additions.append('<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
    opf = opf.replace("</manifest>", f"{''.join(manifest_additions)}</manifest>")
    if include_ncx:
        opf = opf.replace("<spine>", '<spine toc="ncx">')
    if nav_in_spine:
        opf = opf.replace("<itemref idref=\"c1\"/>", '<itemref idref="nav"/><itemref idref="c1"/>')
    if use_foreign_fallback:
        opf = opf.replace(
            '<item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>',
            '<item id="c2" href="missing.bin" media-type="application/octet-stream" fallback="c2-fallback"/>'
            '<item id="c2-fallback" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>',
        )
    chapter_one = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head>
  <link rel="stylesheet" href="../styles/book.css"/>
</head><body>
  <h1 id="chapter-one">Chapter One</h1>
  <p>Hello <strong>bold</strong>, <em>italic</em> and
    <a href="ch2.xhtml#section-two">chapter two</a>.</p>
  <p class="hidden">hidden secret</p>
  <script>alert('active')</script><p><img src="https://example.com/remote.png" alt="Remote image"/></p>
  <ol reversed="reversed" start="3" type="A"><li>Three</li><li>Two</li></ol>
  <p>Inline math <math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mi>x</mi><mn>2</mn></mfrac></math>.</p>
  <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mi>y</mi>
    <annotation encoding="application/x-tex">y^2</annotation></semantics></math>
  <figure><img src="../images/dot.png" alt="Tiny dot"/><figcaption>Dot caption</figcaption></figure>
</body></html>"""
    if dtd_first_chapter:
        chapter_one = chapter_one.replace(
            '<html xmlns="http://www.w3.org/1999/xhtml">',
            '<!DOCTYPE html [<!ENTITY x "hidden">]><html xmlns="http://www.w3.org/1999/xhtml">',
            1,
        )
    chapter_two = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>
  <section id="section-two"><h2>Section Two</h2></section>
  <table><caption>Data table</caption><thead><tr><th>A</th><th>B</th></tr></thead>
    <tbody><tr><td rowspan="2">1</td><td>2</td></tr><tr><td>3</td></tr></tbody></table>
  <pre>if x &lt; 2:\n    print(x)</pre>
  <aside epub:type="footnote"><p>[1] Footnote body</p></aside>
  <p><a href="ch1.xhtml#chapter-one">Back</a></p>
</body></html>"""
    if strip_headings:
        chapter_one = chapter_one.replace('<h1 id="chapter-one">Chapter One</h1>', '<p id="chapter-one">Chapter One</p>')
        chapter_two = chapter_two.replace("<h2>Section Two</h2>", "<p>Section Two</p>")
    nav_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>
  <nav epub:type="toc" role="doc-toc"><ol>
    <li><a href="text/ch1.xhtml#chapter-one"><img src="images/missing.png" alt="NAV Chapter One"/></a>
      <ol><li><a href="text/ch2.xhtml#section-two">NAV Section Two</a></li></ol>
    </li>
    <li><a href="missing.xhtml#appendix">NAV Missing Appendix</a></li>
  </ol></nav>
  <nav epub:type="landmarks"><ol><li><a href="text/ch1.xhtml">Landmark</a></li></ol></nav>
</body></html>"""
    ncx = """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><navMap>
  <navPoint id="n1"><navLabel><text>NCX Chapter One</text></navLabel><content src="text/ch1.xhtml#chapter-one"/>
    <navPoint id="n2"><navLabel><text>NCX Section Two</text></navLabel><content src="text/ch2.xhtml#section-two"/></navPoint>
  </navPoint>
</navMap></ncx>"""
    svg_page = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 100 100">
  <title>Fixed page</title><text x="5" y="20">SVG text</text>
  <image xlink:href="../images/dot.png" x="0" y="30" width="10" height="10"/>
</svg>"""
    encryption = """<?xml version="1.0" encoding="UTF-8"?>
<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container"
 xmlns:enc="http://www.w3.org/2001/04/xmlenc#">
  {}
</encryption>""".format(
        "".join(
            f'<enc:EncryptedData><enc:CipherData><enc:CipherReference URI="{path}"/></enc:CipherData></enc:EncryptedData>'
            for path in encrypted_paths
        )
    )
    output = BytesIO()
    with ZipFile(output, "w") as package:
        if not omit_mimetype:
            package.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        package.writestr("META-INF/container.xml", container, compress_type=ZIP_DEFLATED)
        if encrypted_paths:
            package.writestr("META-INF/encryption.xml", encryption, compress_type=ZIP_DEFLATED)
        package.writestr("EPUB/package.opf", opf, compress_type=ZIP_DEFLATED)
        if include_nav:
            package.writestr("EPUB/nav.xhtml", "<broken" if corrupt_nav else nav_xhtml, compress_type=ZIP_DEFLATED)
        if include_ncx:
            package.writestr("EPUB/toc.ncx", ncx, compress_type=ZIP_DEFLATED)
        package.writestr("EPUB/text/ch1.xhtml", chapter_one, compress_type=ZIP_DEFLATED)
        package.writestr(
            "EPUB/text/ch2.xhtml",
            "<broken" if corrupt_second_chapter else chapter_two,
            compress_type=ZIP_DEFLATED,
        )
        package.writestr("EPUB/fixed/page.svg", svg_page, compress_type=ZIP_DEFLATED)
        package.writestr("EPUB/styles/book.css", ".hidden { display: none; }", compress_type=ZIP_DEFLATED)
        package.writestr("EPUB/images/dot.png", _PNG_BYTES, compress_type=ZIP_DEFLATED)
        if unsafe_member:
            package.writestr(unsafe_member, b"unsafe", compress_type=ZIP_DEFLATED)
    return output.getvalue()


def build_epub2_fixture() -> bytes:
    """构造带 NCX manifest 的最小 EPUB 2，正文仍由 spine XHTML 提供。"""
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        package.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
            compress_type=ZIP_DEFLATED,
        )
        package.writestr(
            "OEBPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf" version="2.0"><metadata '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>EPUB 2</dc:title></metadata><manifest>'
            '<item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
            '</manifest><spine toc="ncx"><itemref idref="chapter"/></spine></package>',
            compress_type=ZIP_DEFLATED,
        )
        package.writestr(
            "OEBPS/chapter.xhtml",
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">'
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>EPUB Two Chapter</h1><p>Legacy&nbsp;body</p></body></html>',
            compress_type=ZIP_DEFLATED,
        )
        package.writestr(
            "OEBPS/toc.ncx",
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap><navPoint id="one">'
            '<navLabel><text>EPUB 2 Entry</text></navLabel><content src="chapter.xhtml"/>'
            "</navPoint></navMap></ncx>",
            compress_type=ZIP_DEFLATED,
        )
    return output.getvalue()


def build_epub_notes_fixture() -> bytes:
    """构造覆盖 Footnote、Endnote、ARIA role、重复 ID 和复杂子块的 EPUB 3。"""
    container = (
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles>'
        '<rootfile full-path="EPUB/package.opf" media-type="application/oebps-package+xml"/>'
        "</rootfiles></container>"
    )
    opf = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Notes Fixture</dc:title></metadata>
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
    <item id="notes" href="notes.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="chapter"/><itemref idref="notes"/></spine>
</package>"""
    chapter = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>
  <h1 id="chapter">Notes Chapter</h1>
  <p>Same-page <a id="ref-one" epub:type="noteref" href="#fn-one">[1]</a>,
    cross-page <a role="doc-noteref" href="notes.xhtml#end-one">[2]</a>,
    duplicate <a href="#duplicate-note">[3]</a>, and empty <a href="#empty-note">[4]</a>.</p>
  <aside id="fn-one" epub:type="footnote">
    <p>First footnote paragraph <a epub:type="backlink" href="#ref-one">back</a>.</p>
    <p>Second footnote paragraph.</p>
    <ul><li>Footnote sibling list</li></ul>
  </aside>
  <aside id="role-note" role="doc-footnote"><p>ARIA footnote.</p></aside>
  <div epub:type="rearnote"><p>Legacy rearnote.</p></div>
  <aside><p>Ordinary aside.</p></aside>
  <aside epub:type="footnotes"><p>Footnotes collection label.</p></aside>
  <aside id="empty-note" epub:type="footnote"><table><tr><td>Complex only</td></tr></table></aside>
  <aside id="duplicate-note" epub:type="footnote"><p>First duplicate note.</p></aside>
  <aside id="duplicate-note" epub:type="footnote"><p>Second duplicate note.</p></aside>
</body></html>"""
    notes = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>
  <h2>Endnotes</h2>
  <section epub:type="endnotes"><ol>
    <li id="end-one" epub:type="endnote"><p>First endnote paragraph.</p><p>Second endnote paragraph.</p></li>
    <li id="end-two" role="doc-endnote">ARIA endnote.</li>
    <li epub:type="endnote">Anonymous endnote.</li>
  </ol></section>
</body></html>"""
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        package.writestr("META-INF/container.xml", container, compress_type=ZIP_DEFLATED)
        package.writestr("EPUB/package.opf", opf, compress_type=ZIP_DEFLATED)
        package.writestr("EPUB/chapter.xhtml", chapter, compress_type=ZIP_DEFLATED)
        package.writestr("EPUB/notes.xhtml", notes, compress_type=ZIP_DEFLATED)
    return output.getvalue()


__all__ = ["build_epub2_fixture", "build_epub_fixture", "build_epub_notes_fixture"]
