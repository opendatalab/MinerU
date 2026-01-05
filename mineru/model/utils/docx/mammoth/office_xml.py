from .lists import flat_map
from .xmlparser import parse_xml, XmlElement, parse_xml_str

_namespaces = [
    # Transitional format
    ("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main"),
    ("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
    ("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
    ("a", "http://schemas.openxmlformats.org/drawingml/2006/main"),
    ("pic", "http://schemas.openxmlformats.org/drawingml/2006/picture"),

    # Strict format
    ("w", "http://purl.oclc.org/ooxml/wordprocessingml/main"),
    ("r", "http://purl.oclc.org/ooxml/officeDocument/relationships"),
    ("wp", "http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing"),
    ("a", "http://purl.oclc.org/ooxml/drawingml/main"),
    ("pic", "http://purl.oclc.org/ooxml/drawingml/picture"),

    # Common
    ("content-types", "http://schemas.openxmlformats.org/package/2006/content-types"),
    ("relationships", "http://schemas.openxmlformats.org/package/2006/relationships"),
    ("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006"),
    ("v", "urn:schemas-microsoft-com:vml"),
    ("office-word", "urn:schemas-microsoft-com:office:word"),

    # [MS-DOCX]: Word Extensions to the Office Open XML (.docx) File Format
    # https://learn.microsoft.com/en-us/openspecs/office_standards/ms-docx/b839fe1f-e1ca-4fa6-8c26-5954d0abbccd
    ("wordml", "http://schemas.microsoft.com/office/word/2010/wordml"),
]


def read(fileobj):
    i = parse_xml(fileobj, _namespaces)
    return _collapse_alternate_content(i)[0]

def read_str(xml_str):
    i = parse_xml_str(xml_str, _namespaces)
    return _collapse_alternate_content(i)[0]

def _collapse_alternate_content(node):
    if isinstance(node, XmlElement):
        if node.name == "mc:AlternateContent":
            return node.find_child_or_null("mc:Fallback").children
        else:
            node.children = flat_map(_collapse_alternate_content, node.children)
            return [node]
    else:
        return [node]
