import xml.dom.minidom

from mammoth.docx.xmlparser import XmlText, XmlElement
from mammoth.docx.office_xml import _collapse_alternate_content, _namespaces


def parse_xml_str(xml_str, namespace_mapping=None):
    if namespace_mapping is None:
        namespace_prefixes = {}
    else:
        namespace_prefixes = dict((uri, prefix) for prefix, uri in namespace_mapping)

    document = xml.dom.minidom.parseString(xml_str)

    def convert_node(node):
        if node.nodeType == xml.dom.Node.ELEMENT_NODE:
            return convert_element(node)
        elif node.nodeType == xml.dom.Node.TEXT_NODE:
            return XmlText(node.nodeValue)
        else:
            return None

    def convert_element(element):
        converted_name = convert_name(element)

        converted_attributes = dict(
            (convert_name(attribute), attribute.value)
            for attribute in element.attributes.values()
            if attribute.namespaceURI != "http://www.w3.org/2000/xmlns/"
        )

        converted_children = []
        for child_node in element.childNodes:
            converted_child_node = convert_node(child_node)
            if converted_child_node is not None:
                converted_children.append(converted_child_node)

        return XmlElement(converted_name, converted_attributes, converted_children)

    def convert_name(node):
        if node.namespaceURI is None:
            return node.localName
        else:
            prefix = namespace_prefixes.get(node.namespaceURI)
            if prefix is None:
                return "{%s}%s" % (node.namespaceURI, node.localName)
            else:
                return "%s:%s" % (prefix, node.localName)

    return convert_node(document.documentElement)


def read_str(xml_str):
    i = parse_xml_str(xml_str, _namespaces)
    return _collapse_alternate_content(i)[0]
