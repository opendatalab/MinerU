import xml.dom.minidom

import cobble


@cobble.data
class XmlElement(object):
    name = cobble.field()
    attributes = cobble.field()
    children = cobble.field()

    def find_child_or_null(self, name):
        return self.find_child(name) or null_xml_element

    def find_child(self, name):
        for child in self.children:
            if isinstance(child, XmlElement) and child.name == name:
                return child


    def find_children(self, name):
        return XmlElementList(filter(
            lambda child: child.node_type == node_types.element and child.name == name,
            self.children
        ))


class XmlElementList(object):
    def __init__(self, elements):
        self._elements = elements

    def __iter__(self):
        return iter(self._elements)

    def find_children(self, name):
        children = []
        for element in self._elements:
            for child in element.find_children(name):
                children.append(child)
        return XmlElementList(children)


class NullXmlElement(object):
    attributes = {}
    children = []

    def find_child_or_null(self, name):
        return self

    def find_child(self, name):
        return None


null_xml_element = NullXmlElement()


@cobble.data
class XmlText(object):
    value = cobble.field()


def element(name, attributes=None, children=None):
    return XmlElement(name, attributes or {}, children or [])

text = XmlText


class node_types(object):
    element = 1
    text = 3


XmlElement.node_type = node_types.element
XmlText.node_type = node_types.text



def parse_xml(fileobj, namespace_mapping=None):
    if namespace_mapping is None:
        namespace_prefixes = {}
    else:
        namespace_prefixes = dict((uri, prefix) for prefix, uri in namespace_mapping)

    document = xml.dom.minidom.parse(fileobj)

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
