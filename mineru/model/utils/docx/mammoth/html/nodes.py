import cobble


class Node(object):
    pass


@cobble.data
class TextNode(Node):
    value = cobble.field()


@cobble.data
class Tag(object):
    tag_names = cobble.field()
    attributes = cobble.field()
    collapsible = cobble.field()
    separator = cobble.field()

    @property
    def tag_name(self):
        return self.tag_names[0]


@cobble.data
class Element(Node):
    tag = cobble.field()
    children = cobble.field()

    @property
    def tag_name(self):
        return self.tag.tag_name

    @property
    def tag_names(self):
        return self.tag.tag_names

    @property
    def attributes(self):
        return self.tag.attributes

    @property
    def collapsible(self):
        return self.tag.collapsible

    @property
    def separator(self):
        return self.tag.separator

    _VOID_TAG_NAMES = set(["br", "hr", "img", "input"])

    def is_void(self):
        return not self.children and self.tag_name in self._VOID_TAG_NAMES


@cobble.visitable
class ForceWrite(Node):
    pass


NodeVisitor = cobble.visitor(Node)
