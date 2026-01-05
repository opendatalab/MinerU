import cobble


class Element(object):
    def copy(self, **kwargs):
        return cobble.copy(self, **kwargs)


class HasChildren(Element):
    children = cobble.field()


@cobble.data
class Document(HasChildren):
    notes = cobble.field()
    comments = cobble.field()

@cobble.data
class Paragraph(HasChildren):
    style_id = cobble.field()
    style_name = cobble.field()
    numbering = cobble.field()
    alignment = cobble.field()
    indent = cobble.field()


@cobble.data
class ParagraphIndent(object):
    start = cobble.field()
    end = cobble.field()
    first_line = cobble.field()
    hanging = cobble.field()


@cobble.data
class Indent(object):
    left = cobble.field()
    right = cobble.field()
    first_line = cobble.field()
    hanging = cobble.field()


@cobble.data
class Run(HasChildren):
    style_id = cobble.field()
    style_name = cobble.field()
    is_bold = cobble.field()
    is_italic = cobble.field()
    is_underline = cobble.field()
    is_strikethrough = cobble.field()
    is_all_caps = cobble.field()
    is_small_caps = cobble.field()
    vertical_alignment = cobble.field()
    font = cobble.field()
    font_size = cobble.field()
    highlight = cobble.field()

@cobble.data
class Text(Element):
    value = cobble.field()

@cobble.data
class Hyperlink(HasChildren):
    href = cobble.field()
    anchor = cobble.field()
    target_frame = cobble.field()

@cobble.data
class Checkbox(Element):
    checked = cobble.field()

checkbox = Checkbox

@cobble.data
class Table(HasChildren):
    style_id = cobble.field()
    style_name = cobble.field()

@cobble.data
class TableRow(HasChildren):
    is_header = cobble.field()

@cobble.data
class TableCell(HasChildren):
    colspan = cobble.field()
    rowspan = cobble.field()

@cobble.data
class TableCellUnmerged:
    children = cobble.field()
    colspan = cobble.field()
    rowspan = cobble.field()
    vmerge = cobble.field()

    def _accept1(self, visitor, arg0):
        return visitor.visit_table_cell(self, arg0)

    def copy(self, **kwargs):
        return cobble.copy(self, **kwargs)

@cobble.data
class Break(Element):
    break_type = cobble.field()

line_break = Break("line")
page_break = Break("page")
column_break = Break("column")


@cobble.data
class Tab(Element):
    pass


@cobble.data
class Image(Element):
    alt_text = cobble.field()
    content_type = cobble.field()
    open = cobble.field()


def document(children, notes=None, comments=None):
    if notes is None:
        notes = Notes({})
    if comments is None:
        comments = []
    return Document(children, notes, comments=comments)

def paragraph(children, style_id=None, style_name=None, numbering=None, alignment=None, indent=None):
    if indent is None:
        indent = paragraph_indent()

    return Paragraph(children, style_id, style_name, numbering, alignment=alignment, indent=indent)

def paragraph_indent(start=None, end=None, first_line=None, hanging=None):
    return ParagraphIndent(start=start, end=end, first_line=first_line, hanging=hanging)

def run(
    children,
    style_id=None,
    style_name=None,
    is_bold=None,
    is_italic=None,
    is_underline=None,
    is_strikethrough=None,
    is_all_caps=None,
    is_small_caps=None,
    vertical_alignment=None,
    font=None,
    font_size=None,
    highlight=None,
):
    if vertical_alignment is None:
        vertical_alignment = VerticalAlignment.baseline
    return Run(
        children=children,
        style_id=style_id,
        style_name=style_name,
        is_bold=bool(is_bold),
        is_italic=bool(is_italic),
        is_underline=bool(is_underline),
        is_strikethrough=bool(is_strikethrough),
        is_all_caps=bool(is_all_caps),
        is_small_caps=bool(is_small_caps),
        vertical_alignment=vertical_alignment,
        font=font,
        font_size=font_size,
        highlight=highlight,
    )

class VerticalAlignment(object):
    baseline = "baseline"
    superscript = "superscript"
    subscript = "subscript"

text = Text

_tab = Tab()

def tab():
    return _tab


image = Image

def hyperlink(children, href=None, anchor=None, target_frame=None):
    return Hyperlink(href=href, anchor=anchor, target_frame=target_frame, children=children)


@cobble.data
class Bookmark(Element):
    name = cobble.field()

bookmark = Bookmark


def table(children, style_id=None, style_name=None):
    return Table(children=children, style_id=style_id, style_name=style_name)

def table_row(children, is_header=None):
    return TableRow(children=children, is_header=bool(is_header))

def table_cell(children, colspan=None, rowspan=None):
    if colspan is None:
        colspan = 1
    if rowspan is None:
        rowspan = 1
    return TableCell(children=children, colspan=colspan, rowspan=rowspan)

def table_cell_unmerged(children, colspan, rowspan, vmerge):
    return TableCellUnmerged(children=children, colspan=colspan, rowspan=rowspan, vmerge=vmerge)

def numbering_level(level_index, is_ordered):
    return _NumberingLevel(str(level_index), bool(is_ordered))

@cobble.data
class _NumberingLevel(object):
    level_index = cobble.field()
    is_ordered = cobble.field()

@cobble.data
class Note(Element):
    note_type = cobble.field()
    note_id = cobble.field()
    body = cobble.field()


note = Note


class Notes(object):
    def __init__(self, notes):
        self._notes = notes

    def find_note(self, note_type, note_id):
        return self._notes[(note_type, note_id)]

    def resolve(self, reference):
        return self.find_note(reference.note_type, reference.note_id)

    def __eq__(self, other):
        return isinstance(other, Notes) and self._notes == other._notes

    def __ne__(self, other):
        return not (self == other)

def notes(notes_list):
    return Notes(dict(
        (_note_key(note), note)
        for note in notes_list
    ))

def _note_key(note):
    return (note.note_type, note.note_id)

@cobble.data
class NoteReference(Element):
    note_type = cobble.field()
    note_id = cobble.field()

note_reference = NoteReference


@cobble.data
class Comment(object):
    comment_id = cobble.field()
    body = cobble.field()
    author_name = cobble.field()
    author_initials = cobble.field()

def comment(comment_id, body, author_name=None, author_initials=None):
    return Comment(
        comment_id=comment_id,
        body=body,
        author_name=author_name,
        author_initials=author_initials,
    )

@cobble.data
class CommentReference(Element):
    comment_id = cobble.field()

comment_reference = CommentReference

def element_visitor(args):
    return cobble.visitor(Element, args=args)
