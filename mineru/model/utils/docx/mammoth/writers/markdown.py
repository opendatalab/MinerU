from __future__ import unicode_literals

from .abc import Writer

import re


class _WriterOutput(object):
    def __init__(self, start, end=None, generate_end=None, anchor_position=None):
        if generate_end is None:
            generate_end = _constant(end)
        
        self.start = start
        self.generate_end = generate_end
        self.anchor_position = anchor_position


def _constant(value):
    def get():
        return value
    
    return get


class _MarkdownState(object):
    def __init__(self):
        self._list_state_stack = []
        self.list_state = None
        self.list_item_has_closed = False
    
    def update_list_state(self, list_state):
        self._list_state_stack.append(self.list_state)
        self.list_state = list_state
    
    def pop_list_state(self):
        self.list_state = self._list_state_stack.pop()


class _MarkdownListState(object):
    def __init__(self, ordered, indentation):
        self.ordered = ordered
        self.count = 0
        self.indentation = indentation


def _symmetric_wrapped(end):
    return _Wrapped(end, end)


class _Wrapped(object):
    def __init__(self, start, end):
        self._start = start
        self._end = end
    
    def __call__(self, attributes, markdown_state):
        return _WriterOutput(self._start, self._end)


def _hyperlink(attributes, markdown_state):
    href = attributes.get("href", "")
    if href:
        return _WriterOutput(
            "[", "]({0})".format(href),
            anchor_position="before",
        )
    else:
        return _default_output


def _image(attributes, markdown_state):
    src = attributes.get("src", "")
    alt_text = attributes.get("alt", "")
    if src or alt_text:
        return _WriterOutput("![{0}]({1})".format(alt_text, src), "")
    else:
        return _default_output


def _list(ordered):
    def call(attributes, markdown_state):
        if markdown_state.list_state is None:
            start = ""
            end_text = "\n"
            indentation = 0
        else:
            start = "\n"
            end_text = ""
            indentation = markdown_state.list_state.indentation + 1
        
        def generate_end():
            markdown_state.pop_list_state()
            return end_text
        
        markdown_state.update_list_state(_MarkdownListState(
            ordered=ordered,
            indentation=indentation,
        ))
        
        return _WriterOutput(start, generate_end=generate_end)
    
    return call


def _list_item(attributes, markdown_state):
    markdown_state.list_item_has_closed = False
    
    list_state = markdown_state.list_state or _MarkdownListState(ordered=False, indentation=0)
    list_state.count += 1
    
    if list_state.ordered:
        bullet = "{0}.".format(list_state.count)
    else:
        bullet = "-"
    
    def generate_end():
        if markdown_state.list_item_has_closed:
            return ""
        else:
            markdown_state.list_item_has_closed = True
            return "\n"
    
    return _WriterOutput(
        start=("\t" * list_state.indentation) + bullet + " ",
        generate_end=generate_end
    )


def _init_writers():
    writers = {
        "p": _Wrapped("", "\n\n"),
        "br": _Wrapped("", "  \n"),
        "strong": _symmetric_wrapped("__"),
        "em": _symmetric_wrapped("*"),
        "a": _hyperlink,
        "img": _image,
        "ol": _list(ordered=True),
        "ul": _list(ordered=False),
        "li": _list_item,
    }
    
    for level in range(1, 7):
        writers["h{0}".format(level)] = _Wrapped("#" * level + " ", "\n\n")
    
    return writers


_writers = _init_writers()
_default_output = _WriterOutput("", "")

def _default_writer(attributes, markdown_state):
    return _default_output


class MarkdownWriter(Writer):
    def __init__(self):
        self._fragments = []
        self._element_stack = []
        self._markdown_state = _MarkdownState()
    
    def text(self, text):
        self._fragments.append(_escape_markdown(text))
    
    def start(self, name, attributes=None):
        if attributes is None:
            attributes = {}
        
        output = _writers.get(name, _default_writer)(attributes, self._markdown_state)
        self._element_stack.append(output.generate_end)
        
        anchor_before_start = output.anchor_position == "before"
        if anchor_before_start:
            self._write_anchor(attributes)
        
        self._fragments.append(output.start)
        
        if not anchor_before_start:
            self._write_anchor(attributes)
        
        

    def end(self, name):
        end = self._element_stack.pop()
        output = end()
        self._fragments.append(output)
    
    def self_closing(self, name, attributes=None):
        self.start(name, attributes)
        self.end(name)
    
    def append(self, other):
        self._fragments.append(other)
    
    def as_string(self):
        return "".join(self._fragments)
    
    def _write_anchor(self, attributes):
        html_id = attributes.get("id")
        if html_id:
            self._fragments.append('<a id="{0}"></a>'.format(html_id))


def _escape_markdown(value):
    return re.sub(r"([\`\*_\{\}\[\]\(\)\#\+\-\.\!])", r"\\\1", re.sub("\\\\", "\\\\\\\\", value))
