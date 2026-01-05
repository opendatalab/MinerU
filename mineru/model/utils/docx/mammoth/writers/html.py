from __future__ import unicode_literals
from xml.sax.saxutils import escape

from .abc import Writer


class HtmlWriter(Writer):
    def __init__(self):
        self._fragments = []
    
    def text(self, text):
        self._fragments.append(_escape_html(text))
    
    def start(self, name, attributes=None):
        attribute_string = _generate_attribute_string(attributes)
        self._fragments.append("<{0}{1}>".format(name, attribute_string))

    def end(self, name):
        self._fragments.append("</{0}>".format(name))
    
    def self_closing(self, name, attributes=None):
        attribute_string = _generate_attribute_string(attributes)
        self._fragments.append("<{0}{1} />".format(name, attribute_string))
    
    def append(self, html):
        self._fragments.append(html)
    
    def as_string(self):
        return "".join(self._fragments)


def _escape_html(text):
    return escape(text, {'"': "&quot;"})


def _generate_attribute_string(attributes):
    if attributes is None:
        return ""
    else:
        return "".join(
            ' {0}="{1}"'.format(key, _escape_html(attributes[key]))
            for key in sorted(attributes)
        )
