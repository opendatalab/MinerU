import contextlib
import re
import sys
from . import documents
from . import results
from . import lists
from . import transforms
from . import complex_fields
from . import dingbats
from .xmlparser import node_types, XmlElement, null_xml_element
from .styles_xml import Styles
from .uris import replace_fragment, uri_to_zip_entry_name

if sys.version_info >= (3, ):
    unichr = chr


def reader(
    numbering=None,
    content_types=None,
    relationships=None,
    styles=None,
    docx_file=None,
    files=None
):

    if styles is None:
        styles = Styles.EMPTY

    read_all = _create_reader(
        numbering=numbering,
        content_types=content_types,
        relationships=relationships,
        styles=styles,
        docx_file=docx_file,
        files=files,
    )
    return _BodyReader(read_all)



class _BodyReader(object):
    def __init__(self, read_all):
        self._read_all = read_all

    def read_all(self, elements):
        result = self._read_all(elements)
        return results.Result(result.elements, result.messages)


def _create_reader(numbering, content_types, relationships, styles, docx_file, files):
    current_instr_text = []
    complex_field_stack = []

    # When a paragraph is marked as deleted, its contents should be combined
    # with the following paragraph. See 17.13.5.15 del (Deleted Paragraph) of
    # ECMA-376 4th edition Part 1.
    deleted_paragraph_contents = []

    _ignored_elements = set([
        "office-word:wrap",
        "v:shadow",
        "v:shapetype",
        "w:annotationRef",
        "w:bookmarkEnd",
        "w:sectPr",
        "w:proofErr",
        "w:lastRenderedPageBreak",
        "w:commentRangeStart",
        "w:commentRangeEnd",
        "w:del",
        "w:footnoteRef",
        "w:endnoteRef",
        "w:pPr",
        "w:rPr",
        "w:tblPr",
        "w:tblGrid",
        "w:trPr",
        "w:tcPr",
    ])

    def text(element):
        return _success(documents.Text(_inner_text(element)))

    def run(element):
        properties = element.find_child_or_null("w:rPr")
        vertical_alignment = properties \
            .find_child_or_null("w:vertAlign") \
            .attributes.get("w:val")
        font = properties.find_child_or_null("w:rFonts").attributes.get("w:ascii")

        font_size_string = properties.find_child_or_null("w:sz").attributes.get("w:val")
        if _is_int(font_size_string):
            # w:sz gives the font size in half points, so halve the value to get the size in points
            font_size = int(font_size_string) / 2
        else:
            font_size = None

        is_bold = read_boolean_element(properties.find_child("w:b"))
        is_italic = read_boolean_element(properties.find_child("w:i"))
        is_underline = read_underline_element(properties.find_child("w:u"))
        is_strikethrough = read_boolean_element(properties.find_child("w:strike"))
        is_all_caps = read_boolean_element(properties.find_child("w:caps"))
        is_small_caps = read_boolean_element(properties.find_child("w:smallCaps"))
        highlight = read_highlight_value(properties.find_child_or_null("w:highlight").attributes.get("w:val"))

        def add_complex_field_hyperlink(children):
            hyperlink_kwargs = current_hyperlink_kwargs()
            if hyperlink_kwargs is None:
                return children
            else:
                return [documents.hyperlink(children=children, **hyperlink_kwargs)]

        return _ReadResult.map_results(
            _read_run_style(properties),
            _read_xml_elements(element.children).map(add_complex_field_hyperlink),
            lambda style, children: documents.run(
                children=children,
                style_id=style[0],
                style_name=style[1],
                is_bold=is_bold,
                is_italic=is_italic,
                is_underline=is_underline,
                is_strikethrough=is_strikethrough,
                is_all_caps=is_all_caps,
                is_small_caps=is_small_caps,
                vertical_alignment=vertical_alignment,
                font=font,
                font_size=font_size,
                highlight=highlight,
            ))

    def _read_run_style(properties):
        return _read_style(properties, "w:rStyle", "Run", styles.find_character_style_by_id)

    def read_boolean_element(element):
        if element is None:
            return False
        else:
            return read_boolean_attribute_value(element.attributes.get("w:val"))

    def read_boolean_attribute_value(value):
        return value not in ["false", "0"]

    def read_underline_element(element):
        return element and element.attributes.get("w:val") not in [None, "false", "0", "none"]

    def read_highlight_value(value):
        if not value or value == "none":
            return None
        else:
            return value

    def paragraph(element):
        properties = element.find_child_or_null("w:pPr")

        is_deleted = properties.find_child_or_null("w:rPr").find_child("w:del")

        if is_deleted is not None:
            for child in element.children:
                deleted_paragraph_contents.append(child)
            return _empty_result

        else:
            alignment = properties.find_child_or_null("w:jc").attributes.get("w:val")
            indent = _read_paragraph_indent(properties.find_child_or_null("w:ind"))

            children_xml = element.children
            if deleted_paragraph_contents:
                children_xml = deleted_paragraph_contents + children_xml
                del deleted_paragraph_contents[:]

            return _ReadResult.map_results(
                _read_paragraph_style(properties),
                _read_xml_elements(children_xml),
                lambda style, children: documents.paragraph(
                    children=children,
                    style_id=style[0],
                    style_name=style[1],
                    numbering=_read_numbering_properties(
                        paragraph_style_id=style[0],
                        element=properties.find_child_or_null("w:numPr"),
                    ),
                    alignment=alignment,
                    indent=indent,
                )).append_extra()

    def _read_paragraph_style(properties):
        return _read_style(properties, "w:pStyle", "Paragraph", styles.find_paragraph_style_by_id)

    def current_hyperlink_kwargs():
        for complex_field in reversed(complex_field_stack):
            if isinstance(complex_field, complex_fields.Hyperlink):
                return complex_field.kwargs

        return None

    def read_fld_char(element):
        fld_char_type = element.attributes.get("w:fldCharType")
        if fld_char_type == "begin":
            complex_field_stack.append(complex_fields.begin(fld_char=element))
            del current_instr_text[:]

        elif fld_char_type == "end":
            complex_field = complex_field_stack.pop()
            if isinstance(complex_field, complex_fields.Begin):
                complex_field = parse_current_instr_text(complex_field)

            if isinstance(complex_field, complex_fields.Checkbox):
                return _success(documents.checkbox(checked=complex_field.checked))

        elif fld_char_type == "separate":
            complex_field_separate = complex_field_stack.pop()
            complex_field = parse_current_instr_text(complex_field_separate)
            complex_field_stack.append(complex_field)

        return _empty_result

    def parse_current_instr_text(complex_field):
        instr_text = "".join(current_instr_text)

        if isinstance(complex_field, complex_fields.Begin):
            fld_char = complex_field.fld_char
        else:
            fld_char = null_xml_element

        return parse_instr_text(instr_text, fld_char=fld_char)

    def parse_instr_text(instr_text, *, fld_char):
        external_link_result = re.match(r'\s*HYPERLINK "(.*)"', instr_text)
        if external_link_result is not None:
            return complex_fields.hyperlink(dict(href=external_link_result.group(1)))

        internal_link_result = re.match(r'\s*HYPERLINK\s+\\l\s+"(.*)"', instr_text)
        if internal_link_result is not None:
            return complex_fields.hyperlink(dict(anchor=internal_link_result.group(1)))

        checkbox_result = re.match(r'\s*FORMCHECKBOX\s*', instr_text)
        if checkbox_result is not None:
            checkbox_element = fld_char \
                .find_child_or_null("w:ffData") \
                .find_child_or_null("w:checkBox")
            checked_element = checkbox_element.find_child("w:checked")

            if checked_element is None:
                checked = read_boolean_element(checkbox_element.find_child("w:default"))
            else:
                checked = read_boolean_element(checked_element)

            return complex_fields.checkbox(checked=checked)

        return None

    def read_instr_text(element):
        current_instr_text.append(_inner_text(element))
        return _empty_result

    def _read_style(properties, style_tag_name, style_type, find_style_by_id):
        messages = []
        style_id = properties \
            .find_child_or_null(style_tag_name) \
            .attributes.get("w:val")

        if style_id is None:
            style_name = None
        else:
            style = find_style_by_id(style_id)
            if style is None:
                style_name = None
                messages.append(_undefined_style_warning(style_type, style_id))
            else:
                style_name = style.name

        return _ReadResult([style_id, style_name], [], messages)

    def _undefined_style_warning(style_type, style_id):
        return results.warning("{0} style with ID {1} was referenced but not defined in the document".format(style_type, style_id))

    def _read_numbering_properties(paragraph_style_id, element):
        num_id = element.find_child_or_null("w:numId").attributes.get("w:val")
        level_index = element.find_child_or_null("w:ilvl").attributes.get("w:val")
        if num_id is not None and level_index is not None:
            return numbering.find_level(num_id, level_index)

        if paragraph_style_id is not None:
            level = numbering.find_level_by_paragraph_style_id(paragraph_style_id)
            if level is not None:
                return level

        # Some malformed documents define numbering levels without an index, and
        # reference the numbering using a w:numPr element without a w:ilvl child.
        # To handle such cases, we assume a level of 0 as a fallback.
        if num_id is not None:
            return numbering.find_level(num_id, "0")

        return None

    def _read_paragraph_indent(element):
        attributes = element.attributes
        return documents.paragraph_indent(
            start=attributes.get("w:start") or attributes.get("w:left"),
            end=attributes.get("w:end") or attributes.get("w:right"),
            first_line=attributes.get("w:firstLine"),
            hanging=attributes.get("w:hanging"),
        )

    def tab(element):
        return _success(documents.tab())


    def no_break_hyphen(element):
        return _success(documents.text(unichr(0x2011)))


    def soft_hyphen(element):
        return _success(documents.text(u"\u00ad"))

    def symbol(element):
        # See 17.3.3.30 sym (Symbol Character) of ECMA-376 4th edition Part 1
        font = element.attributes.get("w:font")
        char = element.attributes.get("w:char")

        unicode_code_point = dingbats.get((font, int(char, 16)))

        if unicode_code_point is None and re.match("^F0..", char):
            unicode_code_point = dingbats.get((font, int(char[2:], 16)))

        if unicode_code_point is None:
            warning = results.warning("A w:sym element with an unsupported character was ignored: char {0} in font {1}".format(
                char,
                font,
            ))
            return _empty_result_with_message(warning)
        else:
            return _success(documents.text(unichr(unicode_code_point)))


    def table(element):
        properties = element.find_child_or_null("w:tblPr")
        return _ReadResult.map_results(
            read_table_style(properties),
            _read_xml_elements(element.children)
                .flat_map(calculate_row_spans),

            lambda style, children: documents.table(
                children=children,
                style_id=style[0],
                style_name=style[1],
            ),
        )


    def read_table_style(properties):
        return _read_style(properties, "w:tblStyle", "Table", styles.find_table_style_by_id)


    def table_row(element):
        properties = element.find_child_or_null("w:trPr")

        # See 17.13.5.12 del (Deleted Table Row) of ECMA-376 4th edition Part 1
        is_deleted = bool(properties.find_child("w:del"))
        if is_deleted:
            return _empty_result

        is_header = bool(properties.find_child("w:tblHeader"))
        return _read_xml_elements(element.children) \
            .map(lambda children: documents.table_row(
                children=children,
                is_header=is_header,
            ))


    def table_cell(element):
        properties = element.find_child_or_null("w:tcPr")
        gridspan = properties \
            .find_child_or_null("w:gridSpan") \
            .attributes.get("w:val")

        if gridspan is None:
            colspan = 1
        else:
            colspan = int(gridspan)

        return _read_xml_elements(element.children) \
            .map(lambda children: documents.table_cell_unmerged(
                children=children,
                colspan=colspan,
                rowspan=1,
                vmerge=read_vmerge(properties),
            ))

    def read_vmerge(properties):
        vmerge_element = properties.find_child("w:vMerge")
        if vmerge_element is None:
            return False
        else:
            val = vmerge_element.attributes.get("w:val")
            return val == "continue" or not val


    def calculate_row_spans(rows):
        unexpected_non_rows = any(
            not isinstance(row, documents.TableRow)
            for row in rows
        )
        if unexpected_non_rows:
            rows = remove_unmerged_table_cells(rows)
            return _elements_result_with_messages(rows, [results.warning(
                "unexpected non-row element in table, cell merging may be incorrect"
            )])

        unexpected_non_cells = any(
            not isinstance(cell, documents.TableCellUnmerged)
            for row in rows
            for cell in row.children
        )
        if unexpected_non_cells:
            rows = remove_unmerged_table_cells(rows)
            return _elements_result_with_messages(rows, [results.warning(
                "unexpected non-cell element in table row, cell merging may be incorrect"
            )])

        columns = {}
        for row in rows:
            cell_index = 0
            for cell in row.children:
                if cell.vmerge and cell_index in columns:
                    columns[cell_index].rowspan += 1
                else:
                    columns[cell_index] = cell
                    cell.vmerge = False
                cell_index += cell.colspan

        for row in rows:
            row.children = [
                documents.table_cell(
                    children=cell.children,
                    colspan=cell.colspan,
                    rowspan=cell.rowspan,
                )
                for cell in row.children
                if not cell.vmerge
            ]

        return _success(rows)


    def remove_unmerged_table_cells(rows):
        return list(map(
            transforms.element_of_type(
                documents.TableCellUnmerged,
                lambda cell: documents.table_cell(
                    children=cell.children,
                    colspan=cell.colspan,
                    rowspan=cell.rowspan,
                ),
            ),
            rows,
        ))


    def read_child_elements(element):
        return _read_xml_elements(element.children)


    def pict(element):
        return read_child_elements(element).to_extra()


    def hyperlink(element):
        relationship_id = element.attributes.get("r:id")
        anchor = element.attributes.get("w:anchor")
        target_frame = element.attributes.get("w:tgtFrame") or None
        children_result = _read_xml_elements(element.children)

        def create(**kwargs):
            return children_result.map(lambda children: documents.hyperlink(
                children=children,
                target_frame=target_frame,
                **kwargs
            ))

        if relationship_id is not None:
            href = relationships.find_target_by_relationship_id(relationship_id)
            if anchor is not None:
                href = replace_fragment(href, anchor)

            return create(href=href)
        elif anchor is not None:
            return create(anchor=anchor)
        else:
            return children_result


    def bookmark_start(element):
        name = element.attributes.get("w:name")
        if name == "_GoBack":
            return _empty_result
        else:
            return _success(documents.bookmark(name))


    def break_(element):
        break_type = element.attributes.get("w:type")

        if not break_type or break_type == "textWrapping":
            return _success(documents.line_break)
        elif break_type == "page":
            return _success(documents.page_break)
        elif break_type == "column":
            return _success(documents.column_break)
        else:
            warning = results.warning("Unsupported break type: {0}".format(break_type))
            return _empty_result_with_message(warning)


    def inline(element):
        properties_element = element.find_child_or_null("wp:docPr")

        properties = properties_element.attributes
        if properties.get("descr", "").strip():
            alt_text = properties.get("descr")
        else:
            alt_text = properties.get("title")

        hlink_click_element = properties_element.find_child_or_null("a:hlinkClick")
        hyperlink_relationship_id = hlink_click_element.attributes.get("r:id")
        if hyperlink_relationship_id:
            href = relationships.find_target_by_relationship_id(hyperlink_relationship_id)
        else:
            href = None

        blips = element.find_children("a:graphic") \
            .find_children("a:graphicData") \
            .find_children("pic:pic") \
            .find_children("pic:blipFill") \
            .find_children("a:blip")
        return _read_blips(blips, alt_text=alt_text, href=href)

    def _read_blips(blips, alt_text, href):
        return _ReadResult.concat(lists.map(lambda blip: _read_blip(blip, alt_text=alt_text, href=href), blips))

    def _read_blip(element, alt_text, href):
        blip_image = _find_blip_image(element)

        if blip_image is None:
            warning = results.warning("Could not find image file for a:blip element")
            return _empty_result_with_message(warning)

        result = _read_image(blip_image, alt_text)
        if href is None:
            return result
        else:
            return result.map(lambda image_elements: documents.hyperlink(
                image_elements,
                href=href,
            ))

    def _read_image(image_file, alt_text):
        image_path, open_image = image_file
        content_type = content_types.find_content_type(image_path)
        image = documents.image(alt_text=alt_text, content_type=content_type, open=open_image)

        if content_type in ["image/png", "image/gif", "image/jpeg", "image/svg+xml", "image/tiff"]:
            messages = []
        else:
            messages = [results.warning("Image of type {0} is unlikely to display in web browsers".format(content_type))]

        return _element_result_with_messages(image, messages)

    def _find_blip_image(element):
        embed_relationship_id = element.attributes.get("r:embed")
        link_relationship_id = element.attributes.get("r:link")
        if embed_relationship_id is not None:
            return _find_embedded_image(embed_relationship_id)
        elif link_relationship_id is not None:
            return _find_linked_image(link_relationship_id)
        else:
            return None

    def _find_embedded_image(relationship_id):
        target = relationships.find_target_by_relationship_id(relationship_id)
        image_path = uri_to_zip_entry_name("word", target)

        def open_image():
            image_file = docx_file.open(image_path)
            if hasattr(image_file, "__exit__"):
                return image_file
            else:
                return contextlib.closing(image_file)

        return image_path, open_image


    def _find_linked_image(relationship_id):
        image_path = relationships.find_target_by_relationship_id(relationship_id)

        def open_image():
            return files.open(image_path)

        return image_path, open_image

    def read_imagedata(element):
        relationship_id = element.attributes.get("r:id")
        if relationship_id is None:
            warning = results.warning("A v:imagedata element without a relationship ID was ignored")
            return _empty_result_with_message(warning)
        else:
            title = element.attributes.get("o:title")
            return _read_image(_find_embedded_image(relationship_id), title)

    def note_reference_reader(note_type):
        def note_reference(element):
            return _success(documents.note_reference(note_type, element.attributes["w:id"]))

        return note_reference

    def read_comment_reference(element):
        return _success(documents.comment_reference(element.attributes["w:id"]))

    def alternate_content(element):
        return read_child_elements(element.find_child_or_null("mc:Fallback"))

    def read_sdt(element):
        content_result = read_child_elements(element.find_child_or_null("w:sdtContent"))

        def handle_content(content):
            # From the WordML standard: https://learn.microsoft.com/en-us/openspecs/office_standards/ms-docx/3350cb64-931f-41f7-8824-f18b2568ce66
            #
            # > A CT_SdtCheckbox element that specifies that the parent
            # > structured document tag is a checkbox when displayed in the
            # > document. The parent structured document tag contents MUST
            # > contain a single character and optionally an additional
            # > character in a deleted run.
            checkbox = element.find_child_or_null("w:sdtPr").find_child("wordml:checkbox")

            if checkbox is None:
                return content

            checked_element = checkbox.find_child("wordml:checked")
            is_checked = (
                checked_element is not None and
                read_boolean_attribute_value(checked_element.attributes.get("wordml:val"))
            )
            document_checkbox = documents.checkbox(checked=is_checked)

            has_checkbox = False

            def transform_text(text):
                nonlocal has_checkbox
                if len(text.value) > 0 and not has_checkbox:
                    has_checkbox = True
                    return document_checkbox
                else:
                    return text

            replaced_content = list(map(
                transforms.element_of_type(documents.Text, transform_text),
                content,
            ))

            if has_checkbox:
                return replaced_content
            else:
                return document_checkbox

        return content_result.map(handle_content)

    handlers = {
        "w:t": text,
        "w:r": run,
        "w:p": paragraph,
        "w:fldChar": read_fld_char,
        "w:instrText": read_instr_text,
        "w:tab": tab,
        "w:noBreakHyphen": no_break_hyphen,
        "w:softHyphen": soft_hyphen,
        "w:sym": symbol,
        "w:tbl": table,
        "w:tr": table_row,
        "w:tc": table_cell,
        "w:ins": read_child_elements,
        "w:object": read_child_elements,
        "w:smartTag": read_child_elements,
        "w:drawing": read_child_elements,
        "v:group": read_child_elements,
        "v:rect": read_child_elements,
        "v:roundrect": read_child_elements,
        "v:shape": read_child_elements,
        "v:textbox": read_child_elements,
        "w:txbxContent": read_child_elements,
        "w:pict": pict,
        "w:hyperlink": hyperlink,
        "w:bookmarkStart": bookmark_start,
        "w:br": break_,
        "wp:inline": inline,
        "wp:anchor": inline,
        "v:imagedata": read_imagedata,
        "w:footnoteReference": note_reference_reader("footnote"),
        "w:endnoteReference": note_reference_reader("endnote"),
        "w:commentReference": read_comment_reference,
        "mc:AlternateContent": alternate_content,
        "w:sdt": read_sdt
    }

    def read(element):
        handler = handlers.get(element.name)
        if handler is None:
            if element.name not in _ignored_elements:
                warning = results.warning("An unrecognised element was ignored: {0}".format(element.name))
                return _empty_result_with_message(warning)
            else:
                return _empty_result
        else:
            return handler(element)


    def _read_xml_elements(nodes):
        elements = filter(lambda node: isinstance(node, XmlElement), nodes)
        return _ReadResult.concat(lists.map(read, elements))

    return _read_xml_elements


def _inner_text(node):
    if node.node_type == node_types.text:
        return node.value
    else:
        return "".join(_inner_text(child) for child in node.children)



class _ReadResult(object):
    @staticmethod
    def concat(results):
        return _ReadResult(
            lists.flat_map(lambda result: result.elements, results),
            lists.flat_map(lambda result: result.extra, results),
            lists.flat_map(lambda result: result.messages, results))


    @staticmethod
    def map_results(first, second, func):
        return _ReadResult(
            [func(first.elements, second.elements)],
            first.extra + second.extra,
            first.messages + second.messages)

    def __init__(self, elements, extra, messages):
        self.elements = elements
        self.extra = extra
        self.messages = messages

    def map(self, func):
        elements = func(self.elements)
        if not isinstance(elements, list):
            elements = [elements]
        return _ReadResult(
            elements,
            self.extra,
            self.messages)

    def flat_map(self, func):
        result = func(self.elements)
        return _ReadResult(
            result.elements,
            self.extra + result.extra,
            self.messages + result.messages)


    def to_extra(self):
        return _ReadResult([], _concat(self.extra, self.elements), self.messages)

    def append_extra(self):
        return _ReadResult(_concat(self.elements, self.extra), [], self.messages)

def _success(elements):
    if not isinstance(elements, list):
        elements = [elements]
    return _ReadResult(elements, [], [])

def _element_result_with_messages(element, messages):
    return _elements_result_with_messages([element], messages)

def _elements_result_with_messages(elements, messages):
    return _ReadResult(elements, [], messages)

_empty_result = _ReadResult([], [], [])

def _empty_result_with_message(message):
    return _ReadResult([], [], [message])

def _concat(*values):
    result = []
    for value in values:
        for element in value:
            result.append(element)
    return result


def _is_int(value):
    if value is None:
        return False

    try:
        int(value)
    except ValueError:
        return False

    return True
