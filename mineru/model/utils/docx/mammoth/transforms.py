from . import documents


def paragraph(transform_paragraph):
    return element_of_type(documents.Paragraph, transform_paragraph)


def run(transform_run):
    return element_of_type(documents.Run, transform_run)


def element_of_type(element_type, transform):
    def transform_element(element):
        if isinstance(element, element_type):
            return transform(element)
        else:
            return element

    return _each_element(transform_element)


def _each_element(transform_element):
    def transform_element_and_children(element):
        if isinstance(element, (documents.HasChildren, documents.TableCellUnmerged)):
            children = list(map(transform_element_and_children, element.children))
            element = element.copy(children=children)

        return transform_element(element)

    return transform_element_and_children


def get_descendants_of_type(element, element_type):
    return list(filter(
        lambda descendant: isinstance(descendant, element_type),
        get_descendants(element),
    ))


def get_descendants(element):
    descendants = []

    def visit(element):
        descendants.append(element)

    _visit_descendants(element, visit)

    return descendants


def _visit_descendants(element, visit):
    if isinstance(element, documents.HasChildren):
        for child in element.children:
            _visit_descendants(child, visit)
            visit(child)

