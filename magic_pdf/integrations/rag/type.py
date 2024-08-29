from enum import Enum

from pydantic import BaseModel, Field


# rag
class CategoryType(Enum):  # py310 not support StrEnum
    text = 'text'
    title = 'title'
    interline_equation = 'interline_equation'
    image = 'image'
    image_body = 'image_body'
    image_caption = 'image_caption'
    table = 'table'
    table_body = 'table_body'
    table_caption = 'table_caption'
    table_footnote = 'table_footnote'


class ElementRelType(Enum):
    sibling = 'sibling'


class PageInfo(BaseModel):
    page_no: int = Field(description='the index of page, start from zero',
                         ge=0)
    height: int = Field(description='the height of page', gt=0)
    width: int = Field(description='the width of page', ge=0)
    image_path: str | None = Field(description='the image of this page',
                                   default=None)


class ContentObject(BaseModel):
    category_type: CategoryType = Field(description='类别')
    poly: list[float] = Field(
        description=('Coordinates, need to convert back to PDF coordinates,'
                     ' order is top-left, top-right, bottom-right, bottom-left'
                     ' x,y coordinates'))
    ignore: bool = Field(description='whether ignore this object',
                         default=False)
    text: str | None = Field(description='text content of the object',
                             default=None)
    image_path: str | None = Field(description='path of embedded image',
                                   default=None)
    order: int = Field(description='the order of this object within a page',
                       default=-1)
    anno_id: int = Field(description='unique id', default=-1)
    latex: str | None = Field(description='latex result', default=None)
    html: str | None = Field(description='html result', default=None)


class ElementRelation(BaseModel):
    source_anno_id: int = Field(description='unique id of the source object',
                                default=-1)
    target_anno_id: int = Field(description='unique id of the target object',
                                default=-1)
    relation: ElementRelType = Field(
        description='the relation between source and target element')


class LayoutElementsExtra(BaseModel):
    element_relation: list[ElementRelation] = Field(
        description='the relation between source and target element')


class LayoutElements(BaseModel):
    layout_dets: list[ContentObject] = Field(
        description='layout element details')
    page_info: PageInfo = Field(description='page info')
    extra: LayoutElementsExtra = Field(description='extra information')


# iter data format
class Node(BaseModel):
    category_type: CategoryType = Field(description='类别')
    text: str | None = Field(description='text content of the object',
                             default=None)
    image_path: str | None = Field(description='path of embedded image',
                                   default=None)
    anno_id: int = Field(description='unique id', default=-1)
    latex: str | None = Field(description='latex result', default=None)
    html: str | None = Field(description='html result', default=None)
