class BlockType:
    IMAGE = 'image'
    TABLE = 'table'
    IMAGE_BODY = 'image_body'
    TABLE_BODY = 'table_body'
    IMAGE_CAPTION = 'image_caption'
    TABLE_CAPTION = 'table_caption'
    IMAGE_FOOTNOTE = 'image_footnote'
    TABLE_FOOTNOTE = 'table_footnote'
    TEXT = 'text'
    TITLE = 'title'
    INTERLINE_EQUATION = 'interline_equation'
    LIST = 'list'
    INDEX = 'index'
    DISCARDED = 'discarded'


class ContentType:
    IMAGE = 'image'
    TABLE = 'table'
    TEXT = 'text'
    INTERLINE_EQUATION = 'interline_equation'
    INLINE_EQUATION = 'inline_equation'


class CategoryId:
    Title = 0
    Text = 1
    Abandon = 2
    ImageBody = 3
    ImageCaption = 4
    TableBody = 5
    TableCaption = 6
    TableFootnote = 7
    InterlineEquation_Layout = 8
    InlineEquation = 13
    InterlineEquation_YOLO = 14
    OcrText = 15
    ImageFootnote = 101


class MakeMode:
    MM_MD = 'mm_markdown'
    NLP_MD = 'nlp_markdown'
    STANDARD_FORMAT = 'standard_format'