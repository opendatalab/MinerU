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


class ContentType:
    IMAGE = 'image'
    TABLE = 'table'
    TEXT = 'text'
    INTERLINE_EQUATION = 'interline_equation'


class MakeMode:
    MM_MD = 'mm_markdown'
    NLP_MD = 'nlp_markdown'
    STANDARD_FORMAT = 'standard_format'