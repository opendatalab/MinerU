class ContentType:
    Image = 'image'
    Table = 'table'
    Text = 'text'
    InlineEquation = 'inline_equation'
    InterlineEquation = 'interline_equation'


class BlockType:
    Image = 'image'
    ImageBody = 'image_body'
    ImageCaption = 'image_caption'
    ImageFootnote = 'image_footnote'
    Table = 'table'
    TableBody = 'table_body'
    TableCaption = 'table_caption'
    TableFootnote = 'table_footnote'
    Text = 'text'
    Title = 'title'
    InterlineEquation = 'interline_equation'
    Footnote = 'footnote'
    Discarded = 'discarded'


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
