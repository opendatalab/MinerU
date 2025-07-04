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
    EQUATION = 'equation'


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
    InterlineEquationNumber_Layout = 9
    InlineEquation = 13
    InterlineEquation_YOLO = 14
    OcrText = 15
    LowScoreText = 16
    ImageFootnote = 101


class MakeMode:
    MM_MD = 'mm_markdown'
    NLP_MD = 'nlp_markdown'
    CONTENT_LIST = 'content_list'


class ModelPath:
    vlm_root_hf = "opendatalab/MinerU2.0-2505-0.9B"
    vlm_root_modelscope = "OpenDataLab/MinerU2.0-2505-0.9B"
    pipeline_root_modelscope = "OpenDataLab/PDF-Extract-Kit-1.0"
    pipeline_root_hf = "opendatalab/PDF-Extract-Kit-1.0"
    doclayout_yolo = "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
    yolo_v8_mfd = "models/MFD/YOLO/yolo_v8_ft.pt"
    unimernet_small = "models/MFR/unimernet_hf_small_2503"
    pytorch_paddle = "models/OCR/paddleocr_torch"
    layout_reader = "models/ReadingOrder/layout_reader"
    slanet_plus = "models/TabRec/SlanetPlus/slanet-plus.onnx"


class SplitFlag:
    CROSS_PAGE = 'cross_page'
    LINES_DELETED = 'lines_deleted'