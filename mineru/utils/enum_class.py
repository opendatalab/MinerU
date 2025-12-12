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

    # Added in vlm 2.5
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    ALGORITHM = "algorithm"
    REF_TEXT = "ref_text"
    PHONETIC = "phonetic"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"


class ContentType:
    IMAGE = 'image'
    TABLE = 'table'
    TEXT = 'text'
    INTERLINE_EQUATION = 'interline_equation'
    INLINE_EQUATION = 'inline_equation'
    EQUATION = 'equation'
    CODE = 'code'


class ContentTypeV2:
    CODE = 'code'
    ALGORITHM = "algorithm"
    EQUATION_INTERLINE = 'equation_interline'
    IMAGE = 'image'
    TABLE = 'table'
    TABLE_SIMPLE = 'simple_table'
    TABLE_COMPLEX = 'complex_table'
    LIST = 'list'
    LIST_TEXT = 'text_list'
    LIST_REF = 'reference_list'
    TITLE = 'title'
    PARAGRAPH = 'paragraph'
    SPAN_TEXT = 'text'
    SPAN_EQUATION_INLINE = 'equation_inline'
    SPAN_PHONETIC = 'phonetic'
    SPAN_MD = 'md'
    SPAN_CODE_INLINE = 'code_inline'
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    PAGE_NUMBER = "page_number"
    PAGE_ASIDE_TEXT = "page_aside_text"
    PAGE_FOOTNOTE = "page_footnote"


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
    CONTENT_LIST_V2 = 'content_list_v2'


class ModelPath:
    vlm_root_hf = "opendatalab/MinerU2.5-2509-1.2B"
    vlm_root_modelscope = "OpenDataLab/MinerU2.5-2509-1.2B"
    pipeline_root_modelscope = "OpenDataLab/PDF-Extract-Kit-1.0"
    pipeline_root_hf = "opendatalab/PDF-Extract-Kit-1.0"
    doclayout_yolo = "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
    yolo_v8_mfd = "models/MFD/YOLO/yolo_v8_ft.pt"
    unimernet_small = "models/MFR/unimernet_hf_small_2503"
    pp_formulanet_plus_m = "models/MFR/pp_formulanet_plus_m"
    pytorch_paddle = "models/OCR/paddleocr_torch"
    layout_reader = "models/ReadingOrder/layout_reader"
    slanet_plus = "models/TabRec/SlanetPlus/slanet-plus.onnx"
    unet_structure = "models/TabRec/UnetStructure/unet.onnx"
    paddle_table_cls = "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx"
    paddle_orientation_classification = "models/OriCls/paddle_orientation_classification/PP-LCNet_x1_0_doc_ori.onnx"


class SplitFlag:
    CROSS_PAGE = 'cross_page'
    LINES_DELETED = 'lines_deleted'


class ImageType:
    PIL = 'pil_img'
    BASE64 = 'base64_img'