from enum import Enum

class BlockType:
    IMAGE = 'image'
    TABLE = 'table'
    CHART = 'chart'
    IMAGE_BODY = 'image_body'
    TABLE_BODY = 'table_body'
    CHART_BODY = 'chart_body'
    CAPTION = 'caption'  # generic caption type (e.g., for Word documents)
    IMAGE_CAPTION = 'image_caption'
    TABLE_CAPTION = 'table_caption'
    CHART_CAPTION = 'chart_caption'
    ALGORITHM_CAPTION = 'algorithm_caption'
    FOOTNOTE = 'footnote'  # pp_layout中的vision_footnote
    IMAGE_FOOTNOTE = 'image_footnote'
    TABLE_FOOTNOTE = 'table_footnote'
    CHART_FOOTNOTE = 'chart_footnote'
    TEXT = 'text'
    TITLE = 'title'
    INTERLINE_EQUATION = 'interline_equation'
    EQUATION = "equation"  # 公式(独立公式)
    LIST = 'list'
    INDEX = 'index'
    DISCARDED = 'discarded'

    # Added in vlm 2.5
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    CODE_FOOTNOTE = "code_footnote"
    ALGORITHM = "algorithm"
    REF_TEXT = "ref_text"
    PHONETIC = "phonetic"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"

    # Added in pp_doclayout_v2
    ABSTRACT = "abstract"
    DOC_TITLE = "doc_title"
    PARAGRAPH_TITLE = "paragraph_title"
    VERTICAL_TEXT = "vertical_text"
    SEAL = "seal"
    HEADER_IMAGE = "header_image"
    FOOTER_IMAGE = "footer_image"
    FORMULA_NUMBER = "formula_number"

class ContentType:
    IMAGE = 'image'
    TABLE = 'table'
    CHART = 'chart'
    TEXT = 'text'
    INTERLINE_EQUATION = 'interline_equation'
    INLINE_EQUATION = 'inline_equation'
    EQUATION = 'equation'
    HYPERLINK = 'hyperlink'
    SEAL = 'seal'


class ContentTypeV2:
    CODE = 'code'
    ALGORITHM = "algorithm"
    EQUATION_INTERLINE = 'equation_interline'
    IMAGE = 'image'
    SEAL = 'seal'
    TABLE = 'table'
    CHART = 'chart'
    TABLE_SIMPLE = 'simple_table'
    TABLE_COMPLEX = 'complex_table'
    LIST = 'list'
    LIST_TEXT = 'text_list'
    LIST_REF = 'reference_list'
    INDEX = 'index'
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
    pp_doclayout_v2 = "models/Layout/PP-DocLayoutV2"
    unimernet_small = "models/MFR/unimernet_hf_small_2503"
    pp_formulanet_plus_m = "models/MFR/pp_formulanet_plus_m"
    pytorch_paddle = "models/OCR/paddleocr_torch"
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


class NotExtractType(Enum):
    TEXT = BlockType.TEXT
    TITLE = BlockType.TITLE
    HEADER = BlockType.HEADER
    FOOTER = BlockType.FOOTER
    PAGE_NUMBER = BlockType.PAGE_NUMBER
    PAGE_FOOTNOTE = BlockType.PAGE_FOOTNOTE
    REF_TEXT = BlockType.REF_TEXT
    TABLE_CAPTION = BlockType.TABLE_CAPTION
    IMAGE_CAPTION = BlockType.IMAGE_CAPTION
    TABLE_FOOTNOTE = BlockType.TABLE_FOOTNOTE
    IMAGE_FOOTNOTE = BlockType.IMAGE_FOOTNOTE
    CODE_CAPTION = BlockType.CODE_CAPTION
