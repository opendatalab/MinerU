from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.dataset import Dataset
from magic_pdf.pdf_parse_union_core_v2 import pdf_parse_union


def parse_pdf_by_txt(
    dataset: Dataset,
    model_list,
    imageWriter,
    start_page_id=0,
    end_page_id=None,
    debug_mode=False,
    lang=None,
):
    return pdf_parse_union(model_list,
                           dataset,
                           imageWriter,
                           SupportedPdfParseMethod.TXT,
                           start_page_id=start_page_id,
                           end_page_id=end_page_id,
                           debug_mode=debug_mode,
                           lang=lang,
                           )
