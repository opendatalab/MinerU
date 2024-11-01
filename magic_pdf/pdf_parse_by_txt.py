from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.pdf_parse_union_core_v2 import pdf_parse_union


def parse_pdf_by_txt(
    pdf_bytes,
    model_list,
    imageWriter,
    start_page_id=0,
    end_page_id=None,
    debug_mode=False,
):
    dataset = PymuDocDataset(pdf_bytes)
    return pdf_parse_union(dataset,
                           model_list,
                           imageWriter,
                           SupportedPdfParseMethod.TXT,
                           start_page_id=start_page_id,
                           end_page_id=end_page_id,
                           debug_mode=debug_mode,
                           )
