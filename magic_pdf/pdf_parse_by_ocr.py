from magic_pdf.pdf_parse_union_core import pdf_parse_union


def parse_pdf_by_ocr(pdf_bytes,
                     model_list,
                     imageWriter,
                     start_page_id=0,
                     end_page_id=None,
                     debug_mode=False,
                     ):
    return pdf_parse_union(pdf_bytes,
                           model_list,
                           imageWriter,
                           "ocr",
                           start_page_id=start_page_id,
                           end_page_id=end_page_id,
                           debug_mode=debug_mode,
                           )
