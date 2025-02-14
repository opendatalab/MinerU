
from magic_pdf.config.drop_reason import DropReason
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.filter.pdf_classify_by_type import classify as do_classify
from magic_pdf.filter.pdf_meta_scan import pdf_meta_scan


def classify(pdf_bytes: bytes) -> SupportedPdfParseMethod:
    """根据pdf的元数据，判断是文本pdf，还是ocr pdf."""
    pdf_meta = pdf_meta_scan(pdf_bytes)
    if pdf_meta.get('_need_drop', False):  # 如果返回了需要丢弃的标志，则抛出异常
        raise Exception(f"pdf meta_scan need_drop,reason is {pdf_meta['_drop_reason']}")
    else:
        is_encrypted = pdf_meta['is_encrypted']
        is_needs_password = pdf_meta['is_needs_password']
        if is_encrypted or is_needs_password:  # 加密的，需要密码的，没有页面的，都不处理
            raise Exception(f'pdf meta_scan need_drop,reason is {DropReason.ENCRYPTED}')
        else:
            is_text_pdf, results = do_classify(
                pdf_meta['total_page'],
                pdf_meta['page_width_pts'],
                pdf_meta['page_height_pts'],
                pdf_meta['image_info_per_page'],
                pdf_meta['text_len_per_page'],
                pdf_meta['imgs_per_page'],
                pdf_meta['text_layout_per_page'],
                pdf_meta['invalid_chars'],
            )
            if is_text_pdf:
                return SupportedPdfParseMethod.TXT
            else:
                return SupportedPdfParseMethod.OCR
