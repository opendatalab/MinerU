from magic_pdf.pdf_parse_union_core import pdf_parse_union


def parse_pdf_by_txt(
    pdf_bytes,
    model_list,
    imageWriter,
    start_page_id=0,
    end_page_id=None,
    debug_mode=False,
):
    return pdf_parse_union(pdf_bytes,
                           model_list,
                           imageWriter,
                           "txt",
                           start_page_id=start_page_id,
                           end_page_id=end_page_id,
                           debug_mode=debug_mode,
                           )


if __name__ == "__main__":
    pass
    # if 1:
    #     import fitz
    #     import json
    #
    #     with open("/opt/data/pdf/20240418/25536-00.pdf", "rb") as f:
    #         pdf_bytes = f.read()
    #     pdf_docs = fitz.open("pdf", pdf_bytes)
    #
    #     with open("/opt/data/pdf/20240418/25536-00.json") as f:
    #         model_list = json.loads(f.readline())
    #
    #     magic_model = MagicModel(model_list, pdf_docs)
    #     for i in range(7):
    #         print(magic_model.get_imgs(i))
    #
    #     for page_no, page in enumerate(pdf_docs):
    #         inline_equations, interline_equations, interline_equation_blocks = (
    #             magic_model.get_equations(page_no)
    #         )
    #
    #         text_raw_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
    #         char_level_text_blocks = page.get_text(
    #             "rawdict", flags=fitz.TEXTFLAGS_TEXT
    #         )["blocks"]
    #         text_blocks = combine_chars_to_pymudict(
    #             text_raw_blocks, char_level_text_blocks
    #         )
    #         text_blocks = replace_equations_in_textblock(
    #             text_blocks, inline_equations, interline_equations
    #         )
    #         text_blocks = remove_citation_marker(text_blocks)
    #
    #         text_blocks = remove_chars_in_text_blocks(text_blocks)
