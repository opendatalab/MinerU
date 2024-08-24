from collections import Counter

from magic_pdf.libs.language import detect_lang

def get_language_from_model(model_list: list):
    language_lst = []
    for ocr_page_info in model_list:
        page_text = ""
        layout_dets = ocr_page_info["layout_dets"]
        for layout_det in layout_dets:
            category_id = layout_det["category_id"]
            allow_category_id_list = [15]
            if category_id in allow_category_id_list:
                page_text += layout_det["text"]
        page_language = detect_lang(page_text)
        language_lst.append(page_language)
    # 统计text_language_list中每种语言的个数
    count_dict = Counter(language_lst)
    # 输出text_language_list中出现的次数最多的语言
    language = max(count_dict, key=count_dict.get)
    return language
