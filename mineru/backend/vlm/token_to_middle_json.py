import time
from loguru import logger
import numpy as np
import cv2
from mineru.utils.config_reader import get_llm_aided_config
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType
from mineru.utils.hash_utils import str_md5
from mineru.backend.vlm.vlm_magic_model import MagicModel
from mineru.utils.pdf_image_tools import get_crop_img
from mineru.version import __version__

heading_level_import_success = False
llm_aided_config = get_llm_aided_config()
if llm_aided_config:
    title_aided_config = llm_aided_config.get('title_aided', {})
    if title_aided_config.get('enable', False):
        try:
            from mineru.utils.llm_aided import llm_aided_title
            from mineru.backend.pipeline.model_init import AtomModelSingleton
            heading_level_import_success = True
        except Exception as e:
            logger.warning("The heading level feature cannot be used. If you need to use the heading level feature, "
                            "please execute `pip install mineru[core]` to install the required packages.")


def token_to_page_info(token, image_dict, page, image_writer, page_index) -> dict:
    """将token转换为页面信息"""
    # 解析token，提取坐标和类型
    # 假设token格式为：<|box_start|>x0 y0 x1 y1<|box_end|><|ref_start|>type<|ref_end|><|md_start|>content<|md_end|>
    # 这里需要根据实际的token格式进行解析
    # 提取所有完整块，每个块从<|box_start|>开始到<|md_end|>或<|im_end|>结束

    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = str_md5(image_dict["img_base64"])
    width, height = map(int, page.get_size())

    magic_model = MagicModel(token, width, height)
    image_blocks = magic_model.get_image_blocks()
    table_blocks = magic_model.get_table_blocks()
    title_blocks = magic_model.get_title_blocks()

    # 如果有标题优化需求，则对title_blocks截图det
    if heading_level_import_success:
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.3,
            lang='ch_lite'
        )
        for title_block in title_blocks:
            title_pil_img = get_crop_img(title_block['bbox'], page_pil_img, scale)
            title_np_img = np.array(title_pil_img)
            # 给title_pil_img添加上下左右各50像素白边padding
            title_np_img = cv2.copyMakeBorder(
                title_np_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            title_img = cv2.cvtColor(title_np_img, cv2.COLOR_RGB2BGR)
            ocr_det_res = ocr_model.ocr(title_img, rec=False)[0]
            if len(ocr_det_res) > 0:
                # 计算所有res的平均高度
                avg_height = np.mean([box[2][1] - box[0][1] for box in ocr_det_res])
                title_block['line_avg_height'] = round(avg_height/scale)

    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    all_spans = magic_model.get_all_spans()
    # 对image/table/interline_equation的span截图
    for span in all_spans:
        if span["type"] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

    page_blocks = []
    page_blocks.extend([*image_blocks, *table_blocks, *title_blocks, *text_blocks, *interline_equation_blocks])
    # 对page_blocks根据index的值进行排序
    page_blocks.sort(key=lambda x: x["index"])

    page_info = {"para_blocks": page_blocks, "discarded_blocks": [], "page_size": [width, height], "page_idx": page_index}
    return page_info


def result_to_middle_json(token_list, images_list, pdf_doc, image_writer):
    middle_json = {"pdf_info": [], "_backend":"vlm", "_version_name": __version__}
    for index, token in enumerate(token_list):
        page = pdf_doc[index]
        image_dict = images_list[index]
        page_info = token_to_page_info(token, image_dict, page, image_writer, index)
        middle_json["pdf_info"].append(page_info)

    """llm优化标题分级"""
    if heading_level_import_success:
        llm_aided_title_start_time = time.time()
        llm_aided_title(middle_json["pdf_info"], title_aided_config)
        logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    # 关闭pdf文档
    pdf_doc.close()
    return middle_json


if __name__ == "__main__":

    output = r"<|box_start|>088 119 472 571<|box_end|><|ref_start|>image<|ref_end|><|md_start|>![]('img_url')<|md_end|>\n<|box_start|>079 582 482 608<|box_end|><|ref_start|>image_caption<|ref_end|><|md_start|>Fig. 2. (a) Schematic of the change in the FDC over time, and (b) definition of model parameters.<|md_end|>\n<|box_start|>079 624 285 638<|box_end|><|ref_start|>title<|ref_end|><|md_start|># 2.2. Zero flow day analysis<|md_end|>\n<|box_start|>079 656 482 801<|box_end|><|ref_start|>text<|ref_end|><|md_start|>A notable feature of Fig. 1 is the increase in the number of zero flow days. A similar approach to Eq. (2), using an inverse sigmoidal function was employed to assess the impact of afforestation on the number of zero flow days per year \((N_{\mathrm{zero}})\). In this case, the left hand side of Eq. (2) is replaced by \(N_{\mathrm{zero}}\) and \(b\) and \(S\) are constrained to negative as \(N_{\mathrm{zero}}\) decreases as rainfall increases, and increases with plantation growth:<|md_end|>\n<|box_start|>076 813 368 853<|box_end|><|ref_start|>equation<|ref_end|><|md_start|>\[\nN_{\mathrm{zero}}=a+b(\Delta P)+\frac{Y}{1+\exp\left(\frac{T-T_{\mathrm{half}}}{S}\right)}\n\]<|md_end|>\n<|box_start|>079 865 482 895<|box_end|><|ref_start|>text<|ref_end|><|md_start|>For the average pre-treatment condition \(\Delta P=0\) and \(T=0\), \(N_{\mathrm{zero}}\) approximately equals \(a\). \(Y\) gives<|md_end|>\n<|box_start|>525 119 926 215<|box_end|><|ref_start|>text<|ref_end|><|md_start|>the magnitude of change in zero flow days due to afforestation, and \(S\) describes the shape of the response. For the average climate condition \(\Delta P=0\), \(a+Y\) becomes the number of zero flow days when the new equilibrium condition under afforestation is reached.<|md_end|>\n<|box_start|>525 240 704 253<|box_end|><|ref_start|>title<|ref_end|><|md_start|># 2.3. Statistical analyses<|md_end|>\n<|box_start|>525 271 926 368<|box_end|><|ref_start|>text<|ref_end|><|md_start|>The coefficient of efficiency \((E)\) (Nash and Sutcliffe, 1970; Chiew and McMahon, 1993; Legates and McCabe, 1999) was used as the 'goodness of fit' measure to evaluate the fit between observed and predicted flow deciles (2) and zero flow days (3). \(E\) is given by:<|md_end|>\n<|box_start|>520 375 735 415<|box_end|><|ref_start|>equation<|ref_end|><|md_start|>\[\nE=1.0-\frac{\sum_{i=1}^{N}(O_{i}-P_{i})^{2}}{\sum_{i=1}^{N}(O_{i}-\bar{O})^{2}}\n\]<|md_end|>\n<|box_start|>525 424 926 601<|box_end|><|ref_start|>text<|ref_end|><|md_start|>where \(O\) are observed data, \(P\) are predicted values, and \(\bar{O}\) is the mean for the entire period. \(E\) is unity minus the ratio of the mean square error to the variance in the observed data, and ranges from \(-\infty\) to 1.0. Higher values indicate greater agreement between observed and predicted data as per the coefficient of determination \((r^{2})\). \(E\) is used in preference to \(r^{2}\) in evaluating hydrologic modelling because it is a measure of the deviation from the 1:1 line. As \(E\) is always \(<r^{2}\) we have arbitrarily considered \(E>0.7\) to indicate adequate model fits.<|md_end|>\n<|box_start|>525 603 926 731<|box_end|><|ref_start|>text<|ref_end|><|md_start|>It is important to assess the significance of the model parameters to check the model assumptions that rainfall and forest age are driving changes in the FDC. The model (2) was split into simplified forms, where only the rainfall or time terms were included by setting \(b=0\), as shown in Eq. (5), or \(Y=0\) as shown in Eq. (6). The component models (5) and (6) were then tested against the complete model, (2).<|md_end|>\n<|box_start|>520 739 735 778<|box_end|><|ref_start|>equation<|ref_end|><|md_start|>\[\nQ_{\%}=a+\frac{Y}{1+\exp\left(\frac{T-T_{\mathrm{half}}^{\prime}}{S}\right)}\n\]<|md_end|>\n<|box_start|>525 787 553 799<|box_end|><|ref_start|>text<|ref_end|><|md_start|>and<|md_end|>\n<|box_start|>520 807 646 825<|box_end|><|ref_start|>equation<|ref_end|><|md_start|>\[\nQ_{\%}=a+b\Delta P\n\]<|md_end|>\n<|box_start|>525 833 926 895<|box_end|><|ref_start|>text<|ref_end|><|md_start|>For both the flow duration curve analysis and zero flow days analysis, a \(t\)-test was then performed to test whether (5) and (6) were significantly different to (2). A critical value of \(t\) exceeding the calculated \(t\)-value<|md_end|><|im_end|>"

    p_info = token_to_page_info(output)
    # 将blocks 转换为json文本
    import json

    json_str = json.dumps(p_info, ensure_ascii=False, indent=4)
    print(json_str)
