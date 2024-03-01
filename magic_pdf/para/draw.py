from magic_pdf.libs.commons import fitz

from magic_pdf.para.commons import *


if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


class DrawAnnos:
    """
    This class draws annotations on the pdf file

    ----------------------------------------
                Color Code
    ----------------------------------------
        Red: (1, 0, 0)
        Green: (0, 1, 0)
        Blue: (0, 0, 1)
        Yellow: (1, 1, 0) - mix of red and green
        Cyan: (0, 1, 1) - mix of green and blue
        Magenta: (1, 0, 1) - mix of red and blue
        White: (1, 1, 1) - red, green and blue full intensity
        Black: (0, 0, 0) - no color component whatsoever
        Gray: (0.5, 0.5, 0.5) - equal and medium intensity of red, green and blue color components
        Orange: (1, 0.65, 0) - maximum intensity of red, medium intensity of green, no blue component
    """

    def __init__(self) -> None:
        pass

    def __is_nested_list(self, lst):
        """
        This function returns True if the given list is a nested list of any degree.
        """
        if isinstance(lst, list):
            return any(self.__is_nested_list(i) for i in lst) or any(isinstance(i, list) for i in lst)
        return False

    def __valid_rect(self, bbox):
        # Ensure that the rectangle is not empty or invalid
        if isinstance(bbox[0], list):
            return False  # It's a nested list, hence it can't be valid rect
        else:
            return bbox[0] < bbox[2] and bbox[1] < bbox[3]

    def __draw_nested_boxes(self, page, nested_bbox, color=(0, 1, 1)):
        """
        This function draws the nested boxes

        Parameters
        ----------
        page : fitz.Page
            page
        nested_bbox : list
            nested bbox
        color : tuple
            color, by default (0, 1, 1)    # draw with cyan color for combined paragraph
        """
        if self.__is_nested_list(nested_bbox):  # If it's a nested list
            for bbox in nested_bbox:
                self.__draw_nested_boxes(page, bbox, color)  # Recursively call the function
        elif self.__valid_rect(nested_bbox):  # If valid rectangle
            para_rect = fitz.Rect(nested_bbox)
            para_anno = page.add_rect_annot(para_rect)
            para_anno.set_colors(stroke=color)  # draw with cyan color for combined paragraph
            para_anno.set_border(width=1)
            para_anno.update()

    def draw_annos(self, input_pdf_path, pdf_dic, output_pdf_path):
        pdf_doc = open_pdf(input_pdf_path)

        if pdf_dic is None:
            pdf_dic = {}

        if output_pdf_path is None:
            output_pdf_path = input_pdf_path.replace(".pdf", "_anno.pdf")

        for page_id, page in enumerate(pdf_doc):  # type: ignore
            page_key = f"page_{page_id}"
            for ele_key, ele_data in pdf_dic[page_key].items():
                if ele_key == "para_blocks":
                    para_blocks = ele_data
                    for para_block in para_blocks:
                        if "paras" in para_block.keys():
                            paras = para_block["paras"]
                            for para_key, para_content in paras.items():
                                para_bbox = para_content["para_bbox"]
                                # print(f"para_bbox: {para_bbox}")
                                # print(f"is a nested list: {self.__is_nested_list(para_bbox)}")
                                if self.__is_nested_list(para_bbox) and len(para_bbox) > 1:
                                    color = (0, 1, 1)
                                    self.__draw_nested_boxes(
                                        page, para_bbox, color
                                    )  # draw with cyan color for combined paragraph
                                else:
                                    if self.__valid_rect(para_bbox):
                                        para_rect = fitz.Rect(para_bbox)
                                        para_anno = page.add_rect_annot(para_rect)
                                        para_anno.set_colors(stroke=(0, 1, 0))  # draw with green color for normal paragraph
                                        para_anno.set_border(width=0.5)
                                        para_anno.update()

                                is_para_title = para_content["is_para_title"]
                                if is_para_title:
                                    if self.__is_nested_list(para_content["para_bbox"]) and len(para_content["para_bbox"]) > 1:
                                        color = (0, 0, 1)
                                        self.__draw_nested_boxes(
                                            page, para_content["para_bbox"], color
                                        )  # draw with cyan color for combined title
                                    else:
                                        if self.__valid_rect(para_content["para_bbox"]):
                                            para_rect = fitz.Rect(para_content["para_bbox"])
                                            if self.__valid_rect(para_content["para_bbox"]):
                                                para_anno = page.add_rect_annot(para_rect)
                                                para_anno.set_colors(stroke=(0, 0, 1))  # draw with blue color for normal title
                                                para_anno.set_border(width=0.5)
                                                para_anno.update()

        pdf_doc.save(output_pdf_path)
        pdf_doc.close()
