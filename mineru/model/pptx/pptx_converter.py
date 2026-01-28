from typing import Final, BinaryIO

from pptx import Presentation, presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from loguru import logger


class PptxConverter:
    _BLIP_NAMESPACES: Final = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    }

    def __init__(self):
        self.file_stream = None
        self.pptx_obj = None
        self.pages = []
        self.cur_page = []

    def convert(
        self,
        file_stream: BinaryIO,
    ):
        self.file_stream = file_stream
        self.pptx_obj = Presentation(self.file_stream)
        if self.pptx_obj:
            self._walk_linear(self.pptx_obj)

    def _walk_linear(self, pptx_obj: presentation.Presentation):
        slide_width = pptx_obj.slide_width
        slide_height = pptx_obj.slide_height

        # 遍历每一张幻灯片
        for _, slide in enumerate(pptx_obj.slides):

            def handle_shapes(shape):
                handle_groups(shape, parent_slide, slide_ind, doc, slide_size)
                if shape.has_table:
                    # 处理表格
                    self._handle_tables(shape, parent_slide, slide_ind, doc, slide_size)
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    # 处理图片
                    if hasattr(shape, "image"):
                        self._handle_pictures(
                            shape, parent_slide, slide_ind, doc, slide_size
                        )
                # 如果形状没有任何文本，则继续处理下一个形状
                if not hasattr(shape, "text"):
                    return
                if shape.text is None:
                    return
                if len(shape.text.strip()) == 0:
                    return
                if not shape.has_text_frame:
                    logger.warning("Warning: shape has text but not text_frame")
                    return
                # 处理其他文本元素，包括列表(项目符号列表、编号列表等)
                self._handle_text_elements(
                    shape, parent_slide, slide_ind, doc, slide_size
                )
                return

            def handle_groups(shape, parent_slide, slide_ind, doc, slide_size):
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    for groupedshape in shape.shapes:
                        handle_shapes(
                            groupedshape, parent_slide, slide_ind, doc, slide_size
                        )

            # 遍历幻灯片中的每一个形状
            for shape in slide.shapes:
                handle_shapes(shape)
