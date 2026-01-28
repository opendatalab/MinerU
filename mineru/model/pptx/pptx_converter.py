from typing import Final, BinaryIO

from pptx import Presentation, presentation


class PptxConverter:
    _BLIP_NAMESPACES: Final = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    }

    def __init__(self):
        self.file_stream = None
        self.pptx_obj = None

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
            pass