"""验证有线表格 OCR 文字的 HTML 输出。"""

from unittest.mock import Mock

import numpy as np
import pytest
from bs4 import BeautifulSoup

from mineru.model.table.rec.unet_table import main as unet


def test_wired_table_preserves_separation_between_ocr_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一行的 OCR 片段仍按原有规则合并，跨行的编号保持可区分。"""
    # 结构模型返回逆时针框，其余匹配、排序及 HTML 渲染使用真实实现。
    polygons = np.asarray([[[0, 0], [0, 50], [100, 50], [100, 0]], [[100, 0], [100, 50], [200, 50], [200, 0]]])
    structure = Mock(return_value=(polygons.copy(), polygons.copy()))
    monkeypatch.setattr(unet, "TSRUnet", Mock(return_value=structure))
    recognizer = unet.WiredTableRecognition(unet.WiredTableInput(model_path="unused"))
    image = np.full((50, 200, 3), 255, dtype=np.uint8)
    ocr = [
        [[[5, 25], [45, 25], [45, 35], [5, 35]], "1582N", 0.9],
        [[[25, 5], [45, 5], [45, 15], [25, 15]], "82", 0.9],
        [[[105, 5], [145, 5], [145, 15], [105, 15]], "single", 0.9],
        [[[5, 5], [25, 5], [25, 15], [5, 15]], "15", 0.9],
    ]

    result = recognizer(image, ocr)

    assert result.pred_html
    cells = BeautifulSoup(result.pred_html, "html.parser").find_all("td")
    assert [cell.get_text() for cell in cells] == ["1582 1582N", "single"]
