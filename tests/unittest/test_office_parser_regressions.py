from pathlib import Path

from pytest import MonkeyPatch

from mineru.backend.analyze import doc_analyze
from mineru.model.flash.office import image as office_image
from mineru.render.markdown import render_markdown


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_vector_image_part_skip_log_is_debug(monkeypatch: MonkeyPatch) -> None:
    """验证 WMF/EMF 占位图使用 debug 日志而不是 warning。"""

    class _Logger:
        """记录图片工具日志级别的测试替身。"""

        def __init__(self) -> None:
            self.debug_messages: list[str] = []
            self.warning_messages: list[str] = []

        def debug(self, message: str) -> None:
            """记录 debug 日志。"""
            self.debug_messages.append(message)

        def warning(self, message: str) -> None:
            """记录 warning 日志。"""
            self.warning_messages.append(message)

    fake_logger = _Logger()
    monkeypatch.setattr(office_image, "logger", fake_logger)
    monkeypatch.setattr(
        office_image,
        "get_standard_vector_placeholder_data_uri",
        lambda: "data:image/jpeg;base64,placeholder",
    )

    assert (
        office_image.serialize_vector_part_with_placeholder("/word/media/image3.wmf", "image/x-wmf")
        == "data:image/jpeg;base64,placeholder"
    )
    assert len(fake_logger.debug_messages) == 1
    assert "Skipping WMF image part before Pillow load" in fake_logger.debug_messages[0]
    assert fake_logger.warning_messages == []


def test_docx_nested_ordered_lists_render_with_local_markers() -> None:
    """验证真实 DOCX 的多级有序列表使用当前层编号，并由 Markdown 缩进表达层级。"""
    file_bytes = (_PROJECT_ROOT / "demo" / "office_docs" / "docx_01.docx").read_bytes()

    middle_json, _ = doc_analyze(file_bytes, file_suffix="docx")
    markdown = render_markdown(middle_json)
    start = markdown.index("1. 如有未尽事宜")
    end_marker = "    - 本合同一式四份，省政府采购中心各一份"
    end = markdown.index(end_marker, start) + len(end_marker)

    assert markdown[start:end] == r"""1. 如有未尽事宜，由双方依法订立[补充合同](https://github.com/opendatalab/MinerU)。
2. 本合同双方应加盖骑缝章。
    - 本合同一式四份，自双方签章并经省政府采购中心审核编号后生效。
    - 甲方、乙方、政府采购管理部门、<u>___广东__</u>省政府采购中心各一份
3. $a^{2}+b^{2}=c^{2}$行内公式后接正文
    - 本合同一式四份，自双方签章并经省政府采购中心审核编号后生效
        1. 本合同双方应加盖骑缝章。
        2. 自双方签章并经省政府采购中心审核编号后生效。
            1. 最里层列表$\left(x+a\right)^{n}=\sum_{k=0}^{n}\left(\genfrac{}{}{0pt}{}{n}{k}\right)x^{k}a^{n-k}$行内公式后接正文
            2. 合同一式四份，自双方签章后生效
        3. <s><strong><u>xyz</u></strong></s>省政府采购中心各一份
            1. 最内层列表$A=\pi r^{2}$行内公式后接正文
            2. 合同一式四份，采购管理部门各一份
        4. 文本[超链接](https://github.com/opendatalab/MinerU)支持，由双方依法订立补充公式$x=\frac{-b\pm \sqrt{b^{2}-4ac}}{2a}$
    - 本合同一式四份，省政府采购中心各一份"""
