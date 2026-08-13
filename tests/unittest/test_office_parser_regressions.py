from pytest import MonkeyPatch

from mineru.model.flash.office import image as office_image


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
