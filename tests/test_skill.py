# Copyright (c) Opendatalab. All rights reserved.
"""ocr-mineru skill 单元测试"""

import asyncio
from pathlib import Path

import pytest

from mineru.skill import ParseOptions, ParseResult, parse_file, parse_file_sync
from mineru.skill.config import SUPPORTED_SUFFIXES


class TestParseResult:
    """ParseResult 工具方法测试"""

    @pytest.fixture
    def sample_result(self):
        return ParseResult(
            markdown="# Title\n\nSome text",
            content_list=[],
            content_list_v2=[
                {"type": "title", "text": "Title"},
                {"type": "paragraph", "text": "Some text"},
                {"type": "table", "text": "| A | B |"},
                {"type": "image", "text": ""},
            ],
            middle_json={"pdf_info": []},
            model_output=None,
            images={"image_001.png": "data:image/png;base64,abc"},
            image_paths=[Path("image_001.png")],
            output_dir=Path("."),
            file_name="test",
        )

    def test_get_text_all(self, sample_result):
        text = sample_result.get_text()
        assert "Title" in text
        assert "Some text" in text
        assert "| A | B |" in text

    def test_get_text_filtered(self, sample_result):
        text = sample_result.get_text(["title", "paragraph"])
        assert "Title" in text
        assert "Some text" in text
        assert "| A | B |" not in text

    def test_get_tables(self, sample_result):
        tables = sample_result.get_tables()
        assert len(tables) == 1
        assert tables[0]["type"] == "table"

    def test_get_images(self, sample_result):
        images = sample_result.get_images()
        assert len(images) == 1
        assert images[0][0] == "image_001.png"

    def test_save_markdown(self, sample_result, tmp_path):
        path = tmp_path / "out.md"
        saved = sample_result.save_markdown(path)
        assert saved == path
        assert path.read_text(encoding="utf-8") == sample_result.markdown

    def test_save_content_list(self, sample_result, tmp_path):
        path = tmp_path / "out.json"
        saved = sample_result.save_content_list(path)
        assert saved == path
        assert path.exists()


class TestParseOptions:
    """ParseOptions 默认值测试"""

    def test_default_values(self):
        opts = ParseOptions()
        assert opts.backend == "hybrid-engine"
        assert opts.parse_method == "auto"
        assert opts.language == "ch"
        assert opts.formula_enable is True
        assert opts.table_enable is True


class TestInputValidation:
    """输入校验测试"""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_file_sync("/path/that/does/not/exist.pdf")

    def test_unsupported_suffix(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not a doc")
        with pytest.raises(ValueError, match="不支持的文件类型"):
            parse_file_sync(bad_file)

    @pytest.mark.parametrize("suffix", ["pdf", "png", "jpg", "jpeg", "docx", "pptx", "xlsx"])
    def test_supported_suffixes(self, suffix, tmp_path):
        # 仅校验后缀，不实际解析
        f = tmp_path / f"test.{suffix}"
        f.write_bytes(b"dummy")
        assert suffix.lstrip(".") in SUPPORTED_SUFFIXES


class TestAsyncInterface:
    """异步接口测试"""

    def test_parse_file_sync_wraps_async(self, tmp_path):
        # 不存在的文件应该同步抛出异常
        with pytest.raises(FileNotFoundError):
            parse_file_sync(tmp_path / "missing.pdf")

    def test_async_parse_file_raises(self, tmp_path):
        async def _run():
            return await parse_file(tmp_path / "missing.pdf")

        with pytest.raises(FileNotFoundError):
            asyncio.run(_run())


@pytest.mark.skip(reason="需要模型权重文件，属于集成测试")
class TestSkillIntegration:
    """集成测试：需要实际模型权重"""

    def test_parse_demo_pdf(self):
        demo_pdf = Path("demo/pdfs/demo3.pdf")
        result = parse_file_sync(demo_pdf)
        assert isinstance(result, ParseResult)
        assert len(result.markdown) > 0
        assert result.file_name == "demo3"
