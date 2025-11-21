import os
from pathlib import Path

import pytest

from mineru.data.utils.uri_io import (
    read_bytes_from_uri,
    prepare_output_dir,
)


def test_read_bytes_from_uri_local_pdf():
    """本地 PDF 路径应能正常读取为 bytes。"""
    pdf_path = Path(__file__).parent / "unittest" / "pdfs" / "test.pdf"
    assert pdf_path.is_file()

    data = read_bytes_from_uri(str(pdf_path))
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_read_bytes_from_uri_unsupported_scheme():
    """非 s3 且带 scheme 的 URI 应抛出友好的错误。"""
    with pytest.raises(ValueError) as exc_info:
        read_bytes_from_uri("http://example.com/foo.pdf")

    msg = str(exc_info.value)
    assert "Unsupported URI scheme" in msg
    assert "Only local paths and s3://" in msg


def test_read_bytes_from_uri_s3_without_backend(monkeypatch):
    """在未配置 S3 backend 时，访问 s3:// 应提示安装 mineru[s3]。"""
    import mineru.data.utils.uri_io as uri_io_mod

    # 强制模拟缺失 S3 backend
    monkeypatch.setattr(uri_io_mod, "S3Reader", None)
    monkeypatch.setattr(uri_io_mod, "S3DataWriter", None)

    with pytest.raises(ImportError) as exc_info:
        uri_io_mod.read_bytes_from_uri("s3://bucket/key")

    msg = str(exc_info.value)
    assert "mineru[s3]" in msg


def test_prepare_output_dir_local(tmp_path):
    """本地输出应使用 fallback_local_dir 并确保目录存在。"""
    fallback_dir = tmp_path / "out"

    actual_dir, is_s3_output, normalized = prepare_output_dir(
        output_uri=None,
        fallback_local_dir=str(fallback_dir),
    )

    assert is_s3_output is False
    assert actual_dir == str(fallback_dir)
    assert normalized == str(fallback_dir)
    assert os.path.isdir(actual_dir)


def test_prepare_output_dir_s3(tmp_path):
    """s3 输出应返回临时目录并标记为 s3 输出。"""
    output_uri = "s3://my-bucket/some/prefix"

    actual_dir, is_s3_output, normalized = prepare_output_dir(
        output_uri=output_uri,
        fallback_local_dir=str(tmp_path),
    )

    assert is_s3_output is True
    assert normalized == output_uri
    assert os.path.isdir(actual_dir)


