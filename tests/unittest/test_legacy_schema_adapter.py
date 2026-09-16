"""旧协议仅作为明确拒绝的负例保留，不再执行历史块迁移。"""

import pytest
from mineru.parser import ParseResult


@pytest.mark.parametrize(
    "payload",
    [
        {"pdf_info": [{"page_idx": 0, "para_blocks": []}]},
        {"schema_version": "1.0", "pages": [{"page_idx": 0, "blocks": []}]},
        {
            "schema_version": "2.0",
            "pages": [],
            "file_suffix": "pdf",
            "effort": "flash",
            "parse_mode": "txt",
            "mineru_version": "3.4.5",
        },
        {
            "schema": "docvortex.middle",
            "schema_version": "1.0",
            "pages": [],
            "file_suffix": "html",
            "producer": {"name": "docvortex", "version": "0.1.0"},
        },
    ],
)
def test_old_document_protocols_require_reparse(payload: dict) -> None:
    """拒绝旧 DocVortex、MinerU 及无协议标识的数据，不静默生成空页。"""
    with pytest.raises(ValueError, match="reparse"):
        ParseResult.from_dict(payload)
