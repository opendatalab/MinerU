"""守卫 MinerU 旧模型路径退役后的宿主合同。"""

import importlib.util

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "mineru.model.doc",
        "mineru.model.docx",
        "mineru.model.ppt",
        "mineru.model.pptx",
        "mineru.model.xls",
        "mineru.model.xlsx",
        "mineru.model.odt",
        "mineru.model.ods",
        "mineru.model.odp",
    ],
)
def test_legacy_office_model_paths_are_removed(module_name: str) -> None:
    """验证迁移后不再暴露旧的 Office 模型包路径。"""

    assert importlib.util.find_spec(module_name) is None
