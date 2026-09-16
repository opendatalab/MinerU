from __future__ import annotations

import importlib

import pytest


def test_old_backend_native_pdf_import_is_removed() -> None:
    """验证旧 backend native_pdf 路径不再兼容。"""

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mineru.backend.flash.native_pdf")
