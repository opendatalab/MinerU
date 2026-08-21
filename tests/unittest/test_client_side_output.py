from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from mineru.types import PageInfo

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CLIENT_SIDE_OUTPUT_PATH = _PROJECT_ROOT / "mineru/cli_old/client_side_output.py"


class _FakeParseResult:
    """为旧客户端输出测试提供最小 ParseResult 行为。"""

    def __init__(
        self,
        pages: list[PageInfo],
        _retained_page_indices: list[int] | None = None,
        _broken_page_indices: list[int] | None = None,
    ) -> None:
        """保存页面和旧客户端会回写的页号元数据。"""
        self.pages = pages
        self._retained_page_indices = _retained_page_indices or []
        self._broken_page_indices = _broken_page_indices or []

    @classmethod
    def from_dict(cls, _payload: dict[str, Any]) -> "_FakeParseResult":
        """从测试 middle json 返回一个空页面结果。"""
        return cls([PageInfo(page_idx=0)])

    def to_dict(self) -> dict[str, Any]:
        """返回旧客户端写文件所需的最小字典。"""
        return {"pages": [page.model_dump(mode="json") for page in self.pages]}


def _load_client_side_output(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """隔离旧 parser/render 基线导入，加载本次迁移的客户端输出模块。"""
    parser_package = ModuleType("mineru.parser")
    parser_package.__path__ = []  # type: ignore[attr-defined]
    parser_base = ModuleType("mineru.parser.base")
    parser_base.ParseResult = _FakeParseResult  # type: ignore[attr-defined]
    render_module = ModuleType("mineru.render")
    render_module.render_content_list = lambda *_args, **_kwargs: []  # type: ignore[attr-defined]
    render_module.render_markdown = lambda *_args, **_kwargs: ""  # type: ignore[attr-defined]
    render_module.render_structured_content = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mineru.parser", parser_package)
    monkeypatch.setitem(sys.modules, "mineru.parser.base", parser_base)
    monkeypatch.setitem(sys.modules, "mineru.render", render_module)

    spec = importlib.util.spec_from_file_location("_client_side_output_under_test", _CLIENT_SIDE_OUTPUT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load client_side_output.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_finalize_module(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """安装旧客户端动态 import 所需的最小 backend.pdf finalize 模块。"""
    pdf_package = ModuleType("mineru.backend.pdf")
    pdf_package.__path__ = []  # type: ignore[attr-defined]
    finalize_module = ModuleType("mineru.backend.pdf.model_output_to_middle_json")

    def finalize_middle_json_from_preproc(_pages: list[PageInfo], effort: str = "medium") -> None:
        """记录旧客户端向动态 finalize 入口转发的 effort。"""
        calls.append(effort)

    finalize_module.finalize_middle_json_from_preproc = finalize_middle_json_from_preproc  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mineru.backend.pdf", pdf_package)
    monkeypatch.setitem(sys.modules, "mineru.backend.pdf.model_output_to_middle_json", finalize_module)


def test_finalize_client_side_pages_passes_hybrid_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验迁移后的客户端 Hybrid finalize 继续转发 effort。"""
    module = _load_client_side_output(monkeypatch)
    calls: list[str] = []
    _install_fake_finalize_module(monkeypatch, calls)

    module._finalize_client_side_pages([PageInfo(page_idx=0)], "hybrid", effort="high")

    assert calls == ["high"]


def test_finalize_client_side_pages_rejects_low_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验迁移后的客户端 Hybrid finalize 仍拒绝 low effort。"""
    module = _load_client_side_output(monkeypatch)
    calls: list[str] = []
    _install_fake_finalize_module(monkeypatch, calls)

    with pytest.raises(ValueError, match="Unsupported effort 'low'"):
        module._finalize_client_side_pages([PageInfo(page_idx=0)], "hybrid", effort="low")

    assert calls == []


def test_regenerate_client_side_outputs_forwards_hybrid_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """校验客户端重建输出时继续把 Hybrid effort 传给迁移后的私有 helper。"""
    module = _load_client_side_output(monkeypatch)
    middle_json_path = tmp_path / "demo_middle.json"
    middle_json_path.write_text(json.dumps({"pages": []}), encoding="utf-8")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        module,
        "_finalize_client_side_pages",
        lambda _pages, backend, effort="medium": calls.append((backend, effort)),
    )

    module.regenerate_client_side_outputs(tmp_path, "demo", "hybrid-engine", effort="high")

    assert calls == [("hybrid", "high")]


def test_finalize_client_side_pages_uses_explicit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """校验迁移后的私有 helper 继续拒绝旧 pipeline backend。"""
    module = _load_client_side_output(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported client-side finalize backend 'pipeline'"):
        module._finalize_client_side_pages([PageInfo(page_idx=0)], "pipeline")
