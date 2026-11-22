"""验证 API 启动时按需预加载公式识别模型。"""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from mineru.kit.commands import api_server as kit_api
from mineru.kit.main import app as kit_app
from mineru.model.runtime.hybrid import HybridLocalModelContextSingleton
from mineru.parser.api_server import create_app


@pytest.mark.parametrize(
    ("tier", "preload_formula_model", "expected"),
    [("basic", True, 1), ("basic", False, 0), ("flash", True, 0)],
)
def test_formula_preload_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tier: str, preload_formula_model: bool, expected: int
) -> None:
    context = MagicMock()
    formula = PropertyMock(return_value=object())
    monkeypatch.setattr(type(context), "mfr_model", formula, raising=False)
    monkeypatch.setattr(HybridLocalModelContextSingleton, "get_model", MagicMock(return_value=context))
    kwargs = {"preload_formula_model": True} if preload_formula_model else {}
    app = create_app(upload_dir=str(tmp_path), tiers={tier}, preload_models=not preload_formula_model, **kwargs)

    with TestClient(app):
        assert app.state.model_preload_error is None
        assert formula.call_count == expected


def test_kit_forwards_formula_preload_option(monkeypatch: pytest.MonkeyPatch) -> None:
    run_server = MagicMock()
    monkeypatch.setattr(kit_api.parser_api_server.main, "main", run_server)

    result = CliRunner().invoke(kit_app, ["api-server", "--tier", "basic", "--preload-formula-model"])

    assert result.exit_code == 0, result.output
    assert "--preload-formula-model" in run_server.call_args.kwargs["args"]
