"""API 精确档位集合、自动推荐、模型合并及三个命令入口的契约。"""

from __future__ import annotations

import asyncio
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from click.testing import CliRunner
from fastapi.testclient import TestClient
import httpx
import pytest
from typer.testing import CliRunner as TyperRunner

from mineru.kit.commands.api_server import main as standalone_main
from mineru.kit.main import app as kit_app
from mineru.kit.gradio.app import build_gradio_app
from mineru.kit.gradio.client import ManagedLocalApiServer, V1ArtifactClient
from mineru.model.runtime import tier_recommendation
from mineru.parser import api_server
from mineru.parser.server_tiers import parse_server_tier
from mineru.types import TIERS, TIERS_BY_SERVER_TIER, Tier, select_default_task_tier


@pytest.fixture
def preflight(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """只记录实际模型依赖需求，避免启动真实推理模型。"""
    check = Mock()
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", check)
    return check


@pytest.mark.parametrize("tiers", [subset for size in range(1, 5) for subset in combinations(TIERS, size)])
def test_every_explicit_subset_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight: Mock, tiers: tuple[Tier, ...]
) -> None:
    """所有非空集合都只发布指定档位，advanced 复用共享模型而不补齐其他档位。"""
    recommend = Mock(side_effect=AssertionError("explicit tiers must skip hardware discovery"))
    monkeypatch.setattr(tier_recommendation, "recommend_tiers", recommend)
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=tiers)
    expected_model = "standard" if {"standard", "advanced"}.intersection(tiers) else "basic" if "basic" in tiers else None
    if expected_model:
        assert preflight.call_count == 1
        assert preflight.call_args.args[0] == expected_model
    else:
        preflight.assert_not_called()
    with TestClient(app) as client:
        assert tuple(item["id"] for item in client.get("/v1/tiers").json()["data"]) == tiers
        assert tuple(app.state.tier_runtime_options) == tiers
        assert not hasattr(app.state, "flash_enabled")
        expected_default = next(t for t in ("standard", "basic", "flash", "advanced") if t in tiers)
        assert app.state.default_tier == select_default_task_tier(tiers) == expected_default
        for options in ({}, {"tier": None}):
            response = client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}], **options})
            assert response.status_code == 202, response.text
            assert app.state.task_store._records[response.json()["task_id"]].options.tier == expected_default
    recommend.assert_not_called()


@pytest.mark.parametrize("tiers", [("flash",), ("basic",), TIERS])
def test_auto_recommendation_is_fixed_at_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight: Mock, tiers: tuple[Tier, ...]
) -> None:
    """自动推荐每个服务只计算一次，生命周期启动与能力查询不重新选择。"""
    recommend = Mock(return_value=SimpleNamespace(pdf_image_tiers=tiers))
    monkeypatch.setattr(tier_recommendation, "recommend_tiers", recommend)
    app = api_server.create_app(upload_dir=str(tmp_path))
    recommend.return_value = SimpleNamespace(pdf_image_tiers=("advanced",))
    with TestClient(app) as client:
        for _ in range(2):
            assert tuple(row["id"] for row in client.get("/v1/tiers").json()["data"]) == TIERS_BY_SERVER_TIER[tiers[-1]]
    recommend.assert_called_once_with(vlm_config=app.state.vlm_config)


@pytest.mark.parametrize(
    ("options", "expected"), [({}, "standard"), ({"tier": None}, "standard"), ({"tier": "advanced"}, "advanced")]
)
def test_auto_server_accepts_task_with_standard_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight: Mock, options: dict[str, str | None], expected: str
) -> None:
    """硬件推荐四档后，经公共接单接口固定默认 standard 或调用方指定的 advanced。"""
    monkeypatch.setattr(tier_recommendation, "recommend_tiers", lambda **kwargs: SimpleNamespace(pdf_image_tiers=TIERS))
    app = api_server.create_app(upload_dir=str(tmp_path))
    with TestClient(app) as client:
        response = client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}], **options})
        assert response.status_code == 202, response.text
        assert app.state.task_store._records[response.json()["task_id"]].options.tier == expected


@pytest.mark.parametrize("tiers,expected", [(("flash",), "flash"), (("flash", "basic"), "basic"), (TIERS, "standard")])
def test_gradio_discovers_auto_server_and_matches_task_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight: Mock, tiers: tuple[Tier, ...], expected: Tier
) -> None:
    """串联托管命令、自动推荐、真实能力发现、界面构建与 Tasks 默认接单。"""
    recommend = Mock(return_value=SimpleNamespace(pdf_image_tiers=tiers))
    monkeypatch.setattr(tier_recommendation, "recommend_tiers", recommend)
    command = ManagedLocalApiServer()._command(8000, tmp_path)
    startup_tier = command[command.index("--tier") + 1]
    assert startup_tier == "auto"
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=parse_server_tier(startup_tier))
    with TestClient(app) as client:
        artifact_client = V1ArtifactClient(api_url="http://testserver", transport=httpx.ASGITransport(app=app))
        capabilities = asyncio.run(artifact_client.discover())
        assert capabilities.tiers == tiers
        demo = build_gradio_app(artifact_client, capabilities, output_root=tmp_path, enable_example=False)
        slider = next(block for block in demo.blocks.values() if block.__class__.__name__ == "Slider")
        assert tiers[int(slider.value)] == app.state.default_tier == expected
        response = client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}]})
        assert response.status_code == 202, response.text
        assert app.state.task_store._records[response.json()["task_id"]].options.tier == expected
    recommend.assert_called_once_with(vlm_config=app.state.vlm_config)


@pytest.mark.parametrize("value", ["", ",", "standard,", "flash,,basic", "ultra", "auto,flash"])
def test_invalid_cli_lists(value: str) -> None:
    """拒绝空列表、空项、未知名称及 auto 混用。"""
    with pytest.raises(ValueError):
        parse_server_tier(value)


def test_cli_normalization() -> None:
    """大小写及空白规范化，上限包含自身和较低档位。"""
    assert parse_server_tier(" AUTO ") is None
    assert parse_server_tier(" STANDARD ") == ("flash", "basic", "standard")


@pytest.mark.parametrize("options", [["--tiers", "standard"], ["--no-flash"], ["--no-advanced"]])
def test_removed_cli_options(options: list[str]) -> None:
    """复数列表与禁用开关不作为旧配置别名保留。"""
    result = CliRunner().invoke(api_server.main, options)
    assert result.exit_code != 0 and "No such option" in result.output
    result = TyperRunner().invoke(kit_app, ["api-server", *options])
    assert result.exit_code != 0 and "No such option" in result.output


@pytest.mark.parametrize("value", [(), [], set(), "standard", ["auto"], [""], [1]])
def test_invalid_python_collection(tmp_path: Path, value: Any) -> None:
    """Python 工厂必须收到非空集合，不能把字符串逐字符解释成档位。"""
    with pytest.raises(ValueError):
        api_server.create_app(upload_dir=str(tmp_path), tiers=value)


@pytest.mark.parametrize("option", ["tier", "no_flash", "no_advanced"])
def test_old_python_options_removed(option: str) -> None:
    """旧工厂参数直接报错，不保留混合配置入口。"""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        api_server.create_app(**{option: "standard"})


@pytest.mark.parametrize("entry", ["module", "kit", "standalone"])
@pytest.mark.parametrize("options", [[], ["--tier", "auto"], *[["--tier", tier] for tier in TIERS]])
def test_cli_entrypoints(monkeypatch: pytest.MonkeyPatch, preflight: Mock, entry: str, options: list[str]) -> None:
    """三个入口共享自动推荐与单档上限，不启动监听进程。"""
    seen: list[tuple[Tier, ...]] = []
    monkeypatch.setattr(tier_recommendation, "recommend_tiers", lambda **kwargs: SimpleNamespace(pdf_image_tiers=TIERS))
    monkeypatch.setattr("uvicorn.Server.run", lambda server: seen.append(server.config.app.state.server_tiers))
    if entry == "module":
        result = CliRunner().invoke(api_server.main, options)
    elif entry == "kit":
        result = TyperRunner().invoke(kit_app, ["api-server", *options])
    else:
        monkeypatch.setattr("sys.argv", ["mineru-api", *options])
        with pytest.raises(SystemExit) as stopped:
            standalone_main()
        assert stopped.value.code == 0
        assert seen == [TIERS_BY_SERVER_TIER[options[1]] if options and options[1] != "auto" else TIERS]
        return
    assert result.exit_code == 0, result.output
    assert seen == [TIERS_BY_SERVER_TIER[options[1]] if options and options[1] != "auto" else TIERS]


@pytest.mark.parametrize("tiers", [("advanced",), ("flash", "standard", "advanced"), ("basic",)])
def test_preload_shared_models_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight: Mock, tiers: tuple[Tier, ...]
) -> None:
    """多个开放档位共享一次模型预加载，只有 advanced 时也可启动。"""
    expected = "standard" if "advanced" in tiers else "basic"
    preload = Mock(return_value=api_server._ModelPreloadResult(expected, "test"))
    monkeypatch.setattr(api_server, "_preload_server_models", preload)
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=tiers, preload_models=True)
    with TestClient(app):
        assert tuple(app.state.tier_runtime_options) == tiers
    assert preload.call_count == 1
    assert preload.call_args.args[0] == expected
