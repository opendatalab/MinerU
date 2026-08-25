from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mineru.config import Config, _collect_env_overrides, _load_config


def test_render_config_defaults() -> None:
    """验证 config.yaml 缺省时使用美元公式定界符。"""
    delimiters = Config().render.latex_delimiters

    assert delimiters.display.left == "$$"
    assert delimiters.display.right == "$$"
    assert delimiters.inline.left == "$"
    assert delimiters.inline.right == "$"


def test_render_config_reads_yaml_shape(tmp_path: Path) -> None:
    """验证 render.latex_delimiters YAML 结构能进入 typed Config。"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
render:
  latex_delimiters:
    display:
      left: '\\['
      right: '\\]'
    inline:
      left: '\\('
      right: '\\)'
""".strip(),
        encoding="utf-8",
    )

    config = Config(**_load_config(str(config_file)))

    assert config.render.latex_delimiters.display.left == r"\["
    assert config.render.latex_delimiters.inline.right == r"\)"


def test_render_config_supports_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证现有环境变量递归覆盖机制可定位 render delimiter。"""
    prefix = "TEST_MINERU_"
    monkeypatch.setenv(f"{prefix}RENDER_LATEX_DELIMITERS_INLINE_LEFT", r"\(")
    monkeypatch.setenv(f"{prefix}RENDER_LATEX_DELIMITERS_INLINE_RIGHT", r"\)")

    overrides, _paths = _collect_env_overrides(prefix=prefix)
    config = Config(**overrides)

    assert config.render.latex_delimiters.inline.left == r"\("
    assert config.render.latex_delimiters.inline.right == r"\)"


@pytest.mark.parametrize("side", ["left", "right"])
def test_render_config_rejects_empty_delimiters(side: str) -> None:
    """验证任一空定界符都会在 typed config 边界报错。"""
    pair = {"left": "$", "right": "$"}
    pair[side] = ""

    with pytest.raises(ValidationError):
        Config(render={"latex_delimiters": {"inline": pair}})


def test_render_config_does_not_read_legacy_root_key() -> None:
    """验证旧根级 latex-delimiter-config 不影响新 render 配置。"""
    config = Config.model_validate(
        {
            "latex-delimiter-config": {
                "inline": {"left": "OLD", "right": "OLD"},
            }
        }
    )

    assert config.render.latex_delimiters.inline.left == "$"
