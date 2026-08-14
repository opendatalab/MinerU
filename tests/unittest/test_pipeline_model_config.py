import pytest

from mineru.backend.pipeline import model_init


def _fake_get_atom_model(
    _self: object,
    atom_model_name: str,
    **_kwargs: object,
) -> object:
    return object()


def _fake_model_path(*_args: object, **_kwargs: object) -> str:
    return "/tmp/mineru-models"


@pytest.mark.parametrize(
    ("config", "formula_enabled", "table_enabled"),
    [
        ({}, True, True),
        ({"formula_config": None, "table_config": None}, True, True),
        (
            {
                "formula_config": {"enable": False},
                "table_config": {"enable": False},
            },
            False,
            False,
        ),
    ],
)
def test_pipeline_model_handles_optional_feature_configs(
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
    formula_enabled: bool,
    table_enabled: bool,
) -> None:
    monkeypatch.setattr(
        model_init.AtomModelSingleton,
        "get_atom_model",
        _fake_get_atom_model,
    )
    monkeypatch.setattr(
        model_init,
        "auto_download_and_get_model_root_path",
        _fake_model_path,
    )

    model = model_init.MineruPipelineModel(**config)

    assert model.apply_formula is formula_enabled
    assert model.apply_table is table_enabled
