"""验证当前 PP-DocLayoutV2 在 Transformers 5 下的构图、设备加载和位置编码契约。"""
# 可选依赖检测必须先于模型模块导入。
# ruff: noqa: E402

from __future__ import annotations

import pytest

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers", minversion="5.10.1")

from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2Config, PPDocLayoutV2ForObjectDetection


def test_layout_configuration_constructs_the_inference_graph() -> None:
    """构造真实版面模型图，覆盖仅运行后处理测试无法发现的配置校验错误。"""
    config = PPDocLayoutV2Config()
    with torch.device("meta"):
        model = PPDocLayoutV2ForObjectDetection(config)
    assert model.config.reading_order_config.hidden_size > 0
    assert any(True for _ in model.parameters())


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("mps", marks=pytest.mark.accelerator), pytest.param("xpu", marks=pytest.mark.accelerator)]
)
def test_layout_position_embedding_preserves_native_math(device: str) -> None:
    """CPU/MPS/XPU 包装保持官方位置编码数值，且没有新增或改名的模型权重。"""
    from transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2 import PPDocLayoutV2SinePositionEmbedding

    from mineru.model.layout.pp_doclayoutv2 import _CpuSinePositionEmbedding

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple Silicon MPS")
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("requires Intel XPU")
    native = PPDocLayoutV2SinePositionEmbedding(embed_dim=16)
    adapter = _CpuSinePositionEmbedding(native)
    actual = adapter(width=3, height=2, device=torch.device(device))
    expected = native(width=3, height=2, device=torch.device("cpu"), dtype=torch.float32)
    assert actual.device.type == device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert not adapter.state_dict()


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "mps", "xpu", "xpu:1"])
def test_layout_loader_adapts_position_embeddings_for_mps_and_xpu(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    """覆盖设备分支，并验证加载时仅提高行内公式阈值，保留其他模型配置。"""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2 import PPDocLayoutV2SinePositionEmbedding

    from mineru.model.layout import pp_doclayoutv2 as layout

    native = PPDocLayoutV2SinePositionEmbedding(embed_dim=16)
    layer = SimpleNamespace(position_embedding=native)
    model = Mock()
    model.model.encoder.aifi = [layer]
    config = SimpleNamespace(class_thresholds=list(layout.DEFAULT_CLASS_THRESHOLDS))
    config.class_thresholds[15] = 0.4
    config.class_thresholds[22] = 0.7
    expected_thresholds = list(config.class_thresholds)
    expected_thresholds[15] = 0.45
    load = Mock(return_value=model)
    monkeypatch.setattr(layout, "load_preprocess_config", lambda _path: {})
    monkeypatch.setattr(layout.PPDocLayoutV2Config, "from_pretrained", Mock(return_value=config))
    monkeypatch.setattr(layout.PPDocLayoutV2ForObjectDetection, "from_pretrained", load)

    loaded = layout.PPDocLayoutV2LayoutModel("local-checkpoint", device=device)

    assert loaded.config.class_thresholds == expected_thresholds
    assert load.call_args.kwargs["config"] is loaded.config
    assert not hasattr(loaded, "conf")
    assert load.call_args.kwargs["device_map"] == {"": device}
    model.to.assert_called_once_with(device)
    model.eval.assert_called_once_with()
    if torch.device(device).type in {"mps", "xpu"}:
        assert isinstance(layer.position_embedding, layout._CpuSinePositionEmbedding)
        # 使用 meta 作为无硬件目标，真实 CPU 算子计算后只将结果送到目标设备。
        embedding_forward = Mock(wraps=native.forward)
        monkeypatch.setattr(native, "forward", embedding_forward)
        result = layer.position_embedding(width=7, height=5, device=torch.device("meta"), dtype=torch.float16)
        assert result.device.type == "meta"
        assert result.dtype == torch.float16
        assert result.shape == (1, 35, 16)
        embedding_forward.assert_called_once_with(width=7, height=5, device=torch.device("cpu"), dtype=torch.float16)
    else:
        assert layer.position_embedding is native
