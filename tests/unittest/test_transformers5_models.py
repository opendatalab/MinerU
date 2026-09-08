"""使用真实的小型网络验证 Transformers 5 的缓存、生成和权重加载契约。"""
# 可选依赖检测必须先于模型模块导入。
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers", minversion="5.10.1")

from transformers import AutoConfig, PreTrainedTokenizerFast, VisionEncoderDecoderModel
from transformers.cache_utils import EncoderDecoderCache

from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2Config, PPDocLayoutV2ForObjectDetection
from mineru.model.mfr.unimernet.unimernet_hf import (
    UnimerMBartConfig,
    UnimerMBartForCausalLM,
    UnimernetConfig,
    UnimernetModel,
    UnimerSwinConfig,
)


def _decoder(attention: str = "eager", *, tied: bool = False) -> UnimerMBartForCausalLM:
    """构造保留 Q/K 压缩、交叉注意力与两层缓存的最小解码器。"""
    config = UnimerMBartConfig(
        vocab_size=19,
        d_model=32,
        qk_squeeze=2,
        encoder_layers=1,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        is_encoder_decoder=False,
        tie_word_embeddings=tied,
    )
    config._attn_implementation = attention
    torch.manual_seed(17)
    return UnimerMBartForCausalLM(config).eval()


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_incremental_cache_matches_full_forward(attention: str, batch_size: int) -> None:
    """逐 token 生成必须与完整前向一致，交叉注意力只物化一次且 K/V 维度不同。"""
    model = _decoder(attention)
    tokens = torch.tensor([[0, 5, 7, 8, 9], [0, 6, 4, 5, 3]])[:batch_size]
    encoder = torch.randn(batch_size, 6, 32)
    mask = torch.ones_like(tokens)
    cross_mask = torch.ones(batch_size, 6, dtype=torch.long)
    cross_mask[:, -1] = 0
    cache = None
    parts = []
    cross_keys = None
    with torch.inference_mode():
        expected = model(
            tokens, encoder_hidden_states=encoder, encoder_attention_mask=cross_mask, attention_mask=mask, use_cache=False
        ).logits
        for index in range(tokens.shape[1]):
            output = model(
                tokens[:, index : index + 1],
                encoder_hidden_states=encoder,
                encoder_attention_mask=cross_mask,
                attention_mask=mask[:, : index + 1],
                past_key_values=cache,
                use_cache=True,
            )
            if cache is not None:
                assert output.past_key_values is cache
            cache = output.past_key_values
            assert isinstance(cache, EncoderDecoderCache)
            assert cache.get_seq_length() == index + 1
            keys = cache.cross_attention_cache.layers[0].keys
            if cross_keys is None:
                cross_keys = keys
            assert keys is cross_keys
            parts.append(output.logits)
    torch.testing.assert_close(torch.cat(parts, dim=1), expected, rtol=1e-4, atol=1e-5)
    assert cache.self_attention_cache.layers[0].keys.shape[-1] == 4
    assert cache.self_attention_cache.layers[0].values.shape[-1] == 8


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_chunked_prefill_and_reordered_cache(attention: str) -> None:
    """多 token 前缀和重排后的 batch 必须保留位置偏移及正确的交叉注意力缓存。"""
    model = _decoder(attention)
    tokens = torch.tensor([[0, 3, 4, 5, 6], [0, 7, 8, 9, 10]])
    encoder = torch.randn(2, 4, 32)
    order = torch.tensor([1, 0])
    with torch.inference_mode():
        prefix = model(tokens[:, :3], encoder_hidden_states=encoder, use_cache=True)
        prefix.past_key_values.reorder_cache(order)
        suffix = model(
            tokens[order, 3:], encoder_hidden_states=encoder[order], past_key_values=prefix.past_key_values, use_cache=True
        )
        expected = model(tokens[order], encoder_hidden_states=encoder[order], use_cache=False)
    torch.testing.assert_close(suffix.logits, expected.logits[:, 3:], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_generate_matches_without_cache_and_stops_at_eos(attention: str) -> None:
    """实际 GenerationMixin 的批量生成须保持 token 序列一致，并在强制 EOS 处停止。"""
    model = _decoder(attention)
    tokens = torch.tensor([[0, 4, 5], [0, 6, 7]])
    with torch.inference_mode():
        cached = model.generate(
            tokens, max_new_tokens=48, do_sample=False, use_cache=True, eos_token_id=None, forced_eos_token_id=2
        )
        uncached = model.generate(
            tokens, max_new_tokens=48, do_sample=False, use_cache=False, eos_token_id=None, forced_eos_token_id=2
        )
    assert torch.equal(cached, uncached)
    assert torch.all(cached[:, -1] == model.config.eos_token_id)


@pytest.mark.parametrize("tied", [False, True])
def test_decoder_save_reload_preserves_weight_tying(tmp_path: Path, tied: bool) -> None:
    """新版权重映射不得将原本独立的 lm_head 与词嵌入意外绑定。"""
    model = _decoder(tied=tied)
    model.save_pretrained(tmp_path)
    restored, info = UnimerMBartForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    assert (restored.lm_head.weight is restored.model.decoder.embed_tokens.weight) is tied
    with torch.inference_mode():
        torch.testing.assert_close(model(torch.tensor([[0, 4]])).logits, restored(torch.tensor([[0, 4]])).logits)


def test_composite_model_round_trip_without_auto_registration(tmp_path: Path) -> None:
    """完整组合模型须显式读取子配置、保存权重并恢复生成，且不污染全局 Auto 映射。"""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    vocabulary = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "x": 4, "+": 5, "1": 6}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="<unk>")),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=16,
    )
    tokenizer.save_pretrained(tmp_path)
    encoder = UnimerSwinConfig(
        image_size=(16, 16),
        patch_size=2,
        embed_dim=8,
        depths=[1, 1],
        num_heads=[2, 4],
        window_size=2,
        use_2d_embeddings=False,
        path_norm=True,
    )
    decoder = UnimerMBartConfig(
        vocab_size=len(tokenizer),
        d_model=16,
        encoder_layers=1,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=16,
        is_decoder=True,
        is_encoder_decoder=False,
        tie_word_embeddings=False,
    )
    config = UnimernetConfig(
        encoder=encoder.to_dict(),
        decoder=decoder.to_dict(),
        decoder_start_token_id=0,
        pad_token_id=1,
        tie_word_embeddings=False,
        _name_or_path=str(tmp_path),
    )
    model = UnimernetModel(config).eval()
    model.save_pretrained(tmp_path)
    restored, info = UnimernetModel.from_pretrained(tmp_path, output_loading_info=True)
    assert restored.config.model_type == "vision-encoder-decoder"
    assert isinstance(restored.config.encoder, UnimerSwinConfig)
    assert isinstance(restored.config.decoder, UnimerMBartConfig)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    with pytest.raises(ValueError):
        AutoConfig.for_model("unimer-swin")
    # 既有带 model.model. 前缀的 PyTorch checkpoint 入口也使用显式子配置。
    torch.save(
        {"model": {"model.model." + key: value for key, value in model.state_dict().items()}}, tmp_path / "checkpoint.pth"
    )
    checkpoint_model = UnimernetModel.from_checkpoint(str(tmp_path), model_filename="checkpoint.pth").eval()
    pixels = torch.randn(1, 3, 16, 16)
    with torch.inference_mode():
        expected = VisionEncoderDecoderModel.generate(model, pixel_values=pixels, max_new_tokens=4)
        actual = VisionEncoderDecoderModel.generate(restored, pixel_values=pixels, max_new_tokens=4)
        checkpoint_result = VisionEncoderDecoderModel.generate(checkpoint_model, pixel_values=pixels, max_new_tokens=4)
    assert torch.equal(actual, expected)
    assert torch.equal(checkpoint_result, expected)


def test_layout_configuration_constructs_the_inference_graph() -> None:
    """构造真实版面模型图，覆盖仅运行后处理测试无法发现的配置校验错误。"""
    config = PPDocLayoutV2Config()
    with torch.device("meta"):
        model = PPDocLayoutV2ForObjectDetection(config)
    assert model.config.reading_order_config.hidden_size > 0
    assert any(True for _ in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Silicon MPS")
def test_layout_position_embedding_preserves_native_math_on_mps() -> None:
    """MPS 路径仍使用官方位置编码数值，并且没有新增或改名的模型权重。"""
    from transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2 import PPDocLayoutV2SinePositionEmbedding
    from mineru.model.layout.pp_doclayoutv2 import _CpuSinePositionEmbedding

    native = PPDocLayoutV2SinePositionEmbedding(embed_dim=16)
    adapter = _CpuSinePositionEmbedding(native)
    actual = adapter(width=3, height=2, device=torch.device("mps"))
    expected = native(width=3, height=2, device=torch.device("cpu"), dtype=torch.float32)
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert not adapter.state_dict()


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_unimernet_stages_only_mps_weights_on_cpu(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    """MPS 在目标精度下经 CPU 物化后串行搬运，CPU/CUDA 保持直接加载。"""
    from types import SimpleNamespace

    from mineru.model.mfr.unimernet.Unimernet import UnimernetModel as Wrapper
    from mineru.model.mfr.unimernet.unimernet_hf import UnimernetModel as Model

    calls = []
    moves = []
    model = SimpleNamespace(to=lambda target: moves.append(str(target)), eval=lambda: None)

    def load(*args: object, **kwargs: object) -> object:
        """只记录公开加载协议，测试不分配真实设备或模型权重。"""
        calls.append(kwargs)
        return model

    monkeypatch.setattr(Model, "from_pretrained", load)
    Wrapper("local-checkpoint", device)
    assert str(calls[0]["device_map"][""]) == ("cpu" if device == "mps" else device)
    assert calls[0]["dtype"] == (torch.float32 if device == "cpu" else torch.float16)
    assert moves == (["mps"] if device == "mps" else [])
