import argparse
import colorsys
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from tqdm import tqdm

import torch
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3TextEmbeddings
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.modeling_rt_detr import RTDetrForObjectDetection, RTDetrModel, RTDetrPreTrainedModel
from transformers.utils import ModelOutput

DEFAULT_IMAGE_SIZE = (800, 800)
DEFAULT_RESCALE_FACTOR = 1.0 / 255.0

PP_DOCLAYOUT_V2_LABELS = [
    "abstract",           # 0
    "algorithm",          # 1
    "aside_text",         # 2
    "chart",              # 3
    "content",            # 4
    "formula",            # 5
    "doc_title",          # 6
    "figure_title",       # 7
    "footer",             # 8
    "footer",             # 9
    "footnote",           # 10
    "formula_number",     # 11
    "header",             # 12
    "header",             # 13
    "image",              # 14
    "formula",            # 15
    "number",             # 16
    "paragraph_title",    # 17
    "reference",          # 18
    "reference_content",  # 19
    "seal",               # 20
    "table",              # 21
    "text",               # 22
    "text",               # 23
    "vision_footnote",    # 24
]

# Per-class confidence threshold used before reading-order decoding.
DEFAULT_CLASS_THRESHOLDS = [
    0.5,   # 0  abstract
    0.5,   # 1  algorithm
    0.5,   # 2  aside_text
    0.5,   # 3  chart
    0.5,   # 4  content
    0.4,   # 5  formula
    0.4,   # 6  doc_title
    0.5,   # 7  figure_title
    0.5,   # 8  footer
    0.5,   # 9  footer
    0.5,   # 10 footnote
    0.5,   # 11 formula_number
    0.5,   # 12 header
    0.5,   # 13 header
    0.5,   # 14 image
    0.4,   # 15 formula
    0.5,   # 16 number
    0.4,   # 17 paragraph_title
    0.5,   # 18 reference
    0.5,   # 19 reference_content
    0.45,  # 20 seal
    0.5,   # 21 table
    0.4,   # 22 text
    0.4,   # 23 text
    0.5,   # 24 vision_footnote
]

# Reading-order head class remap used by the original upstream model.
DEFAULT_CLASS_ORDER = [
    4,   # 0  abstract
    2,   # 1  algorithm
    14,  # 2  aside_text
    1,   # 3  chart
    5,   # 4  content
    7,   # 5  formula
    8,   # 6  doc_title
    6,   # 7  figure_title
    11,  # 8  footer
    11,  # 9  footer
    9,   # 10 footnote
    13,  # 11 formula_number
    10,  # 12 header
    10,  # 13 header
    1,   # 14 image
    2,   # 15 formula
    3,   # 16 number
    0,   # 17 paragraph_title
    2,   # 18 reference
    2,   # 19 reference_content
    12,  # 20 seal
    1,   # 21 table
    2,   # 22 text
    15,  # 23 text
    6,   # 24 vision_footnote
]


def _build_default_backbone_config():
    return AutoConfig.for_model(
        "hgnet_v2",
        arch="L",
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        lr_mult_list=[0, 0.05, 0.05, 0.05, 0.05],
        out_features=["stage2", "stage3", "stage4"],
    )


def _create_bidirectional_mask(
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    encoder_hidden_states: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 4:
        return attention_mask
    if attention_mask.ndim != 2:
        raise ValueError(
            f"PP-DocLayoutV2 reading-order mask must be 2D or 4D, got shape {tuple(attention_mask.shape)}"
        )

    embeds = encoder_hidden_states if encoder_hidden_states is not None else inputs_embeds
    batch_size, query_length = inputs_embeds.shape[:2]
    key_length = attention_mask.shape[1]

    if attention_mask.shape[0] != batch_size:
        raise ValueError(
            f"Attention mask batch size {attention_mask.shape[0]} does not match embeddings batch size {batch_size}"
        )

    expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, query_length, key_length)
    expanded_mask = expanded_mask.to(device=embeds.device, dtype=embeds.dtype)
    min_value = torch.finfo(embeds.dtype).min
    return torch.where(
        expanded_mask > 0,
        torch.zeros(1, dtype=embeds.dtype, device=embeds.device),
        torch.full((1,), min_value, dtype=embeds.dtype, device=embeds.device),
    )


def _load_preprocess_config(model_dir: str) -> Dict:
    config_path = os.path.join(model_dir, "preprocessor_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_runtime_device(device: Optional[Union[str, torch.device]]) -> str:
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device_str = str(device)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device_str.startswith("mps") and not torch.backends.mps.is_available():
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _label_to_color(label: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    saturation = 0.65 + (digest[1] / 255.0) * 0.2
    value = 0.85 + (digest[2] / 255.0) * 0.1
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


@dataclass
class PPDocLayoutV2ForObjectDetectionOutput(ModelOutput):
    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    order_logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_logits: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    intermediate_predicted_corners: torch.FloatTensor | None = None
    initial_reference_points: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    init_reference_points: torch.FloatTensor | None = None
    enc_topk_logits: torch.FloatTensor | None = None
    enc_topk_bboxes: torch.FloatTensor | None = None
    enc_outputs_class: torch.FloatTensor | None = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None
    denoising_meta_values: dict | None = None


class PPDocLayoutV2ReadingOrderConfig(PretrainedConfig):
    model_type = "pp_doclayout_v2_reading_order"

    def __init__(
        self,
        hidden_size=512,
        num_attention_heads=8,
        attention_probs_dropout_prob=0.1,
        has_relative_attention_bias=False,
        has_spatial_attention_bias=True,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.1,
        intermediate_size=2048,
        hidden_act="gelu",
        num_hidden_layers=6,
        rel_pos_bins=32,
        max_rel_pos=128,
        rel_2d_pos_bins=64,
        max_rel_2d_pos=256,
        max_position_embeddings=514,
        max_2d_position_embeddings=1024,
        type_vocab_size=1,
        vocab_size=4,
        initializer_range=0.01,
        start_token_id=0,
        pad_token_id=1,
        end_token_id=2,
        pred_token_id=3,
        coordinate_size=171,
        shape_size=170,
        num_classes=20,
        relation_bias_embed_dim=16,
        relation_bias_theta=10000,
        relation_bias_scale=100,
        global_pointer_head_size=64,
        gp_dropout_value=0.0,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.max_rel_2d_pos = max_rel_2d_pos
        self.max_position_embeddings = max_position_embeddings
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.start_token_id = start_token_id
        self.pad_token_id = pad_token_id
        self.end_token_id = end_token_id
        self.pred_token_id = pred_token_id
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.num_classes = num_classes
        self.relation_bias_embed_dim = relation_bias_embed_dim
        self.relation_bias_theta = relation_bias_theta
        self.relation_bias_scale = relation_bias_scale
        self.global_pointer_head_size = global_pointer_head_size
        self.gp_dropout_value = gp_dropout_value
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class PPDocLayoutV2Config(RTDetrConfig):
    model_type = "pp_doclayout_v2"
    sub_configs = {"backbone_config": AutoConfig, "reading_order_config": PPDocLayoutV2ReadingOrderConfig}

    def __init__(
        self,
        backbone_config=None,
        class_thresholds: Optional[List[float]] = None,
        class_order: Optional[List[int]] = None,
        reading_order_config: Optional[Union[PPDocLayoutV2ReadingOrderConfig, Dict]] = None,
        **kwargs,
    ):
        if backbone_config is None:
            backbone_config = _build_default_backbone_config()
        if isinstance(reading_order_config, PPDocLayoutV2ReadingOrderConfig):
            reading_order = reading_order_config
        else:
            reading_order = PPDocLayoutV2ReadingOrderConfig(**(reading_order_config or {}))

        super().__init__(
            backbone_config=backbone_config,
            class_thresholds=class_thresholds or list(DEFAULT_CLASS_THRESHOLDS),
            class_order=class_order or list(DEFAULT_CLASS_ORDER),
            **kwargs,
        )
        self.class_thresholds = list(class_thresholds or DEFAULT_CLASS_THRESHOLDS)
        self.class_order = list(class_order or DEFAULT_CLASS_ORDER)
        self.reading_order_config = reading_order

    def to_dict(self):
        output = super().to_dict()
        output["reading_order_config"] = self.reading_order_config.to_dict()
        return output


class PPDocLayoutV2GlobalPointer(nn.Module):
    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__()
        self.head_size = config.global_pointer_head_size
        self.dense = nn.Linear(config.hidden_size, self.head_size * 2)
        self.dropout = nn.Dropout(config.gp_dropout_value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        projection = self.dense(inputs).reshape(batch_size, sequence_length, 2, self.head_size)
        projection = self.dropout(projection)
        queries, keys = torch.unbind(projection, dim=2)
        logits = (queries @ keys.transpose(-2, -1)) / (self.head_size**0.5)
        mask = torch.tril(torch.ones(sequence_length, sequence_length, device=logits.device)).bool()
        return logits.masked_fill(mask.unsqueeze(0), -1e4)


class PPDocLayoutV2PositionRelationEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig, device=None):
        super().__init__()
        self.config = config
        self.embed_dim = config.relation_bias_embed_dim
        self.scale = config.relation_bias_scale
        self.pos_proj = nn.Conv2d(
            in_channels=self.embed_dim * 4,
            out_channels=config.num_attention_heads,
            kernel_size=1,
        )
        inv_freq, _ = self.compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: PPDocLayoutV2ReadingOrderConfig,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, float]:
        base = config.relation_bias_theta
        dim = config.relation_bias_embed_dim
        half_dim = dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / half_dim)
        )
        return inv_freq, 1.0

    def box_relative_encoding(
        self,
        source_boxes: torch.Tensor,
        target_boxes: Optional[torch.Tensor] = None,
        epsilon: float = 1e-5,
    ) -> torch.Tensor:
        source_boxes = source_boxes.unsqueeze(-2)
        target_boxes = source_boxes if target_boxes is None else target_boxes.unsqueeze(-3)
        source_coordinates, source_dim = source_boxes[..., :2], source_boxes[..., 2:]
        target_coordinates, target_dim = target_boxes[..., :2], target_boxes[..., 2:]
        coordinate_difference = torch.abs(source_coordinates - target_coordinates)
        relative_coordinates = torch.log(coordinate_difference / (source_dim + epsilon) + 1.0)
        relative_dim = torch.log((source_dim + epsilon) / (target_dim + epsilon))
        return torch.cat([relative_coordinates, relative_dim], dim=-1)

    def get_position_embedding(self, x: torch.Tensor, scale: float = 100.0) -> torch.Tensor:
        embedding = (x * scale).unsqueeze(-1) * self.inv_freq
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1).flatten(start_dim=-2).to(x.dtype)

    def forward(self, source_boxes: torch.Tensor, target_boxes: Optional[torch.Tensor] = None) -> torch.Tensor:
        target_boxes = source_boxes if target_boxes is None else target_boxes
        with torch.no_grad():
            relative_encoding = self.box_relative_encoding(source_boxes, target_boxes)
            position_embedding = self.get_position_embedding(relative_encoding, self.scale).permute(0, 3, 1, 2)
        return self.pos_proj(position_embedding)


class PPDocLayoutV2ReadingOrderSelfAttention(nn.Module):
    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads "
                f"({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    @staticmethod
    def cogview_attention(attention_scores: torch.Tensor, alpha: float = 32.0) -> torch.Tensor:
        scaled_attention_scores = attention_scores / alpha
        max_value = scaled_attention_scores.amax(dim=-1, keepdim=True)
        return nn.Softmax(dim=-1)((scaled_attention_scores - max_value) * alpha)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rel_pos: Optional[torch.Tensor] = None,
        rel_2d_pos: Optional[torch.Tensor] = None,
    ):
        batch_size, _, _ = hidden_states.shape
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        attention_scores = torch.matmul(query_layer / (self.attention_head_size**0.5), key_layer.transpose(-1, -2))
        if rel_2d_pos is not None:
            attention_scores = attention_scores + rel_2d_pos
        elif self.has_relative_attention_bias and rel_pos is not None:
            attention_scores = attention_scores + rel_pos / (self.attention_head_size**0.5)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = self.cogview_attention(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*context_layer.size()[:-2], self.all_head_size)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class PPDocLayoutV2ReadingOrderSelfOutput(nn.Module):
    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.norm(hidden_states + input_tensor)


class PPDocLayoutV2ReadingOrderIntermediate(nn.Module):
    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class PPDocLayoutV2ReadingOrderOutput(nn.Module):
    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.norm(hidden_states + input_tensor)


class PPDocLayoutV2ReadingOrderAttention(nn.Module):
    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__()
        self.self = PPDocLayoutV2ReadingOrderSelfAttention(config)
        self.output = PPDocLayoutV2ReadingOrderSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rel_pos: Optional[torch.Tensor] = None,
        rel_2d_pos: Optional[torch.Tensor] = None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class PPDocLayoutV2ReadingOrderLayer(nn.Module):
    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__()
        self.attention = PPDocLayoutV2ReadingOrderAttention(config)
        self.intermediate = PPDocLayoutV2ReadingOrderIntermediate(config)
        self.output = PPDocLayoutV2ReadingOrderOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rel_pos: Optional[torch.Tensor] = None,
        rel_2d_pos: Optional[torch.Tensor] = None,
    ):
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = attention_outputs[0]
        layer_output = self.output(self.intermediate(attention_output), attention_output)
        return (layer_output,) + attention_outputs[1:]


class PPDocLayoutV2ReadingOrderEncoder(nn.Module):
    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PPDocLayoutV2ReadingOrderLayer(config) for _ in range(config.num_hidden_layers)])
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        self.gradient_checkpointing = False

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config.num_attention_heads, bias=False)

        self.rel_bias_module = PPDocLayoutV2PositionRelationEmbedding(config)

    @staticmethod
    def relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = torch.max(-relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        val_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return ret + torch.where(is_small, relative_position, val_if_large)

    def _cal_1d_pos_emb(self, position_ids: torch.Tensor) -> torch.Tensor:
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        with torch.no_grad():
            rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        return rel_pos.contiguous()

    def _cal_2d_pos_emb(self, bbox: torch.Tensor) -> torch.Tensor:
        x_min, y_min, x_max, y_max = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        width = (x_max - x_min).clamp(min=1e-3)
        height = (y_max - y_min).clamp(min=1e-3)
        center_x = (x_min + x_max) * 0.5
        center_y = (y_min + y_max) * 0.5
        boxes = torch.stack([center_x, center_y, width, height], dim=-1)
        return self.rel_bias_module(boxes)

    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        position_ids: Optional[torch.Tensor] = None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias and position_ids is not None else None
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias and bbox is not None else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PPDocLayoutV2TextEmbeddings(LayoutLMv3TextEmbeddings):
    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__(config)
        del self.LayerNorm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        spatial_embed_dim = 4 * config.coordinate_size + 2 * config.shape_size
        self.spatial_proj = nn.Linear(spatial_embed_dim, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids)
        embeddings = embeddings + self.position_embeddings(position_ids)
        spatial_position_embeddings = self.spatial_proj(self.calculate_spatial_position_embeddings(bbox))
        return embeddings + spatial_position_embeddings


class PPDocLayoutV2PreTrainedModel(RTDetrPreTrainedModel):
    config_class = PPDocLayoutV2Config

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, PPDocLayoutV2TextEmbeddings):
            module.position_ids.copy_(torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, PPDocLayoutV2PositionRelationEmbedding):
            inv_freq, _ = module.compute_default_rope_parameters(module.config, module.inv_freq.device)
            module.register_buffer("inv_freq", inv_freq, persistent=False)


class PPDocLayoutV2ReadingOrder(PPDocLayoutV2PreTrainedModel):
    config_class = PPDocLayoutV2ReadingOrderConfig

    def __init__(self, config: PPDocLayoutV2ReadingOrderConfig):
        super().__init__(config)
        self.embeddings = PPDocLayoutV2TextEmbeddings(config)
        self.label_embeddings = nn.Embedding(config.num_classes, config.hidden_size)
        self.label_features_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.encoder = PPDocLayoutV2ReadingOrderEncoder(config)
        self.relative_head = PPDocLayoutV2GlobalPointer(config)
        self.post_init()

    def forward(
        self,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            raise ValueError("PP-DocLayoutV2 reading-order inference requires a mask tensor.")

        device = mask.device
        batch_size, seq_len = mask.shape
        num_pred = mask.sum(dim=1)

        input_ids = torch.full(
            (batch_size, seq_len + 2),
            self.config.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        input_ids[:, 0] = self.config.start_token_id
        pred_col_idx = torch.arange(seq_len + 2, device=device).unsqueeze(0)
        pred_mask = (pred_col_idx >= 1) & (pred_col_idx <= num_pred.unsqueeze(1))
        input_ids[pred_mask] = self.config.pred_token_id
        input_ids[torch.arange(batch_size, device=device), num_pred + 1] = self.config.end_token_id

        pad_box = torch.zeros((batch_size, 1, boxes.shape[-1]), dtype=boxes.dtype, device=device)
        pad_boxes = torch.cat([pad_box, boxes, pad_box], dim=1)
        bbox_embedding = self.embeddings(input_ids=input_ids, bbox=pad_boxes.long())

        if labels is not None:
            label_proj = self.label_features_projection(self.label_embeddings(labels))
            pad_label = torch.zeros((batch_size, 1, label_proj.shape[-1]), dtype=label_proj.dtype, device=device)
            label_proj = torch.cat([pad_label, label_proj, pad_label], dim=1)
        else:
            label_proj = torch.zeros_like(bbox_embedding)

        final_embeddings = self.embeddings.norm(bbox_embedding + label_proj)
        final_embeddings = self.embeddings.dropout(final_embeddings)

        attention_mask = pred_col_idx < (num_pred + 2).unsqueeze(1)
        attention_mask = _create_bidirectional_mask(final_embeddings, attention_mask)
        position_ids = self.embeddings.create_position_ids_from_input_ids(input_ids, self.embeddings.padding_idx)
        encoder_output = self.encoder(
            hidden_states=final_embeddings,
            bbox=pad_boxes,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        token = encoder_output.last_hidden_state[:, 1 : 1 + seq_len, :]
        return self.relative_head(token)


class PPDocLayoutV2Model(RTDetrModel):
    def __init__(self, config: PPDocLayoutV2Config):
        super().__init__(config)
        self.denoising_class_embed = nn.Embedding(config.num_labels, config.d_model)
        nn.init.xavier_uniform_(self.denoising_class_embed.weight)


class PPDocLayoutV2ForObjectDetection(RTDetrForObjectDetection):
    config_class = PPDocLayoutV2Config
    _keys_to_ignore_on_load_missing = ["num_batches_tracked", "rel_pos_y_bias", "rel_pos_x_bias"]

    def __init__(self, config: PPDocLayoutV2Config):
        super().__init__(config)
        self.model = PPDocLayoutV2Model(config)
        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed
        self.reading_order = PPDocLayoutV2ReadingOrder(config.reading_order_config)
        self.num_queries = config.num_queries
        self.config = config
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[list[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if labels is not None:
            raise ValueError("PPDocLayoutV2ForObjectDetection only supports inference.")

        use_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        raw_bboxes = outputs.intermediate_reference_points[:, -1]
        logits = outputs.intermediate_logits[:, -1]

        box_centers, box_sizes = raw_bboxes.split(2, dim=-1)
        bboxes = torch.cat([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], dim=-1) * 1000.0
        bboxes = bboxes.clamp_(0.0, 1000.0)

        max_logits, class_ids = logits.max(dim=-1)
        max_probs = max_logits.sigmoid()
        class_thresholds = torch.tensor(self.config.class_thresholds, dtype=torch.float32, device=logits.device)
        thresholds = class_thresholds[class_ids]
        mask = max_probs >= thresholds
        indices = torch.argsort(mask.to(torch.int8), dim=1, descending=True)

        sorted_class_ids = torch.take_along_dim(class_ids, indices, dim=1)
        sorted_boxes = torch.take_along_dim(bboxes, indices[..., None].expand(-1, -1, 4), dim=1)
        pred_boxes = torch.take_along_dim(raw_bboxes, indices[..., None].expand(-1, -1, 4), dim=1)
        logits = torch.take_along_dim(logits, indices[..., None].expand(-1, -1, logits.size(-1)), dim=1)
        sorted_mask = torch.take_along_dim(mask, indices, dim=1)

        pad_boxes = torch.where(sorted_mask[..., None], sorted_boxes, torch.zeros_like(sorted_boxes))
        pad_class_ids = torch.where(sorted_mask, sorted_class_ids, torch.zeros_like(sorted_class_ids))
        class_order = torch.tensor(self.config.class_order, dtype=torch.long, device=logits.device)
        pad_class_ids = class_order[pad_class_ids]

        order_logits = self.reading_order(
            boxes=pad_boxes,
            labels=pad_class_ids,
            mask=mask,
        )[:, :, : self.num_queries]

        result = PPDocLayoutV2ForObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            order_logits=order_logits,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_logits=outputs.intermediate_logits,
            intermediate_reference_points=outputs.intermediate_reference_points,
            intermediate_predicted_corners=outputs.intermediate_predicted_corners,
            initial_reference_points=outputs.initial_reference_points,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            init_reference_points=outputs.init_reference_points,
            enc_topk_logits=outputs.enc_topk_logits,
            enc_topk_bboxes=outputs.enc_topk_bboxes,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            denoising_meta_values=outputs.denoising_meta_values,
        )
        return result if use_return_dict else result.to_tuple()


class PPDocLayoutV2LayoutModel:
    def __init__(
        self,
        weight: str,
        device: Optional[str] = "cuda",
        imgsz: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        conf: float = 0.5,
    ):
        self.device = device
        self.conf = conf
        self.model_dir = weight
        self.preprocess_config = _load_preprocess_config(self.model_dir)
        size = self.preprocess_config.get("size", {})
        self.imgsz = (
            int(size.get("width", imgsz[0])),
            int(size.get("height", imgsz[1])),
        )
        self.rescale_factor = float(self.preprocess_config.get("rescale_factor", DEFAULT_RESCALE_FACTOR))
        self.config = PPDocLayoutV2Config.from_pretrained(self.model_dir)
        self.model = PPDocLayoutV2ForObjectDetection.from_pretrained(self.model_dir, config=self.config)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _get_order_seqs(order_logits: torch.Tensor) -> torch.Tensor:
        order_scores = torch.sigmoid(order_logits)
        batch_size, sequence_length, _ = order_scores.shape
        order_votes = order_scores.triu(diagonal=1).sum(dim=1) + (
            1.0 - order_scores.transpose(1, 2)
        ).tril(diagonal=-1).sum(dim=1)
        order_pointers = torch.argsort(order_votes, dim=1)
        order_seq = torch.empty_like(order_pointers)
        ranks = torch.arange(sequence_length, device=order_pointers.device, dtype=order_pointers.dtype).expand(
            batch_size, -1
        )
        order_seq.scatter_(1, order_pointers, ranks)
        return order_seq

    def _preprocess_single_image(self, image: Union[np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type for PP-DocLayoutV2: {type(image)}")

        pil_image = pil_image.convert("RGB")
        target_size = pil_image.size[1], pil_image.size[0]
        resized = pil_image.resize(self.imgsz, resample=Image.BICUBIC)
        pixel_values = np.asarray(resized, dtype=np.float32) * self.rescale_factor
        pixel_values = np.transpose(pixel_values, (2, 0, 1))
        return torch.from_numpy(pixel_values), target_size

    def _post_process_object_detection(
        self,
        outputs: PPDocLayoutV2ForObjectDetectionOutput,
        target_sizes: Sequence[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        boxes = outputs.pred_boxes
        logits = outputs.logits
        order_logits = outputs.order_logits
        order_seqs = self._get_order_seqs(order_logits)

        box_centers, box_dims = torch.split(boxes, 2, dim=-1)
        boxes = torch.cat([box_centers - 0.5 * box_dims, box_centers + 0.5 * box_dims], dim=-1)

        img_height, img_width = torch.as_tensor(target_sizes, device=boxes.device).unbind(1)
        scale_factor = torch.stack([img_width, img_height, img_width, img_height], dim=1).to(boxes.device)
        boxes = boxes * scale_factor[:, None, :]

        num_top_queries = logits.shape[1]
        num_classes = logits.shape[2]
        scores = torch.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
        labels = index % num_classes
        index = index // num_classes
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        order_seqs = order_seqs.gather(dim=1, index=index)

        results = []
        for score, label, box, order_seq in zip(scores, labels, boxes, order_seqs):
            keep = score >= self.conf
            order_seq = order_seq[keep]
            order_seq, indices = torch.sort(order_seq)
            results.append(
                {
                    "scores": score[keep][indices],
                    "labels": label[keep][indices],
                    "boxes": box[keep][indices],
                    "order_seq": order_seq,
                }
            )
        return results

    def _label_id_to_label_name(self, label_id: int) -> str:
        return str(self.config.id2label.get(label_id, self.config.id2label.get(str(label_id), label_id)))

    @staticmethod
    def _clip_bbox(box: Sequence[float], image_size: Tuple[int, int]) -> Optional[List[float]]:
        height, width = image_size
        xmin, ymin, xmax, ymax = [float(v) for v in box]
        xmin = max(0.0, min(float(width), xmin))
        xmax = max(0.0, min(float(width), xmax))
        ymin = max(0.0, min(float(height), ymin))
        ymax = max(0.0, min(float(height), ymax))
        if xmax <= xmin or ymax <= ymin:
            return None
        return [round(xmin, 4), round(ymin, 4), round(xmax, 4), round(ymax, 4)]

    def _parse_prediction(self, result: Dict[str, torch.Tensor], image_size: Tuple[int, int]) -> List[Dict]:
        layout_res = []
        for index, (score, label_id, box, _order_seq) in enumerate(
            zip(
                result["scores"],
                result["labels"],
                result["boxes"],
                result["order_seq"],
            ),
            start=1,
        ):
            bbox = self._clip_bbox(box.tolist(), image_size)
            if bbox is None:
                continue

            cls_id = int(label_id.item())
            layout_res.append(
                {
                    "cls_id": cls_id,
                    "label": self._label_id_to_label_name(cls_id),
                    "score": round(float(score.item()), 4),
                    "bbox": bbox,
                    "index": index,
                }
            )
        return layout_res

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        return self.batch_predict([image], batch_size=1)[0]

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 1,
    ) -> List[List[Dict]]:
        if len(images) == 0:
            return []

        results: List[List[Dict]] = []
        with torch.no_grad():
            with tqdm(total=len(images), desc="Layout Predict") as pbar:
                for start in range(0, len(images), batch_size):
                    batch_images = images[start : start + batch_size]
                    pixel_values_list = []
                    target_sizes = []
                    for image in batch_images:
                        pixel_values, target_size = self._preprocess_single_image(image)
                        pixel_values_list.append(pixel_values)
                        target_sizes.append(target_size)

                    batch_tensor = torch.stack(pixel_values_list, dim=0).to(self.device)
                    outputs = self.model(pixel_values=batch_tensor)
                    predictions = self._post_process_object_detection(outputs, target_sizes)
                    for prediction, image_size in zip(predictions, target_sizes):
                        results.append(self._parse_prediction(prediction, image_size))
                    pbar.update(len(batch_images))
        return results

    def visualize(
        self,
        image: Union[np.ndarray, Image.Image],
        results: List[Dict],
    ) -> Image.Image:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for res in sorted(results, key=lambda item: (item.get("index", 10**9), item.get("bbox", [0, 0, 0, 0])[1])):
            xmin, ymin, xmax, ymax = res["bbox"]
            color = _label_to_color(res["label"])
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

            text = f"{res['index']}: {res['label']} {res['score']:.2f}"
            text_top = int(round(ymin))
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            pad = 3

            if text_top - text_height - pad * 2 >= 0:
                text_bg_top = text_top - text_height - pad * 2
            else:
                text_bg_top = text_top
            text_bg_bottom = text_bg_top + text_height + pad * 2
            text_bg_right = int(round(xmax))
            text_bg_left = text_bg_right - text_width - pad * 2

            draw.rectangle(
                [text_bg_left, text_bg_top, text_bg_right, text_bg_bottom],
                fill=color,
            )
            draw.text(
                (text_bg_left + pad, text_bg_top + pad),
                text,
                fill="white",
                font=font,
            )
        return image


__all__ = [
    "DEFAULT_PP_DOCLAYOUT_V2_REPO",
    "PPDocLayoutV2Config",
    "PPDocLayoutV2ForObjectDetection",
    "PPDocLayoutV2LayoutModel",
    "PPDocLayoutV2ReadingOrder",
    "PPDocLayoutV2ReadingOrderConfig",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP-DocLayoutV2 local inference smoke test")
    parser.add_argument("image", nargs="?", help="Path to an input image. If omitted, only model loading is tested.")
    parser.add_argument("--model", default=DEFAULT_PP_DOCLAYOUT_V2_REPO, help="HF repo id or local model directory.")
    parser.add_argument("--device", default=None, help="Runtime device, e.g. cpu/mps/cuda.")
    parser.add_argument("--output", default=None, help="Optional path to save the visualization image.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the visualization window.")
    args = parser.parse_args()

    model = PPDocLayoutV2LayoutModel(weight=args.model, device=args.device)
    print(f"model loaded on {model.device}")

    args.image = "/Users/myhloli/pdf/png/academic_paper_img_formula.png"
    
    if args.image:
        with Image.open(args.image) as img:
            results = model.predict(img)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            vis_img = model.visualize(img, results)
            if args.output:
                vis_img.save(args.output)
            if not args.no_show:
                vis_img.show()
