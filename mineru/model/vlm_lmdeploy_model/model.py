# Copyright (c) OpenMMLab. All rights reserved.
import math
import re
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from mineru.model.vlm_lmdeploy_model.utils import get_anyres_image_grid_shape


class SiglipVisionTower(nn.Module):

    def __init__(self,
                 vision_tower,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = SiglipVisionConfig.from_pretrained(vision_tower)
        assert isinstance(self.config, SiglipVisionConfig)
        self.config.num_hidden_layers -= 1  # drop the last hidden layer
        self.config.vision_use_head = False
        self.vision_tower = SiglipVisionModel(config=self.config)
        self.vision_tower.requires_grad_(False)

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device,
                             dtype=torch.bfloat16).unsqueeze(0),
                    output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[-1].to(
                    image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(
                device=self.device, dtype=torch.bfloat16),
                                                   output_hidden_states=True)
            image_features = image_forward_outs.hidden_states[-1].to(
                images.dtype)

        return image_features.to(torch.bfloat16)

    @property
    def dummy_feature(self):
        return torch.zeros(1,
                           self.hidden_size,
                           device=self.device,
                           dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size)**2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


def build_vision_tower(config: PretrainedConfig,
                       ctx_mgr: StepContextManager,
                       dtype: torch.dtype = None,
                       device: torch.device = None):
    vision_tower = getattr(config, 'mm_vision_tower',
                           getattr(config, 'vision_tower', ''))
    model_path = getattr(config, "_name_or_path", "")
    if 'siglip' in vision_tower.lower():
        vistion_tower_path = vision_tower if os.path.isabs(
            vision_tower) else os.path.join(model_path, vision_tower)
        return SiglipVisionTower(vistion_tower_path,
                                 ctx_mgr,
                                 dtype=torch.bfloat16,
                                 device=device)
    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_vision_projector(config: PretrainedConfig):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())  # type: ignore
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return nn.Identity()

    raise ValueError(f'Unknown projector type: {projector_type}')


class Mineru2QwenModel(Qwen2Model):
    config_class = PretrainedConfig

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config, dtype, device)
        self.vision_tower = build_vision_tower(config, ctx_mgr, dtype, device)
        self.mm_projector = build_vision_projector(config)

        if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))


class Mineru2QwenForCausalLM(Qwen2ForCausalLM):
    config_class = PretrainedConfig

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config, ctx_mgr, dtype=dtype, device=device)
        self.dtype = dtype
        self.device = device
        self.ignore_index = config.ignore_index
        self.image_token_index = config.image_token_index

        self.model = Mineru2QwenModel(config, ctx_mgr, dtype, device)
        self.input_processor = Mineru2VLInputProcessor(self.config)
        self.input_processor.prepare_shape = self.prepare_shape
        self.input_processor.prepare_inputs_labels_for_multimodal = self.prepare_inputs_labels_for_multimodal

    def get_input_processor(self) -> BaseModelInputProcessor:
        """get input processor."""
        return self.input_processor

    def get_model(self):
        return self.model

    def encode_images(self, images: torch.Tensor):
        image_features = self.get_model().vision_tower(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_shape(self, images, image_sizes):
        if type(images) is list or images.ndim == 5:
            image_features = torch.empty((images.shape[1], 729, 2),
                                         dtype=images.dtype,
                                         device="npu")
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type',
                                          'flat')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_model(
                        ).vision_tower.num_patches_per_side
                        max_num_patches = 9
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                            image_sizes[image_idx],
                            self.config.image_grid_pinpoints,
                            self.get_model().vision_tower.config.image_size,
                        )
                        image_feature = image_feature.view(
                            -1, num_patch_height * height,
                            num_patch_width * height)

                        unit = height
                        c, h, w = image_feature.shape

                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            return int(h // times) * (int(
                                w // times) + 1) + base_image_feature.shape[0]
                        return image_feature.shape[1] * (
                            image_feature.shape[2] +
                            1) + base_image_feature.shape[0]
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature,
                                 self.model.image_newline[None].to(
                                     image_feature.device)),
                                dim=0)
                        return image_feature.shape[0]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def prepare_inputs_labels_for_multimodal(self, input_ids, images,
                                             image_sizes):
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type',
                                          'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio',
                                         'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_model(
                        ).vision_tower.num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if 'anyres_max' in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(
                                r'square_anyres_max_(\d+)', image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(
                                    matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == 'anyres' or 'anyres_max' in image_aspect_ratio:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx],
                                self.config.image_grid_pinpoints,
                                self.get_model().vision_tower.config.
                                image_size,
                            )
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height,
                                width, -1)
                        else:
                            raise NotImplementedError
                        if ('unpad' in mm_patch_merge_type
                                and 'anyres_max' in image_aspect_ratio
                                and matched_anyres_max_num_patches):
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1,
                                                                  2).flatten(
                                                                      2, 3)
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w /
                                              (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                origin_type = image_feature.dtype
                                image_feature = image_feature.to(torch.float16)
                                image_feature = nn.functional.interpolate(
                                    image_feature,
                                    [int(h // times),
                                     int(w // times)],
                                    mode='bilinear')[0]
                                image_feature = image_feature.to(origin_type)
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None].
                                    expand(*image_feature.shape[:-1], 1).to(
                                        image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1,
                                                                  2).transpose(
                                                                      0, 1)
                        elif 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1,
                                                                  2).flatten(
                                                                      2, 3)
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None].
                                    expand(*image_feature.shape[:-1], 1).to(
                                        image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1,
                                                                  2).transpose(
                                                                      0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature,
                                 self.model.image_newline[None].to(
                                     image_feature.device)),
                                dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f'Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}'
                )
        else:
            image_features = self.encode_images(images)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.full_like(input_ids, self.ignore_index)

        # remove the padding using attention_mask -- FIXME
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(
                input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.image_token_index).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                cur_image_idx += 1
                continue

            image_token_indices = ([-1] + torch.where(
                cur_input_ids == self.image_token_index)[0].tolist() +
                                   [cur_input_ids.shape[0]])
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[image_token_indices[i] +
                                  1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] +
                                                  1:image_token_indices[i +
                                                                        1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds,
                                                 split_sizes,
                                                 dim=0)
            cur_new_input_embeds = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)

            cur_new_input_embeds = [
                x.to(self.device) for x in cur_new_input_embeds
            ]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)

            new_input_embeds.append(cur_new_input_embeds)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config,
                                             'tokenizer_model_max_length',
                                             None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
        return new_input_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        ret = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return ret

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # vision inputs
        if context.input_multimodals is not None:
            real_input_ids = []
            pixel_values = []
            image_sizes = []
            decode_inputs_embeds = []
            # try:
            assert inputs_embeds is None
            cur_pos = 0
            for bs, input_multimodal in enumerate(context.input_multimodals):
                if input_multimodal:
                    cur_pos += context.q_seqlens[bs]
                    real_input_ids.append(
                        input_multimodal['image'][1].data.unsqueeze(0))
                    pixel_values.append(input_multimodal['image'][0].data[0])
                    image_sizes.append(
                        input_multimodal['image'][0].meta['image_size'])
                else:
                    temp_embeds = self.model.embed_tokens(input_ids[:,
                                                                    cur_pos])
                    cur_pos += 1
                    decode_inputs_embeds.append(temp_embeds)

            real_input_ids = torch.cat(real_input_ids)

            prefill_inputs_embeds = self.prepare_inputs_labels_for_multimodal(
                real_input_ids, pixel_values, image_sizes)

            inputs_embeds = prefill_inputs_embeds + decode_inputs_embeds
            inputs_embeds = torch.cat(inputs_embeds).unsqueeze(0)
            inputs_embeds = inputs_embeds.to(self.dtype)

        else:
            pass
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                #if weight_name not in name:
                if weight_name not in name or "vision_model" in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)


class Mineru2VLInputProcessor(BaseModelInputProcessor):
    """mineru2 input processor."""

    def __init__(self, config: PretrainedConfig) -> None:
        self.config = config

    @torch.no_grad()
    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals
        assert len(input_multimodals) == 1

        ori_input_len = len(input_ids)

        input_imgs = []
        fake_input_ids = None
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.config.torch_dtype)
            image_token_id = input_mm['image_token_id']
            image_size = input_mm['image_size']
            offset = 10
            num_pad = 2
            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_pad,
                                       meta=dict(image_token_id=image_token_id,
                                                 image_size=image_size))
            input_imgs.append(mm_data)

        input_multimodals = dict(image=input_imgs)

        input_ids = torch.tensor(input_ids)
        pixel_values = [
            input_mm.data for input_mm in input_multimodals['image']
        ]
        image_sizes = [
            input_mm.meta['image_size']
            for input_mm in input_multimodals['image']
        ]
        if len(pixel_values) > 0:
            pixel_values = torch.cat([data for data in pixel_values])

        fake_input_len = self.prepare_shape(pixel_values,
                                            image_sizes) + ori_input_len - 1

        fake_input_ids = [1 for i in range(0, fake_input_len)]

        mm_data = MultiModalTensor(data=input_ids,
                                   start=offset,
                                   end=offset + num_pad,
                                   meta=dict(image_token_id=image_token_id,
                                             image_size=image_size))
        input_multimodals['image'].append(mm_data)

        # input_multimodals['image'][0].input_ids = input_ids
        if fake_input_ids is not None:
            input_ids = fake_input_ids
        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=input_multimodals,
        )
        return result
