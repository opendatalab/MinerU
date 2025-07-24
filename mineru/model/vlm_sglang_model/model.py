import math
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sglang.version import __version__ as sglang_version
from packaging import version
if version.parse(sglang_version) >= version.parse("0.4.9"):
    # sglang >= 0.4.9
    from sglang.srt.multimodal.mm_utils import (
            get_anyres_image_grid_shape,
        )
else:
    # 0.4.7 <= sglang < 0.4.9
    from sglang.srt.mm_utils import (
        get_anyres_image_grid_shape,
    )

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from ..vlm_hf_model.configuration_mineru2 import Mineru2QwenConfig
from ..vlm_hf_model.modeling_mineru2 import build_vision_projector
from ...utils.models_download_utils import auto_download_and_get_model_root_path


def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [item for sublist in nested_list for item in flatten_nested_list(sublist)]
    else:
        return [nested_list]


def downgrade_modality(modality):
    modality_str = str(modality)
    if "MULTI_IMAGES" in modality_str:
        return "multi-images"
    if "IMAGE" in modality_str:
        return "image"
    if "VIDEO" in modality_str:
        return "video"
    if "AUDIO" in modality_str:
        return "audio"
    raise ValueError(f"Unexpected modality: {modality_str}")


class Mineru2QwenForCausalLM(nn.Module):
    def __init__(
        self,
        config: Mineru2QwenConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        if getattr(self.config, "projector_hidden_act", None) is None:
            self.config.projector_hidden_act = "gelu"
        if getattr(self.config, "image_token_index", None) is None:
            self.config.image_token_index = 151646

        # load vision tower
        mm_vision_tower = self.config.mm_vision_tower
        model_root_path = auto_download_and_get_model_root_path(mm_vision_tower, "vlm")
        mm_vision_tower = f"{model_root_path}/{mm_vision_tower}"

        if "clip" in mm_vision_tower:
            vision_config = CLIPVisionConfig.from_pretrained(mm_vision_tower)
            self.vision_tower = CLIPVisionModel(vision_config)  # type: ignore
        elif "siglip" in mm_vision_tower:
            vision_config = SiglipVisionConfig.from_pretrained(mm_vision_tower)
            self.vision_tower = SiglipVisionModel(vision_config)  # type: ignore
            # Siglip needs all feature tokens
            self.config.mm_vision_select_feature = "full"
        else:
            raise ValueError(f"Unexpected mm_vision_tower: {mm_vision_tower}")

        ### EDIT: change projector
        # the name `projector` contains `proj` which is often used in attention layers, which can cause bugs in quantization.
        self.multi_modal_mlp = build_vision_projector(config)

        self.language_model = Qwen2ForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(torch.empty(config.hidden_size))

        language_model_device = next(self.language_model.parameters()).device
        self.vision_tower = self.vision_tower.to(language_model_device)
        self.vision_tower.eval()

        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None)

        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)
        if self.vision_feature_select_strategy in ("patch", "full"):
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

    def pad_input_ids(self, input_ids: List[int], image_inputs):

        image_sizes = flatten_nested_list([item.image_sizes for item in image_inputs.mm_items])
        pad_values = [item.pad_value for item in image_inputs.mm_items]

        # hardcode for spatial_unpad + anyres
        # if image_inputs.modalities is not None and (
        #     "multi-images" in image_inputs.modalities or "video" in image_inputs.modalities
        # ):
        #     image_aspect_ratio = "pad"
        # else:
        #     image_aspect_ratio = "anyres"

        offset_list = []
        image_inputs.image_pad_len = []
        for image_idx, image_s in enumerate(image_sizes):
            if len(image_sizes) > 16:
                # 2x2 pooling with stride 2
                new_image_feature_len = math.ceil(self.image_size / self.patch_size / 2) ** 2
            else:
                new_image_feature_len = self.image_feature_len  # multiimage

            height = width = self.num_patches_per_side
            if "anyres" in self.config.image_aspect_ratio:
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_s,
                    self.image_grid_pinpoints,
                    self.vision_tower.config.image_size,
                )
                h = num_patch_height * height
                w = num_patch_width * width

                ### EDIT: remove `unpad_image_shape`
                # new_h, new_w = unpad_image_shape(h, w, image_s)
                new_h, new_w = h, w

                if "anyres_max" in self.config.image_aspect_ratio:
                    matched_anyres_max_num_patches = re.match(r".*anyres_max_(\d+)", self.config.image_aspect_ratio)
                    if matched_anyres_max_num_patches:
                        max_num_patches = int(matched_anyres_max_num_patches.group(1))
                        times = math.sqrt(new_h * new_w / (max_num_patches * self.image_feature_len))
                        if times > 1.1:
                            new_h = int(new_h // times)
                            new_w = int(new_w // times)
                new_image_feature_len += new_h * (new_w + 1)

            try:
                offset = input_ids.index(self.config.image_token_index)
            except ValueError:
                offset = 0
            # old_len + pad_len - 1, because we need to remove image_token_id
            input_ids = input_ids[:offset] + [pad_values[image_idx]] * new_image_feature_len + input_ids[offset + 1 :]
            offset_list.append(offset)
            image_inputs.image_pad_len.append(new_image_feature_len)

        image_inputs.image_offsets = offset_list
        return input_ids

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype)
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # NOTE: This is not memory efficient. (output_hidden_states=True) will save all the hidden stated.

        selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy in ["default", "patch"]:
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.vision_feature_select_strategy}")

        image_features = self.multi_modal_mlp(selected_image_feature)
        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        image_inputs = forward_batch.mm_inputs

        if image_inputs is None:
            image_inputs = []

        if forward_batch.forward_mode.is_extend():
            # Clamp input ids. This is because the input_ids for the image tokens are
            # filled with the hash values of the image for the prefix matching in the radix attention.
            # There values are useless because their embeddings will be replaced by vision embeddings anyway.
            input_ids.clamp_(min=0, max=self.config.vocab_size - 1)

            # Embed text inputs
            input_embeds = self.language_model.model.embed_tokens(input_ids)

            # Got List[List[str]] extend it to List[str]
            # The length of the List should be equal to batch size
            modalities_list = []
            max_image_offset = []
            for im in image_inputs:
                if im:
                    modalities_list.extend([downgrade_modality(item.modality) for item in im.mm_items])
                if im and im.image_offsets:
                    max_image_offset.append(np.max(np.array(im.image_offsets) + np.array(im.image_pad_len)))
                else:
                    max_image_offset.append(-1)

            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy()
            need_vision = start_positions <= np.array(max_image_offset)

            if need_vision.any():
                bs = forward_batch.batch_size

                if version.parse(sglang_version) >= version.parse("0.4.9.post3"):
                    # sglang >= 0.4.9.post3
                    pixel_values = flatten_nested_list(
                        [[item.feature for item in image_inputs[i].mm_items] for i in range(bs) if need_vision[i]]
                    )  # image_inputs[batch_idx].mm_items[item_idx].pixel_values is Tensor
                    image_sizes = [
                        flatten_nested_list([item.model_specific_data["image_sizes"] for item in image_inputs[i].mm_items])
                        for i in range(bs)
                        if need_vision[i]
                    ]  # image_inputs[batch_idx].mm_items[item_idx].image_sizes should be tuple, but is list of tuple for now.
                else:
                    # 0.4.7 <= sglang <= 0.4.9.post2
                    pixel_values = flatten_nested_list(
                        [[item.pixel_values for item in image_inputs[i].mm_items] for i in range(bs) if need_vision[i]]
                    )  # image_inputs[batch_idx].mm_items[item_idx].pixel_values is Tensor
                    image_sizes = [
                        flatten_nested_list([item.image_sizes for item in image_inputs[i].mm_items])
                        for i in range(bs)
                        if need_vision[i]
                    ]  # image_inputs[batch_idx].mm_items[item_idx].image_sizes should be tuple, but is list of tuple for now.

                ########## Encode Image ########

                if pixel_values[0].ndim == 4:
                    # llava-hd: BS, num_patch, C=3, H=336, W=336, num_patch obtained from process_images
                    np.concatenate(pixel_values, axis=0)
                    # ndim=4
                    concat_images = torch.tensor(
                        np.concatenate(pixel_values, axis=0),
                        device=self.vision_tower.device,
                    )
                    image_features = self.encode_images(concat_images)
                    split_sizes = [image.shape[0] for image in pixel_values]
                    image_features = torch.split(image_features, split_sizes, dim=0)
                    # hd image_features: BS, num_patch, 576, 4096
                else:
                    # normal pixel: BS, C=3, H=336, W=336
                    pixel_values = torch.tensor(np.array(pixel_values), device=self.vision_tower.device)
                    image_features = self.encode_images(pixel_values)
                    # image_features: BS, 576, 4096

                if self.mm_patch_merge_type.startswith("spatial"):
                    new_image_features = []
                    height = width = self.num_patches_per_side
                    for image_idx, image_feature in enumerate(image_features):
                        if modalities_list[image_idx] == "image":
                            image_aspect_ratio = self.config.image_aspect_ratio  # single image
                        elif modalities_list[image_idx] == "multi-images" or modalities_list[image_idx] == "video":
                            image_aspect_ratio = "pad"  # multi image
                        # image_aspect_ratio = (
                        #     "anyres" if len(image_sizes[image_idx]) == 1 else "pad"
                        # )
                        if (
                            image_feature.shape[0] > 1
                            and "anyres" in image_aspect_ratio
                            and modalities_list[image_idx] == "image"
                        ):
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            assert height * width == base_image_feature.shape[0]

                            if "anyres_max" in image_aspect_ratio:
                                matched_anyres_max_num_patches = re.match(r".*anyres_max_(\d+)", image_aspect_ratio)
                                if matched_anyres_max_num_patches:
                                    max_num_patches = int(matched_anyres_max_num_patches.group(1))

                            if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                                vision_tower_image_size = self.image_size
                                try:
                                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                        image_sizes[image_idx][0],
                                        self.config.image_grid_pinpoints,
                                        vision_tower_image_size,
                                    )
                                except Exception as e:
                                    print(f"Error: {e}")
                                    num_patch_width, num_patch_height = 2, 2
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                image_feature = image_feature.view(2, 2, height, width, -1)

                            if "unpad" in self.mm_patch_merge_type:
                                unit = image_feature.shape[2]
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)

                                ### EDIT: remove `unpad_image`
                                # image_feature = unpad_image(image_feature, image_sizes[image_idx][0])

                                if "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                                    c, h, w = image_feature.shape
                                    times = math.sqrt(h * w / (max_num_patches * unit**2))
                                    if times > 1.1:
                                        image_feature = image_feature[None]
                                        image_feature = nn.functional.interpolate(
                                            image_feature,
                                            [int(h // times), int(w // times)],
                                            mode="bilinear",
                                        )[0]
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.language_model.model.image_newline[:, None, None].expand(
                                            *image_feature.shape[:-1], 1
                                        ),
                                    ),
                                    dim=-1,
                                )
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            image_feature = image_feature.unsqueeze(0)
                        else:
                            if modalities_list[image_idx] == "video":  # video
                                # 2x2 pooling
                                num_of_frames = image_feature.shape[0]
                                image_feature = image_feature.view(num_of_frames, height, width, -1)
                                image_feature = image_feature.permute(0, 3, 1, 2).contiguous()  # N, C, H, W
                                height, weight = image_feature.shape[2:]
                                scaled_shape = [
                                    math.ceil(height / 2),
                                    math.ceil(weight / 2),
                                ]
                                image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode="bilinear")
                                image_feature = image_feature.flatten(2).transpose(1, 2).contiguous()  # N, C, H*W
                            if "unpad" in self.mm_patch_merge_type:
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        # Expand to (bs, 1, hidden_dim) and concat at the end of the image tokens
                                        self.language_model.model.image_newline[None, None].expand(
                                            image_feature.shape[0],
                                            1,
                                            image_feature.shape[-1],
                                        ),
                                    ),
                                    dim=1,
                                )

                        new_image_features.append(image_feature)
                    image_features = new_image_features

                # Fill in the placeholder for the image
                extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                extend_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
                prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
                pt = 0
                for i in range(bs):
                    if not need_vision[i]:
                        continue

                    start_idx = extend_start_loc_cpu[i]
                    seq_len = extend_seq_lens[i]
                    prefix_len = prefix_lens_cpu[i]

                    # Multiple images
                    for image_idx, image_offset in enumerate(image_inputs[i].image_offsets):
                        if image_offset + image_inputs[i].image_pad_len[image_idx] <= prefix_len:
                            continue
                        if image_offset >= prefix_len + seq_len:
                            break

                        tmp_image_feature = image_features[pt][image_idx]
                        pad_len = tmp_image_feature.shape[0]

                        input_offset = image_offset - prefix_len
                        left_idx = start_idx + input_offset
                        right_idx = left_idx + pad_len
                        assert right_idx > start_idx
                        if input_offset < 0:
                            left_idx = start_idx
                            tmp_image_feature = tmp_image_feature[-input_offset:]
                        if right_idx > start_idx + seq_len:
                            tmp_image_feature = tmp_image_feature[: start_idx + seq_len - right_idx]
                            right_idx = start_idx + seq_len
                        try:
                            input_embeds[left_idx:right_idx] = tmp_image_feature
                        except RuntimeError as e:
                            print(f"RuntimeError in image encoding: {e}")
                            print(f"{input_embeds.shape=}, {tmp_image_feature.shape=}")
                            print(f"{start_idx=}, {image_offset=}, {prefix_len=}, {pad_len=}")
                    pt += 1

            return self.language_model(input_ids, positions, forward_batch, input_embeds=input_embeds)
        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)
        else:
            raise ValueError(f"Unexpected forward mode: {forward_batch.forward_mode}")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        projector_weights = {
            "model.mm_projector": "multi_modal_mlp",
            "model.vision_tower.vision_tower": "vision_tower",
            # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
            "model.image_newline": "language_model.model.image_newline",
        }
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "projector" in name or "vision_tower" in name or "image_newline" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                self.language_model.load_weights([(name, loaded_weight)])

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size


EntryClass = [Mineru2QwenForCausalLM]
