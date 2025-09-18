# Copyright (c) OpenMMLab. All rights reserved.
"""Mineru vision model."""
import ast
import math
import re
import os
from functools import partial, reduce
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb, normalize, rescale, resize, to_channel_dimension_format
from transformers.image_utils import ChannelDimension, PILImageResampling, to_numpy_array
from transformers.utils import TensorType

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel

from mineru.model.vlm_lmdeploy_model.utils import select_best_resolution


def divide_to_patches(image, patch_size):
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if pil_img.mode == 'L':
        pil_img = pil_img.convert('RGB')
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_anyres_image(image, processor, grid_pinpoints):
    if isinstance(grid_pinpoints, str) and 'x' in grid_pinpoints:
        patch_size = processor.crop_size['height']
        assert patch_size in [
            224, 336, 384, 448, 512
        ], 'patch_size should be in [224, 336, 384, 448, 512]'
        matches = re.findall(r'\((\d+)x(\d+)\)', grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [(i, j)
                          for i in range(range_start[0], range_end[0] + 1)
                          for j in range(range_start[1], range_end[1] + 1)]
        grid_pinpoints = [[dim * patch_size for dim in pair]
                          for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)  # type: ignore
    best_resolution = select_best_resolution(image.size, possible_resolutions)

    image_padded = image.resize(best_resolution)
    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize(
        (processor.crop_size['height'], processor.crop_size['height']))
    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch,
                             return_tensors='pt')['pixel_values'][0]
        for image_patch in image_patches
    ]
    return torch.stack(image_patches, dim=0)


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, 'image_aspect_ratio', '')
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == 'anyres' or 'anyres_max' in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(image, image_processor,
                                         model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


class Mineru2ImageProcessor(BaseImageProcessor):
    model_input_names = ['pixel_values']

    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Optional[Dict[str, int]] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[list] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        crop_size = crop_size if crop_size is not None else {
            'height': 384,
            'width': 384
        }
        crop_size = get_size_dict(crop_size,
                                  default_to_square=True,
                                  param_name='crop_size')

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints
        self.in_e2e_processing = False

    def _preprocess(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize,
                    size=self.size,
                    resample=self.resample,
                    data_format=self.data_format),
            partial(rescale,
                    scale=self.rescale_factor,
                    data_format=self.data_format),
            partial(normalize,
                    mean=self.image_mean,
                    std=self.image_std,
                    data_format=self.data_format),
            partial(to_channel_dimension_format,
                    channel_dim=self.data_format,
                    input_channel_dim=self.data_format),
        ]
        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        return {'pixel_values': images}

    def _preprocess_end_to_end(self, images):
        image_aspect_ratio = self.image_aspect_ratio
        image_grid_pinpoints = self.image_grid_pinpoints
        assert image_aspect_ratio is not None
        assert image_grid_pinpoints is not None

        pixel_values = []
        if image_aspect_ratio == 'pad':
            for image in images:
                image = expand2square(
                    image, tuple(int(x * 255) for x in self.image_mean))
                image = self._preprocess(image)['pixel_values'][0]
                pixel_values.append(image)
        elif image_aspect_ratio == 'anyres' or 'anyres_max' in image_aspect_ratio:
            for image in images:
                image = process_anyres_image(image, self,
                                             self.image_grid_pinpoints)
                pixel_values.append(image.numpy())
        else:
            pixel_values = self._preprocess(images)['pixel_values']

        if isinstance(pixel_values, list) and all(
                x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = np.stack(pixel_values, axis=0)

        # CAUTION: here used (height, width).
        image_sizes = [(image.height, image.width) for image in images]
        assert len(pixel_values) == len(image_sizes)
        return {'pixel_values': pixel_values, 'image_sizes': image_sizes}

    def preprocess(
        self,
        images,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        if self.image_aspect_ratio is None or self.in_e2e_processing:
            data = self._preprocess(images)
        else:
            assert self.image_grid_pinpoints is not None
            self.in_e2e_processing = True
            try:
                data = self._preprocess_end_to_end(images)
            finally:
                self.in_e2e_processing = False
        return BatchFeature(data=data, tensor_type=return_tensors)


@VISION_MODELS.register_module()
class MinerUModel(VisonModel):
    """Mineru vision model."""

    _arch = 'Mineru2QwenForCausalLM'

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = ''):
        super().__init__(
            model_path,
            with_llm,
            max_memory,
            hf_config,
            backend,
        )
        self.config = hf_config
        self.hf_config = hf_config

    def build_preprocessor(self):
        self.mm_vistion_tower_path = self.config.mm_vision_tower if os.path.isabs(
            self.config.mm_vision_tower) else os.path.join(
                self.model_path, self.config.mm_vision_tower)
        self.mm_vistion_tower_config = AutoConfig.from_pretrained(
            self.mm_vistion_tower_path)

        self.image_processor = Mineru2ImageProcessor()
        image_size = self.mm_vistion_tower_config.vision_config.image_size
        patch_size = self.mm_vistion_tower_config.vision_config.patch_size

        self.n_token_per_image = (image_size // patch_size)**2
        if self.hf_config.mm_vision_select_feature == 'cls_patch':
            self.n_token_per_image += 1

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refer to `super().preprocess() for spec."""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            pixel_values = process_images([image], self.image_processor,
                                          self.config)
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_size=image.size,
                     image_tokens=self.n_token_per_image,
                     image_token_id=self.image_token_id))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            n_images = len(
                [1 for x in message['content'] if x['type'] == 'image'])
            content = [
                x.get('text', '') for x in message['content']
                if x['type'] == 'text'
            ]
            prompt = content[0]
            if IMAGE_TOKEN in prompt:
                pass
            else:
                prompt = ''.join([f'<image>\n'
                                  for i in range(n_images)]) + prompt
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch_aux(self, messages, prompt, IMAGE_TOKEN, tokenizer,
                       sequence_start):
        """auxiliary function to pack the preprocessing results in a format
        compatible with what is required by pytorch engine.

        Args:
            messages(List[Dict]): the output of `preprocess`
            prompt(str): the prompt after applying chat template
            IMAGE_TOKEN(str): a placeholder where image tokens will be
                inserted
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
        """
        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        # split prompt into segments and validate data
        segs = prompt.split(IMAGE_TOKEN)

        # calculate the image token offset for each image
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                preps[i - 1].update(offset=len(input_ids))
                image_tokens = preps[i - 1]['image_tokens']
                assert self.image_token_id == preps[i - 1]['image_token_id']
                input_ids.extend([self.image_token_id] * image_tokens)
            token_ids = tokenizer.encode(seg,
                                         add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)
        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start,
                   **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer,
                                   sequence_start)
