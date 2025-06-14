import ast
import math
import re
from functools import partial, reduce
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.utils import TensorType


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    original_width, original_height = original_size
    best_fit = (0, 0)
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


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
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [
            (i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)
        ]
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)  # type: ignore
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


# This functions is not used.
def resize_and_pad_image(image, target_resolution):
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


# DIFFERENT from sglang.srt.mm_utils.process_anyres_image
def process_anyres_image(image, processor, grid_pinpoints):
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        patch_size = processor.crop_size["height"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [
            (i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)
        ]
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)  # type: ignore
    best_resolution = select_best_resolution(image.size, possible_resolutions)

    # image_padded = resize_and_pad_image(image, best_resolution)
    image_padded = image.resize(best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    image_original_resize = image.resize((processor.crop_size["height"], processor.crop_size["height"]))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", "")
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


class Mineru2ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

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

        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

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
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        return {"pixel_values": images}

    def _preprocess_end_to_end(self, images):
        image_aspect_ratio = self.image_aspect_ratio
        image_grid_pinpoints = self.image_grid_pinpoints
        assert image_aspect_ratio is not None
        assert image_grid_pinpoints is not None

        pixel_values = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = expand2square(image, tuple(int(x * 255) for x in self.image_mean))
                image = self._preprocess(image)["pixel_values"][0]
                pixel_values.append(image)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            for image in images:
                image = process_anyres_image(image, self, self.image_grid_pinpoints)
                pixel_values.append(image.numpy())
        else:
            pixel_values = self._preprocess(images)["pixel_values"]

        if isinstance(pixel_values, list) and all(x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = np.stack(pixel_values, axis=0)

        # CAUTION: here used (height, width).
        image_sizes = [(image.height, image.width) for image in images]
        assert len(pixel_values) == len(image_sizes)

        return {"pixel_values": pixel_values, "image_sizes": image_sizes}

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
