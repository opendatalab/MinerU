import ast
import asyncio
import re
from typing import List, Optional, Union

import numpy as np

try:
    # sglang==0.4.5.post3
    from sglang.srt.managers.multimodal_processors.base_processor import (
        BaseMultimodalProcessor as BaseProcessor,
    )

    get_global_processor = None
except ImportError:
    # sglang==0.4.4.post1
    from sglang.srt.managers.image_processors.base_image_processor import (
        BaseImageProcessor as BaseProcessor,
        get_global_processor,
    )

from sglang.srt.mm_utils import divide_to_patches, expand2square, select_best_resolution
from sglang.srt.utils import load_image, logger
from sglang.utils import get_exception_traceback

from .model import Mineru2QwenForCausalLM


# image_best_res is only resized (not padded).
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
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)

    image_best_res = image.resize(best_resolution)  # <<<<<<< Here changed
    patches = divide_to_patches(image_best_res, processor.crop_size["height"])
    image_original_resize = image.resize((processor.crop_size["height"], processor.crop_size["height"]))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch)["pixel_values"][0] for image_patch in image_patches]
    return np.stack(image_patches, axis=0)


class Mineru2ImageProcessor(BaseProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes],
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[str] = None,
        image_processor=None,
    ):
        if image_processor is None:
            assert get_global_processor is not None
            image_processor = get_global_processor().image_processor

        try:
            image, image_size = load_image(image_data)
            if image_size is not None:
                # It is a video with multiple images
                image_hash = hash(image_data)
                pixel_values = image_processor(image)["pixel_values"]
                pixel_values = np.stack(pixel_values, axis=0)
                return pixel_values, image_hash, image_size
            else:
                # It is an image
                image_hash = hash(image_data)
                if image_aspect_ratio == "pad":
                    image = expand2square(
                        image,
                        tuple(int(x * 255) for x in image_processor.image_mean),
                    )
                    pixel_values = image_processor(image.convert("RGB"))["pixel_values"][0]
                elif image_aspect_ratio == "anyres" or (image_aspect_ratio is not None and "anyres_max" in image_aspect_ratio):
                    pixel_values = process_anyres_image(image, image_processor, image_grid_pinpoints)
                else:
                    pixel_values = image_processor(image)["pixel_values"][0]
                return pixel_values, image_hash, image.size
        except Exception:
            logger.error("Exception in TokenizerManager:\n" + get_exception_traceback())

    async def _process_single_image(self, image_data: Union[bytes, str], aspect_ratio: str, grid_pinpoints: str):
        if hasattr(self, "cpu_executor"):
            executor = self.cpu_executor
        else:
            executor = self.executor

        if get_global_processor is not None:
            image_processor = None  # save ipc cost
        else:
            image_processor = self._processor.image_processor

        if executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                executor,
                Mineru2ImageProcessor._process_single_image_task,
                image_data,
                aspect_ratio,
                grid_pinpoints,
                image_processor,
            )
        else:
            return self._process_single_image_task(
                image_data,
                aspect_ratio,
                grid_pinpoints,
                image_processor,
            )

    # sglang==0.4.4.post1
    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None

        modalities = request_obj.modalities or ["image"]
        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", "")

        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints
            if hasattr(self.hf_config, "image_grid_pinpoints") and "anyres" in aspect_ratio
            else None
        )

        if isinstance(image_data, str):
            image_data = [image_data]

        if isinstance(image_data, list) and len(image_data) > 0:
            if "multi-images" in modalities or "video" in modalities:
                # Multiple images
                aspect_ratio = "pad"  # LLaVA OneVision Handling: more than one image --> interleaved image mode or video mode. We do not use anyres
                pixel_values, image_hashes, image_sizes = [], [], []
                res = []
                for img_data in image_data:
                    res.append(self._process_single_image(img_data, aspect_ratio, grid_pinpoints))
                res = await asyncio.gather(*res)
                for pixel_v, image_h, image_s in res:
                    pixel_values.append(pixel_v)
                    image_hashes.append(image_h)
                    image_sizes.append(image_s)

                if isinstance(pixel_values[0], np.ndarray):
                    pixel_values = np.stack(pixel_values, axis=0)
            else:
                # A single image
                pixel_values, image_hash, image_size = await self._process_single_image(
                    image_data[0], aspect_ratio, grid_pinpoints
                )
                image_hashes = [image_hash]
                image_sizes = [image_size]
        else:
            raise ValueError(f"Invalid image data: {image_data}")

        return {
            "pixel_values": pixel_values,
            "image_hashes": image_hashes,
            "image_sizes": image_sizes,
            "modalities": request_obj.modalities or ["image"],
        }

    # sglang==0.4.5.post3
    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

        result = await self.process_images_async(image_data, input_text, request_obj, *args, **kwargs)

        if result is None:
            return None

        modality = Modality.IMAGE
        if isinstance(request_obj.modalities, list):
            if request_obj.modalities[0] == "multi-images":
                modality = Modality.MULTI_IMAGES
            elif request_obj.modalities[0] == "video":
                modality = Modality.VIDEO

        return {
            "mm_items": [
                MultimodalDataItem(
                    pixel_values=result["pixel_values"],
                    image_sizes=result["image_sizes"],
                    modality=modality,
                )
            ],
        }


ImageProcessorMapping = {Mineru2QwenForCausalLM: Mineru2ImageProcessor}
