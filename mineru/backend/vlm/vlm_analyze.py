# Copyright (c) Opendatalab. All rights reserved.
import time

from loguru import logger

from .model_output_to_middle_json import result_to_middle_json
from ...data.data_reader_writer import DataWriter
from mineru.utils.pdf_image_tools import load_images_from_pdf

from ...utils.enum_class import ImageType
from ...utils.models_download_utils import auto_download_and_get_model_root_path

from mineru_vl_utils import MinerUClient


class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        backend: str,
        model_path: str | None,
        server_url: str | None,
        **kwargs,
    ) -> MinerUClient:
        key = (backend, model_path, server_url)
        if key not in self._models:
            if backend in ['transformers', 'vllm-engine'] and not model_path:
                model_path = auto_download_and_get_model_root_path("/","vlm")
            self._models[key] = MinerUClient(
                backend=backend,
                model_path=model_path,
                server_url=server_url,
                **kwargs,
            )
        return self._models[key]


def doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    **kwargs,
):
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)

    # load_images_start = time.time()
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    # load_images_time = round(time.time() - load_images_start, 2)
    # logger.info(f"load images cost: {load_images_time}, speed: {round(len(images_base64_list)/load_images_time, 3)} images/s")

    # infer_start = time.time()
    results = predictor.batch_two_step_extract(images=images_pil_list)
    # infer_time = round(time.time() - infer_start, 2)
    # logger.info(f"infer finished, cost: {infer_time}, speed: {round(len(results)/infer_time, 3)} page/s")

    middle_json = result_to_middle_json(results, images_list, pdf_doc, image_writer)
    return middle_json, results


async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    **kwargs,
):
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)

    # load_images_start = time.time()
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    # load_images_time = round(time.time() - load_images_start, 2)
    # logger.info(f"load images cost: {load_images_time}, speed: {round(len(images_base64_list)/load_images_time, 3)} images/s")

    # infer_start = time.time()
    results = await predictor.aio_batch_two_step_extract(images=images_pil_list)
    # infer_time = round(time.time() - infer_start, 2)
    # logger.info(f"infer finished, cost: {infer_time}, speed: {round(len(results)/infer_time, 3)} page/s")
    middle_json = result_to_middle_json(results, images_list, pdf_doc, image_writer)
    return middle_json, results
