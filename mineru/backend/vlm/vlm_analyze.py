# Copyright (c) Opendatalab. All rights reserved.
import time

from loguru import logger

from ...data.data_reader_writer import DataWriter
from mineru.utils.pdf_image_tools import load_images_from_pdf
from .base_predictor import BasePredictor
from .predictor import get_predictor
from .token_to_middle_json import result_to_middle_json
from ...utils.models_download_utils import auto_download_and_get_model_root_path


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
    ) -> BasePredictor:
        key = (backend, model_path, server_url)
        if key not in self._models:
            if backend in ['transformers', 'sglang-engine'] and not model_path:
                model_path = auto_download_and_get_model_root_path("/","vlm")
            self._models[key] = get_predictor(
                backend=backend,
                model_path=model_path,
                server_url=server_url,
                **kwargs,
            )
        return self._models[key]


def doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: BasePredictor | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    **kwargs,
):
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)

    # load_images_start = time.time()
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes)
    images_base64_list = [image_dict["img_base64"] for image_dict in images_list]
    # load_images_time = round(time.time() - load_images_start, 2)
    # logger.info(f"load images cost: {load_images_time}, speed: {round(len(images_base64_list)/load_images_time, 3)} images/s")

    # infer_start = time.time()
    results = predictor.batch_predict(images=images_base64_list)
    # infer_time = round(time.time() - infer_start, 2)
    # logger.info(f"infer finished, cost: {infer_time}, speed: {round(len(results)/infer_time, 3)} page/s")

    middle_json = result_to_middle_json(results, images_list, pdf_doc, image_writer)
    return middle_json, results


async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: BasePredictor | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    **kwargs,
):
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)

    # load_images_start = time.time()
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes)
    images_base64_list = [image_dict["img_base64"] for image_dict in images_list]
    # load_images_time = round(time.time() - load_images_start, 2)
    # logger.info(f"load images cost: {load_images_time}, speed: {round(len(images_base64_list)/load_images_time, 3)} images/s")

    # infer_start = time.time()
    results = await predictor.aio_batch_predict(images=images_base64_list)
    # infer_time = round(time.time() - infer_start, 2)
    # logger.info(f"infer finished, cost: {infer_time}, speed: {round(len(results)/infer_time, 3)} page/s")
    middle_json = result_to_middle_json(results, images_list, pdf_doc, image_writer)
    return middle_json, results
