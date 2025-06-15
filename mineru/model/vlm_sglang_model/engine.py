import asyncio
import time
from types import MethodType
from typing import AsyncIterator, Dict, Iterator, List, Optional, Union

import fastapi
from sglang.srt.entrypoints.engine import Engine as _Engine
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    dataclass_to_string_truncated,
    logger,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs

from ...utils.run_async import run_async
from .logit_processor import Mineru2LogitProcessor


class BatchEngine(_Engine):
    """
    The engine is patched to support batch multi-modal generate, and early image preprocessing.
    """

    def __init__(self, server_args: ServerArgs, **kwargs):
        server_args.enable_custom_logit_processor = True
        super().__init__(server_args=server_args, **kwargs)
        _patch_tokenizer_manager(self.tokenizer_manager)

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be a file name, a url, or base64 encoded string.
        # See also python/sglang/srt/utils.py:load_image.
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        modalities_list = []

        # EDIT
        if isinstance(image_data, list):
            for _ in range(len(image_data)):
                modalities_list.append(["image"])
        elif image_data is not None:
            modalities_list.append("image")

        # ADD
        if custom_logit_processor is None:
            custom_logit_processor = Mineru2LogitProcessor().to_str()

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            modalities=modalities_list,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
        )
        generator = _generate_request(self.tokenizer_manager, obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = run_async(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = run_async(generator.__anext__())
            return ret

    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be a file name, a url, or base64 encoded string.
        # See also python/sglang/srt/utils.py:load_image.
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
    ) -> Union[Dict, AsyncIterator[Dict], Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        modalities_list = []

        # EDIT
        if isinstance(image_data, list):
            for _ in range(len(image_data)):
                modalities_list.append(["image"])
        elif image_data is not None:
            modalities_list.append("image")

        # ADD
        if custom_logit_processor is None:
            custom_logit_processor = Mineru2LogitProcessor().to_str()

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            modalities=modalities_list,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
        )
        generator = _generate_request(self.tokenizer_manager, obj, None)

        if stream is True:
            return generator
        else:
            return await generator.__anext__()


def _auto_create_handle_loop(self: TokenizerManager):
    """
    patch the original `auto_create_handle_loop()` method to reset `no_create_loop`
    when the event loop changes.
    """
    try:
        curr_handle_loop = asyncio.get_running_loop()
    except RuntimeError:
        curr_handle_loop = None

    last_handle_loop = getattr(self, "_last_handle_loop", None)
    if last_handle_loop != curr_handle_loop:
        self.no_create_loop = False
        setattr(self, "_last_handle_loop", curr_handle_loop)
    return TokenizerManager.auto_create_handle_loop(self)


def _patch_tokenizer_manager(self: TokenizerManager):
    self.auto_create_handle_loop = MethodType(_auto_create_handle_loop, self)


async def _one_request(
    self: TokenizerManager,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request],
    created_time: Optional[float],
):
    tokenized_obj = await self._tokenize_one_request(obj)
    state = self._send_one_request(obj, tokenized_obj, created_time)
    async for out in self._wait_one_response(obj, state, request):
        yield out


async def _handle_batch_request(
    self: TokenizerManager,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request] = None,
    created_time: Optional[float] = None,
):
    batch_size = obj.batch_size

    generators = []
    rids = []

    if getattr(obj, "parallel_sample_num", 1) != 1:
        raise Exception("parallel_sample_num != 1 is not supported in this patched code.")

    # Send all requests
    for i in range(batch_size):
        tmp_obj = obj[i]
        generators.append(_one_request(self, tmp_obj, request, created_time))
        rids.append(tmp_obj.rid)

    # Wait for all requests
    is_stream = hasattr(obj, "stream") and obj.stream
    if not is_stream:
        outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
        yield outputs
    else:
        rid_to_index = {rid: i for i, rid in enumerate(rids)}
        task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
        while task_map:
            done, _ = await asyncio.wait(task_map.keys(), return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                gen = task_map.pop(task)
                try:
                    result = task.result()
                    result["index"] = rid_to_index[result["meta_info"]["id"]]
                    yield result
                    new_task = asyncio.create_task(gen.__anext__())
                    task_map[new_task] = gen
                except StopAsyncIteration:
                    pass


async def _generate_request(
    self: TokenizerManager,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request] = None,
):
    created_time = time.time()

    self.auto_create_handle_loop()

    if isinstance(obj, EmbeddingReqInput) and self.is_generation:
        raise ValueError(
            "This model does not appear to be an embedding model by default. "
            "Please add `--is-embedding` when launching the server or try another model."
        )

    obj.normalize_batch_and_arguments()

    if self.log_requests:
        max_length, skip_names, _ = self.log_request_metadata
        logger.info(f"Receive: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}")

    async with self.model_update_lock.reader_lock:
        is_single = obj.is_single
        if is_single:
            tokenized_obj = await self._tokenize_one_request(obj)
            state = self._send_one_request(obj, tokenized_obj, created_time)
            async for response in self._wait_one_response(obj, state, request):
                yield response
        else:
            async for response in _handle_batch_request(self, obj, request, created_time):
                yield response
