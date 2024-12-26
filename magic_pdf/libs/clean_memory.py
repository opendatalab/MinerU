# Copyright (c) Opendatalab. All rights reserved.
import torch
import gc


def clean_memory(device='cuda'):
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif str(device).startswith("npu"):
        import torch_npu
        if torch.npu.is_available():
            torch_npu.empty_cache()
            torch_npu.ipc_collect()
    gc.collect()