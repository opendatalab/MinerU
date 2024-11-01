# Copyright (c) Opendatalab. All rights reserved.
import torch
import gc


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()