## 基于Triton的ROCm 不同后端实现优化，基本实现vllm后端正常推理，以及pipeline后端中第一步layout用的DocLayout-YOLO

**已有完整python vllm和mineru环境直接跳转第五步！！！**
**其他GPU执行问题可以参考，先prof查看定位找到哪个算子问题，然后triton后端实现即可**
测试了一下，基本和MinerU官网效果差不多，用AMD的人也不是很多，就在评论区分享给大家了

### 1.结果介绍
**补充一个200页的PDF python编程书测试一下速度，可以到1.99it/s：**
Two Step Extraction: 100%|████████████████████████████████████████| 200/200 [01:40<00:00,  1.99it/s]

**下面为之前14学术论文测试结果：**
7900xtx mineru-gradio --server-name 0.0.0.0 --server-port 7860 --enable-vllm-engine true 速度大概为**1.6-1.8s/it**，没有仔细测试，简单试了两个文档。第二种矩阵乘法代替原来的dots点乘可以进一步提速到1.3s/it，优化后的主要算子耗时在hipblast(这个没法提升了)和vllm triton后端，各占25%耗时吧，vllm tirion后端这个这个只能等官方优化了。。。。
doclayout-yolo的layout速度从原来的1.6it/s提高到15it/s，注意需要缓存一下输入的pdf尺寸后，triton必须要缓存尺寸没办法。主要是为了保留模型输入输出接口，最小代码改动。
采用-b vlm-vllm-engine模式举个例子

---
**测试结果为优化为5d矩阵乘代替原来的点积结果：**
2025-10-05 15:45:12.985 | INFO     | mineru.backend.vlm.vlm_analyze:get_model:128 - get vllm-engine predictor cost: 18.45s
Adding requests: 100%|████████████████████████████████████████████████████████████████████████████████| 14/14 [00:01<00:00, 12.20it/s]
Processed prompts: 100%|█████████████████████| 14/14 [00:08<00:00,  1.56it/s, est. speed input: 2174.18 toks/s, output: 791.87 toks/s]
Adding requests: 100%|█████████████████████████████████████████████████████████████████████████████| 278/278 [00:00<00:00, 323.03it/s]
Processed prompts: 100%|██████████████████| 278/278 [00:07<00:00, 37.63it/s, est. speed input: 5264.66 toks/s, output: 2733.31 toks/s]

mineru-gradio --server-name 0.0.0.0 --server-port 7860 --enable-vllm-engine true测试：
2025-10-05 15:46:55.953 | WARNING  | mineru.cli.common:convert_pdf_bytes_to_bytes_by_pypdfium2:54 - end_page_id is out of range, use pdf_docs length
Two Step Extraction: 100%|████████████████████████████████████████████████████████████████████████████| 14/14 [00:18<00:00,  1.30s/it]

---

### 2.原因介绍
AMD RDNA使用vllm后端有严重的性能问题，原因是因为vllm的**qwen2_vl.py**中有一个算子在rocm kernel上没有对应的实现，导致性能出现严重的卷积计算回退，一次执行花了12s，。。。。。。。。一言难尽。即**MIOpen 库中缺少模型中特定 Conv3d(bfloat16) 的优化内核**。
DocLayout-YOLO的**g2l_crm.py**空洞卷积也是这个问题，专业的CDNA MI210也没解决这个问题
正好一起处理了。

---

### 3.环境介绍
System: Ubuntu 24.04.3        Kernel: Linux 6.14.0-33-generic      ROCm version: 7.0.1
python环境：
python 3.12
pytorch-triton-rocm   3.5.0+gitbbb06c03 
torch                            2.10.0.dev20251001+rocm7.0
torchvision                  0.25.0.dev20251003+rocm7.0
vllm                              0.11.0rc2.dev198+g736fbf4c8.rocm701
不同版本无所谓，处理方法是一样的。

---

### 4.前置环境安装
```
uv venv --python python3.12
source .venv/bin/activate
uv pip install --pre torch torchvision   -i https://pypi.tuna.tsinghua.edu.cn/simple/   --extra-index-url https://download.pytorch.org/whl/nightly/rocm7.0
uv pip install pip
# 避免覆盖我们本地的pytorch，改用pip而没有继续使用uv pip
pip install -U "mineru[core]" -i https://pypi.mirrors.ustc.edu.cn/simple/
```
vllm 安装参考官方手册[Vllm](https://docs.vllm.com.cn/en/latest/getting_started/installation/gpu.html#amd-rocm)
```
#手动安装aiter，vllm，amd-smi等，自行找一个位置clone，然后进入该目录吧
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
git submodule sync; git submodule update --init --recursive
python setup.py develop
cd ..
git clone https://github.com/vllm-project/vllm.git
cd vllm/
cp -r /opt/rocm/share/amd_smi ~/Pytorch/vllm/
pip install amd_smi/
pip install --upgrade numba \
    scipy \
    huggingface-hub[cli,hf_transfer] \
    setuptools_scm
pip install -r requirements/rocm.txt
export PYTORCH_ROCM_ARCH="gfx1100"   #根据自己的GPU架构 rocminfo | grep gfx
python setup.py develop
```
---

### 5.vllm中关键triton算子添加
#### 这里我给出两种解决方法，第一种解决方法就是前面提到的优化到1.5到1.8s/it，第二种方法有手动优化算子到矩阵乘法，7900xtx肯定适用，大概1.3s/it，其他AMD GPU相对方案一也有提速，但是不一定是最佳速度实现，里面的手动部分可能需要微调。
**注意pip把triton 后端的flash_attn卸载了，搞了半天各种尝试还是报错，问题比较大，直接不用就行了**
```
#定位自己vllm位置XXX
pip show vllm
```
**关键更改**
XXX/vllm/model_executor/models/qwen2_vl.py文件：
**1.qwen2_vl.py文件33行下增加from .qwen2_vl_vision_kernels import triton_conv3d_patchify**
```
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Callable, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .qwen2_vl_vision_kernels import triton_conv3d_patchify
```
**接下来分为方案一(2.1和3.1)和方案二(2.2和3.2)，选取一种实现即可**

---
**方案1**
**2.1qwen2_vl.py文件498行class Qwen2VisionPatchEmbed(nn.Module),PS.就是这玩意AMD没有现成的内核算子导致回退**
```
class Qwen2VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels,
                              embed_dim,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x_reshaped = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                            self.patch_size)
        
        # Call your custom Triton kernel instead of self.proj
        x_out = triton_conv3d_patchify(x_reshaped, self.proj.weight)
        
        # The output of our kernel is already the correct shape [L, embed_dim]
        return x_out
```
**3.1XXX/vllm/model_executor/models/目录下创建qwen2_vl_vision_kernels.py文件，用triton实现**
```
import torch
from vllm.triton_utils import tl, triton

@triton.jit
def _conv3d_patchify_kernel(
    # Pointers to tensors
    X, W, Y,
    # Tensor dimensions
    N, C_in, D_in, H_in, W_in,
    C_out, KD, KH, KW,
    # Stride and padding for memory access
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_yn, stride_yc,
    # Triton-specific metaparameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for a non-overlapping 3D patching convolution.
    Each kernel instance computes one output value for one patch.
    """
    # Get the program IDs for the N (patch) and C_out (output channel) dimensions
    pid_n = tl.program_id(0)  # The index of the patch we are processing
    pid_cout = tl.program_id(1) # The index of the output channel we are computing

    # --- Calculate memory pointers ---
    # Pointer to the start of the current input patch
    x_ptr = X + (pid_n * stride_xn)
    # Pointer to the start of the current filter (weight)
    w_ptr = W + (pid_cout * stride_wn)
    # Pointer to where the output will be stored
    y_ptr = Y + (pid_n * stride_yn + pid_cout * stride_yc)

    # --- Perform the convolution (element-wise product and sum) ---
    # This is a dot product between the flattened patch and the flattened filter.
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the elements of the patch/filter
    for c_offset in range(0, C_in):
        for d_offset in range(0, KD):
            for h_offset in range(0, KH):
                # Unrolled loop for the innermost dimension (width) for performance
                for w_offset in range(0, KW, BLOCK_SIZE):
                    # Create masks to handle cases where KW is not a multiple of BLOCK_SIZE
                    w_range = w_offset + tl.arange(0, BLOCK_SIZE)
                    w_mask = w_range < KW

                    # Calculate offsets to load data
                    patch_offset = (c_offset * stride_xc + d_offset * stride_xd +
                                    h_offset * stride_xh + w_range * stride_xw)
                    filter_offset = (c_offset * stride_wc + d_offset * stride_wd +
                                     h_offset * stride_wh + w_range * stride_ww)

                    # Load patch and filter data, applying masks
                    patch_vals = tl.load(x_ptr + patch_offset, mask=w_mask, other=0.0)
                    filter_vals = tl.load(w_ptr + filter_offset, mask=w_mask, other=0.0)

                    # Multiply and accumulate
                    accumulator += patch_vals.to(tl.float32) * filter_vals.to(tl.float32)

    # Sum the accumulator block and store the single output value
    output_val = tl.sum(accumulator, axis=0)
    tl.store(y_ptr, output_val)


def triton_conv3d_patchify(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for the 3D patching convolution Triton kernel.
    """
    # Get tensor dimensions
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, KD, KH, KW = weight.shape

    # Create the output tensor
    # The output of this specific conv is (N, C_out, 1, 1, 1), which we squeeze
    Y = torch.empty((N, C_out), dtype=x.dtype, device=x.device)

    # Define the grid for launching the Triton kernel
    # Each kernel instance handles one patch (N) for one output channel (C_out)
    grid = (N, C_out)

    # Launch the kernel
    # We pass all strides to make the kernel flexible
    _conv3d_patchify_kernel[grid](
        x, weight, Y,
        N, C_in, D_in, H_in, W_in,
        C_out, KD, KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        Y.stride(0), Y.stride(1),
        BLOCK_SIZE=16, # A reasonable default, can be tuned
    )

    return Y
```
---
**方案2**
**2.2qwen2_vl.py文件498行class Qwen2VisionPatchEmbed(nn.Module)函数,PS.就是这玩意AMD没有现成的内核算子导致回退，这里我们直接5D张量一步到位，改为矩阵乘法**
```
class Qwen2VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = (temporal_patch_size, patch_size, patch_size)

        self.proj = nn.Conv3d(in_channels,
                              embed_dim,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x_reshaped_5d = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                               self.patch_size)

        return triton_conv3d_patchify(x_reshaped_5d, self.proj.weight)
```
**3.2XXX/vllm/model_executor/models/目录下创建qwen2_vl_vision_kernels.py文件，用triton实现**
```
import torch
from vllm.triton_utils import tl, triton

@triton.jit
def _conv_gemm_kernel(
    A, B, C, M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_conv3d_patchify(x_5d: torch.Tensor, weight_5d: torch.Tensor) -> torch.Tensor:
    N_patches, _, _, _, _ = x_5d.shape
    C_out, _, _, _, _ = weight_5d.shape
    A = x_5d.view(N_patches, -1)
    B = weight_5d.view(C_out, -1).transpose(0, 1).contiguous()
    M, K = A.shape
    _K, N = B.shape
    assert K == _K
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # --- 针对7900xtx的手动调优配置，其他GPU的最优组合可能需要自行寻找，AMD的autotune效果就是没有效果 ---
    best_config = {
        'BLOCK_M': 128,
        'BLOCK_N': 128,
        'BLOCK_K': 32,
    }
    num_stages = 4
    num_warps = 8

    grid = (triton.cdiv(M, best_config['BLOCK_M']),
            triton.cdiv(N, best_config['BLOCK_N']))

    _conv_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        **best_config,
        num_stages=num_stages,
        num_warps=num_warps
    )

    return C
```
---
**4.关闭终端后再次使用mineru-gradio会报一个Lora错误，修改代码跳过它**
```
pip show mineru_vl_utils
```

打开该文件XXX/mineru_vl_utils/vlm_client/vllm_async_engine_client.py修改第58行self.tokenizer = vllm_async_llm.tokenizer.get_lora_tokenizer()为：
```
        try:
            self.tokenizer = vllm_async_llm.tokenizer.get_lora_tokenizer()
        except AttributeError:
            # 如果没有 get_lora_tokenizer 方法，直接使用原始 tokenizer
            self.tokenizer = vllm_async_llm.tokenizer
```

**最后整两个环境变量后愉快玩耍即可**
```
export MINERU_MODEL_SOURCE=modelscope
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```
---

### 6.vllm后端已经没有问题，下面是pipeline 中layout用的doclayout-yolo模型空洞卷积问题
### 我在 [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO/issues/120#issuecomment-3368144275) 下做了一个回答，因此 pipeline 的空洞卷积问题不在这里赘述，直接点击链接查看即可。
查看自己doclayout-yolo安装位置如下，然后进入修改链接中回复介绍的文件即可
```
pip show doclayout-yolo
```

