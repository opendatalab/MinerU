import os
import sys

from mineru.backend.vlm.utils import set_default_gpu_memory_utilization, enable_custom_logits_processors
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

from vllm.entrypoints.cli.main import main as vllm_main


def main():
    args = sys.argv[1:]

    has_port_arg = False
    has_gpu_memory_utilization_arg = False
    has_logits_processors_arg = False
    model_path = None
    model_arg_indices = []

    # 检查现有参数
    for i, arg in enumerate(args):
        if arg == "--port" or arg.startswith("--port="):
            has_port_arg = True
        if arg == "--gpu-memory-utilization" or arg.startswith("--gpu-memory-utilization="):
            has_gpu_memory_utilization_arg = True
        if arg == "--logits-processors" or arg.startswith("--logits-processors="):
            has_logits_processors_arg = True
        if arg == "--model":
            if i + 1 < len(args):
                model_path = args[i + 1]
                model_arg_indices.extend([i, i + 1])
        elif arg.startswith("--model="):
            model_path = arg.split("=", 1)[1]
            model_arg_indices.append(i)

    # 从参数列表中移除 --model 参数
    if model_arg_indices:
        for index in sorted(model_arg_indices, reverse=True):
            args.pop(index)

    custom_logits_processors = enable_custom_logits_processors()

    # 添加默认参数
    if not has_port_arg:
        args.extend(["--port", "30000"])
    if not has_gpu_memory_utilization_arg:
        gpu_memory_utilization = str(set_default_gpu_memory_utilization())
        args.extend(["--gpu-memory-utilization", gpu_memory_utilization])
    if not model_path:
        model_path = auto_download_and_get_model_root_path("/", "vlm")
    if (not has_logits_processors_arg) and custom_logits_processors:
        args.extend(["--logits-processors", "mineru_vl_utils:MinerULogitsProcessor"])

    # 重构参数，将模型路径作为位置参数
    sys.argv = [sys.argv[0]] + ["serve", model_path] + args

    if os.getenv('OMP_NUM_THREADS') is None:
        os.environ["OMP_NUM_THREADS"] = "1"

    # 启动vllm服务器
    print(f"start vllm server: {sys.argv}")
    vllm_main()


if __name__ == "__main__":
    main()
