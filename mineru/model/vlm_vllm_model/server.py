import sys

from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from vllm.entrypoints.cli.main import main as vllm_main


def main():
    args = sys.argv[1:]

    has_port_arg = False
    has_gpu_memory_utilization_arg = False
    has_model_arg = False

    for i, arg in enumerate(args):
        if arg == "--port" or arg.startswith("--port="):
            has_port_arg = True
        if arg == "--gpu-memory-utilization" or arg.startswith("--gpu-memory-utilization="):
            has_gpu_memory_utilization_arg = True
        if arg == "--model" or arg.startswith("--model="):
            has_model_arg = True

    if not has_port_arg:
        args.extend(["--port", "30000"])
    if not has_gpu_memory_utilization_arg:
        args.extend(["--gpu-memory-utilization", "0.5"])
    if not has_model_arg:
        default_path = auto_download_and_get_model_root_path("/", "vlm")
        args.extend([default_path])

    # 重新构造sys.argv，以便透传所有参数给vllm
    sys.argv = [sys.argv[0]] + ["serve"] + args

    # 启动vllm服务器
    print(f"start vllm server: {sys.argv}")
    vllm_main()


if __name__ == "__main__":
    main()
