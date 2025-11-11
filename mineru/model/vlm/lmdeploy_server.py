import os
import sys

from loguru import logger

from mineru.backend.vlm.utils import set_lmdeploy_backend
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


def main():
    args = sys.argv[1:]

    has_port_arg = False
    has_gpu_memory_utilization_arg = False
    has_log_level_arg = False
    has_backend_arg = False
    device_type = "cuda"
    lm_backend = ""

    # 检查现有参数
    for i, arg in enumerate(args):
        if arg == "--server-port" or arg.startswith("--server-port="):
            has_port_arg = True
        if arg == "--cache-max-entry-count" or arg.startswith("--cache-max-entry-count="):
            has_gpu_memory_utilization_arg = True
        if arg == "--log-level" or arg.startswith("--log-level="):
            has_log_level_arg = True
        if arg == "--backend":
            has_backend_arg = True
            if i + 1 < len(args):
                lm_backend = args[i + 1]
        if arg.startswith("--backend="):
            has_backend_arg = True
            lm_backend = arg.split("=", 1)[1]
        if arg == "--device":
            if i + 1 < len(args):
                device_type = args[i + 1]
        if arg.startswith("--device="):
            device_type = arg.split("=", 1)[1]

    # 添加默认参数
    if not has_port_arg:
        args.extend(["--server-port", "30000"])
    if not has_gpu_memory_utilization_arg:
        args.extend(["--cache-max-entry-count", "0.5"])
    if not has_log_level_arg:
        args.extend(["--log-level", "ERROR"])

    if ":" in device_type:
        device_type = device_type.split(":")[0]
    if lm_backend == "":
        lm_backend = set_lmdeploy_backend(device_type)
    logger.info(f"Set lmdeploy_backend to: {lm_backend}")

    # args中如果有--backend参数，则不设置
    if not has_backend_arg:
        args.extend(["--backend", lm_backend])

    model_path = auto_download_and_get_model_root_path("/", "vlm")

    # 重构参数，将模型路径作为位置参数
    sys.argv = [sys.argv[0]] + ["serve", "api_server", model_path] + args

    if os.getenv('OMP_NUM_THREADS') is None:
        os.environ["OMP_NUM_THREADS"] = "1"

    # 启动 lmdeploy 服务器
    print(f"start lmdeploy server: {sys.argv}")

    # 使用os.system调用启动lmdeploy服务器
    os.system("lmdeploy " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
