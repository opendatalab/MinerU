import sys

from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from vllm.entrypoints.openai_api_server import main as vllm_serve


def main():
    # 检查命令行参数中是否包含--model-path
    args = sys.argv[1:]
    has_model_path_arg = False

    for i, arg in enumerate(args):
        if arg == "--model" or arg.startswith("--model="):
            has_model_path_arg = True
            break

    # 如果没有--model-path参数，在参数列表中添加它
    if not has_model_path_arg:
        default_path = auto_download_and_get_model_root_path("/", "vlm")
        args.extend(["--model", default_path])

    # 重新构造sys.argv，以便透传所有参数给vllm
    sys.argv = [sys.argv[0]] + args

    # 启动vllm服务器
    print(f"start vllm server: {sys.argv}")
    vllm_serve()


if __name__ == "__main__":
    main()
