import os
import sys

from fastapi import Request
from sglang.srt.entrypoints.http_server import app, generate_request, launch_server
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.conversation import Conversation

from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from .logit_processor import Mineru2LogitProcessor

# mineru2.0的chat_template与chatml在换行上有微小区别
def custom_get_prompt(self) -> str:
    system_prompt = self.system_template.format(system_message=self.system_message)
    if self.system_message == "":
        ret = ""
    else:
        ret = system_prompt + self.sep

    for role, message in self.messages:
        if message:
            ret += role + "\n" + message + self.sep
        else:
            ret += role + "\n"
    return ret

_custom_logit_processor_str = Mineru2LogitProcessor().to_str()

# remote the existing /generate route
for route in app.routes[:]:
    if hasattr(route, "path") and getattr(route, "path") == "/generate":
        app.routes.remove(route)


# add the custom /generate route
@app.api_route("/generate", methods=["POST", "PUT"])
async def custom_generate_request(obj: GenerateReqInput, request: Request):
    if obj.custom_logit_processor is None:
        obj.custom_logit_processor = _custom_logit_processor_str
    return await generate_request(obj, request)


def main():
    # 检查命令行参数中是否包含--model-path
    args = sys.argv[1:]
    has_model_path_arg = False

    for i, arg in enumerate(args):
        if arg == "--model-path" or arg.startswith("--model-path="):
            has_model_path_arg = True
            break

    # 如果没有--model-path参数，在参数列表中添加它
    if not has_model_path_arg:
        default_path = auto_download_and_get_model_root_path("/", "vlm")
        args.extend(["--model-path", default_path])

    server_args = prepare_server_args(args)

    if server_args.chat_template is None:
        server_args.chat_template = "chatml"
        Conversation.get_prompt = custom_get_prompt

    server_args.enable_custom_logit_processor = True

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
