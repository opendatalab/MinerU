import os
import sys

from fastapi import Request
from sglang.srt.entrypoints.http_server import app, generate_request, launch_server
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from .logit_processor import Mineru2LogitProcessor

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
    server_args = prepare_server_args(sys.argv[1:])

    if server_args.chat_template is None:
        server_args.chat_template = "chatml"

    server_args.enable_custom_logit_processor = True

    if server_args.model_path is None:
        server_args.model_path = auto_download_and_get_model_root_path("/","vlm")

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
