# Deployment

## Choose a service {#services}

| Deployment goal | Entrypoint | How to check it |
| --- | --- | --- |
| Parse interactively on one machine | `mineru-kit webui` manages a local parsing worker by default. | Open the address printed at startup and convert a file using the [WebUI guide](../usage/webui.md). |
| Offer document parsing to applications | `mineru-kit api-server` exposes the asynchronous Tasks API. | Inspect `/v1/health` and `/v1/tiers`, then run the [single-file HTTP example](../usage/http_api.md#curl-example). |
| Put multiple GPUs or existing parsing services behind one address | Deploy a [Router](router.md). | Submit the same Tasks example to the Router's address. |
| Return an attachment after a single-file upload | Run [file-parse](../usage/file_parse.md) in front of a Tasks service. | Check the attachment or the status URL in a `202` response. |
| Provide remote VLM inference | Run `mineru-kit vlm-server` or use an existing OpenAI-compatible model service. | Set `model.vlm.server_url` on the parser and try a Standard or Advanced parse. [Remote service configuration](../usage/model_source.md) |

## Prepare hardware and models {#prepare}

1. Use `mineru-kit hardware info --json` to inspect devices, runtimes, and existing calibration records. Use `mineru-kit hardware detect` for a tier recommendation. Initial ONNX calibration may download a model. [Tiers and runtimes](../usage/tiers.md)
2. The base package can parse native documents directly. For model inference, select the [optional modules](../quick_start/extension_modules.md) or an [NVIDIA Docker image](../quick_start/docker_deployment.md) for the target device. For multi-GPU assignments, see [GPU and engine parameters](../usage/advanced_cli_parameters.md).
3. [Download and verify the models](../usage/model_source.md) for the backend and engine you will run. `mineru-kit models show` reports selection and readiness. Use the exit code from `models verify` to check required files. Complete both steps before going offline.
4. Start the service and inspect a result. The server's `--tier` sets the ceiling for PDFs and images. Native documents use Flash. Use native HTML to check the request and download path, then parse a representative PDF or image at the inference tier you need.
