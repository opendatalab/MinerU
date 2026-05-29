# Using SGLang as the Inference Backend

[SGLang](https://github.com/sgl-project/sglang) is a fast serving engine for large language and vision-language models. `MinerU2.5-2509-1.2B` is built on the `Qwen2-VL` architecture (`Qwen2VLForConditionalGeneration`), which SGLang implements natively. This means you can serve the model with SGLang's OpenAI-compatible server and point MinerU's `vlm-http-client` backend at it — **no code changes are required, and no local `torch` is needed on the MinerU side**.

> [!NOTE]
> MinerU shipped a native `vlm-sglang-engine` backend for older releases. `MinerU2.5` switched its default acceleration backend to `vllm`. The OpenAI-server + `vlm-http-client` path documented here is the supported, zero-code-change way to run `MinerU2.5` on SGLang today, and it works with any OpenAI-compatible server.

## Prerequisites

You need two environments. They can live on the same machine or on different machines (server and client communicate over HTTP).

- **Client side (where MinerU runs):** install the lightweight client. The `vlm-http-client` backend does not require local `torch`.
  ```bash
  uv pip install mineru
  ```
  If you also want to run the full MinerU pipeline locally, install the full package instead:
  ```bash
  uv pip install "mineru[core]"
  ```

- **Server side (where the model runs):** install SGLang in a **separate** environment, following the [SGLang installation guide](https://docs.sglang.ai/start/install.html). Serving `MinerU2.5-2509-1.2B` was verified end-to-end on **SGLang 0.5.12.post1**.

> [!TIP]
> Keep SGLang in its own virtual environment. The MinerU client environment only needs the `mineru` package and network access to the server.

## Step 1: Start the SGLang server

Launch SGLang's OpenAI-compatible server with the `MinerU2.5-2509-1.2B` model:

```bash
python3 -m sglang.launch_server \
  --model-path opendatalab/MinerU2.5-2509-1.2B \
  --host 0.0.0.0 --port 30000
```

Newer SGLang versions also accept the shorter entrypoint alias:

```bash
sglang serve opendatalab/MinerU2.5-2509-1.2B --host 0.0.0.0 --port 30000
```

> [!TIP]
> - `--chat-template` is **not** required: the tokenizer already ships the `Qwen2-VL` chat template. Passing `--chat-template qwen2-vl` also works but is unnecessary.
> - `--trust-remote-code` is **not** required, because this is the standard `qwen2_vl` architecture. It is harmless if added.
> - Port `30000` is SGLang's default and matches MinerU's `vlm-http-client` examples.
> - Optional: add `--mem-fraction-static 0.6` to free VRAM when this 1.2B model shares a large GPU with other processes. SGLang otherwise reserves most of the GPU for the KV cache. This is a tuning knob only and is not required for correctness.

## Step 2: Run MinerU against it

In another terminal (on the client machine), point MinerU's `vlm-http-client` backend at the server:

```bash
mineru -p <input_path> -o <output_path> -b vlm-http-client -u http://127.0.0.1:30000
```

> [!TIP]
> - `<input_path>`: a local PDF or image file, or a directory of them
> - `<output_path>`: output directory
> - `-u`: the OpenAI-compatible server URL. If SGLang runs on another machine, use `http://<server_ip>:30000`.
> - `vlm-http-client` is the lightweight remote client option and does not require local `torch`.

If you prefer to drive the model from Python, you can use the [`mineru-vl-utils`](https://github.com/opendatalab/mineru-vl-utils) library with the `http-client` backend directly:

```python
from mineru_vl_utils import MinerUClient

client = MinerUClient(backend="http-client", server_url="http://127.0.0.1:30000")
blocks = client.two_step_extract(image)  # image is a PIL.Image
```

## Repetition control

The `vllm` serving path enforces anti-repetition with `MinerULogitsProcessor` (passed to `vllm serve` via `--logits-processors mineru_vl_utils:MinerULogitsProcessor`), which implements the `no_repeat_ngram_size` sampling parameter.

> [!NOTE]
> SGLang does **not** support `no_repeat_ngram_size`. `mineru-vl-utils`' http-client sends it inside the `vllm_xargs` field, and the SGLang server silently ignores unknown xargs.
>
> - On clean documents this makes no difference — output matches the `transformers` / `vllm` path exactly.
> - On pathological inputs (dense repeated table cells, heavy noise or blur), the absence of a hard n-gram block could in principle allow a repetition loop that the `vllm` path would have prevented.
> - In practice, `mineru-vl-utils` already defaults to `presence_penalty=1.0` and `frequency_penalty=0.05` for content / table / equation extraction, which mitigates repetition. SGLang's available anti-repetition knobs are `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `min_p`, and `top_k`.
> - For stricter control, the n-gram block can be ported through SGLang's custom logit processor mechanism (server flag `--enable-custom-logit-processor`).

## Reference accuracy

For reference, `MinerU2.5-2509-1.2B` served with SGLang `0.5.12.post1` on **a single NVIDIA H200 GPU** was scored on OmniDocBench v1.6 (full set, 1651 pages) with the official scorer (`pdf_validation.py`, end2end `quick_match`):

| Metric | Result |
|---|---|
| Text-block edit distance (lower is better) | 0.0453 |
| Table TEDS | 87.94 (structure-only 91.78) |
| Reading-order edit distance | 0.1304 |
| Display-formula edit distance | ~0.49 |

Throughput: 1651 pages extracted in ~23 minutes at client concurrency 24 (`aio_concurrent_two_step_extract`), about 0.84 s/page for the two-step (layout + content) flow.

> [!NOTE]
> These numbers are on par with `MinerU2.5`'s published `transformers` / `vllm` results, confirming SGLang serving parity. (The display-formula CDM metric was skipped because it requires LaTeX / Ghostscript / ImageMagick.)
