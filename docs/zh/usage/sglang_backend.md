# 使用 SGLang 作为推理后端

[SGLang](https://github.com/sgl-project/sglang) 是面向大语言模型与视觉语言模型的高性能推理引擎。`MinerU2.5-2509-1.2B` 基于 `Qwen2-VL` 架构（`Qwen2VLForConditionalGeneration`）构建，而该架构正是 SGLang 原生支持的，因此你可以用 SGLang 启动 OpenAI 兼容服务，再让 MinerU 的 `vlm-http-client` 后端连接到它——**无需任何代码改动，MinerU 端也无需本地安装 `torch`**。

> [!NOTE]
> MinerU 早期版本曾提供原生的 `vlm-sglang-engine` 后端，`MinerU2.5` 已将默认加速后端切换为 `vllm`。本文所述的 OpenAI 服务 + `vlm-http-client` 方式，是当前在 SGLang 上运行 `MinerU2.5` 的、零代码改动的受支持方式，并且适用于任意 OpenAI 兼容服务。

## 前置准备

你需要两个环境，它们可以在同一台机器上，也可以分布在不同机器上（服务端与客户端通过 HTTP 通信）。

- **客户端（运行 MinerU 的一侧）：** 安装轻量客户端即可，`vlm-http-client` 后端不要求本地安装 `torch`。
  ```bash
  uv pip install mineru
  ```
  如果你还想在本地运行完整的 MinerU pipeline，则改为安装完整包：
  ```bash
  uv pip install "mineru[core]"
  ```

- **服务端（运行模型的一侧）：** 请在**独立**的环境中安装 SGLang，可参考 [SGLang 安装文档](https://docs.sglang.ai/start/install.html)。`MinerU2.5-2509-1.2B` 的部署已在 **SGLang 0.5.12.post1** 上完成端到端验证。

> [!TIP]
> 建议将 SGLang 放在独立的虚拟环境中。MinerU 客户端环境只需 `mineru` 包以及到服务端的网络连通即可。

## 第一步：启动 SGLang 服务

使用 `MinerU2.5-2509-1.2B` 模型启动 SGLang 的 OpenAI 兼容服务：

```bash
python3 -m sglang.launch_server \
  --model-path opendatalab/MinerU2.5-2509-1.2B \
  --host 0.0.0.0 --port 30000
```

较新版本的 SGLang 也支持更简短的入口别名：

```bash
sglang serve opendatalab/MinerU2.5-2509-1.2B --host 0.0.0.0 --port 30000
```

> [!TIP]
> - **无需** `--chat-template`：tokenizer 已自带 `Qwen2-VL` 的对话模板。传入 `--chat-template qwen2-vl` 也能工作，但属于多余操作。
> - **无需** `--trust-remote-code`：这是标准的 `qwen2_vl` 架构。即使加上也无害。
> - `30000` 是 SGLang 的默认端口，与 MinerU 的 `vlm-http-client` 示例一致。
> - 可选：当这个 1.2B 模型与其他进程共用一张大显存 GPU 时，可加上 `--mem-fraction-static 0.6` 释放显存。否则 SGLang 会为 KV cache 预留大部分显存。这只是一个调优开关，并非正确性所必需。

## 第二步：让 MinerU 连接该服务

在另一个终端（客户端机器上），让 MinerU 的 `vlm-http-client` 后端连接到该服务：

```bash
mineru -p <input_path> -o <output_path> -b vlm-http-client -u http://127.0.0.1:30000
```

> [!TIP]
> - `<input_path>`：本地 PDF 或图片文件，或包含它们的目录
> - `<output_path>`：输出目录
> - `-u`：OpenAI 兼容服务的地址。若 SGLang 运行在另一台机器上，请使用 `http://<server_ip>:30000`。
> - `vlm-http-client` 是轻量远程 client，用法上不要求本地安装 `torch`。

如果你更习惯用 Python 驱动模型，也可以直接使用 [`mineru-vl-utils`](https://github.com/opendatalab/mineru-vl-utils) 库的 `http-client` 后端：

```python
from mineru_vl_utils import MinerUClient

client = MinerUClient(backend="http-client", server_url="http://127.0.0.1:30000")
blocks = client.two_step_extract(image)  # image 为 PIL.Image
```

## 重复抑制（Repetition control）

`vllm` 部署路径通过 `MinerULogitsProcessor` 来强制抑制重复（在 `vllm serve` 中通过 `--logits-processors mineru_vl_utils:MinerULogitsProcessor` 传入），它实现了 `no_repeat_ngram_size` 采样参数。

> [!NOTE]
> SGLang **不**支持 `no_repeat_ngram_size`。`mineru-vl-utils` 的 http-client 会把它放在 `vllm_xargs` 字段里发送，而 SGLang 服务端会静默忽略未知的 xargs。
>
> - 对于干净文档，这没有任何差异——输出与 `transformers` / `vllm` 路径完全一致。
> - 对于病态输入（密集重复的表格单元、严重噪声或模糊），缺少硬性的 n-gram 阻断，理论上可能出现 `vllm` 路径本可避免的重复循环。
> - 实际上，`mineru-vl-utils` 在正文 / 表格 / 公式提取中已默认 `presence_penalty=1.0`、`frequency_penalty=0.05`，可在实践中缓解重复。SGLang 可用的重复抑制开关包括 `repetition_penalty`、`frequency_penalty`、`presence_penalty`、`min_p`、`top_k`。
> - 如需更严格的控制，可通过 SGLang 的自定义 logit processor 机制（服务端开关 `--enable-custom-logit-processor`）移植该 n-gram 阻断逻辑。

## 参考精度

作为参考，在 **单张 NVIDIA H200 GPU** 上以 SGLang `0.5.12.post1` 部署的 `MinerU2.5-2509-1.2B`，使用官方评测脚本（`pdf_validation.py`，end2end `quick_match`）在 OmniDocBench v1.6（全量，1651 页）上的得分如下：

| 指标 | 结果 |
|---|---|
| 正文块编辑距离（越低越好） | 0.0453 |
| 表格 TEDS | 87.94（仅结构 91.78） |
| 阅读顺序编辑距离 | 0.1304 |
| 行间公式编辑距离 | ~0.49 |

吞吐：在客户端并发 24（`aio_concurrent_two_step_extract`）下，1651 页约 23 分钟完成提取，即约 0.84 秒/页（两步：版面 + 内容）。

> [!NOTE]
> 这些数字与 `MinerU2.5` 公布的 `transformers` / `vllm` 结果相当，印证了 SGLang 部署的精度一致性。（行间公式的 CDM 指标因需要 LaTeX / Ghostscript / ImageMagick 而被跳过。）
