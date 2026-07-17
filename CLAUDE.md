# MinerU 两阶段解耦 — 环境配置与验证指南

本文档面向在新机器上配置本项目的 Claude Code 会话：按顺序执行即可完成环境搭建与验证。
当前工作分支：`feat/decouple-layout-vlm`（layout 与 VLM 识别两阶段解耦）。

## 项目背景（30 秒版）

vlm 后端的"版面分析(layout) + 内容识别(VLM)"已拆分为两个可独立替换的阶段，
中间以标准 `layout.json` 交接。关键代码：

| 文件 | 内容 |
|---|---|
| `mineru/backend/vlm/stages.py` | `LayoutDetector` / `ContentRecognizer` 接口、layout.json 序列化、`PrecomputedLayoutDetector` |
| `mineru/backend/vlm/pipeline_layout_detector.py` | PP-DocLayoutV2 layout 实现（CPU 可跑） |
| `mineru/backend/vlm/ovis_ocr_recognizer.py` | OvisOCR2 识别适配器（纯 HTTP） |
| `mineru/backend/vlm/vlm_analyze.py` | `doc_analyze(..., layout_detector=, content_recognizer=, layout_writer=)`；默认路径与原版逐字一致 |
| `demo/vlm_two_stage_demo.py` | 模式 a/b/c/layout-only + `--recognizer ovis` + `--layout-json` |
| `demo/pp_doclayoutv3_layout.py` | PP-DocLayoutV3 导出脚本（独立 paddle 环境运行） |
| `demo/vlm_layout_visualize.py` | layout.json 可视化（画框标注） |

## 环境总览：三个必须隔离的 Python 环境

依赖版本互相冲突，**绝不能合并安装**：

| 环境 | 用途 | 冲突原因 |
|---|---|---|
| ① MinerU 主环境 | 跑解析、测试、demo | 锁定 `vllm>=0.10.1.1,<0.22.0` |
| ② OvisOCR2 服务环境 | `vllm serve` 起识别服务 | 要求 `vllm==0.22.1`，与 ① 冲突 |
| ③ Paddle 环境（可选） | PP-DocLayoutV3 出 layout.json | paddlepaddle 钉死 numpy/opencv 版本 |

②③ 与 ① 只通过 HTTP / layout.json 文件交互，无代码依赖。

## 第 1 步：MinerU 主环境

```bash
git checkout feat/decouple-layout-vlm
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[core]"
export MINERU_MODEL_SOURCE=modelscope   # 国内网络必设；海外可省略
```

**验证门槛 1（无需 GPU/模型，必须全绿再继续）：**

```bash
pytest tests/unittest/test_vlm_stages.py tests/unittest/test_ovis_recognizer.py \
       tests/unittest/test_pp_doclayoutv3_export.py -q -o addopts=""
# 预期：27 passed
```

> 坑：pyproject 配了 `--cov` 但通常没装 pytest-cov，跑 pytest **必须**带 `-o addopts=""`。

## 第 2 步：判断路线（先确认 OS 和 GPU）

- **vLLM 只支持 Linux**。Windows 机器起服务必须在 WSL2 内，或用 Docker Desktop。
- 无 GPU / 纯 Windows 不想装 WSL：走"transformers 免服务路线"，跳过第 3 步的服务部分。

## 第 3 步：起服务（Linux / WSL2）

**MinerU VLM 服务**（环境 ①，终端 1）：

```bash
pip install -e ".[vllm]"
mineru-vllm-server --port 30000 --gpu-memory-utilization 0.4
```

**OvisOCR2 服务**（环境 ②，终端 2，独立 venv）：

```bash
python -m venv ~/ovis-env && source ~/ovis-env/bin/activate
pip install "vllm==0.22.1" pillow
# HF 访问受限时：export HF_ENDPOINT=https://hf-mirror.com
vllm serve "ATH-MaaS/OvisOCR2" --port 8000 --gpu-memory-utilization 0.4
```

- 两个 `--gpu-memory-utilization 0.4` 是为单卡共存设计；单独跑可去掉。
- 就绪判断：`curl http://127.0.0.1:<port>/v1/models` 有响应即可用，启动加载模型约需 1-3 分钟。

## 第 4 步：真模型验证

**验证门槛 2（e2e 测试）：**

```bash
# 服务路线：
export MINERU_TEST_VLM_BACKEND=http-client
export MINERU_TEST_VLM_SERVER_URL=http://127.0.0.1:30000
# 或免服务路线（GPU 本机直推，慢但零服务）：
# export MINERU_TEST_VLM_BACKEND=transformers

pytest tests/unittest/test_vlm_stages_e2e.py -q -o addopts="" -v
# 预期：4 passed（覆盖默认不变性、模式b、模式c一致性、异步一致性）
```

**演示命令（按需）：**

```bash
# 模式 b：PP-DocLayoutV2 layout + MinerU VLM 识别
python demo/vlm_two_stage_demo.py --pdf demo/pdfs/demo1.pdf --mode b \
    --backend http-client --server-url http://127.0.0.1:30000

# 全替换链路：小模型 layout + OvisOCR2 识别（不含 MinerU 原模型）
python demo/vlm_two_stage_demo.py --pdf demo/pdfs/demo1.pdf --mode b \
    --recognizer ovis --ovis-url http://127.0.0.1:8000

# 可视化任一 layout.json
python demo/vlm_layout_visualize.py --pdf demo/pdfs/demo1.pdf \
    --layout output/vlm_two_stage_demo/demo1/b/layout.json \
    --output output/vis
```

## 第 5 步（可选）：PP-DocLayoutV3 链路

```bash
# 环境 ③（独立 venv；GPU 版 paddlepaddle 见 paddlepaddle.org.cn/en/install）
python -m venv ~/paddle-env && source ~/paddle-env/bin/activate
pip install paddlepaddle paddleocr pypdfium2 pillow
python demo/pp_doclayoutv3_layout.py --pdf demo/pdfs/demo1.pdf --output output/v3/layout.json

# 回到环境 ①：V3 layout + OvisOCR2 识别
python demo/vlm_two_stage_demo.py --pdf demo/pdfs/demo1.pdf --mode c \
    --layout-json output/v3/layout.json --recognizer ovis --ovis-url http://127.0.0.1:8000
```

## 已知坑速查

| 症状 | 处理 |
|---|---|
| pytest 报 `unrecognized arguments: --cov` | 加 `-o addopts=""` |
| 模型下载慢/失败 | `MINERU_MODEL_SOURCE=modelscope`；Ovis/HF 用 `HF_ENDPOINT=https://hf-mirror.com` |
| 显存不足（<8G） | 两个服务别共存；MinerU 侧改 transformers 路线 |
| Windows 起 vllm 失败 | vLLM 不支持原生 Windows，进 WSL2 |
| e2e 测试全 skip | 未设 `MINERU_TEST_VLM_BACKEND`（这是设计行为） |
| 想缩小处理批次 | `MINERU_PROCESSING_WINDOW_SIZE=<页数>`（默认 64） |

## 约定

- 默认路径（不传 layout_detector/content_recognizer）必须与上游 MinerU 行为完全一致；改动 `vlm_analyze.py` 时不得触碰默认分支。
- layout.json schema 变更需同步：`stages.py`、`pp_doclayoutv3_layout.py`（静态复制）、契约测试 `test_pp_doclayoutv3_export.py`。
- 提交前跑验证门槛 1；动了识别/解析逻辑再跑门槛 2。
