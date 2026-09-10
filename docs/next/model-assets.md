# MinerU 4 模型资产

`mineru/model/registry.py` 定义仓库、必需文件和档位映射；`download.py` 负责显式下载、目录锁和完整性检查。
两个资源包常量为 `MINERU_4_MODELS_TORCH` 和 `MINERU_4_MODELS_ONNX`；
`mineru_4_models_for_stack()` 按 stack 选择资源包。

| Stack | basic | standard |
|---|---|---|
| full | MinerU-4_models_torch | Torch 资源包 + MinerU2.5-Pro-2605-1.2B |
| light | MinerU-4_models_onnx | ONNX 资源包 + 现有 Q8_0 GGUF 和 mmproj |

新资源包使用各自的新缓存目录，不复用 PDF-Extract-Kit-1.0 或原来分散的 ONNX 仓库缓存。
当前只配置 Hugging Face 来源，ModelScope 镜像发布并核对文件后再添加来源地址。

## 下载与离线运行

```bash
export MINERU_MODEL_STACK=light
export MINERU_MODEL_SOURCE=huggingface
mineru-kit models download --tier standard --stack light --source huggingface
mineru-kit models verify --tier standard --stack light

export MINERU_MODEL_SOURCE=local
mineru-kit parse input.pdf --tier basic -o output.md
```

`MINERU_MODEL_BASE_DIR` 可指定独立模型根目录。`--stack` 只覆盖当前模型管理命令；解析和服务运行时读取环境变量或配置文件。
所有 ONNX 会话固定使用 CPU，最低版本为 ONNX Runtime 1.20.1，支持资源包所需的 ONNX IR 10。
llama.cpp 的设备参数独立于 ONNX provider；full 的 Torch 和 VLM 仍沿用各自设备策略。

## 资源结构

两个资源包均使用 `Layout/`、`OCR/paddleocr/`、`MFR/pp_formulanet_plus_m/`、`Table/`。

- Layout：PP-DocLayoutV2。ONNX 版读取官方 `inference.yml`，按调用方 batch 合并固定尺寸输入，并用逐页框数量拆分结果，保持页序。
- OCR：PP-OCRv6 Tiny Det + Small Rec。OCR TXT 字符表位于 `mineru/model/ocr/data/ppocrv6_dict.txt`，Torch/ONNX 共用。
- Seal：PP-OCRv4 mobile detector，使用多边形后处理、弯曲文字裁剪和共享 Small Rec。没有普通 OCR 替代 seal 的隐式降级。
- MFR：PP-FormulaNet plus-M，共用预处理、tokenizer 和公式修复；ONNX 自回归循环包含在图中；CPU 按原图面积排序，以最多 8 张一次合批推理，并恢复原顺序。
- Table：SlanetPlus、Unet 和 PP-LCNet 表格分类文件在两仓库中逐字节一致。

OCR YAML 保留上游配置，`MinerU` 字段另外记录普通 OCR 的实际阈值、宽度边界和字典位置。调用方传入的表格专用检测参数仍显式覆盖默认值。

## 可重复准备与验证

转换依赖只安装在开发环境，不属于 MinerU 的运行依赖：

```bash
uv pip install --python .venv1/bin/python 'onnx>=1.17,<2'
.venv1/bin/python -m scripts.prepare_onnx_models prepare --output-dir output/model-migration/models
.venv1/bin/python -m scripts.prepare_onnx_models verify \
  --repo-dir output/model-migration/models/MinerU-4_models_onnx
```

脚本使用固定源 revision；下载可利用 HF 缓存恢复。Seal 使用公开 `torch.onnx.export` API 导出 FP32、opset 17、动态 batch/高/宽概率图。
`manifest.json` 记录来源、哈希、大小、ONNX 签名、工具版本和导出验证。权重和验证产物保存在忽略的 `output/` 目录中。

组件验证示例：

```bash
.venv1/bin/python -m scripts.validate_onnx_models seal seal.png \
  --output-dir output/model-migration/validation/seal
.venv1/bin/python -m scripts.validate_onnx_models layout page1.png page2.png \
  --output-dir output/model-migration/validation/layout
.venv1/bin/python -m scripts.validate_onnx_models mfr formula1.png formula2.png \
  --output-dir output/model-migration/validation/mfr
```

使用独立 core 环境验证 light，而不是通过隐藏已安装的重依赖模拟：

```bash
uv venv output/model-migration/core-venv
uv pip install --python output/model-migration/core-venv/bin/python -e .
MINERU_INTRA_OP_NUM_THREADS=2 output/model-migration/core-venv/bin/python \
  -m scripts.validate_onnx_models parse input.pdf --stack light --tier basic \
  --rounds 2 --output-dir output/model-migration/validation/light-basic
```

验证报告分别保存初始化、首次/重复推理时间、进程峰值 RSS、结果包和叠图。公式逐项比较文本；版面检查页序、标签、几何和阅读顺序。

Seal 首先以 `rtol=1e-4, atol=1e-5` 比较概率图。若超限，报告保留失败标记并追加 Torch FP64 参考误差，
同时严格要求检测阈值掩码、多边形、裁剪图和最终文本一致；不会静默扩大概率图容差。完整验收结论应同时说明数值结果与语义结果。

发布按“本地验证 → HF 上传 → 新目录真实下载 → 本地来源解析”执行。ModelScope 镜像必须复制相同文件并核对哈希，不能指向旧模型仓库。

## Plus-M PTH 到 ONNX 对照实验

`scripts.export_formula_onnx` 从当前 Torch 资源包的 PTH 导出 CPU FP32 模型：
先导出编码器和单 token KV 缓存解码步骤，再用 ONNX Loop 组成独立的 image → token IDs 模型。
导出保留动态 batch、动态序列长度、EOS/PAD 和当前 Torch 路径的强制 EOS 行为。
导出中间产物写到独立目录；正式资源包采用这份经过验证的 PTH 导出模型。

```bash
.venv1/bin/python -m scripts.export_formula_onnx \
  --weights-dir output/model-migration/models/MinerU-4_models_torch/MFR/pp_formulanet_plus_m \
  --output-dir output/formula-onnx-comparison

.venv1/bin/python -m scripts.compare_formula_onnx \
  --reference output/model-migration/models/MinerU-4_models_onnx/MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M.onnx \
  --candidate output/formula-onnx-comparison/PP-FormulaNet_plus-M.from_torch.onnx \
  --config output/model-migration/models/MinerU-4_models_onnx/MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M_inference.yml \
  --weights-dir output/model-migration/models/MinerU-4_models_torch/MFR/pp_formulanet_plus_m \
  --inputs-json output/model-migration/formula-inputs.json \
  --output-dir output/formula-onnx-comparison
```

`--inputs-json` 接收公式图像路径的 JSON 数组。报告分别记录完整 token、LaTeX、原始权重按值匹配、
图节点/算子统计、编码器特征与前 16 步 logits 的三方数值对照，以及 batch=2 与逐项推理的结果。
权重匹配递归访问控制流子图，并考虑 Paddle/Torch Linear 权重的转置；未匹配项不能直接解释为权重不同。

`scripts.check_formula_generation_limit` 使用同样的 `--reference`、`--candidate` 和 `--output-dir` 参数，
在诊断副本中固定 argmax 为非 EOS token，保留原图的序列更新和停止逻辑，验证超长序列边界。
当前 Torch 的 `ForcedEOSTokenLogitsProcessor(max_length=1537)` 会在第 1536 个生成 token 强制 EOS；
现用 Paddle 来源 ONNX 没有这一强制步骤，最多生成 2560 个 token。
因此短公式结果一致不足以证明两个导出版本在超长公式上等价。

`scripts.benchmark_formula_onnx` 用 `--images-dir` 指定含 `manifest.json` 的公式样本目录，
用 `--reference`、`--candidate`、`--config`、`--output-dir` 指定两个模型、字典配置及输出目录。
它将原始 36 张和 demo1/demo2 的 117 张分开，按原图像素面积升序排列，测试真实 batch=1/2/4/8/16。
每个配置在独立 CPU ORT 进程中预热后测两轮，固定 intra-op=2、inter-op=1；配置间交替模型先后次序。
计时只包含 `session.run`，原始数据同时保存逐批耗时、EOS 截断后的 token、LaTeX 和进程峰值 RSS。
生产公式 CPU 封装采用 batch=8 上限；调用方请求更小批次时仍遵守其上限。

## 其他 ONNX 模型的 batch 现状

- Layout 图和封装均支持真 batch；通过逐页框数量拆分合并输出，再独立执行过滤与阅读顺序处理。
- OCR det 按预处理后完全相同的尺寸分桶，桶满或处理尾批时一次调用 ORT；不补边，结果恢复原输入顺序。
- OCR rec 图和封装都支持真实批量，默认 batch=6，按宽高比排序并将同批文本行补齐到共同宽度。

模型文件更新不会自动替换已标记完成的本地缓存。已有缓存需要更新对应权重，或使用新的模型目录下载。

Layout 与 OCR det 保留现有批次参数，PDF 主流程当前上限分别为 2 和 16；OCR det 尺寸不匹配时自然形成不同的桶，实际批次可小于上限。
单图检测共用批处理路径；无效批次上限、Layout 数量数组异常或检测输出 batch 不匹配时明确报错。

真批处理验证：

```bash
MINERU_INTRA_OP_NUM_THREADS=2 MINERU_INTER_OP_NUM_THREADS=1 .venv1/bin/python \
  -m scripts.validate_detection_batching components --output-dir output/detection-batching/components
MINERU_INTRA_OP_NUM_THREADS=2 MINERU_INTER_OP_NUM_THREADS=1 .venv1/bin/python \
  -m scripts.validate_detection_batching parse demo/pdfs/demo1.pdf demo/pdfs/demo2.pdf \
  --output-dir output/detection-batching/parse
```

验证脚本记录实际 ORT 输入形状，比较逐图/合批结果，并输出检测叠图、真实解析结果、耗时及进程峰值 RSS。
