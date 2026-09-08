# Transformers 5 推理栈

本分支只支持 `transformers>=5.10.1,<6`，回归锚点为 5.10.1、5.14.0 与 5.16.1；macOS full/all 的 MLX-VLM 0.7 只允许 >=5.14。
`mineru-vl-utils` 配套版本为 2.0.0；该版本提高框架最低版本，并将 LMDeploy 参数契约更新为公开的 `Pipeline`。
基础安装仍使用轻量模型栈，不安装或提前导入 Torch/Transformers。

## 依赖和实现边界

| 组件 | 范围 |
| --- | --- |
| Transformers | `>=5.10.1,<6` |
| Tokenizers | `>=0.22.0,<0.24`；5.10.1 与 5.16.1 由解析器选择不同兼容版本 |
| PyTorch | `>=2.6,<3`，TorchVision 与其配套 |
| vLLM | `>=0.19.1,<0.29.0` |
| LMDeploy | `>=0.17.0,<0.18` |
| MLX-VLM | `>=0.7.0,<0.8.0`，MLX 版本由其依赖决定 |
| mineru-vl-utils | `>=2.0.0,<3` |

- UniMERNet 保留现有权重、词表、Q/K 压缩、token 上限及分批规则。组合配置显式构造 Swin/MBart，不修改全局 Auto 注册表。
- 自注意力和交叉注意力使用 `Cache`/`EncoderDecoderCache`；权重初始化使用 Transformers 5 的保护机制，防止已加载参数被重新随机初始化。
- PP-DocLayoutV2 复用官方模型，保留 MinerU 的预处理、阈值、类别、阅读顺序和后处理。MPS 的正弦位置编码在 CPU 调用官方实现，然后送回模型设备，避开 float64 限制。
- Qwen2-VL 从文本子配置读取上下文长度，并合并生成配置、文本配置及 tokenizer 的特殊 token。
- LMDeploy 同步/异步客户端使用同一个 `Pipeline.infer`。异步调用在线程中执行；取消时等待在途同步调用结束后再释放名额，防止共享引擎被提前卸载。
- vLLM 客户端不再安装依赖已删除 `AnyTokenizer` 的导入时全局 logprobs 补丁。

## 本地联合验证

utils 2.0.0 正式发布前，先构建 DocVortex 0.3.0、utils 2.0.0 和 MinerU 三个工作树的 wheel，并在独立环境中一起安装，避免解析到旧版 DocVortex 或 utils：

```bash
# 在 mineru-vl-utils 工作树中
uv build --wheel

# 在 MinerU 工作树中
uv build --wheel
uv venv .venv-validation
uv pip install --python .venv-validation/bin/python \
  /path/to/docvortex/dist/docvortex-0.3.0-py3-none-any.whl \
  /path/to/mineru-vl-utils/dist/mineru_vl_utils-2.0.0-py3-none-any.whl \
  './dist/mineru-4.0.0b1-py3-none-any.whl[torch]' \
  'transformers==5.16.1' pytest
uv pip check --python .venv-validation/bin/python
```

安装矩阵直接读取已构建 wheel 的元数据，覆盖三个平台、Python 3.10–3.14 与基础/torch/full/all 模式：

```bash
python scripts/check_transformers5_dependencies.py \
  --mineru-wheel dist/mineru-4.0.0b1-py3-none-any.whl \
  --utils-wheel /path/to/mineru_vl_utils-2.0.0-py3-none-any.whl \
  --docvortex-wheel /path/to/docvortex-0.3.0-py3-none-any.whl --utils-matrix --check-wheels \
  --platform linux --transformers 5.10.1 --output dependency-matrix.json
```

Linux 解析目标使用 glibc 2.34，Apple Silicon 使用 macOS 14；这是所选二进制依赖的安装基线。
元数据可解不代表 GPU 推理已通过。

真实模型验证工具为 `scripts/validate_transformers_migration.py`。输入清单包含
`pages`（每项有 `name`、`pdf`、一基 `page`、渲染后 `image` 路径）及 `formulas`（固定公式裁剪路径列表）。
同一清单分别交给保存的 4.57.6 基线源码和迁移源码，输出模型结果、Markdown/MiddleJson、耗时及内存采样：

```bash
python scripts/validate_transformers_migration.py \
  --repo . --manifest /path/to/inputs.json --kind parse --tier basic \
  --device mps --rounds 3 --output validation/parse-basic.json
```

`layout`/`mfr` 验证独立模型；`vlm-transformers`/`vlm-mlx` 验证真实 VLM 的短生成链路。
VLM 组件 smoke 将图像最长边限制为 448 像素，不代替 `parse` 场景的原分辨率整页质量验收。
内存报告中的 `rss_bytes` 为主进程 RSS，`torch_device_bytes` 为每 20ms 采样的 Torch 活跃设备内存。
Transformers 5.16 使用 safetensors 的 MPS `pread` 路径，框架 allocator 计数不包含所有权重存储；RSS 与 GPU allocator 数值不能直接跨加载方式比较。
macOS 同时记录系统 `proc_pid_rusage` 的 `physical_footprint_bytes`，用于统一内存的真实占用比较。保留 RSS 和框架计数供诊断，不把它们误报为 MLX 或外部 MPS 存储的 GPU 峰值。

## CI 与发布门槛

`transformers5.yml` 对 next/PR 运行模型图、增量缓存、重载、惰性导入、HTTP VLM 与生命周期回归，并解析安装矩阵。
手动触发时可传入 `utils-wheel-url` 与 `docvortex-wheel-url` 验证未发布的联合构建。utils 仓库也提供客户端双版本回归。

发布前需要：固定权重和输入的内容回归通过；同一设备预热后三轮耗时、内存/显存无超过 10% 的回退；Linux CUDA、Windows GPU、macOS 实机推理有对应证据。
单元测试替身和安装解析不替代实机结果。旧国产硬件镜像继续使用旧发布系列，单独迁移。
当前工作按用户要求暂缓发布，不据此文档声明所有平台已经通过。


## Python 3.14 联合升级

- 三仓 Requires-Python 均为 `>=3.10,<3.15`；不包含 free-threaded ABI。
- MinerU 依赖 `docvortex>=0.3.0,<1`，共享纯字符边界拼接，完整移除 fast-langdetect/fasttext-predict 下载链。
- 同步 vLLM.generate 自行渲染 raw prompt；异步入口保留 EngineInput。corex CompilationConfig 使用 mode，不使用旧 level。
- MLX 同模型生成串行化，取消任务时等待在途线程结束；异步模型锁等待不在后台遗留 acquire 线程。
- 检查脚本显式验证 wheel 的 Requires-Python 和每个目标 ABI 的发行文件；jieba 为唯一允许的纯 Python sdist。
- macOS full/all + Transformers 5.10.1 应拒绝；Python 3.14 + vLLM 0.19.1 应拒绝，其余范围由解析器选择兼容版本。
- 发布顺序为 DocVortex、utils、MinerU。本次只做本地合入、构建与验证，未发布依赖必须显式提供三个 wheel。
