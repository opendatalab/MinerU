# MinerU 测试职责

原生解析、共享 Schema、确定性后处理、渲染与导出的完整算法回归已迁入 DocVortex。配套迁移清单和验证结果位于 DocVortex 仓库的 `docs/validation/test-migration.md`、`docs/validation/test-migration.json`。

本仓库保留 OCR/Hybrid/VLM 编排、Flash 模式路由、CLI/API/Doclib、产品元数据与协议适配、渲染参数传递、旧宿主路径移除、LLM 增强及 Content List V1/V2 测试。需要小型 Office 输入的集成测试继续使用本地构造器，不从 DocVortex 的测试目录导入代码。

## 本地运行

```sh
.venv4/bin/python -m pytest tests/unittest -o addopts= -q -m 'not torch'
.venv4/bin/python -m pytest tests/unittest -o addopts= -q -m 'torch and not accelerator'
```

第二条命令需要安装 `.[test,torch]`。`torch` 标记用于真实依赖 Torch/Transformers 的测试；仅模拟设备策略的测试仍属于基础集合。
`accelerator` 标记用于必须访问实际 MPS/XPU 等设备的测试，独立 CPU CI 不运行这些用例，本地可用 `-m accelerator` 验证。

## CI 分层与报告

- 上游 `master/dev` 的 push、PR 和手动触发运行 `mineru-cli-test`，fork 的主测试保持跳过。
- 代码测试保留 Linux Python 3.11–3.14 全量基础回归，以及 Windows/macOS 的公开接口矩阵；Linux Python 3.12 单独采集覆盖率。
- Torch CPU job 显式安装 CPU 依赖并验证导入，执行所有 `torch and not accelerator` 用例。
- 纯文档变更跳过代码矩阵，`CI result` 始终汇总适用任务；分支保护可把该名称设置为 required check。
- 覆盖率使用 pytest-cov 的原生 `--cov-fail-under=58 --cov-precision=2` 门槛。2026-10-24 本地 Python 3.12 无 Torch 环境实测 59.87%，为平台差异保留余量，并避免整数四舍五入影响门槛判断。
- 每个测试 job 输出 JUnit XML 和最慢 30 项，覆盖率 job 额外输出 XML；成功或失败均保留报告 14 天。
- 全仓 Ruff 先检查 `E9,F63,F7,F82`，新增 CI 脚本和 `tests/ci` 执行项目完整规则及格式检查；其余存量规则逐步治理。

CI 脚本的轻依赖验证不加载产品测试的自动 fixture：

```sh
.venv4/bin/python -m pytest tests/ci --confcutdir=tests/ci -o addopts= -q
actionlint
```

文档 job 只安装 `docs/requirements.txt`；HTTP 示例回归由 Linux 基础套件执行。发布文档时直接上传已经通过 strict build 的 `site/`，不再次构建。

## 发布与 DocVortex 依赖

正式发布前先提交 `mineru/version.py` 的目标版本，再在该提交创建 `mineru-<version>-released` 标签。
构建和测试只检出事件的固定 SHA；标签和源码版本不一致会立即失败，不再自动改写或推送 `master`。
完整 CI 与跨平台 wheel 安装检查通过后才发布同一份 wheel。GitHub Release 附带 `release.json`，记录源码 SHA、版本、标签和 wheel SHA-256。

MinerU CI 按 `pyproject.toml` 的版本约束直接安装 PyPI 上的 DocVortex wheel，使用 `--only-binary=docvortex` 禁止回退到源码构建。
DocVortex 的源码构建、原生协议和产物身份验证由其自身仓库负责；MinerU 保留常规测试中的导入边界、共享协议、路由及真实文档集成回归。

## Hybrid 输入与原始语料

`tests/unittest/pdfs` 已移除，MinerU 的真实 PDF 集成输入继续使用 `demo/pdfs/demo1.pdf`、`demo2.pdf`。完整 PDF/Office/EMF+ 语料、表格评测器和合成 PDF 生成器由 DocVortex 维护。

- `fixtures/hybrid_native_script_inputs.json`：原有 33 组字符与几何快照，内容及历史来源信息保持不变。
- `fixtures/hybrid_pdf_script_inputs.json`：新增四个 CJK 探针和一个旋转页场景，包含原始字符、字体、索引、tight/origin 几何及所需页面属性。
- `fixtures/pdf_mixed_font_script_line.json`：继续覆盖 Hybrid 的真实混合字体回填行为。
- `fixtures/rtf/semantic.rtf`：保留产品接入测试仍使用的小型输入。

普通测试只读取本仓库快照。显式再生新增快照时，由调用者提供 DocVortex 源码目录：

```sh
.venv4/bin/python tests/fixtures/capture_hybrid_native_script_inputs.py \
  --fixture tests/fixtures/hybrid_pdf_script_inputs.json \
  --source-root /path/to/DocVortex --check
```

不带 `--check` 时会写回指定快照。旧夹具仍使用原默认 `--fixture` 和 DocVortex 的 `demo/pdfs` 作为 source root。再生前先核对源文件哈希；不要为消除回归差异覆盖已有真值。
