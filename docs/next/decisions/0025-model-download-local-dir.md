# ADR-0025: 模型下载与本地模型目录

状态: Accepted
日期: 2026-07-13
相关文档:
- ../cli/mineru-kit-models.md
- ../config.md
- 0019-mineru-kit-models-command.md

本 ADR supersedes ADR-0019 中关于模型配置落点、`pipeline` / `vlm` / `all` bundle 命令语义，以及下载后写入 `mineru.json` 的决策。ADR-0019 中关于建立 `mineru-kit models` 命令组的总体方向仍保留。

## 背景

MinerU 当前模型下载逻辑仍依赖旧 `mineru.json` 中的 `models-dir` 和 `model-source` 字段。下载完成后，MinerU 会记录 Hugging Face 或 ModelScope 返回的模型根目录。这个设计有几个问题:

- 模型路径/source 配置与新的 `config.yaml` 配置体系割裂。
- 下载结果被写入配置文件，配置文件既表达用户意图，又混入运行时探测结果。
- Hugging Face 和 ModelScope 默认 cache 目录结构不同，难以形成统一的本地模型库语义。
- 首次使用本地 managed parser 时，模型是否已准备好不容易在配置阶段判断。
- `mineru-kit models verify`、managed tier readiness 检查、parse 懒下载需要共享一套模型定位和校验规则。
- `pipeline` / `vlm` 是旧实现分组名，不是用户真正关心的模型库或解析能力；继续暴露这些 bundle 会让模型下载命令和 tier 语义脱节。

经过验证，Hugging Face 和 ModelScope 都支持 `local_dir`，并且可以把同一模型仓库展开到同一个目录。两者的 provider 元数据不会互相覆盖:

- Hugging Face 使用 `.cache/huggingface/...`
- ModelScope 使用 `.msc`、`.mv`、`._____temp`

在不并发写同一目录的前提下，两个 provider 可以顺序 materialize 到同一个 `local_dir`。payload 文件最终只存储一份；但切换 provider 时仍可能重复下载大文件，因此不能把该目录视为 provider 共享 cache。

## 决策

### 1. 模型配置迁移到 `config.yaml`

在顶层 `Config` 中增加 `model` 配置，与 `doclib` 同级:

```yaml
model:
  base_dir: ${MINERU_HOME:-~/.mineru}/models
  source: auto
```

`model.source` 支持以下取值:

| 值 | 语义 |
|----|------|
| `auto` | 探测网络可达性后选择 `huggingface` 或 `modelscope` |
| `huggingface` | 强制使用 Hugging Face 下载 |
| `modelscope` | 强制使用 ModelScope 下载 |
| `local` | 禁止 parse 懒下载和自动下载，只使用已经存在且 ready 的本地模型；显式 `mineru-kit models download` 例外 |

`MINERU_MODEL_SOURCE` 作为 `config.model.source` 的环境变量覆盖项，沿用 `config.py` 现有环境变量合并规则。

### 2. `auto` 探测结果写回当前配置文件

当 `model.source=auto` 时，MinerU 会探测当前环境应使用 Hugging Face 还是 ModelScope。探测成功后，可以把实际来源写回当前上下文中的配置文件:

- 如果 `MINERU_CONFIG` 指向自定义配置文件，则写入该文件。
- 否则写入默认 `~/.mineru/config.yaml`。

写回只更新:

```yaml
model:
  source: huggingface
```

或:

```yaml
model:
  source: modelscope
```

下载结果、模型路径和 provider 返回的 snapshot/cache 路径不写入配置文件。

当前进程应继续使用本次解析出的 source，不依赖重新 import `config` 后再生效。

### 2.1 配置文件原子写入与来源追踪

`config.py` 需要提供通用配置文件写入和来源查询能力，而不是只为 `model.source` 写专用逻辑。

新增能力:

```python
ConfigSource = Literal["default", "file", "env"]

def get_config_source(path: str | Sequence[str]) -> ConfigSource:
    ...

def get_config_file_path() -> str:
    ...

def get_config_file_exists() -> bool:
    ...

def update_config_file(patch: dict[str, Any]) -> None:
    ...
```

来源追踪规则:

- 内置默认值来源为 `default`。
- YAML 文件中出现的叶子字段来源为 `file`。
- 环境变量覆盖的叶子字段来源为 `env`。

例如 `get_config_source("model.source")` 返回 `default`、`file` 或 `env`。

未设置 `MINERU_CONFIG` 且默认配置文件不存在时，默认路径仍是当前配置上下文路径，读取结果等价于空配置文件。`get_config_file_path()` 返回默认路径，`get_config_file_exists()` 返回 `False`。如果显式设置 `MINERU_CONFIG` 且文件不存在，仍应报错，避免把拼写错误静默变成新配置文件。

`update_config_file` 必须原子写入:

- 对目标配置文件加锁，例如 `{config_path}.lock`。
- 读取现有 YAML；缺失或为空时按 `{}` 处理。
- deep merge patch。
- 写到同目录临时文件。
- 使用 `os.replace(tmp, config_path)` 原子替换。
- 保留未知字段。

写入目标:

- 如果 `MINERU_CONFIG` 指向自定义配置文件，写该文件。
- 否则写默认 `~/.mineru/config.yaml`。

`auto` 探测后的写回规则:

- 只有 `model.source` 的来源是 `default` 或 `file`，且有效值为 `auto` 时，才写回 resolved source。
- 如果 `model.source` 来源是 `env`，即使环境变量值是 `auto`，也不写回配置文件。
- 如果 `model.source=local`，显式 download 命令可临时按 auto 下载，但不得把 `local` 覆盖掉。

### 3. 模型目录由 MinerU 管理

所有 MinerU 使用的模型仓库都展开到:

```text
{config.model.base_dir}/{spec.local_name}
```

例如:

```text
~/.mineru/models/PDF-Extract-Kit-1.0
~/.mineru/models/MinerU2.5-Pro-2605-1.2B
```

Hugging Face 和 ModelScope 共用同一个 `local_dir`。该目录语义是 MinerU 管理的模型展开目录，不是 Hugging Face / ModelScope 的共享 cache。

### 4. 移除模型路径/source 的 `mineru.json` 依赖

新模型下载功能不再读取或写入 `mineru.json` 中的:

- `models-dir`
- `model-source`

`mineru.json` 不再提供任何与模型路径或模型 source 有关的配置。仍需要保留的旧工具配置应与模型下载逻辑解耦。

### 5. 增加模型 registry，并移除 bundle 概念

新增:

```text
mineru/utils/model_registry.py
```

用于集中描述模型仓库和 repo 内部路径。

```python
@dataclass(frozen=True)
class ModelPath:
    repo: ModelRepo
    name: str
    relative_path: str

    def ensure(self, *, source: ModelSource | None = None) -> Path:
        ...

    def local_path(self) -> Path:
        ...

    def path(self, relative_path: str, /, *children: str) -> ModelPath:
        ...


@dataclass(frozen=True)
class ModelRepo:
    name: str
    local_name: str
    repos: dict[str, str]
    paths: dict[str, str]
    download_mode: Literal["full", "required_paths"] = "full"

    def ensure(self, *, source: ModelSource | None = None) -> Path:
        ...

    def path(self, name: str) -> ModelPath:
        ...

    def __getattr__(self, name: str) -> ModelPath:
        ...
```

不再使用 `pipeline` / `vlm` 作为用户可见 bundle 名。registry 使用模型仓库原名:

- `PDF-Extract-Kit-1.0`
- `MinerU2.5-Pro-2605-1.2B`

`PIPELINE_MODEL_PATHS` 和 `VLM_MODEL_MARKERS` 迁入 registry，避免 core 下载逻辑依赖 `mineru/kit/common.py`。旧 `ModelPath` 类中的模型仓库 root 和 repo 内路径也迁入 registry；`mineru/utils/enum_class.py` 不再承载模型路径配置。

示例:

```python
PDF_EXTRACT_KIT = ModelRepo(
    name="PDF-Extract-Kit-1.0",
    local_name="PDF-Extract-Kit-1.0",
    repos={
        "huggingface": "opendatalab/PDF-Extract-Kit-1.0",
        "modelscope": "OpenDataLab/PDF-Extract-Kit-1.0",
    },
    paths={
        "pp_doclayout_v2": "models/Layout/PP-DocLayoutV2",
        "unimernet_small": "models/MFR/unimernet_hf_small_2503",
        "pytorch_paddle": "models/OCR/paddleocr_torch",
        "slanet_plus": "models/TabRec/SlanetPlus/slanet-plus.onnx",
        "unet_structure": "models/TabRec/UnetStructure/unet.onnx",
        "paddle_table_cls": "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx",
    },
)
```

调用方可以使用绑定 repo 的路径对象:

```python
PDF_EXTRACT_KIT.ensure()
PDF_EXTRACT_KIT.pp_doclayout_v2.ensure()
PDF_EXTRACT_KIT.path("pp_doclayout_v2").ensure()
```

`PDF_EXTRACT_KIT.pp_doclayout_v2` 不提供静态 type hint。为保持实现简单，第一版接受 `__getattr__` 形式，并同时提供 `path(name)` 作为显式入口。

`ModelRepo.path()` 和 `ModelPath.path()` 都返回绑定同一 repo 的 `ModelPath`，用于支持更细粒度的局部下载。`path()` 参数语义是相对路径片段，不混用 registry path key。

示例:

```python
PDF_EXTRACT_KIT.path("models/OCR/paddleocr_torch").ensure()
PDF_EXTRACT_KIT.pytorch_paddle.path(det_model_name).ensure()
PDF_EXTRACT_KIT.pytorch_paddle.path(rec_model_name).ensure()
```

### 6. Tier 到模型仓库的映射

模型下载命令不再使用 `pipeline` / `vlm` bundle。需要下载哪些模型由 tier 决定:

```python
REPOS_FOR_TIER = {
    "flash": (),
    "medium": (PDF_EXTRACT_KIT,),
    "high": (PDF_EXTRACT_KIT, MINERU_2_5_PRO_2605_1_2B),
    "xhigh": (PDF_EXTRACT_KIT, MINERU_2_5_PRO_2605_1_2B),
}
```

语义:

- `flash` 不需要本地模型下载。
- `medium` 需要 `PDF-Extract-Kit-1.0`。
- `high` 和 `xhigh` 需要 `PDF-Extract-Kit-1.0` 与 `MinerU2.5-Pro-2605-1.2B`。

repo registry 不反向依赖 tier。tier 只选择需要的 repo 集合，repo 自身仍只描述模型仓库。

repo 同时声明默认下载策略:

- `download_mode="full"`: `download_model_repo(repo)` 下载完整仓库。
- `download_mode="required_paths"`: `download_model_repo(repo)` 只下载 `repo.required_paths()` 对应的文件/目录。

当前 `PDF-Extract-Kit-1.0` 使用 `required_paths`，避免拉取仓库中未被 runtime 使用的大量历史/可选权重。`MinerU2.5-Pro-2605-1.2B` 保持 `full`，因为 VLM 模型加载通常需要完整 tokenizer/config/weights 文件集；整仓 readiness 由仓库根目录的 `.mineru_complete` 表示，不再维护逐文件 paths 清单。

### 7. 下载 API

提供两个公开下载 API:

```python
def download_model_repo(repo: ModelRepo, *, source: ModelSource | None = None) -> Path:
    ...

def download_model_files(repo: ModelRepo, paths: Sequence[str | ModelPath], *, source: ModelSource | None = None) -> Path:
    ...
```

内部使用一个统一实现，避免整仓下载和局部下载逻辑分叉:

```python
def _download_model(
    repo: ModelRepo,
    *,
    paths: Sequence[str | ModelPath] | None,
    source: ModelSource | None,
) -> Path:
    ...
```

规则:

- `download_model_repo` 按 `repo.download_mode` 下载整仓或 required paths，并验证 repo 的完整 required paths。
- `download_model_files` 只下载指定子集，并只验证本次请求的 paths。
- 文件路径直接加入 `allow_patterns`。
- 目录路径展开为 `path` 和 `path/*`。
- Hugging Face 与 ModelScope 都通过 `snapshot_download(..., local_dir=...)` 实现。
- 第一版不为单文件下载分叉到 `hf_hub_download` 或 `model_file_download`。

### 8. 移除旧兼容 wrapper

不再保留旧函数:

```python
auto_download_and_get_model_root_path(relative_path: str, repo_mode: str = "pipeline") -> str
```

原因:

- 当前内部调用点已迁移到 registry-aware API。
- `repo_mode="pipeline"` / `repo_mode="vlm"` 会继续泄漏旧 bundle 名。
- 旧 wrapper 返回 repo root，调用方再拼接相对路径，容易重新引入路径拼接错误。

代码应直接使用:

```python
PDF_EXTRACT_KIT.pp_doclayout_v2.ensure()
PDF_EXTRACT_KIT.ensure()
```

### 9. Readiness 校验

模型是否 ready 由 MinerU 自己判断，不依赖 provider 元数据。

提供校验函数:

```python
def verify_model_repo(repo: ModelRepo) -> ModelReadyResult:
    ...

def is_model_repo_ready(repo: ModelRepo) -> bool:
    ...

def verify_model_tier(tier: Tier) -> ModelReadyResult:
    ...
```

校验规则:

- `model.source=local` 时禁止下载，只做 ready 检查。
- `mineru-kit models verify <repo>` 检查指定 repo 的完整 required paths。
- `mineru-kit models verify --tier <tier>` 检查该 tier 需要的 repo 集合。
- managed local parse server 的 tier readiness 检查基于 `REPOS_FOR_TIER`。
- partial download 只验证本次请求 paths，不要求整仓 ready。
- 最终文件存在时视为 ready；Hugging Face 和 ModelScope 都先写临时文件，完成后再移动到最终路径。
- `full` 模式在 ModelRepo 根目录写入空白 `.mineru_complete`，表示整仓 ready；`required_paths` 模式在每个目录型 ModelPath 下分别写入 marker，文件型 ModelPath 仍按最终文件是否存在判断。
- marker 只由 MinerU 在 provider 下载成功返回且下载后 payload 检查通过后创建。
- 下载或更新目录前删除旧 marker；下载失败、payload 缺失或进程中断时不创建 marker。
- provider 的 `.cache/huggingface`、`._____temp`、`.msc`、`.mdl` 和 `.mv` 等元数据或临时内容不能单独作为模型 payload。
- 旧版本留下但没有 marker 的目录按未确认完整处理；远端 source 会通过 provider 增量补齐并创建 marker，`local` source 则报告 not ready。

### 10. 并发安全

引入第三方依赖 `filelock`，并写入项目 dependencies。

每个模型目录对应一个锁文件:

```text
{config.model.base_dir}/.locks/{safe_local_name}.lock
```

锁保护整个模型 `local_dir`。以下操作必须在锁内执行:

- 整仓下载
- 局部文件下载
- 下载后的校验
- 需要防止与下载并发冲突的模型删除或清理操作

锁文件不放在模型 `local_dir` 内，避免与 provider 写入的 payload 和元数据混在一起。

等待策略:

- `mineru-kit models download` 默认无限等待锁。
- parse 懒下载默认等待锁，并记录正在等待模型下载锁。
- 第一版不新增锁 timeout 配置。

文件锁只能保护遵守 MinerU 锁协议的进程。用户手工修改模型目录或外部脚本直接写入同一目录，不在该锁的保护范围内。

### 11. `mineru-kit models` 命令语义

模型管理命令继续挂在 `mineru-kit models` 下，不新增 `mineru models` 子命令。

`mineru-kit models download` 支持按 tier 或按模型仓库下载:

```bash
mineru-kit models download --tier medium
mineru-kit models download --tier high
mineru-kit models download --tier xhigh
mineru-kit models download PDF-Extract-Kit-1.0
mineru-kit models download MinerU2.5-Pro-2605-1.2B
```

规则:

- repo 参数和 `--tier` 二选一。
- 两者都不传时报错，不默认下载大模型。
- 两者同时出现时报错。
- `--tier flash` 成功 no-op，并提示 flash 不需要本地模型下载。
- 不支持 repo alias；repo 参数只接受 registry 中的模型仓库主名。错误时列出可用 repo 名。
- 继续保留 `--source`。
- `--source` 只接受 `auto`、`huggingface`、`modelscope`；不接受 `local`。
- `--source` 未传时为 `None`，表示使用 `config.model.source` 的有效值。

source 解析规则:

| 场景 | 行为 |
|------|------|
| `--source huggingface` | 本次强制使用 Hugging Face |
| `--source modelscope` | 本次强制使用 ModelScope |
| `--source auto` | 本次自动探测 |
| 未传 `--source` 且 `config.model.source=huggingface` | 使用 Hugging Face |
| 未传 `--source` 且 `config.model.source=modelscope` | 使用 ModelScope |
| 未传 `--source` 且 `config.model.source=auto` | 自动探测，并可按 2.1 规则写回 |
| 未传 `--source` 且 `config.model.source=local` | 显式 download 命令临时按 auto 下载，但不写回覆盖 `local` |

除 `local` 外，配置或环境变量中的无效 `model.source` 值按 `auto` 处理并记录 warning。CLI `--source` 的无效值必须报错，避免用户 typo 被静默解释为 auto。

`MINERU_MODEL_SOURCE=auto` 是合法环境覆盖，但由于来源是 `env`，不会触发写回配置文件。

`mineru-kit models show` 展示:

- 当前配置文件路径
- `model.base_dir`
- `model.source`
- `MINERU_MODEL_SOURCE`
- 每个模型仓库的 expected local dir
- 每个 tier 需要的模型仓库
- readiness 状态

`mineru-kit models verify` 使用与 download 相同的目标选择语义:

```bash
mineru-kit models verify --tier high
mineru-kit models verify PDF-Extract-Kit-1.0
mineru-kit models verify
```

不带 repo 且不带 `--tier` 时，`verify` 检查全部 registry repo。`verify` 不读取 `mineru.json`。

## 替代方案

### 方案 A: 继续记录 provider 返回的 cache/snapshot 路径

未采用。

原因:

- Hugging Face 与 ModelScope 的 cache 结构不同。
- 配置文件会继续混入运行时下载结果。
- readiness 检查必须理解 provider cache 细节，不利于稳定。

### 方案 B: HF 和 ModelScope 使用不同本地目录

未采用。

原因:

- 同一模型会长期占用两份空间。
- 用户看到的本地模型库结构更复杂。
- 已验证顺序写同一个 `local_dir` 基本可行，只需加锁和 readiness 校验。

### 方案 C: 使用 provider 共享 `cache_dir`

未采用。

原因:

- HF 和 ModelScope 的 cache 元数据、目录布局和一致性协议不同。
- 共享 `cache_dir` 不是两边 SDK 的稳定契约。
- 本 ADR 只共享最终展开的 `local_dir`，不共享 provider cache。

### 方案 D: 自己实现跨平台锁

未采用。

原因:

- `fcntl` 与 `msvcrt` 需要分别处理，超时和异常释放逻辑容易出错。
- `filelock` 已是成熟小依赖，且 Hugging Face 生态中已有类似依赖使用经验。
- 模型下载锁不是 MinerU 的核心差异化能力，不值得维护自研跨平台锁。

### 方案 E: 彻底移除 `auto_download_and_get_model_root_path`

采用。

原因:

- 内部调用点已经迁移到 registry-aware API。
- 继续保留 wrapper 会保留 `pipeline` / `vlm` 旧概念。
- 删除 wrapper 后，模型路径解析只有 `ModelRepo` / `ModelPath` 一条主路径。

### 方案 F: 继续使用 `pipeline` / `vlm` bundle 作为下载目标

未采用。

原因:

- `pipeline` / `vlm` 是旧实现分组名，不是模型仓库名。
- 用户关心的是准备某个 tier 所需模型，或下载某个具体模型仓库。
- 继续暴露 bundle 会让 `medium/high/xhigh` 与模型准备动作之间多一层间接概念。
- 当前实际远端模型仓库只有两个，用真实仓库名表达更直接。

### 方案 G: 为每个 repo 编写强类型路径类或 decorator 语法

未采用。

候选形式包括:

```python
PDF_EXTRACT_KIT.pp_doclayout_v2.ensure()
```

并为 `pp_doclayout_v2` 提供完整静态 type hint，或通过 decorator 把类定义转换为 repo spec。

未采用的原因:

- 强类型 repo class 会让 registry 定义明显变复杂。
- decorator 会引入运行时转换，不符合本项目“确定性优先”的编码原则。
- 当前只有两个 repo，路径数量有限，使用 `ModelRepo` + `ModelPath` 已能显著降低调用端冗余。
- 第一版接受 `PDF_EXTRACT_KIT.pp_doclayout_v2` 无静态属性 type hint，并保留 `PDF_EXTRACT_KIT.path("pp_doclayout_v2")` 作为显式入口。

## 影响

### 对配置

- 模型路径和 source 迁移到 `config.yaml`。
- `MINERU_MODEL_SOURCE` 成为 `config.model.source` 的自然环境变量覆盖项。
- `mineru.json` 不再承载模型路径/source。

### 对下载行为

- HF/MS 下载到同一个 MinerU-owned `local_dir`。
- 最终模型 payload 只存储一份。
- 切换 provider 时可能重复下载大文件，这是可接受的 provider fallback 成本。
- 下载结果不再写入配置文件。

### 对 managed local parser

- 设置 `parse_server.local.mode=managed` 或 `parse_server.local.managed_tier` 时，可以基于 `config.model.base_dir` 做 readiness 检查。
- 如果模型未 ready，应提示用户显式执行 `mineru-kit models download ...`。
- 不改变 parse server 启动流程，不在 parse 阶段新增额外 readiness gate。

### 对实现边界

- `mineru/utils/model_registry.py` 成为模型仓库、repo 内路径和 tier-to-repo 映射的定义边界。
- `models_download_utils.py` 负责 source 解析、锁、下载和 ready 检查。
- `mineru/kit/common.py` 不再持有模型 required path 常量。
- `mineru/utils/enum_class.py` 不再持有 `ModelPath`；该文件暂时只保留仍被非模型下载逻辑使用的枚举/常量。

### 对用户体验

- 本地模型目录稳定、可解释。
- `mineru-kit models show/verify` 能直接展示模型仓库、tier 需求、expected local dir 和 readiness。
- `mineru-kit models download --tier high` 直接表达“准备 high tier”，不要求用户理解旧 `pipeline` / `vlm` 分组。
- 使用 `model.source=local` 时，离线部署行为更明确。
- 首次模型准备可以从“parse 时静默下载”转为“配置/下载命令显式准备”。

## 后续动作

1. 在 `mineru/config.py` 增加 `ModelConfig`。
2. 在 `mineru/config.py` 增加字段级来源查询和原子配置写入函数。
3. 增加 `mineru/utils/model_registry.py`，定义 `PDF_EXTRACT_KIT`、`MINERU_2_5_PRO_2605_1_2B`、repo 内 `ModelPath` 和 `REPOS_FOR_TIER`。
4. 重构 `mineru/utils/models_download_utils.py`，移除 `mineru.json` 模型路径/source 读写。
5. 引入 `filelock` 依赖。
6. 改造 `mineru-kit models download/show/verify`，支持 repo 参数和 `--tier` 选择器，移除 `pipeline` / `vlm` bundle 参数。
7. 移除旧 `auto_download_and_get_model_root_path` wrapper。
8. 在 managed local parser 配置变更时接入 readiness 检查。
9. 将 API server tier metadata 中的模型名同步为 registry 中的 repo name。
10. 更新 CLI 与配置文档，说明 `model.base_dir`、`model.source` 和 `local` source。
11. 增加测试覆盖:
   - `MINERU_MODEL_SOURCE` 覆盖 `config.model.source`
   - `get_config_source("model.source")` 区分 `default`、`file`、`env`
   - `update_config_file` 并发/原子写入
   - `auto` 探测写回当前配置文件
   - `MINERU_MODEL_SOURCE=auto` 不写回配置文件
   - HF/MS 共用 `local_dir` 时锁路径稳定
   - `model.source=local` 缺失模型时报错
   - partial download 只验证请求路径
   - `ModelRepo.path()` 和 `ModelPath.path()` 支持更细粒度 partial download
   - `mineru-kit models download` 的 repo 参数与 `--tier` 互斥
   - `mineru-kit models download --tier flash` 为 no-op
   - `mineru-kit models verify` 检查 repo required paths 和 tier 所需 repo 集合
