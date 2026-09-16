# MinerU Kit 模型命令 stack 参数设计

日期：2026-08-05
范围：`mineru/kit/commands/models.py`、`mineru/utils/model_registry.py`、相关测试

## 背景

MinerU 已支持两套模型 stack：

- **light**：ONNX 推理（layout、OCR、公式）+ 未来 llama.cpp VLM，不需要 torch/transformers
- **full**：PyTorch 推理 + transformers/vllm/lmdeploy VLM

`config.model.stack` 字段已落地，`get_model_stack()` 把 `auto` 解析成 `cpu→light, 其他→full`。
`ModelRepo` 已带 `stack` 字段：`PDF_EXTRACT_KIT`、`MINERU_2_5_PRO_2605_1_2B` 为 `full`，6 个 Paddle ONNX repos 为 `light`。

但 `mineru-kit models` 子命令仍只支持 `--tier`，无法按 stack 选择下载集：
`REPOS_FOR_TIER` 是静态字典，只映射到 full repos，ONNX repos 没有归到任何 tier。
用户在 CPU 机器上跑 `--tier basic`，运行时会用 light ONNX 模型，但 `models download --tier basic` 仍下载 full 的 `PDF_EXTRACT_KIT`（约 6GB），完全浪费。

## 目标

给 `mineru-kit models` 子命令加 `--stack` 参数，让下载/验证的模型集与运行时实际使用的 stack 一致。

## 非目标

- 不实现 light stack 的 VLM 下载（llama.cpp/GGUF 未落地）
- 不动 `mineru-kit parse`、`api-server` 等其他子命令
- 不改 `config.model.stack` 已有语义
- 不拆分 `PDF_EXTRACT_KIT`（表格模型拆分另行处理）

## 设计

### CLI 接口

```
mineru-kit models download [REPO] [--tier T] [--stack S] [--source S] [-v]
mineru-kit models show                  [--stack S]
mineru-kit models verify   [REPO] [--tier T] [--stack S]
```

`--stack` 取值：

- `auto`（默认）：调用 `get_model_stack()` 解析，`cpu→light`，其他→`full`
- `light`：使用 light repos
- `full`：使用 full repos

参数互斥规则：

- `download` 传 repo 名时忽略 `--stack`（repo 自身已声明 stack 字段）
- `download` 同时传 repo 和 `--tier` 仍报错（已有逻辑保留）
- `verify` 传 repo 名时忽略 `--stack` 与 `--tier`

### tier → repos 映射

`REPOS_FOR_TIER` 静态字典改为函数 `model_repos_for_tier(tier, *, stack=None)`：

| tier \ stack | light | full |
|--------------|-------|------|
| basic | `PP_DOCLAYOUT_V2_ONNX` + `PP_OCR_V6_SMALL_DET_ONNX` + `PP_OCR_V6_SMALL_REC_ONNX` + `PP_FORMULANET_PLUS_M_ONNX` | `PDF_EXTRACT_KIT` |
| standard | 同 basic（暂无 light VLM） | `PDF_EXTRACT_KIT` + `MINERU_2_5_PRO_2605_1_2B` |

设计要点：

- light basic 与运行时 `atom_model_init()` 在 `stack=="light"` 下加载的模型一一对应（layout + small OCR det/rec + formula）
- light standard 暂时与 light basic 相同：light VLM（llama.cpp/GGUF）未实现，不能让用户以为下载完就能跑 standard
- full stack 保持原映射不变

向后兼容：

- `model_repos_for_tier(tier)` 单参数调用仍可工作，默认 `stack=None` → 解析 auto
- `model_repos_for_tier("flash")` / `"advanced"` 仍按原逻辑报错 `Supported model tiers: basic, standard`

### stack 解析

新增 helper `resolve_model_stack(stack: str | None) -> Literal["light", "full"]`：

```python
def resolve_model_stack(stack: str | None) -> Literal["light", "full"]:
    """把 --stack 参数或 config 值解析为 light/full。None 或 'auto' 走 get_model_stack()。"""
    from ..utils.config_reader import get_model_stack
    if stack in ("light", "full"):
        return stack
    if stack is None or stack == "auto":
        return get_model_stack()
    raise ValueError(f"Unsupported stack '{stack}'. Expected one of: auto, light, full.")
```

放在 `mineru/utils/model_registry.py`，与 `validate_model_tier` 并列。

### show 命令输出

新增字段：

```
Config: ...
Config exists: ...
MINERU_MODEL_SOURCE=...
model.base_dir: ...
model.base_dir.source: ...
model.source: ...
model.source.source: ...
model.stack: <配置值>           # 新增
model.stack.source: <source>    # 新增
Effective stack: <light|full>   # 新增，--stack 解析后的值
Repos:
  PDF-Extract-Kit-1.0: ready (...) [stack=full]      # 新增 [stack=...] 标记
  PP-DocLayoutV2_onnx: ready (...) [stack=light]
  ...
Model tiers:                     # 改为按 effective stack 显示
  basic: <该 stack 下的 repos>
  standard: <该 stack 下的 repos>
```

`show` 的 `--stack` 参数会覆盖 `config.model.stack` 用于解析 effective stack。

### verify 命令行为

- 无 repo 无 tier：验证 effective stack 下的所有 repos（不再验证全部 8 个）
- 带 `--tier`：验证该 tier 在 effective stack 下的 repos
- 带 repo 名：忽略 `--stack` 与 `--tier`，验证该 repo

### download 命令行为

- `download --tier basic`：用 effective stack 下的 basic repos
- `download --tier basic --stack light`：强制 light 下的 basic repos
- `download PP-DocLayoutV2_onnx`：忽略 `--stack`，下载该 repo
- `download --tier basic --stack full`：保持原行为（下载 `PDF_EXTRACT_KIT`）

## 涉及文件

- `mineru/utils/model_registry.py`
  - 新增 `resolve_model_stack(stack)` helper
  - `model_repos_for_tier(tier, *, stack=None)` 改为函数（替换静态字典导出）
  - 内部保留 `REPOS_FOR_TIER_FULL`、新增 `REPOS_FOR_TIER_LIGHT` 两个静态字典作为实现
  - `__all__` 导出 `resolve_model_stack`
- `mineru/kit/commands/models.py`
  - 三个子命令加 `--stack` 参数（默认 `None`，help 文本说明 `auto/light/full`）
  - 加 `_resolve_effective_stack(stack)` 局部 helper 调用 `resolve_model_stack`
  - `_select_target_repos` 接受 `stack` 参数，传给 `model_repos_for_tier`
  - `show_cmd` 输出新增 `model.stack`、`Effective stack` 字段，repos 行追加 `[stack=...]` 标记，tiers 部分按 effective stack 显示
  - `verify_cmd` 在无 repo 无 tier 时按 effective stack 过滤 repos
- `tests/unittest/test_kit_commands.py`
  - 现有 `test_models_download_tier_basic/standard` 等用例：monkeypatch `get_model_stack` 返回 `"full"`，避免依赖运行时设备
  - 新增 light stack 用例：
    - `test_models_download_tier_basic_light`：验证下载 4 个 light repos
    - `test_models_download_tier_standard_light`：验证与 basic light 相同
    - `test_models_show_displays_stack_fields`：验证 `model.stack`、`Effective stack`、`[stack=...]` 标记
    - `test_models_verify_filters_by_effective_stack`：验证只验证 effective stack 下的 repos
    - `test_models_download_repo_ignores_stack`：验证传 repo 名时 `--stack` 被忽略
    - `test_models_download_rejects_invalid_stack`：验证 `--stack foo` 报错

## 兼容性

- `model_repos_for_tier(tier)` 单参数调用向后兼容（默认 `stack=None`）
- `REPOS_FOR_TIER` 名字不再导出（仅作内部实现），如有外部引用需更新
- `config.model.stack` 已有字段，`show` 只是把它显示出来
- 现有测试若依赖 `REPOS_FOR_TIER` 静态字典导出，需改为调用 `model_repos_for_tier`

## 风险

- **`REPOS_FOR_TIER` 移除导出的破坏性**：需 grep 确认无外部引用。如有，保留导出但内部委托函数版。
- **light standard 与 basic 相同的语义**：用户可能困惑。通过 `show` 输出明确标注 `standard(light): same as basic` 或在 download 时 logger.warning 提示 "light stack does not yet include VLM"。
- **测试 monkeypatch 时机**：`get_model_stack` 在 `model_repos_for_tier` 内部 lazy import，monkeypatch 需针对 `mineru.utils.config_reader.get_model_stack` 或在测试开始前设置 `config.model.stack="full"`。后者更稳，优先用。
- **现有 `test_models_show_and_verify` 依赖 full stack**：该用例断言输出含 `PDF-Extract-Kit-1.0: ok` 与 `MinerU2.5-Pro-2605-1.2B: ok`。在 CPU 机器上 effective stack=light，verify 默认只验证 light repos，断言会失败。修法：用例开头 monkeypatch `config.model.stack="full"`，或在测试中显式传 `--stack full`。前者更稳，因为 `show` 输出也会随之显示 full tiers，与现有断言一致。

## 验收

- `mineru-kit models download --tier basic --stack light` 下载 4 个 light repos
- `mineru-kit models download --tier basic --stack full` 下载 `PDF_EXTRACT_KIT`
- `mineru-kit models download --tier basic`（无 --stack）在 CPU 机器上等同 `--stack light`
- `mineru-kit models show` 输出包含 `model.stack`、`Effective stack`、repos 行的 `[stack=...]` 标记
- `mineru-kit models verify` 默认只验证 effective stack 下的 repos
- `mineru-kit models download PP-DocLayoutV2_onnx --stack full` 仍下载 ONNX repo（--stack 被忽略）
- 所有现有测试通过，新增 light stack 测试通过
