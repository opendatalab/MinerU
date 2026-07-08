# Middle JSON 总览

状态: Draft
读者: backend 开发者、输出开发者、SDK 开发者、Agent 能力开发者
范围: Middle JSON 的目标、当前事实和工作分层
来源: 由根目录旧 Middle JSON 底稿迁移整理而来

## 当前事实

当前 Middle JSON 的事实标准是 `mineru/types.py` 中的 typed document model:

```text
PageInfo
  -> preproc_blocks: list[Block]
  -> para_blocks: list[Block]
  -> discarded_blocks: list[Block]

Block
  -> lines: list[Line]
  -> blocks: list[Block]

Line
  -> spans: list[Span]
```

这套结构已经被 parser、doclib、render 和 parse-server 共同使用。旧底稿中“需要定义 canonical schema”的方向仍然成立，但下一步应以当前 dataclass 为基准，而不是另起一套独立结构。

## 为什么还需要整理

当前混乱主要来自四类不一致:

| 类别 | 现状 |
|------|------|
| 顶层 envelope | 当前 `ParseResult` 使用 `schema_version + pages`；顶层 `_backend` 是临时兼容 metadata；历史 `_version_name` 仅作为离线迁移输入处理；底稿希望有 `_meta`。 |
| backend 细节 | Pipeline/VLM/Hybrid/Office/HTML 对 bbox、index、page_size、preproc_blocks 的质量不同。 |
| render 消费 | 已有统一 render facade，但内部仍按 backend dispatch。 |
| Agent 能力 | 引用定位、稳定 page/block 地址和隐私边界还没有落地。 |

## 下一版目标

Middle JSON 下一版要达到以下目标:

1. 统一顶层 envelope，运行时只接受当前 `pages` 结构。
2. 明确 `PageInfo` / `Block` / `Line` / `Span` 字段契约。
3. 为每个 backend 提供 normalization 任务清单。
4. 定义 Agent 可引用的稳定 locator 规则。
5. 明确 render 层依赖哪些字段，逐步移除对 `_backend` 的长期依赖。
6. 提供 migration / validation 的验收标准。

## 工作分层

| 层 | 工作内容 | 产物 |
|----|----------|------|
| Schema | 稳定 dataclass 字段和 envelope。 | `current-medium.md`、`envelope.md` |
| Normalization | 补齐 backend 差异。 | `backend-gaps.md` |
| Agent-native | 引用、locator。 | `agent-gaps.md` |
| Structured Content | 盘点当前 structured_content，作为新 schema 起点。 | `structured-content-current.md` |
| Structured Content Schema | 定义 NEXT 版目标 schema 草案。 | `structured-content-schema.md` |
| Rendering | 统一 Markdown / Content List / Structured Content 消费。 | `rendering.md` |
| Migration | 分阶段落地。 | `migration.md` |

## P0 判断

下面这些是 P0，因为它们直接阻塞 Agent 场景:

- 稳定 page/block locator。
- 可从 Agent answer 追溯到 page / block / bbox / source hash。
- 历史 middle json 可迁移到当前 envelope。
- `ParseResult.from_dict()` 可以恢复 API / 缓存结果。
- 默认选择 / tier 产生的不同质量结果能被 doclib 正确区分和缓存。

## 非目标

- 不重新设计 OCR、layout、VLM 模型输出。
- 不把所有 backend 内部临时字段公开为 schema。
- 不在 Middle JSON 层定义 chunking 算法本身；这里只定义可寻址基础。
