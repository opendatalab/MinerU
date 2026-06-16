# ADR-0011: Doclib Doc Short ID

状态: Accepted
日期: 2026-06-15
相关文档: 0012-doclib-block-locator.md, ../middle-json/agent-gaps.md, ../middle-json/structured-content-schema.md, ../workflows.md

## 背景

Agent citation、Markdown locator marker 和 Structured Content 都需要一个短、稳定、可读的文档引用前缀。

完整 `sha256` 适合作为数据库主键和严格校验字段，但直接放入 block reference 会过长:

```text
doc:{sha256}/tier:{tier}/page:{page}/block:{block}
```

Git/GitHub 常展示短 SHA，但 Git 的短 SHA 是按需在当前 repository object store 中做 prefix disambiguation，并不会把 short id 作为对象字段持久化。MinerU 的场景不同:

- citation / marker 可能长期出现在 Agent 输出、日志、外部工具和用户复制的文本中。
- doclib 是本地长期数据库，引用应尽量稳定。
- 如果每次动态计算“当前最短唯一前缀”，后续新文档可能改变旧文档的最短前缀，历史引用会变得不稳定。

因此需要在 `docs` 表中持久化一个稳定的短文档 ID。

## 决策

在 `docs` 表中增加 `short_id` 字段，放在 `sha256` 主键列之后:

```sql
CREATE TABLE docs (
    sha256    TEXT PRIMARY KEY NOT NULL,
    short_id  TEXT NOT NULL UNIQUE,
    ...
);
```

`short_id` 的生成规则:

1. 新 doc 插入时，默认尝试 `sha256[:7]`。
2. 如果 `short_id` 与已有 doc 冲突，则长度递增:

   ```text
   sha256[:8]
   sha256[:9]
   sha256[:10]
   ...
   ```

3. 直到满足 `UNIQUE(short_id)`。
4. 一旦写入，`short_id` 不再因为后续新 doc 插入而改变。
5. 如果极端情况下所有前缀都冲突，最终可使用完整 `sha256`。

`short_id` 是 doclib 内稳定 doc identifier，不替代 `sha256`:

- `sha256` 仍是内容主键。
- `short_id` 用于人类可读引用、Agent marker、block reference 和 CLI/SDK 展示。
- API / SDK 响应在需要严格校验时仍应返回完整 `sha256`。

`short_id` 会被 [ADR-0012](0012-doclib-block-locator.md) 用于组成全局 block reference。ADR-0011 只定义 doc 级短 ID 的存储和生成规则，不定义 page/block locator 语义。

## 插入与并发

生成 `short_id` 不能只依赖插入前查询，因为多个 worker 可能并发插入前缀相同的不同 doc。

实现应以数据库唯一约束为最终仲裁:

1. 生成候选 `short_id`。
2. 尝试插入 `docs`。
3. 如果 SQLite 返回 `UNIQUE(short_id)` 冲突，则增加前缀长度并重试。
4. 如果冲突来自 `sha256` 已存在，则读取现有 doc，不创建新 doc，也不改变现有 `short_id`。

伪代码:

```python
for length in range(7, len(sha256) + 1):
    short_id = sha256[:length]
    try:
        insert_doc(sha256=sha256, short_id=short_id, ...)
        return short_id
    except UniqueViolation as exc:
        if conflict_is_sha256(exc):
            return existing_doc.short_id
        if conflict_is_short_id(exc):
            continue
        raise
```

SQLite 的错误信息未必稳定区分具体 unique constraint。实现时可以在冲突后查询:

- `SELECT short_id FROM docs WHERE sha256=?`
- `SELECT sha256 FROM docs WHERE short_id=?`

再决定是返回已有 doc，还是递增前缀重试。

## 替代方案

### 方案 A: 直接在引用中使用完整 SHA256

拒绝。完整 SHA256 过长，不适合 Markdown marker、Agent 输出和人工调试。

### 方案 B: 每次动态计算最短唯一前缀

拒绝作为 P0 主方案。它更接近 Git 的 abbreviation 机制，但 MinerU citation 需要长期稳定。动态前缀可能因为后续新 doc 插入而变化，导致历史引用不稳定。

### 方案 C: 固定使用 12 位 SHA256 前缀，不入库

拒绝。固定 12 位实现简单，但仍存在理论碰撞；一旦碰撞，需要额外 fallback 规则。持久化 `short_id` 可以在保持短引用的同时由 DB 提供唯一性保证。

### 方案 D: 不持久化 `short_id`，在 block reference 中直接使用 hash 型 `chunk_id`

拒绝作为 P0 主方案。hash 型 chunk id 更短且不可枚举，但不如基于 `short_id` 的引用可读。P0 先持久化 `short_id`；hash 型 chunk id 后续可作为补充字段。

## 影响

### 数据库

- `docs` 表新增 `short_id TEXT NOT NULL UNIQUE`。
- 开发阶段可直接更新 `001_init.sql`；如果已有用户数据需要保留，应提供 migration。
- 所有 doc 创建路径必须通过同一个 helper 生成并写入 `short_id`。

### Interface / SDK / API

`DocInfo` 应增加:

```python
short_id: str
```

需要引用文档的响应可同时返回:

```json
{
  "sha256": "...",
  "short_id": "ab12cd3"
}
```

调用方需要严格身份校验时使用 `sha256`；需要人类可读引用时使用 `short_id`。

### Agent 输出

Agent 输出中的局部 locator、全局 `block_ref`、Markdown marker 和 citation record 由 [ADR-0012](0012-doclib-block-locator.md) 定义。本 ADR 只保证 `short_id` 可以作为这些引用的稳定 doc 前缀。

### 回查

如果用户或 Agent 输入包含 `doc:{short_id}` 的引用，doclib 应按 `docs.short_id` 精确查找。

由于 `short_id` 有唯一约束，正常情况下不会出现 ambiguous prefix。只有当调用方传入的不是完整 `short_id`，而是更短手写前缀时，才需要返回类似 `ambiguous_doc_ref` 的错误；P0 可以不支持短于 `short_id` 的 prefix lookup。

## 后续动作

1. 更新 `docs` schema，增加 `short_id`。
2. 实现 doc 插入 helper，处理 `short_id` 冲突和并发重试。
3. 扩展 `DocInfo` / list docs / show doc / show file 响应。
4. 在 locator / citation helper 中使用 `short_id`，详见 [ADR-0012](0012-doclib-block-locator.md)。
5. 增加测试:
   - 默认生成 7 位 short_id。
   - 前缀冲突时自动增长。
   - 已存在 sha256 不改变 short_id。
   - `short_id` 不随后续新 doc 改变。
