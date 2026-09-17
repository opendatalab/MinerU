# Remote 路径补 3.x Middle JSON 兼容（设计）

日期：2026-09-17
状态：已确认，待实现
分支：opendatalab-master

## 背景与动机

MinerU 4.0 把 Middle JSON 收敛到统一共享协议（schema 2.0）。`ca15b938`（2026-09-07，
"remove legacy support"）删除了通用入口 `ParseResult.from_dict` 与 HTTP 客户端里的 legacy 兼容，
当天 `332bb866` 只把旧格式读取恢复到 doclib 持久化缓存边界
（`mineru/doclib/core/middle_json.py::read_cached_middle_json`），并有意让 HTTP/ZIP/Gradio
继续严格读取新协议——前提假设 remote parse server 已经升级到 4.0。

但现实是仍有 3.x remote server 在服务，返回 3.4.5 `pdf_info` 或旧 schema 1.0/2.0 的 middle_json。
`parser/api_client.py::_parse_result_from_middle_json` 现在只接受新协议，使 doclib 走 remote
解析这类 server 时抛 `invalid_middle_json_output`。

本次打破「HTTP/ZIP 严格读新协议」的边界，为 remote 路径临时补回 3.x 兼容，
待 remote server 全部升级到 4.0 后移除。

## 目标

- remote JSON 与 ZIP 两条输出路径都能正确读取 3.x 历史 middle_json
  （3.4.5 `pdf_info`、旧 schema `1.0` pages、旧 schema `2.0`）。
- 4.0 新协议路径行为保持不变。
- 兼容逻辑保持单一实现、单一归属，未来一处删除。

## 设计

### 模块布局

- 删除 `mineru/doclib/core/middle_json.py`（对外仅 `read_cached_middle_json` 一个符号）。
- 新建 `mineru/backend/postprocess/legacy_middle_json.py`，承载历史 middle_json 读取：

```python
def read_legacy_middle_json(payload: dict[str, Any]) -> MiddleJson:
    """读取 3.x 历史 middle_json（pdf_info / 旧 1.0 / 旧 2.0），返回当前 MiddleJson。"""
```

  移入原 `read_cached_middle_json` 的全部识别逻辑，方法名与错误信息去「cached」化。

### 数据流

remote 与 doclib 两个消费方共用同一读取器：

```
doclib 缓存读取（parse_svc / compaction）
        └─> read_legacy_middle_json(payload) -> MiddleJson

remote/HTTP/ZIP（parser.api_client._parse_result_from_middle_json）
        └─> ParseResult.from_dict（严格新协议）
            └─ 失败则 read_legacy_middle_json(payload) -> MiddleJson -> ParseResult
```

### remote 侧实现

`parser/api_client.py::_parse_result_from_middle_json` 改为：

1. 非 `dict` → `_V1APIError("invalid_middle_json_output", ...)`。
2. `ParseResult.from_dict`（4.0 新协议原路径）。
3. 失败后 fallback：`read_legacy_middle_json`，包成 `ParseResult(middle_json=...)`。
4. fallback 也失败 → `_V1APIError("invalid_middle_json_output", str(exc))`。

ZIP 路径经 `_parse_result_from_zip_bytes` 复用同一入口，自动获得兼容。

### 错误处理

- 合法 3.x 旧格式 → 迁移成功。
- 新协议数据损坏 / 未知 / 混合信封 → 仍抛 `invalid_middle_json_output`，不跨协议兜底。

## 测试

### 翻转（原锁定「拒绝」）

| 测试 | 位置 | 变更 |
|---|---|---|
| `test_api_client_rejects_remote_pdf_info_middle_json` | `tests/unittest/test_parser_api_contract.py:1055` | 拒绝 → 接受 |
| `test_async_api_client_rejects_remote_pdf_info_middle_json` | `:1078` | 拒绝 → 接受 |
| `test_api_client_rejects_legacy_official_layout_json` | `:814` | 拒绝 → 接受 |

### 更新

- `tests/unittest/test_doclib_middle_json.py`：import 与调用从
  `mineru.doclib.core.middle_json.read_cached_middle_json` 改为
  `mineru.backend.postprocess.legacy_middle_json.read_legacy_middle_json`。
- 新增正向用例：3.x `pdf_info` 经 remote JSON 与 ZIP 路径正确恢复 pages。

## 文档更新

- `docs/next/middle-json/doclib-compatibility.md`：删除「HTTP/ZIP 与 Gradio 继续严格读取新协议」，
  改为「HTTP/ZIP 临时兼容 3.x remote server，待其升级 4.0 后移除」。
- `docs/doclib-middle-json-compat-validation.md`：同步修订，并记录移除点。

## 临时性与移除计划

本兼容是过渡措施。remote server 全部升级 4.0 后，移除路径：

1. 删除 `parser/api_client.py::_parse_result_from_middle_json` 的 fallback 分支。
2. 删除（或先收窄回 doclib 缓存）`backend/postprocess/legacy_middle_json.py`。
3. 恢复相应测试与文档。

## 风险与注意事项

- `test_doclib_middle_json.py` 等多处调用点要随函数改名同步更新，避免漏改。
- 与 memory `legacy-migration-consumer-checklist` 呼应：删除/重建兼容层须「删哪几处、接回哪几处」逐一对照。