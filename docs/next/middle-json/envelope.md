# Model / Middle JSON 统一外层协议

状态：Implemented，DocVortex 0.2.0 起直接切换。

两边共用 `docvortex.schema` 定义的类型和 JSON 2.0，MinerU 不再维护独立的文档封装。

```json
{
  "schema": "docvortex.middle",
  "schema_version": "2.0",
  "metadata": {
    "file_suffix": "pdf",
    "producer": {"name": "mineru", "version": "4.0.0b1"}
  },
  "extensions": {"mineru": {"tier": "standard", "parse_mode": "txt"}},
  "is_full_document": true,
  "pages": []
}
```

Model 使用 `docvortex.model`、`page_index_map` 及原来的逐页 raw block 列表；
Middle 保留 `is_full_document`、PageInfo、Block 和 Span。默认写出保留协议头、
完整 metadata 和空 extensions。JSON Schema 由同一组声明字段生成。

Python 使用 `document.metadata.file_suffix` 和 `document.metadata.producer`，
旧顶层字段和转发属性全部移除。构造器要求明确提供格式和生产者；解析入口参数不变。
`ModelJson.from_dict/from_json`、`MiddleJson.from_dict/from_json` 是统一读取入口，
`ParseResult.from_dict/from_json` 直接委托并校验可选的 MinerU 扩展。

## 来源与产品扩展

`producer` 只表达真实生成来源。加载、渲染、保存不改写，Model 到 Middle 深拷贝
来源和扩展。没有 MinerU 扩展的 DocVortex 文档可以直接在 MinerU 渲染。

`extensions.mineru` 仅含实际 `tier` 和最终 `parse_mode`；
`flash/medium/high/xhigh` 映射为 `flash/basic/standard/advanced`。
仅接受 `txt/ocr`，不存请求值或 `auto`，不重复存 `effort/mineru_version`。
MinerU 校验已存在的自身扩展；DocVortex 不依赖产品枚举，完整保留其他应用扩展。

## 序列化和素材

DocumentResult 与 ParseResult 继续存在。MinerU 的 PDF ParseResult 序列化仍省略
递归 image_base64，非 PDF 保持原有图片表示；源对象不被修改。因此协议统一不代表
所有结果包装的图片字段完全相同。DocVortex 继续使用 AssetStore 和显式素材外置。
Structured Content 迁移 metadata/extensions，但不带 Model/Middle 协议标识；
Content List V1/V2 内容契约不变。
HTTP 请求、输出文件名和 ZIP 布局保持不变；ZIP 内文档必须使用完整新版协议。

## 严格协议与 Doclib 历史缓存

通用 ParseResult、DocVortex codec、HTTP/ZIP 和 Gradio 仍联合校验 schema 与版本，
不接受历史协议，不提供旧格式写出。Bundle 继续使用 docvortex.bundle 2.0。

Doclib 的持久化读取边界单独兼容 3.4.5（含旧 1.0 pages 包装）和旧 MinerU Schema 2.0，
在内存中转换为当前 MiddleJson。兼容成功的批次可用于读取、缓存命中、覆盖范围和 FTS。
压缩先转换并比较来源、扩展及整本标识，一致才合并并写出新协议；损坏或冲突时保留源数据。
详见 [Doclib 历史兼容](doclib-compatibility.md)。普通读取不修改历史文件和数据库记录。
