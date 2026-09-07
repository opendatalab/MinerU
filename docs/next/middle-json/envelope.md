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

## 直接迁移与缓存

读取联合检查 schema 和 schema_version；不接受旧 DocVortex 1.0、缺少 schema 的
MinerU 2.0、旧 1.0 pages 或 pdf_info。不猜测、不自动转换，也不提供旧格式写出。
需要从源文件重新解析。Bundle 使用 docvortex.bundle 2.0 和原有素材摘要清单。

Doclib 仅将协议有效且覆盖记录页范围的批次视为可用缓存。旧缓存不计入缓存命中、
可用页范围和默认读取档位。请求解析时重新生成缺失结果；直接读取旧页返回重新解析提示。
压缩仅合并 metadata、extensions、is_full_document 一致的新批次，重复页仍取最新值；
遇到旧格式、损坏数据或冲突时跳过压缩并保留源数据。
