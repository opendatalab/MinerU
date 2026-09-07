# JSON 2.0 直接迁移

采用[统一外层协议](envelope.md)，通用解析接口保持严格；仅 Doclib 持久化读取提供[历史转换](doclib-compatibility.md)。

1. 升级 DocVortex 至 0.2.x，MinerU 要求 `docvortex>=0.2.1,<0.3.0`。
2. 将 Model/Middle 构造器的 file_suffix、producer 放入 DocumentMetadata；
   Python 访问同步改为 metadata.file_suffix、metadata.producer。
3. JSON 使用完整 schema/schema_version/metadata/extensions；两边统一 from_dict/from_json。
4. 产品记录只输出实际 tier 与 txt/ocr；删除旧 effort、顶层 mineru_version 和协议适配调用。
5. Doclib 中可识别的历史 Middle JSON 在读取时自动转换，无需重新解析或改写旧文件。
   DocVortex 旧原生协议、旧结果包和通用接口收到的历史 JSON 仍需重新生成。
6. 同步升级自建 HTTP 服务与客户端。旧远端协议会明确拒绝，不尝试迁移。

验证协议 Schema、跨入口往返、真实生产者与扩展保留、旧协议拒绝、ZIP 和素材，
以及 Doclib 新旧缓存混存、有效页覆盖、重复页与压缩冲突。冻结迁移前页面、正文、
几何及渲染语义，不把外层迁移变成解析算法调整。
