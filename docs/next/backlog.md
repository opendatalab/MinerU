# 当前待办与待决问题

本页是后续工作的唯一索引，不授权自动执行历史计划。整理基于 4.1.2；标为“待复核”的项目来自旧评审，
领取前必须用当前调用链和回归验证仍然存在，不把旧报告的“未处理”直接当作当前缺陷。

## 已收敛，不再重复排期

- Tasks 是唯一异步解析接口，创建／查询／取消／上传响应独立；公共协议位于双语用户文档。
- API 共享模型已集中在 `mineru/tasks/contracts.py`，Doclib 解析职责已拆分为入库、队列、执行、读取服务。
- Middle JSON 采用 DocVortex schema 2.0；历史兼容仅限 Doclib 缓存。旧 HTTP 3.x 兼容计划已结束。
- `mineru-kit version`、四档名称、页码规范和服务启动上限已有实现，不再按早期计划重做。
- 结构优化 PR #5665 的完成记录已归档。

## 待办

| ID | 范围与下一步 | 状态／验收入口 |
|---|---|---|
| D-01 | 嵌入式宿主的自定义公式分隔符：核对初次 FTS 写入与重建索引是否使用同一宿主配置 | 待复核；源自结构优化记录的非阻断后续项；比较真实 DB 中两条路径的文本 |
| D-02 | Doclib ingest 失败后的显式重试、错误阶段及时间元数据 | 待设计；合并旧 E-003/E-004 与 G-004；保持 scan 刷新语义，不引入无界自动重试 |
| D-03 | Doclib 解析后索引／元数据持久化失败、服务重启中断的错误分类 | 待复核；旧 E-005/E-007；检查 parse_execution_svc、parse_errors、app startup 及失败恢复测试 |
| D-04 | failed parse 的显式重试语义与历史成功缓存保护 | 待设计；旧 E-006；先决定新建批次或重置批次，不能覆盖已完成产物 |
| D-05 | compaction 的数据库替换事务与磁盘产物的一致性 | 待复核；旧 E-008；检查当前事务与进程中断恢复，而非重新实现已有 JSON-first 写入 |
| D-06 | watch 失败的结构化状态、scan/refresh 日志与内部扫描重试策略 | 待设计；旧 E-009/E-011、G-006；区分人工 scan 与后台 watch |
| D-07 | CLI/Doclib 错误的 HTTP 状态、诊断信息和可操作扩展 | 待设计；旧 E-012/E-013、OQ-B-004/OQ-S-015；与公共 Tasks 及 file-parse 协议分开 |
| D-08 | Doclib 作为第三方 SDK/API 的稳定性等级、服务发现命名空间及便利接口 | 待决策；旧 OQ-C-012/C-013、OQ-S 系列；不恢复已经删除的别名或旧参数 |
| D-09 | 搜索结果可信度、更细粒度引用与解析版本追踪 | 待设计；旧 G-008/G-009/G-010；先复核现有 locator、metadata 和缓存能力 |
| D-10 | 公共 JSON Schema 发布方式及新增文档字段／结构化输出提案 | 待决策；共享 schema 变更归 DocVortex，MinerU 产品扩展归宿主；不把旧草案当成现行协议 |
| D-11 | 质量评测语料、性能基线、可靠性指标、语言覆盖与公开路线图 | 待确定范围；旧 OQ-R 系列；历史性能结果只能证明其记录的版本和设备 |
| D-12 | 第三方插件的本地 4.1 Tasks 集成验收及可导入工作流 | 待上游兼容证据；分别记录插件版本、云／本地模式和真实调用结果，不从旧截图推断兼容 |

## 证据与历史背景

- [结构优化验收](https://github.com/opendatalab/MinerU/blob/master/docs/archive/reviews/architecture-optimization-status.md)
- [Doclib 错误评审](https://github.com/opendatalab/MinerU/blob/master/docs/archive/design/doclib-error-handling-gap-review.md)
- [EveryDoc 差异评审](https://github.com/opendatalab/MinerU/blob/master/docs/archive/design/everydoc-doclib-gap-review.md)
- [原开放问题](https://github.com/opendatalab/MinerU/blob/master/docs/archive/design/open-questions.md)
- [原实现计划](https://github.com/opendatalab/MinerU/blob/master/docs/archive/design/implementation-plan.md)
- [历史性能报告及原始数据](https://github.com/opendatalab/MinerU/tree/master/docs/archive/performance)

新工作在此登记目标和验证边界；形成决定后更新相应规范或 ADR，再从本表移除。旧实验失败、环境限制和未实现提案不追溯抹去。
