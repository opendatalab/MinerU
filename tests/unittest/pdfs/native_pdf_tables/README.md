# Native PDF Table 真实语料

本目录保存 Native PDF 表格结构恢复的仓库内回归输入。文件名采用稳定的英文语义名，评测真值位于 `tests/fixtures/native_pdf_table_cross_page_manifest.json`。

| 文件 | 页数 | 场景 |
|---|---:|---|
| `conference_schedule_tables.pdf` | 12 | 会议日程中的多页规则表格 |
| `manufacturing_facilities_cross_page_table.pdf` | 2 | 生产设施跨页表格 |
| `procurement_document_blank_page_tables.pdf` | 4 | 含空白页的采购文档表格 |
| `pollutant_discharge_tables.pdf` | 89 | 排污许可多样式表格 |
| `procurement_contract_tables.pdf` | 2 | 采购合同付款及账户表格 |
| `annual_report_research_projects_table.pdf` | 2 | 年报研发项目跨页表格 |
| `annual_report_fundraising_projects_table.pdf` | 2 | 年报募投项目跨页表格 |
| `annual_report_management_roles_table.pdf` | 2 | 年报任职情况跨页表格 |
| `quarterly_report_financial_tables.pdf` | 2 | 季报财务数据跨页表格 |
| `fund_manager_profile_table.pdf` | 2 | 基金经理简介跨页表格 |
| `fund_asset_and_transaction_tables.pdf` | 2 | 基金资产及交易表格 |
| `engineering_process_restrictions_table.pdf` | 9 | 工艺禁限用目录跨页表格 |

## 隐私处理

- 所有 PDF 均通过逐页重写移除了文档信息字典和 XMP 元数据，包括 Title、Author、Subject、Keywords、Creator、Producer 和日期字段。
- `pollutant_discharge_tables.pdf` 仅保留原文档第 3 页及之后的页面；含企业与个人信息的前两页未进入仓库。
- 除上述裁页外，页面可见内容、页面尺寸、旋转和绘图对象保持不变，并由结构真值与渲染回归共同校验。
