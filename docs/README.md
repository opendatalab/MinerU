# 文档维护

## 目录与发布边界

| 目录 | 用途 | GitHub Pages |
|---|---|---|
| `en/`、`zh/` | 双语安装、使用与公开接口参考 | 发布，进入搜索 |
| `next/` | 现行开发规范、模块边界、ADR 与待办；保留目录名以稳定既有链接 | 发布，通过开发参考页访问 |
| `native-async-analysis.md` | 原生异步运行时说明 | 发布，通过开发参考页访问 |
| `archive/` | 历史设计、迁移、验收、性能证据及旧平台／第三方教程 | 不发布正文，不进入搜索 |
| `proposals/` | 未采纳或尚未实现的接口草案 | 不发布正文，不进入搜索 |

用户文档从 `en/index.md` 和 `zh/index.md` 进入；开发文档从 `next/README.md` 进入。
Middle JSON 现行规范只在 `next/middle-json/` 维护；官方用户文档引用该规范，不另建一套 schema。
Tasks、file-parse 与版本迁移的公开契约在双语用户文档维护，开发文档链接到它们。

## 编辑规则

- 同一主题只设一个目录首页。不要同时新增 `topic.md` 与 `topic/README.md`，两者会生成相同网址。
- 面向用户的行为修改同步中英文；代码路径、示例字段和默认值应从当前实现核对。
- 中文小节被跨页引用时使用显式、稳定的英文锚点，如 `{#tier-selection}`。
- 归档记录保留当时的版本、失败和测试数据；尚未解决的问题迁入 `next/backlog.md`，不把历史结果改写成当前验收。
- 页面迁移后更新源文件链接，在 `_redirects.json` 登记旧地址。构建为旧页面生成无索引跳转；历史 JSON/PDF 附件的旧下载地址仍返回原字节。
- 归档或提案的站内入口使用 GitHub 源文件链接。截图随使用它的教程保留，不按目录年龄批量删除。

## 验证

```bash
uv pip install -r docs/requirements.txt
python -m mkdocs build --strict
python docs/_hooks/check_site.py site
```

失效的本地文件或锚点会导致严格构建失败。归档与提案不参与现行页面构建；检查搜索索引时也不应出现它们的正文。

`docs/plans/` 与 `docs/superpowers/` 已删除，旧内容分别保存在 `archive/plans/` 与 `archive/superpowers/`。不要重新建立现行计划的第二个目录。

ADR 保留决策背景，不作为当前源码路径清单。现行接口说明不得依赖历史报告或未实施提案；需要继续实施的产品事项只进入 `next/backlog.md`。
