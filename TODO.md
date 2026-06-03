# MinerU TODO 清单

## mineru server

- [ ] **`GET /parse/content` 应生成 markdown-with-markers**（`<!-- page N of M -->`）
  - 依赖 ParseResult.markdown_with_markers()（已实现）
  - 等 middle_json 对齐后，Generate markdown from middle_json 时顺便插入 marker
  - CLI `--no-marker` 开关
- [ ] **`-o` 文件输出 CLI 端支持** — 当前 server 端写文件，未来需 CLI 端 fallback（Docker/远程场景）
- [ ] **standard/pro 引擎实际接入** — 当前仅 flash 引擎可用
- [ ] **`parse()` pages range 字符串重构** — 当前用临时 `_split_ranges` 方案，需改为原生支持 `"1~5,10~15"` 格式
- [ ] **`--remote` + `--api-key`** — 与 mineru.net/api 集成
- [ ] **隐私优先决策链** — 根据 NEXT-CLI.md §隐私优先 实现
- [ ] **单元测试** — test_db.py, test_parse_svc.py, test_search_svc.py, test_config_svc.py, test_cleanup_svc.py
- [ ] **`on_event` → lifespan** — FastAPI deprecated 迁移
- [ ] **server 重启时保留 DB** — 当前种子数据每次启动重复插入（已部分修复 rules 表）

## mineru CLI

- [ ] **`-f --format`** — markdown/text/json/html 多格式输出
- [ ] **`--language` 参数** — 文档语言提示
- [ ] **`-V --version`** — 显示版本号
- [ ] **`-v --verbose`** — 详细日志（当前只有 parse --json）
- [ ] **`mineru-kit parse`** — 开发者批处理 CLI
- [ ] **`mineru-kit api-server`** — 无状态解析 API（parser/api_server.py）
- [ ] **`mineru-kit vlm-server`** — 本地 VLM 服务

## middle_json 对齐（NEXT-JSON.md）

- [ ] **团队讨论项 #1**: preproc_blocks 是否从序列化中移除 → **已决议：保留 dataclass 字段，不从 middle.json 序列化**
- [ ] **团队讨论项 #2**: equation vs interline_equation 同义合并
- [ ] **团队讨论项 #3**: algorithm_caption 补生产 / 删除
- [ ] **团队讨论项 #4**: DISCARDED 从 BlockType enum 拆出
- [ ] **团队讨论项 #5**: index（目录类型）的归属
- [ ] **团队讨论项 #6**: span index 的作用域与必带性
- [ ] **团队讨论项 #7**: _meta.file.filename 是否纳入
- [ ] **团队讨论项 #8**: _meta.models 字段粒度
- [ ] **团队讨论项 #9**: _meta.parsed_at 是否纳入
- [ ] **团队讨论项 #10**: merge_prev 与 CROSS_PAGE 是否合并
- [ ] **result_to_middle_json 统一** — Pipeline/VLM/Hybrid 三合一
- [ ] **union_make 收敛** — 四套 → 一套，依赖 middle_json 对齐
- [ ] **canonical JSON Schema** — 写入 `mineru/schema/middle_json/v1.json`
- [ ] **校验工具** — `mineru.schema.validate_middle_json(data)`

## 设计文档

- [ ] NEXT-JSON.md 团队讨论项决议后更新为定稿
- [ ] NEXT-DESIGN.md → 同步代码变更（SQLiteConfig pragmas、config 分层、无 lock 文件）
