# my-pdf: opendataloader-pdf 代码迁移 + 裁剪方案

**日期**: 2026-05-30  
**状态**: 设计阶段（待审批）

## 1. 目标

将 opendataloader-pdf 中 **PDF → JSON → Markdown** 链路及其依赖的 veraPDF 代码迁移到新项目 `my-pdf`，使用 `my.pdf` 作为统一 package。要求：Maven 单模块可编译，精确裁剪不需要的代码。

## 2. 范围

### 2.1 保留

```
PDF 文件
  → DocumentProcessor (veraPDF 解析 + WCAG 版面分析)
    → ExtractionResult (内部中间结构)
      → JsonWriter → JSON 输出
      → MarkdownGenerator → Markdown 输出
```

来自 opendataloader-pdf 的包：
- `json/` — JSON 输出（含 serializers/）
- `markdown/` — Markdown 输出（含 MarkdownHTMLGenerator）
- `processors/` — 主解析管线，仅保留 `DocumentProcessor` 引用的 processor
- `cli/` — CLI 入口
- `utils/`、`entities/`、`exceptions/` 等支撑类

来自 veraPDF 的模块（精确裁剪后）：
- `parser` — COS 对象模型、Content Stream 解析、PDDocument 加载
- `wcag-validation` — ChunkParser、SA 层结构元素
- `wcag-algorithms` — 全部实体模型 + 版面分析管线
- `validation-model` — GF COS 适配层（仅保留被引用的类）
- `core` — 少量基础类型（仅保留被引用的类）
- `xmp-core` — 极少量 ThreadLocal 容器类

### 2.2 砍掉

| 模块 | 原因 |
|------|------|
| `hybrid/` (18 文件) | 不需要 hybrid 模式 |
| `html/` (3 文件) | 不需要 HTML 输出 |
| `text/` (1 文件) | 不需要纯文本输出 |
| `pdf/` (2 文件) | 不需要标注覆盖 PDF 输出 |
| `AutoTaggingProcessor` | 不需要 tagged-pdf |
| `TaggedDocumentProcessor` | 不需要结构树解析 |
| `HybridDocumentProcessor` | 不需要 hybrid |
| `PDFStreamWriter` | tagged-pdf 内部实现 |
| `autotagging/` | 被 AutoTaggingProcessor 独占 |
| 整个 **Apache PDFBox** (~853 文件) | 唯一使用者 PDFWriter 已砍掉 |
| veraPDF `feature-reporting` | 整模块排除 |
| veraPDF `metadata-fixer` | 整模块排除 |

### 2.3 外部依赖（保留在 pom.xml 中）

| 依赖 | 来源 | 用途 |
|------|------|------|
| `com.fasterxml.jackson.core:jackson-databind` | odl-pdf | JSON 序列化 |
| `org.aspectj:aspectjrt` + `aspectjtools` | odl-pdf | AOP 编译时织入 |
| `org.mozilla:rhino` | veraPDF core | 待裁剪后判定 |
| `javax.xml.bind:jaxb-api` + impl | veraPDF core | 待裁剪后判定 |
| `net.sf.saxon:Saxon-HE` | veraPDF core | 待裁剪后大概率不需要 |

外部依赖的最终列表在阶段 4 据实确定。

## 3. 项目位置

```
MinerU-Repo/my-pdf/
```

## 4. 目录结构

```
my-pdf/
├── pom.xml
└── src/main/java/
    └── my/pdf/
        ├── cli/                         ← odl-pdf-cli
        ├── [odl-pdf-core 各子包]        ← json/, markdown/, processors/, utils/, entities/, exceptions/
        └── verapdf/                     ← veraPDF 裁剪后全部
            ├── as/
            ├── containers/
            ├── cos/
            ├── exceptions/
            ├── factory/
            ├── gf/
            ├── io/
            ├── operator/
            ├── parser/
            ├── pd/
            ├── tools/
            ├── wcag/
            └── xmp/
```

## 5. Package 映射

| 原始 package | 目标 package |
|-------------|-------------|
| `org.opendataloader.pdf.*` | `my.pdf.*` |
| `org.verapdf.*` | `my.pdf.verapdf.*` |
| `java.*`, `javax.*` | 不变 |
| `com.fasterxml.*`, `org.aspectj.*` | 不变 |
| `org.mozilla.*`, `net.sf.*` | 不变 |

注：PDFBox 整棵砍掉，不需要映射。

## 6. 实施步骤

### 阶段 1：依赖扫描（产出精确文件清单）

1. 从保留的 odl-pdf 入口文件（~80 个 .java）出发
2. 提取所有 `import org.verapdf.*`
3. 将每个 import 映射到 competitors 下的具体源文件
4. 递归追踪 veraPDF 模块间的内部交叉引用
5. 输出完整的"可达文件集"清单

### 阶段 2：拷贝 + import 重写（Python 脚本一次性完成）

```python
# 伪代码
for each file in reachable_set:
    target_path = rewrite_path(file)  # 目录 + package 映射
    ensure_parent_dir(target_path)
    copy_and_rewrite(file, target_path)  # 逐行替换 package + import
```

重写规则：
```
^package org\.opendataloader\.pdf  →  package my.pdf
^package org\.verapdf              →  package my.pdf.verapdf
^import org\.opendataloader\.pdf   →  import my.pdf
^import org\.verapdf               →  import my.pdf.verapdf
(其他行不动)
```

### 阶段 3：边缘情况修复

- 代码中的 FQN 引用（`org.verapdf.xxx.Y` 形式）→ grep + 替换
- `META-INF/services/` 文件中的全限定类名 → 如有则同步改
- `module-info.java` → 检查是否需要

### 阶段 4：pom.xml 编写

- 扫描新目录树中所有外部 import
- 编写 pom.xml（Java 11 + AspectJ 编译插件 + 外部依赖坐标）
- `mvn compile` 试编译

### 阶段 5：编译迭代修复

- `mvn compile` 循环修复，直到通过
- 常见错误类型：
  - 缺类 → 回到阶段 1 补充遗漏文件
  - 缺外部依赖 → 补充 pom.xml
  - import 遗漏 → 补充替换规则

### 阶段 6（可选）：功能验证

- 用 PDF 测试 JSON → Markdown 输出，对比 odl-pdf 原始结果

## 7. 风险

| 风险 | 缓解 |
|------|------|
| veraPDF core/validation-model 内部耦合高，裁剪后编译通不过 | 阶段 1 先跑，如验证规则类大量相互引用，考虑保守保留 validation-model 全部 |
| AspectJ 织入导致切面相关类缺失 | 阶段 5 检查，必要时保留 `DumpAspect` 等 |
| 包名改后反射/配置字符串断裂 | 阶段 3 grep `"org.verapdf` 等字符串字面量 |
| 预估规模不准 | 阶段 1 产出实际清单后重新评估 |

## 8. 预估

| 来源 | 原始文件数 | 预估裁剪后 |
|------|----------|----------|
| odl-pdf | 111 | ~70 |
| veraPDF | ~1159 | ~600-800 |
| **合计** | **~1270** | **~700-900** |
