# DocGale HTML 标记迁移验证

HTML 解析和渲染继续委托给 DocGale。新产物使用 `docgale-*`、
`data-docgale-html-version="1"` 和 `data-docgale-latex`；EPUB XHTML/CSS 与
Markdown 内嵌 HTML 同步切换。旧 HTML 标记不再进入精确往返路径，按普通网页解析。

本次更新相关测试调用与断言，并保持 JSON 产品封装、tier 路由、Gradio UI 标记、
表格识别和非 HTML 输出标识不变。原有全类型往返、公式、链接、表格、代码、
HTML 安全及 EPUB 资源验证均保留，没有通过放宽断言处理差异。

本地 Python 3.13.5、PDFium 5.10.1 环境中，HTML、EPUB、Markdown、渲染 API、
anchor 与 DocGale 路由测试 **459 项通过**。DocGale 全量本地回归 **396 项通过**，
已有少线表诊断单独排除；独立 Python 3.14 wheel 验证 **80 项通过**。

[三平台 CI](https://github.com/myhloli/docgale/actions/runs/34042886135) 已通过，
代码提交为 `5f2fd59acefb2f3ed247d1e59c63fcb7cd73aae4`。
Linux/macOS 每组 396 项通过；Windows 每组 395 项通过、1 项 POSIX 检查跳过。
已有少线表诊断仍保留其独立失败，不计入本次修复。

三平台各三份实际 PDF 产物的 ModelJson、MiddleJson、原始几何和全部页面 PNG
与此前一致；HTML/Markdown 仅改变实现标记。中文论文3/4的 Windows/Linux HTML
和 Layout 对照已刷新，浏览器中的公式、Mermaid、代码高亮与素材加载检查通过。

已有 JSON 和结果包不自动修改。具备完整保存结果时，重新渲染即可得到新 HTML，
无需重新解析源 PDF。本次更新 0.1.0 草稿安装包，不发布至 PyPI。
