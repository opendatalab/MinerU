# 在 WebUI 中解析和核对文档

使用[在线演示](https://www.modelscope.cn/studios/OpenDataLab/MinerU)，或安装 MinerU 后在终端启动本地界面。

```bash
mineru-kit webui
```

在浏览器打开终端打印的地址。本地 WebUI 默认启动一个 Router 和一个解析 worker。已有 V1 服务时，使用 `mineru-kit webui --api-url http://127.0.0.1:8000`，文件会提交到指定服务。

## 上传和选页 {#upload}

1. 在上传区选择、拖放文件，或使用**粘贴**放入剪贴板里的文件或图片。选择 PDF 后即可浏览原文。
2. 为 PDF 或图片选择服务提供的档位。原生办公文档、网页等格式使用 Flash 并解析整份文档。
3. 要解析 PDF 的连续几页，拖动页范围的两个端点。页码从 1 开始，包含首尾页。默认选全部页，部署设置了 `--max-pages` 时，初始选区受该上限约束。翻动预览只改变正在看的页面。
4. 点击**转换**。默认自动判断 PDF 是否需要 OCR。需要忽略文字层并重新识别时，可在转换前打开**强制 OCR**。

## 对照原文检查结果 {#source-tracing}

在 **Markdown** 标签中查看排版结果，用**目录**跳到对应标题。对于 PDF 或图片，点击左侧原文区域会高亮右侧结果；点击右侧有来源位置的段落、标题、公式或图表，会定位到左侧原文。跨页合并的段落或表格可以通过**来源页**切换关联页面。

标框叠加在原文预览上。只解析部分页面时，预览仍使用原始页码，可继续翻阅整份 PDF。

## 下载结果 {#downloads}

在结果栏打开**下载**并选择格式。Markdown 和 JSON 分别下载为 ZIP，各包含正文和配套的 `images/`。**JSON** 标签及下载给出 Structured Content。需要 Middle JSON 时，可使用 Tasks 的 `default` 包或 `mineru-kit parse --format middle_json`。

菜单还提供 HTML、DOCX、LaTeX、EPUB 和 PDF。LaTeX 包含 `.tex` 及图片；PDF 从解析结果渲染，可用于分享。单文件 HTML 引用当前服务上的图片，离线携带图片可选择 Markdown ZIP。[各入口的输出格式](../reference/output_files.md)
