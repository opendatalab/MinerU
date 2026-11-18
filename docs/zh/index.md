# MinerU

![MinerU](../images/MinerU-logo.png){ width="300" }

MinerU 将 PDF、图片和办公文档转换为 Markdown 和结构化数据。你可以在浏览器中对照原文检查结果，在本地批量转换文件，或从自己的应用调用解析服务。

## 开始使用 {#_2}

| 想做什么 | 从这里开始 |
| --- | --- |
| 在线试一份文档 | 打开 [ModelScope 演示](https://www.modelscope.cn/studios/OpenDataLab/MinerU)，按 [WebUI 指南](usage/webui.md)上传、核对和下载结果。 |
| 在本机把文件转成 Markdown | 按[快速入门](quick_start/index.md)安装基础包，转换一个文件并打开生成的 `.md`。 |
| 在应用中解析文档 | 在 Python 进程内使用 [SDK](usage/sdk_api.md)，或向服务发送[第一次 HTTP 请求](usage/http_api.md)。 |
| 搜索和继续阅读已解析的内容 | 使用[文档库与 Agent 阅读](usage/quick_usage.md)保存解析结果，按定位符读取原文内容。 |

## 产品概览 {#_1}

PDF 和图片支持 Flash、Basic、Standard、Advanced 四档，可按速度、硬件和版面复杂度选择。[档位与运行环境](usage/tiers.md)

DOC/DOCX、PPT/PPTX、XLS/XLSX、RTF、ODT/ODS/ODP、EPUB、OFD、HTML/MHTML 和 CSV/TSV 通过 DocVortex 做本地 Flash 原生解析。解析服务、文档库和 WebUI 各自提供不同的导出选项。[输出格式与结果协议](reference/output_files.md)

## 版本与项目资源

[更新日志](reference/changelog.md)记录各版本的功能与兼容性变化。升级旧部署可查阅 [3.x → 4.0](reference/migration_4.md) 和 [4.0.x → 4.1](reference/migration_4_1.md){#mineru-41} 迁移说明。

[GitHub](https://github.com/opendatalab/MinerU) · [官网](https://mineru.net/) · [Colab 示例](https://colab.research.google.com/gist/myhloli/a3cb16570ab3cfeadf9d8f0ac91b4fca/mineru_demo.ipynb) · [许可证](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)

项目的技术报告包括 [MinerU](https://arxiv.org/abs/2409.18839) · [MinerU2.5](https://arxiv.org/abs/2509.22186) · [MinerU2.5 Pro](https://arxiv.org/abs/2604.04771)

[![PyPI 版本](https://img.shields.io/pypi/v/mineru)](https://pypi.org/project/mineru/) [![Python 版本](https://img.shields.io/pypi/pyversions/mineru)](https://pypi.org/project/mineru/)
