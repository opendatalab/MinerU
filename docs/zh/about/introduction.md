<!-- logo -->
<p align="center">
  <img src="https://opendatalab.github.io/MinerU/images/MinerU-logo.png" width="300px" style="vertical-align:middle;">
</p>

<!-- icon -->

<div align="center">

</div>

<div align="center" style="font-size:16px;">

</div>


## 项目介绍

MinerU 是由上海人工智能实验室OpenDataLab团队研发的一款**面向 AI 大模型时代的高性能文档解析引擎**，能够将 PDF、Word、PPT 等格式文档高保真地转换为 Markdown、JSON、LaTeX 等结构化数据。MinerU 可准确解析复杂排版、跨页表格、复杂数学公式、图像、代码及化学分子结构等多样化文档元素。
在 OmniDocBench、OlmOCR-bench 和 Ocean-OCR 等主流文档解析评测基准中，MinerU 均处于优势地位。基于模型核心能力，项目已衍生出多平台桌面客户端、网页端及在线 API 服务，支持批量处理与灵活集成。

目前，MinerU 的代码与模型已在 GitHub 和 HuggingFace 平台开源，提供模型下载、在线演示，便于科研人员、开发者及企业用户使用与二次开发，诚挚欢迎社区开发者参与贡献与反馈。
基于超强解析能力，MinerU 能够为 RAG 知识库构建、Agent 工具调用及大模型训练提供高质量的 AI-Ready 语料，已在金融、学术、教育等场景中展现出卓越的应用价值。

与成熟的商用产品相比，MinerU 仍处于快速迭代阶段。如遇到问题或者结果不及预期请到 [issue](https://github.com/opendatalab/MinerU/issues) 提交问题，同时**附上相关PDF**，更好地支持开源事业。

![type:video](https://github.com/user-attachments/assets/4bea02c9-6d54-4cd6-97ed-dff14340982c)

## 功能定位

|  | 网页端 / 桌面客户端 | 本地部署开源版 | 在线 API 服务 |
| :--- | :--- | :--- | :--- |
| **适用用户** | 常规用户 / 研究者 | 研究者 / 开发者 / 高级用户 | 开发者 |
| **解决的核心问题** | 快速、直观地使用 MinerU 的核心功能，无需关心技术细节，集成 MinerU4Chem 化学功能 | 在本地环境进行深度定制、二次开发和离线数据处理 | 将 MinerU 的数据处理能力集成到自己的应用程序或服务中 |
| **使用方式** | 图形化界面 (GUI) | WebUI / 命令行 / 代码 | HTTP 请求 |
| **部署要求** | 无需部署，浏览器或桌面客户端即开即用 | 需要 Python/Docker 环境，依赖本地硬件资源 | 无需部署，通过网络调用 |
| **定制能力** | 无 | 高（可修改源代码，自由扩展功能） | 中（通过 API 参数进行结果控制） |
| **更新速度** | 稳定版本（定期更新，保障用户体验） | 最新特性（代码首发在 GitHub，第一时间体验新功能） | 稳定版本（与网页端 / 桌面客户端保持一致） |
| **快速开始** | 访问 [MinerU 官网](https://mineru.com){:target="_blank" rel="noopener"} | 查看 [GitHub 开源项目](https://github.com/example/mineru){:target="_blank" rel="noopener"} | 阅读[ API 文档](https://mineru.net/apiManage/docs) |



      