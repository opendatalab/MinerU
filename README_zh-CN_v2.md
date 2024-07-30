<div align="center">
<!-- logo -->
<p align="center">
  <img src="docs/images/MinerU-logo.png" width="300px" style="vertical-align:middle;">
</p>


<!-- icon -->
[![stars](https://img.shields.io/github/stars/opendatalab/MinerU.svg)](https://github.com/opendatalab/MinerU)
[![forks](https://img.shields.io/github/forks/opendatalab/MinerU.svg)](https://github.com/opendatalab/MinerU)
[![open issues](https://img.shields.io/github/issues-raw/opendatalab/MinerU)](https://github.com/opendatalab/MinerU/issues)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/opendatalab/MinerU)](https://github.com/opendatalab/MinerU/issues)
[![PyPI version](https://badge.fury.io/py/magic-pdf.svg)](https://badge.fury.io/py/magic-pdf)
[![Downloads](https://static.pepy.tech/badge/magic-pdf)](https://pepy.tech/project/magic-pdf)
[![Downloads](https://static.pepy.tech/badge/magic-pdf/month)](https://pepy.tech/project/magic-pdf)
<a href="https://trendshift.io/repositories/11174" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11174" alt="opendatalab%2FMinerU | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<!-- language -->
[English](README.md) | [简体中文](README_zh-CN.md) | [日本語](README_ja-JP.md)


<!-- hot link -->
<p align="center">
<a href="https://github.com/opendatalab/MinerU">MinerU: 端到端的PDF解析工具（基于PDF-Extract-Kit）支持PDF转Markdown</a>🚀🚀🚀<br>
<a href="https://github.com/opendatalab/PDF-Extract-Kit">PDF-Extract-Kit: 高质量PDF解析工具箱</a>🔥🔥🔥
</p>

<!-- join us -->
<p align="center">
    👋 join us on <a href="https://discord.gg/AsQMhuMN" target="_blank">Discord</a> and <a href="https://cdn.vansin.top/internlm/mineru.jpg" target="_blank">WeChat</a>
</p>

</div>


# 更新记录

- 2024/07/18 首次开源


<!-- TABLE OF CONTENT -->
<details open="open">
  <summary><h2 style="display: inline-block">文档目录</h2></summary>
  <ol>
    <li>
      <a href="#mineru">MinerU</a>
      <ul>
        <li><a href="#项目简介">项目简介</a></li>
        <li><a href="#主要功能">主要功能</a></li>
        <li><a href="#快速开始">快速开始</a>
            <ul>
            <li><a href="#在线体验">在线体验</a></li>
            <li><a href="#使用cpu快速体验">使用CPU快速体验</a></li>
            <li><a href="#使用gpu">使用GPU</a></li>
            </ul>
        </li>
        <li><a href="#使用">使用</a>
            <ul>
            <li><a href="#命令行">命令行</a></li>
            <li><a href="#api">API</a></li>
            <li><a href="#二次开发">二次开发指南</a></li>
            </ul>
        </li>
      </ul>
    </li>
    <li><a href="#todo">TODO List</a></li>
    <li><a href="#known-issue">Known Issue</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#all-thanks-to-our-contributors">Contributors</a></li>
    <li><a href="#license-information">License Information</a></li>
    <li><a href="#acknowledgments">Acknowledgements</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#star-history">Star History</a></li>
    <li><a href="#magic-doc">magic-doc快速提取PPT/DOC/PDF</a></li>
    <li><a href="#magic-html">magic-html提取混合网页内容</a></li>
    <li><a href="#links">Links</a></li>
  </ol>
</details>



# MinerU
## 项目简介
MinerU是一款将PDF转化为机器可读格式的工具（如markdown、json），可以很方便地抽取为任意格式。
MinerU诞生于[书生-浦语](https://github.com/InternLM/InternLM)的预训练过程中，我们将会集中精力解决科技文献中的符号转化问题，以此在大模型时代推动人类科技的发展。

## 主要功能

- 删除页眉、页脚、脚注、页码等元素，保持语义连贯
- 符合人类阅读顺序的排版格式
- 保留原文档的结构和格式，包括标题、段落、列表等
- 提取图像、图片标题、表格、表格标题
- 自动识别文档中的公式并将公式转换成latex
- 乱码PDF自动检测并启用OCR
- 支持CPU和GPU环境
- 支持windows/linux/mac平台


## 快速开始

如果遇到任何问题，请先查询<a href="#faq">FAQ</a>
如果遇到效果不及预期，查询<a href="#known-issue">Known Issue</a>
有3种不同方式可以体验MinerU的效果：
- 在线体验
- 使用CPU快速体验（Windows，Linux，Mac）
- Linux/Windows + GPU

### 在线体验

[在线体验点击这里](TODO)

### 使用CPU快速体验

```bash
 command to install magic-pdf[full]
```


### 使用GPU
- [ubuntu22.04 + GPU]()
- [windows10/11 + GPU]()


## 使用

### 命令行

TODO

### API

处理本地磁盘上的文件
```python
image_writer = DiskReaderWriter(local_image_dir)
image_dir = str(os.path.basename(local_image_dir))
jso_useful_key = {"_pdf_type": "", "model_list": model_json}
pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
pipe.pipe_classify()
pipe.pipe_parse()
md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
```

处理对象存储上的文件
```python
s3pdf_cli = S3ReaderWriter(pdf_ak, pdf_sk, pdf_endpoint)
image_dir = "s3://img_bucket/"
s3image_cli = S3ReaderWriter(img_ak, img_sk, img_endpoint, parent_path=image_dir)
pdf_bytes = s3pdf_cli.read(s3_pdf_path, mode=s3pdf_cli.MODE_BIN)
jso_useful_key = {"_pdf_type": "", "model_list": model_json}
pipe = UNIPipe(pdf_bytes, jso_useful_key, s3image_cli)
pipe.pipe_classify()
pipe.pipe_parse()
md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
```

详细实现可参考 
- [demo.py 最简单的处理方式](demo/demo.py)
- [magic_pdf_parse_main.py 能够更清晰看到处理流程](demo/magic_pdf_parse_main.py)


### 二次开发

TODO

# TODO

- [ ] 基于语义的阅读顺序
- [ ] 正文中列表识别
- [ ] 正文中代码块识别
- [ ] 目录识别
- [ ] 表格识别
- [ ] 化学式识别
- [ ] 几何图形识别


# Known Issue
- 阅读顺序基于规则的分割，在一些情况下会乱序
- 列表、代码块、目录在layout模型里还没有支持
- 漫画书、艺术图册、小学教材、习题尚不能很好解析

好消息是，这些我们正在努力实现！

# FAQ
[常见问题](docs/FAQ_zh_cn.md)
[FAQ](docs/FAQ.md)


# All Thanks To Our Contributors

<a href="https://github.com/opendatalab/MinerU/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=opendatalab/MinerU" />
</a>

# License Information

[LICENSE.md](LICENSE.md)

The project currently leverages PyMuPDF to deliver advanced functionalities; however, its adherence to the AGPL license may impose limitations on certain use cases. In upcoming iterations, we intend to explore and transition to a more permissively licensed PDF processing library to enhance user-friendliness and flexibility.

# Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [fast-langdetect](https://github.com/LlmKira/fast-langdetect)
- [pdfminer.six](https://github.com/pdfminer/pdfminer.six)

# Citation

```bibtex
@article{he2024opendatalab,
  title={Opendatalab: Empowering general artificial intelligence with open datasets},
  author={He, Conghui and Li, Wei and Jin, Zhenjiang and Xu, Chao and Wang, Bin and Lin, Dahua},
  journal={arXiv preprint arXiv:2407.13773},
  year={2024}
}

@misc{2024mineru,
    title={MinerU: A One-stop, Open-source, High-quality Data Extraction Tool},
    author={MinerU Contributors},
    howpublished = {\url{https://github.com/opendatalab/MinerU}},
    year={2024}
}
```

# Star History

<a>
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date" />
 </picture>
</a>

# Magic-doc
[Magic-Doc](https://github.com/InternLM/magic-doc) Fast speed ppt/pptx/doc/docx/pdf extraction tool

# Magic-html
[Magic-HTML](https://github.com/opendatalab/magic-html) Mixed web page extraction tool

# Links

- [LabelU (A Lightweight Multi-modal Data Annotation Tool)](https://github.com/opendatalab/labelU)
- [LabelLLM (An Open-source LLM Dialogue Annotation Platform)](https://github.com/opendatalab/LabelLLM)
- [PDF-Extract-Kit (A Comprehensive Toolkit for High-Quality PDF Content Extraction)](https://github.com/opendatalab/PDF-Extract-Kit)






