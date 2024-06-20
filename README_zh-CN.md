<div id="top"></div>
<div align="center">

[![stars](https://img.shields.io/github/stars/magicpdf/Magic-PDF.svg)](https://github.com/magicpdf/Magic-PDF)
[![forks](https://img.shields.io/github/forks/magicpdf/Magic-PDF.svg)](https://github.com/magicpdf/Magic-PDF)
[![license](https://img.shields.io/github/license/magicpdf/Magic-PDF.svg)](https://github.com/magicpdf/Magic-PDF/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/magicpdf/Magic-PDF)](https://github.com/magicpdf/Magic-PDF/issues)
[![open issues](https://img.shields.io/github/issues-raw/magicpdf/Magic-PDF)](https://github.com/magicpdf/Magic-PDF/issues)

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

<div align="center">

</div>

# Magic-PDF

### 简介

Magic-PDF 是一款将 PDF 转化为 markdown 格式的工具。支持转换本地文档或者位于支持S3协议对象存储上的文件。

主要功能包含

- 支持多种前端模型输入
- 删除页眉、页脚、脚注、页码等元素
- 符合人类阅读顺序的排版格式
- 保留原文档的结构和格式，包括标题、段落、列表等
- 提取图像和表格并在markdown中展示
- 将公式转换成latex
- 乱码PDF自动识别并转换
- 支持cpu和gpu环境
- 支持windows/linux/mac平台

### 上手指南

###### 配置要求

python 3.9+

###### 使用说明

1.安装Magic-PDF

```bash
pip install magic-pdf[cpu] # 安装 cpu 版本 
或 
pip install magic-pdf[gpu] # 安装 gpu 版本
```

2.通过命令行使用

```bash
magic-pdf --help
```

### 版权说明

[LICENSE.md](https://github.com/magicpdf/Magic-PDF/blob/master/LICENSE.md)

### 鸣谢
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)


