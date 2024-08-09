<div align="center" xmlns="http://www.w3.org/1999/html">
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
[English](README.md) | [简体中文](README_zh-CN.md)


<!-- hot link -->
<p align="center">
<a href="https://github.com/opendatalab/PDF-Extract-Kit">PDF-Extract-Kit: 高质量PDF解析工具箱</a>🔥🔥🔥
</p>

<!-- join us -->
<p align="center">
    👋 join us on <a href="https://discord.gg/gPxmVeGC" target="_blank">Discord</a> and <a href="https://cdn.vansin.top/internlm/mineru.jpg" target="_blank">WeChat</a>
</p>

</div>


# 更新记录
- 2024/08/09 0.7.0b1发布，简化安装步骤提升易用性，加入表格识别功能
- 2024/08/01 0.6.2b1发布，优化了依赖冲突问题和安装文档
- 2024/07/05 首次开源


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
        <li><a href="#使用">使用方式</a>
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
MinerU诞生于[书生-浦语](https://github.com/InternLM/InternLM)的预训练过程中，我们将会集中精力解决科技文献中的符号转化问题，希望在大模型时代为科技发展做出贡献。
相比国内外知名商用产品MinerU还很年轻，如果遇到问题或者结果不及预期请到[issue](https://github.com/opendatalab/MinerU/issues)提交问题，同时**附上相关PDF**。

https://github.com/user-attachments/assets/4bea02c9-6d54-4cd6-97ed-dff14340982c

## 主要功能

- 删除页眉、页脚、脚注、页码等元素，保持语义连贯
- 对多栏输出符合人类阅读顺序的文本
- 保留原文档的结构，包括标题、段落、列表等
- 提取图像、图片标题、表格、表格标题
- 自动识别文档中的公式并将公式转换成latex
- 自动识别文档中的表格并将表格转换成latex
- 乱码PDF自动检测并启用OCR
- 支持CPU和GPU环境
- 支持windows/linux/mac平台


## 快速开始

如果遇到任何安装问题，请先查询 <a href="#faq">FAQ</a> </br>
如果遇到解析效果不及预期，参考 <a href="#known-issue">Known Issue</a></br>
有3种不同方式可以体验MinerU的效果：
- [在线体验(无需任何安装)](#在线体验)
- [使用CPU快速体验（Windows，Linux，Mac）](#使用cpu快速体验)
- [Linux/Windows + CUDA](#使用gpu)


**⚠️安装前必看——软硬件环境支持说明**

为了确保项目的稳定性和可靠性，我们在开发过程中仅对特定的软硬件环境进行优化和测试。这样当用户在推荐的系统配置上部署和运行项目时，能够获得最佳的性能表现和最少的兼容性问题。

通过集中资源和精力于主线环境，我们团队能够更高效地解决潜在的BUG，及时开发新功能。

在非主线环境中，由于硬件、软件配置的多样性，以及第三方依赖项的兼容性问题，我们无法100%保证项目的完全可用性。因此，对于希望在非推荐环境中使用本项目的用户，我们建议先仔细阅读文档以及FAQ，大多数问题已经在FAQ中有对应的解决方案，除此之外我们鼓励社区反馈问题，以便我们能够逐步扩大支持范围。

<table>
    <tr>
        <td colspan="3" rowspan="2">操作系统</td>
    </tr>
    <tr>
        <td>Ubuntu 22.04 LTS</td>
        <td>Windows 10 / 11</td>
        <td>macOS 11+</td>
    </tr>
    <tr>
        <td colspan="3">CPU</td>
        <td>x86_64</td>
        <td>x86_64</td>
        <td>x86_64 / arm64</td>
    </tr>
    <tr>
        <td colspan="3">内存</td>
        <td colspan="3">大于等于16GB，推荐32G以上</td>
    </tr>
    <tr>
        <td colspan="3">python版本</td>
        <td colspan="3">3.10</td>
    </tr>
    <tr>
        <td colspan="3">Nvidia Driver 版本</td>
        <td>latest(专有驱动)</td>
        <td>latest</td>
        <td>None</td>
    </tr>
    <tr>
        <td colspan="3">CUDA环境</td>
        <td>自动安装[12.1(pytorch)+11.8(paddle)]</td>
        <td>11.8(手动安装)+cuDNN v8.7.0(手动安装)</td>
        <td>None</td>
    </tr>
    <tr>
        <td rowspan="2">GPU硬件支持列表</td>
        <td colspan="2">最低要求 8G+显存</td>
        <td colspan="2">3060ti/3070/3080/3080ti/4060/4070/4070ti<br>
        8G显存仅可开启lavout和公式识别加速</td>
        <td rowspan="2">None</td>
    </tr>
    <tr>
        <td colspan="2">推荐配置 16G+显存</td>
        <td colspan="2">3090/3090ti/4070tisuper/4080/4090<br>
        16G及以上可以同时开启layout，公式识别和ocr加速</td>
    </tr>
</table>

### 在线体验

[在线体验点击这里](https://opendatalab.com/OpenSourceTools/Extractor/PDF)


### 使用CPU快速体验

#### 1. 安装magic-pdf
最新版本国内镜像源同步可能会有延迟，请耐心等待
```bash
conda create -n MinerU python=3.10
conda activate MinerU
pip install magic-pdf[full]==0.6.2b1 detectron2 --extra-index-url https://wheels.myhloli.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```
#### 2. 下载模型权重文件

详细参考 [如何下载模型文件](docs/how_to_download_models_zh_cn.md)
> ❗️模型下载后请务必检查模型文件是否下载完整
> 
> 请检查目录下的模型文件大小与网页上描述是否一致，如果可以的话，最好通过sha256校验模型是否下载完整

#### 3. 拷贝配置文件并进行配置
在仓库根目录可以获得 [magic-pdf.template.json](magic-pdf.template.json) 配置模版文件
> ❗️务必执行以下命令将配置文件拷贝到【用户目录】下，否则程序将无法运行
> 
>  windows的用户目录为 "C:\Users\用户名", linux用户目录为 "/home/用户名", macOS用户目录为 "/Users/用户名"
```bash
cp magic-pdf.template.json ~/magic-pdf.json
```

在用户目录中找到magic-pdf.json文件并配置"models-dir"为[2. 下载模型权重文件](#2-下载模型权重文件)中下载的模型权重文件所在目录
> ❗️务必正确配置模型权重文件所在目录的【绝对路径】，否则会因为找不到模型文件而导致程序无法运行
>
> windows系统中此路径应包含盘符，且需把路径中所有的"\"替换为"/",否则会因为转义原因导致json文件语法错误。
> 
> 例如：模型放在D盘根目录的models目录，则model-dir的值应为"D:/models"
```json
{
  // other config
  "models-dir": "D:/models",
  "table-config": {
        "is_table_recog_enable": false, // 表格识别功能默认是关闭的，如果需要修改此处的值
        "max_time": 400
    }
}
```


### 使用GPU
如果您的设备支持CUDA，且满足主线环境中的显卡要求，则可以使用GPU加速，请根据自己的系统选择适合的教程：

- [Ubuntu22.04LTS + GPU](docs/README_Ubuntu_CUDA_Acceleration_zh_CN.md)
- [Windows10/11 + GPU](docs/README_Windows_CUDA_Acceleration_zh_CN.md)


## 使用

### 命令行

```bash
magic-pdf --help
Usage: magic-pdf [OPTIONS]

Options:
  -v, --version                display the version and exit
  -p, --path PATH              local pdf filepath or directory  [required]
  -o, --output-dir TEXT        output local directory
  -m, --method [ocr|txt|auto]  the method for parsing pdf.  
                               ocr: using ocr technique to extract information from pdf,
                               txt: suitable for the text-based pdf only and outperform ocr,
                               auto: automatically choose the best method for parsing pdf
                                  from ocr and txt.
                               without method specified, auto will be used by default. 
  --help                       Show this message and exit.


## show version
magic-pdf -v

## command line example
magic-pdf -p {some_pdf} -o {some_output_dir} -m auto
```

其中 `{some_pdf}` 可以是单个pdf文件，也可以是一个包含多个pdf文件的目录。
运行完命令后输出的结果会保存在`{some_output_dir}`目录下, 输出的文件列表如下

```text
├── some_pdf.md                 # markdown 文件
├── images                      # 存放图片目录
├── layout.pdf                  # layout 绘图
├── middle.json                 # minerU 中间处理结果
├── model.json                  # 模型推理结果
├── origin.pdf                  # 原 pdf 文件
└── spans.pdf                   # 最小粒度的bbox位置信息绘图
```

更多有关输出文件的信息，请参考[输出文件说明](docs/output_file_zh_cn.md)


### API

处理本地磁盘上的文件
```python
image_writer = DiskReaderWriter(local_image_dir)
image_dir = str(os.path.basename(local_image_dir))
jso_useful_key = {"_pdf_type": "", "model_list": []}
pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
pipe.pipe_classify()
pipe.pipe_analyze()
pipe.pipe_parse()
md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
```

处理对象存储上的文件
```python
s3pdf_cli = S3ReaderWriter(pdf_ak, pdf_sk, pdf_endpoint)
image_dir = "s3://img_bucket/"
s3image_cli = S3ReaderWriter(img_ak, img_sk, img_endpoint, parent_path=image_dir)
pdf_bytes = s3pdf_cli.read(s3_pdf_path, mode=s3pdf_cli.MODE_BIN)
jso_useful_key = {"_pdf_type": "", "model_list": []}
pipe = UNIPipe(pdf_bytes, jso_useful_key, s3image_cli)
pipe.pipe_classify()
pipe.pipe_analyze()
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
- [x] 表格识别
- [ ] 化学式识别
- [ ] 几何图形识别


# Known Issue
- 阅读顺序基于规则的分割，在一些情况下会乱序
- 不支持竖排文字
- 列表、代码块、目录在layout模型里还没有支持
- 漫画书、艺术图册、小学教材、习题尚不能很好解析
- 在一些公式密集的PDF上强制启用OCR效果会更好
- 如果您要处理包含大量公式的pdf,强烈建议开启OCR功能。使用pymuPDF提取文字的时候会出现文本行互相重叠的情况导致公式插入位置不准确。
- **表格识别**目前处于测试阶段，识别速度较慢，识别准确度有待提升。以下是我们在Ubuntu 22.04 LTS + NVIDIA GeForce RTX 4090环境下的一些性能测试结果，可供参考。

| 表格大小     | 解析耗时        | 
|---------------|----------------------------| 
| 6\*5 55kb     | 37s                   | 
| 16\*12 284kb  | 3m18s                 | 
| 44\*7 559kb   | 4m12s                 | 
 


# FAQ
[常见问题](docs/FAQ_zh_cn.md)
[FAQ](docs/FAQ.md)


# All Thanks To Our Contributors

<a href="https://github.com/opendatalab/MinerU/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=opendatalab/MinerU" />
</a>

# License Information

[LICENSE.md](LICENSE.md)

本项目目前采用PyMuPDF以实现高级功能，但因其遵循AGPL协议，可能对某些使用场景构成限制。未来版本迭代中，我们计划探索并替换为许可条款更为宽松的PDF处理库，以提升用户友好度及灵活性。

# Acknowledgments
- [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)
- [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy)
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






