.. xtuner documentation master file, created by
   sphinx-quickstart on Tue Jan  9 16:33:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎来到 MinerU 文档库
==============================================

.. figure:: ./_static/image/logo.png
  :align: center
  :alt: mineru
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong> 一站式、高质量的开源文档提取工具
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/opendatalab/MinerU" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/opendatalab/MinerU/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/opendatalab/MinerU/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>


项目介绍
--------------------

MinerU是一款将PDF转化为机器可读格式的工具（如markdown、json），可以很方便地抽取为任意格式。
MinerU诞生于\ `书生-浦语 <https://github.com/InternLM/InternLM>`__\ 的预训练过程中，我们将会集中精力解决科技文献中的符号转化问题，希望在大模型时代为科技发展做出贡献。
相比国内外知名商用产品MinerU还很年轻，如果遇到问题或者结果不及预期请到\ `issue <https://github.com/opendatalab/MinerU/issues>`__\ 提交问题，同时\ **附上相关PDF**\ 。

.. video:: https://github.com/user-attachments/assets/4bea02c9-6d54-4cd6-97ed-dff14340982c

主要功能
--------

-  删除页眉、页脚、脚注、页码等元素，确保语义连贯
-  输出符合人类阅读顺序的文本，适用于单栏、多栏及复杂排版
-  保留原文档的结构，包括标题、段落、列表等
-  提取图像、图片描述、表格、表格标题及脚注
-  自动识别并转换文档中的公式为LaTeX格式
-  自动识别并转换文档中的表格为LaTeX或HTML格式
-  自动检测扫描版PDF和乱码PDF，并启用OCR功能
-  OCR支持84种语言的检测与识别
-  支持多种输出格式，如多模态与NLP的Markdown、按阅读顺序排序的JSON、含有丰富信息的中间格式等
-  支持多种可视化结果，包括layout可视化、span可视化等，便于高效确认输出效果与质检
-  支持CPU和GPU环境
-  兼容Windows、Linux和Mac平台


用户指南
-------------
.. toctree::
   :maxdepth: 2
   :caption: 用户指南

   user_guide


API 接口
-------------
本章节主要介绍函数、类、类方法的细节信息

目前只提供英文版本的接口文档，请切换到英文版本的接口文档！


附录
------------------
.. toctree::
   :maxdepth: 1
   :caption: 附录

   additional_notes/known_issues
   additional_notes/faq
   additional_notes/glossary


