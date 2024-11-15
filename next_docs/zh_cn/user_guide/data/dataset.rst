
数据集
======

导入数据类
-----------

数据集
^^^^^^^^

每个 PDF 或图像将形成一个 Dataset。众所周知，PDF 有两种类别：:ref:`TXT <digital_method_section>` 或 :ref:`OCR <ocr_method_section>` 方法部分。从图像中可以获得 ImageDataset，它是 Dataset 的子类；从 PDF 文件中可以获得 PymuDocDataset。ImageDataset 和 PymuDocDataset 之间的区别在于 ImageDataset 仅支持 OCR 解析方法，而 PymuDocDataset 支持 OCR 和 TXT 两种方法。

.. note::

    实际上，有些 PDF 可能是由图像生成的，这意味着它们不支持 `TXT` 方法。目前，由用户保证不会调用 `TXT` 方法来解析图像生成的 PDF

PDF 解析方法
---------------

.. _ocr_method_section:

OCR
^^^^
通过 光学字符识别 技术提取字符。

.. _digital_method_section:

TXT
^^^^^^^^
通过第三方库提取字符，目前我们使用的是 pymupdf。

