下载模型权重文件
==================

模型下载分为初始下载和更新到模型目录。请参考相应的文档以获取如何操作的指示。

首次下载模型文件
-----------------

模型文件可以从 Hugging Face 或 Model Scope下载，由于网络原因，国内用户访问HF可能会失败，请使用 ModelScope。


方法一：从 Hugging Face 下载模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用python脚本 从Hugging Face下载模型文件

.. code:: bash

   pip install huggingface_hub
   wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models_hf.py -O download_models_hf.py
   python download_models_hf.py

python脚本会自动下载模型文件并配置好配置文件中的模型目录

方法二：从 ModelScope 下载模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用python脚本从 ModelScope 下载模型文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   pip install modelscope
   wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
   python download_models.py

python脚本会自动下载模型文件并配置好配置文件中的模型目录

配置文件可以在用户目录中找到，文件名为\ ``magic-pdf.json``

.. admonition:: Tip
    :class: tip

    windows的用户目录为 “C:\Users\用户名”, linux用户目录为 “/home/用户名”, macOS用户目录为 “/Users/用户名”

此前下载过模型，如何更新
--------------------

1. 通过 git lfs 下载过模型
^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Important
    :class: tip

    由于部分用户反馈通过git lfs下载模型文件遇到下载不全和模型文件损坏情况，现已不推荐使用该方式下载。

    0.9.x及以后版本由于PDF-Extract-Kit 1.0更换仓库和新增layout排序模型，不能通过 ``git pull``\命令更新，需要使用python脚本一键更新。

当magic-pdf <= 0.8.1时，如此前通过 git lfs 下载过模型文件，可以进入到之前的下载目录中，通过 ``git pull`` 命令更新模型。

2. 通过 Hugging Face 或 Model Scope 下载过模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如此前通过 HuggingFace 或 Model Scope 下载过模型，可以重复执行此前的模型下载 python 脚本，将会自动将模型目录更新到最新版本。