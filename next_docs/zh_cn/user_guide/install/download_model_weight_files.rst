下载模型权重文件
===============

模型下载分为初始下载和更新到模型目录。请参考相应的文档以获取如何操作的指示。

初始下载模型文件
--------------
从 Hugging Face 下载模型


使用 Python 脚本从 Hugging Face 下载模型文件

.. code:: bash

   pip install huggingface_hub
   wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
   python download_models_hf.py

该 Python 脚本将自动下载模型文件，并在配置文件中配置模型目录。

配置文件可以在用户目录中找到，文件名为 ``magic-pdf.json``。

如何更新先前下载的模型
-----------------------------------------

1. 通过 Git LFS 下载的模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   由于一些用户的反馈指出使用 git lfs 下载模型文件会出现不完整或导致模型文件损坏的情况，因此不再推荐使用这种方法。

如果您之前通过 git lfs 下载了模型文件，您可以导航到之前的下载目录并使用 ``git pull`` 命令来更新模型。

2. 通过 Hugging Face 或 ModelScope 下载的模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您之前通过 Hugging Face 或 ModelScope 下载了模型，您可以重新运行用于初始下载的 Python 脚本。这将自动将模型目录更新到最新版本。