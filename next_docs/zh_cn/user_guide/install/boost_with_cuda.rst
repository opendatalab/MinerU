使用 CUDA 加速
================

如果您的设备支持 CUDA 并符合主线环境的 GPU 要求，您可以使用 GPU 加速。请选择适合您系统的指南：

-  :ref:`ubuntu_22_04_lts_section`
-  :ref:`windows_10_or_11_section`
-  使用 Docker 快速部署
 
.. admonition:: Important
    :class: tip

    Docker 需要至少 16GB 显存的 GPU，并且所有加速功能默认启用。
   
    在运行此 Docker 容器之前，您可以使用以下命令检查您的设备是否支持 Docker 上的 CUDA 加速。

    .. code-block:: sh

      bash  docker run --rm --gpus=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

.. code:: sh

   wget https://github.com/opendatalab/MinerU/raw/master/Dockerfile
   docker build -t mineru:latest .
   docker run --rm -it --gpus=all mineru:latest /bin/bash
   magic-pdf --help


.. _ubuntu_22_04_lts_section:

Ubuntu 22.04 LTS
----------------

1.检查 NVIDIA 驱动程序是否已安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: sh

   nvidia-smi

如果您看到类似以下的信息，则表示 NVIDIA 驱动程序已安装，可以跳过第 2 步。

.. note::

   ``CUDA 版本`` 应 >= 12.1，如果显示的版本号小于 12.1，请升级驱动程序。

.. code:: text

   +---------------------------------------------------------------------------------------+
   | NVIDIA-SMI 537.34                 Driver Version: 537.34       CUDA Version: 12.2     |
   |-----------------------------------------+----------------------+----------------------+
   | GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
   |                                         |                      |               MIG M. |
   |=========================================+======================+======================|
   |   0  NVIDIA GeForce RTX 3060 Ti   WDDM  | 00000000:01:00.0  On |                  N/A |
   |  0%   51C    P8              12W / 200W |   1489MiB /  8192MiB |      5%      Default |
   |                                         |                      |                  N/A |
   +-----------------------------------------+----------------------+----------------------+


2. 安装驱动程序
~~~~~~~~~~~~~~~~~~~~~

如果没有安装驱动程序，请使用以下命令：

.. code:: sh

   sudo apt-get update
   sudo apt-get install nvidia-driver-545

安装专有驱动程序并在安装后重启计算机。

.. code:: sh

   reboot

3. 安装 Anaconda
~~~~~~~~~~~~~~~~~~

如果已经安装了 Anaconda，请跳过此步骤。

.. code:: sh

   wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
   bash Anaconda3-2024.06-1-Linux-x86_64.sh

在最后一步中输入 ``yes``，关闭终端并重新打开。

4. 使用 Conda 创建环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

指定 Python 版本为 3.10。

.. code:: sh

   conda create -n MinerU python=3.10
   conda activate MinerU

5. 安装应用程序
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com

.. admonition:: Important
    :class: tip

    ❗ 安装完成后，请确保使用以下命令检查 ``magic-pdf`` 的版本：

.. code:: sh

   magic-pdf --version

如果版本号小于 0.7.0，请报告问题。

6. 下载模型
~~~~~~~~~~~~~~~~~~

参考详细说明 :doc:`下载模型权重文件 <download_model_weight_files>`

7. 了解配置文件的位置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

完成 `6. 下载模型 <#6-download-models>`__ 步骤后，脚本将自动在用户目录中生成一个 ``magic-pdf.json`` 文件并配置默认模型路径。您可以在用户目录中找到 ``magic-pdf.json`` 文件。

.. admonition:: Tip
    :class: tip
   
    Linux 用户目录是 “/home/用户名”。

8. 首次运行
~~~~~~~~~~~~

从仓库下载示例文件并测试它。

.. code:: sh

   wget https://github.com/opendatalab/MinerU/raw/master/demo/small_ocr.pdf
   magic-pdf -p small_ocr.pdf -o ./output

9. 测试 CUDA 加速
~~~~~~~~~~~~~~~~~~~~~~~~~

如果您的显卡至少有 **8GB** 显存，请按照以下步骤测试 CUDA 加速：

1. 修改位于用户目录中的 ``magic-pdf.json`` 配置文件中的 ``"device-mode"`` 值。

   .. code:: json

      {
        "device-mode": "cuda"
      }

2. 使用以下命令测试 CUDA 加速：

   .. code:: sh

      magic-pdf -p small_ocr.pdf -o ./output

.. admonition:: Tip
    :class: tip

    CUDA加速是否生效可以根据log中输出的各个阶段cost耗时来简单判断，通常情况下，``layout detection cost`` 和 ``mfr time`` 应提速10倍以上。

10. 启用 OCR 的 CUDA 加速
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 下载 ``paddlepaddle-gpu``。安装将自动启用 OCR 加速。

   .. code:: sh

      python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

2. 使用以下命令测试 OCR 加速：

   .. code:: sh

      magic-pdf -p small_ocr.pdf -o ./output

.. admonition:: Tip
    :class: tip

    CUDA加速是否生效可以根据log中输出的各个阶段cost耗时来简单判断，通常情况下， ``ocr cost`` 应提速10倍以上。

.. _windows_10_or_11_section:

Windows 10/11
--------------

1. 安装 CUDA 和 cuDNN
~~~~~~~~~~~~~~~~~~~~~~~~~

所需版本：CUDA 11.8 + cuDNN 8.7.0

-  CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
-  cuDNN v8.7.0（2022年11月28日发布），适用于 CUDA 11.x：
   https://developer.nvidia.com/rdp/cudnn-archive

2. 安装 Anaconda
~~~~~~~~~~~~~~~~~~

如果已经安装了 Anaconda，您可以跳过此步骤。

下载链接：https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Windows-x86_64.exe

3. 使用 Conda 创建环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 版本必须是 3.10。

.. code:: bash

   conda create -n MinerU python=3.10
   conda activate MinerU

4. 安装应用程序
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com

.. admonition:: Important
    :class: tip

    ❗️安装完成后，请验证 ``magic-pdf`` 的版本：

    .. code:: bash

      magic-pdf --version

    如果版本号小于 0.7.0，请在问题部分报告。

5. 下载模型
~~~~~~~~~~~~~~~~~~

参考详细说明 :doc:`下载模型权重文件 <download_model_weight_files>`

6. 了解配置文件的位置
~~~~~~~~~~~~~~~~~~~~

完成 `5. 下载模型 <#5-download-models>__` 步骤后，脚本将自动在用户目录中生成一个 magic-pdf.json 文件并配置默认模型路径。您可以在【用户目录】中找到 magic-pdf.json 文件。

.. admonition:: Tip
    :class: tip

    Windows 用户目录是 “C:/Users/用户名”。

7. 首次运行
~~~~~~~~~~

从仓库下载示例文件并测试它。

.. code:: powershell

     wget https://github.com/opendatalab/MinerU/raw/master/demo/small_ocr.pdf -O small_ocr.pdf
     magic-pdf -p small_ocr.pdf -o ./output

8. 测试CUDA加速
~~~~~~~~~~~~~~~~

如果您的显卡显存大于等于 **8GB**
，可以进行以下流程，测试CUDA解析加速效果

**1.覆盖安装支持cuda的torch和torchvision**

.. code:: bash

   pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

.. admonition:: Important
    :class: tip

    ❗️务必在命令中指定以下版本

    .. code:: bash

      torch==2.3.1 torchvision==0.18.1

    这是我们支持的最高版本，如果不指定版本会自动安装更高版本导致程序无法运行

**2.修改【用户目录】中配置文件magic-pdf.json中”device-mode”的值**

.. code:: json

   {
     "device-mode":"cuda"
   }

**3.运行以下命令测试cuda加速效果**

.. code:: bash

   magic-pdf -p small_ocr.pdf -o ./output

.. admonition:: Tip
    :class: tip

    提示：CUDA加速是否生效可以根据log中输出的各个阶段的耗时来简单判断，通常情况下，\ ``layout detection time`` 和 ``mfr time`` 应提速10倍以上。

9. 为ocr开启cuda加速
~~~~~~~~~~~~~~~~~~~~~~~

**1.下载paddlepaddle-gpu, 安装完成后会自动开启ocr加速**

.. code:: bash

   pip install paddlepaddle-gpu==2.6.1

**2.运行以下命令测试ocr加速效果**

.. code:: bash

   magic-pdf -p small_ocr.pdf -o ./output

.. admonition:: Tip
    :class: tip   

    提示：CUDA加速是否生效可以根据log中输出的各个阶段cost耗时来简单判断，通常情况下，\ ``ocr time``\ 应提速10倍以上。
