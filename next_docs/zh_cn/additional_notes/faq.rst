常见问题解答
============

1.在较新版本的mac上使用命令安装pip install magic-pdf[full] zsh: no matches found: magic-pdf[full]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 macOS 上，默认的 shell 从 Bash 切换到了 Z shell，而 Z shell 对于某些类型的字符串匹配有特殊的处理逻辑，这可能导致no matches found错误。 可以通过在命令行禁用globbing特性，再尝试运行安装命令

.. code:: bash

   setopt no_nomatch
   pip install magic-pdf[full]

2.使用过程中遇到_pickle.UnpicklingError: invalid load key, ‘v’.错误
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可能是由于模型文件未下载完整导致，可尝试重新下载模型文件后再试。参考：https://github.com/opendatalab/MinerU/issues/143

3.模型文件应该下载到哪里/models-dir的配置应该怎么填
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

模型文件的路径输入是在”magic-pdf.json”中通过

.. code:: json

   {
     "models-dir": "/tmp/models"
   }

进行配置的。这个路径是绝对路径而不是相对路径，绝对路径的获取可在models目录中通过命令 “pwd” 获取。
参考：https://github.com/opendatalab/MinerU/issues/155#issuecomment-2230216874

4.在WSL2的Ubuntu22.04中遇到报错\ ``ImportError: libGL.so.1: cannot open shared object file: No such file or directory``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WSL2的Ubuntu22.04中缺少\ ``libgl``\ 库，可通过以下命令安装\ ``libgl``\ 库解决：

.. code:: bash

   sudo apt-get install libgl1-mesa-glx

参考：https://github.com/opendatalab/MinerU/issues/388

5.遇到报错 ``ModuleNotFoundError : Nomodulenamed 'fairscale'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

需要卸载该模块并重新安装

.. code:: bash

   pip uninstall fairscale
   pip install fairscale

参考：https://github.com/opendatalab/MinerU/issues/411

6.在部分较新的设备如H100上，使用CUDA加速OCR时解析出的文字乱码。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cuda11对新显卡的兼容性不好，需要升级paddle使用的cuda版本

.. code:: bash

   pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

参考：https://github.com/opendatalab/MinerU/issues/558

7.在部分Linux服务器上，程序一运行就报错 ``非法指令 (核心已转储)`` 或 ``Illegal instruction (core dumped)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可能是因为服务器CPU不支持AVX/AVX2指令集，或cpu本身支持但被运维禁用了，可以尝试联系运维解除限制或更换服务器。

参考：https://github.com/opendatalab/MinerU/issues/591 ，https://github.com/opendatalab/MinerU/issues/736
