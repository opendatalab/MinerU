
安装
=====

如果您遇到任何安装问题，请首先查阅 :doc:`../../additional_notes/faq`。如果解析结果不如预期，可参考 :doc:`../../additional_notes/known_issues`。

.. admonition:: Warning
    :class: tip

    **预安装须知—硬件和软件环境支持**
    
    为了确保项目的稳定性和可靠性，在开发过程中我们仅对特定的硬件和软件环境进行了优化和测试。这确保了在推荐系统配置上部署和运行项目的用户能够获得最佳性能，并且兼容性问题最少。

    通过将资源集中在主线环境中，我们的团队可以更高效地解决潜在的错误并开发新功能。

    在非主线环境中，由于硬件和软件配置的多样性以及第三方依赖项的兼容性问题，我们无法保证100%的项目可用性。因此，对于希望在非推荐环境中使用该项目的用户，我们建议首先仔细阅读文档和常见问题解答。大多数问题在常见问题解答中已经有相应的解决方案。我们也鼓励社区反馈，以帮助我们逐步扩大支持。


.. raw:: html

    <style>
        table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
        }
    </style>
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
            <td>x86_64(暂不支持ARM Linux)</td>
            <td>x86_64(暂不支持ARM Windows)</td>
            <td>x86_64 / arm64</td>
        </tr>
        <tr>
            <td colspan="3">内存</td>
            <td colspan="3">大于等于16GB，推荐32G以上</td>
        </tr>
        <tr>
            <td colspan="3">python版本</td>
            <td colspan="3">3.10 (请务必通过conda创建3.10虚拟环境)</td>
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
            <td colspan="2">3060ti/3070/4060<br>
            8G显存可开启layout、公式识别和ocr加速</td>
            <td rowspan="2">None</td>
        </tr>
        <tr>
            <td colspan="2">推荐配置 10G+显存</td>
            <td colspan="2">3080/3080ti/3090/3090ti/4070/4070ti/4070tisuper/4080/4090<br>
            10G显存及以上可以同时开启layout、公式识别和ocr加速和表格识别加速<br>
            </td>
        </tr>
    </table>


创建环境
~~~~~~~~~~

.. code-block:: shell

    conda create -n MinerU python=3.10
    conda activate MinerU
    pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com


下载模型权重文件
~~~~~~~~~~~~~~~

.. code-block:: shell

    pip install huggingface_hub
    wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
    python download_models_hf.py    

MinerU 已安装，查看 :doc:`../quick_start` 或阅读 :doc:`boost_with_cuda` 以加速推理。

