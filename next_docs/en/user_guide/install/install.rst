
Install 
===============================================================
If you encounter any installation issues, please first consult the FAQ.
If the parsing results are not as expected, refer to the Known Issues.
There are three different ways to experience MinerU

Pre-installation Notice—Hardware and Software Environment Support
------------------------------------------------------------------

To ensure the stability and reliability of the project, we only optimize
and test for specific hardware and software environments during
development. This ensures that users deploying and running the project
on recommended system configurations will get the best performance with
the fewest compatibility issues.

By focusing resources on the mainline environment, our team can more
efficiently resolve potential bugs and develop new features.

In non-mainline environments, due to the diversity of hardware and
software configurations, as well as third-party dependency compatibility
issues, we cannot guarantee 100% project availability. Therefore, for
users who wish to use this project in non-recommended environments, we
suggest carefully reading the documentation and FAQ first. Most issues
already have corresponding solutions in the FAQ. We also encourage
community feedback to help us gradually expand support.

.. raw:: html

   <style>
      table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
      }
   </style>
   <table>
    <tr>
        <td colspan="3" rowspan="2">Operating System</td>
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
        <td colspan="3">Memory</td>
        <td colspan="3">16GB or more, recommended 32GB+</td>
    </tr>
    <tr>
        <td colspan="3">Python Version</td>
        <td colspan="3">3.10</td>
    </tr>
    <tr>
        <td colspan="3">Nvidia Driver Version</td>
        <td>latest (Proprietary Driver)</td>
        <td>latest</td>
        <td>None</td>
    </tr>
    <tr>
        <td colspan="3">CUDA Environment</td>
        <td>Automatic installation [12.1 (pytorch) + 11.8 (paddle)]</td>
        <td>11.8 (manual installation) + cuDNN v8.7.0 (manual installation)</td>
        <td>None</td>
    </tr>
    <tr>
        <td rowspan="2">GPU Hardware Support List</td>
        <td colspan="2">Minimum Requirement 8G+ VRAM</td>
        <td colspan="2">3060ti/3070/3080/3080ti/4060/4070/4070ti<br>
        8G VRAM enables layout, formula recognition acceleration and OCR acceleration</td>
        <td rowspan="2">None</td>
    </tr>
    <tr>
        <td colspan="2">Recommended Configuration 16G+ VRAM</td>
        <td colspan="2">3090/3090ti/4070ti super/4080/4090<br>
        16G VRAM or more can enable layout, formula recognition, OCR acceleration and table recognition acceleration simultaneously
        </td>
    </tr>
   </table>


Create an environment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    conda create -n MinerU python=3.10
    conda activate MinerU
    pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com


Download model weight files
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    pip install huggingface_hub
    wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
    python download_models_hf.py    


The MinerU is installed, Check out :doc:`../quick_start` or reading :doc:`boost_with_cuda` for accelerate inference