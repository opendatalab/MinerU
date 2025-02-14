

Config
=========

File **magic-pdf.json** is typically located in the **${HOME}** directory under a Linux system or in the **C:\Users\{username}** directory under a Windows system.

.. admonition:: Tip 
    :class: tip

    You can override the default location of config file via the following command:
    
    export MINERU_TOOLS_CONFIG_JSON=new_magic_pdf.json



magic-pdf.json
----------------

.. code:: json 

    {
        "bucket_info":{
            "bucket-name-1":["ak", "sk", "endpoint"],
            "bucket-name-2":["ak", "sk", "endpoint"]
        },
        "models-dir":"/tmp/models",
        "layoutreader-model-dir":"/tmp/layoutreader",
        "device-mode":"cpu",
        "layout-config": {
            "model": "layoutlmv3"
        },
        "formula-config": {
            "mfd_model": "yolo_v8_mfd",
            "mfr_model": "unimernet_small",
            "enable": true
        },
        "table-config": {
            "model": "rapid_table",
            "enable": false,
            "max_time": 400    
        },
        "config_version": "1.0.0"
    }




bucket_info
^^^^^^^^^^^^^^
Store the access_key, secret_key and endpoint of AWS S3 Compatible storage config

Example: 

.. code:: text

        {
            "image_bucket":[{access_key}, {secret_key}, {endpoint}],
            "video_bucket":[{access_key}, {secret_key}, {endpoint}]
        }


models-dir
^^^^^^^^^^^^

Store the models download from **huggingface** or **modelshop**. You do not need to modify this field if you download the model using the scripts shipped with **MinerU**


layoutreader-model-dir
^^^^^^^^^^^^^^^^^^^^^^^

Store the models download from **huggingface** or **modelshop**. You do not need to modify this field if you download the model using the scripts shipped with **MinerU**


devide-mode
^^^^^^^^^^^^^^

This field have two options, **cpu** or **cuda**.

**cpu**: inference via cpu

**cuda**: using cuda to accelerate inference


layout-config 
^^^^^^^^^^^^^^^

.. code:: json

    {
        "model": "layoutlmv3"  
    }

layout model can not be disabled now, And we have only kind of layout model currently.


formula-config
^^^^^^^^^^^^^^^^

.. code:: json

    {
        "mfd_model": "yolo_v8_mfd",   
        "mfr_model": "unimernet_small",
        "enable": true 
    }


mfd_model
""""""""""

Specify the formula detection model, options are ['yolo_v8_mfd']


mfr_model
""""""""""
Specify the formula recognition model, options are ['unimernet_small']

Check `UniMERNet <https://github.com/opendatalab/UniMERNet>`_ for more details


enable
""""""""

on-off flag, options are [true, false]. **true** means enable formula inference, **false** means disable formula inference


table-config
^^^^^^^^^^^^^^^^

.. code:: json

   {
        "model": "rapid_table",
        "enable": false,
        "max_time": 400    
    }

model
""""""""

Specify the table inference model, options are ['rapid_table', 'tablemaster', 'struct_eqtable']


max_time
"""""""""

Since table recognition is a time-consuming process, we set a timeout period. If the process exceeds this time, the table recognition will be terminated.



enable
"""""""

on-off flag, options are [true, false]. **true** means enable table inference, **false** means disable table inference


config_version
^^^^^^^^^^^^^^^^

The version of config schema.


.. admonition:: Tip
    :class: tip
    
    Check `Config Schema <https://github.com/opendatalab/MinerU/blob/master/magic-pdf.template.json>`_ for the latest details

