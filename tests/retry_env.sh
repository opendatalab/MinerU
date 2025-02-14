#!/bin/bash

max_retries=5
retry_count=0

while true; do
    # prepare env
    #python -m pip install -r requirements-qa.txt
    #python -m pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple
    pip install -e .
    python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    pip install modelscope
    wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
    python download_models.py
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "test.sh 成功执行！"
        break
    else
        let retry_count+=1
        if [ $retry_count -ge $max_retries ]; then
            echo "达到最大重试次数 ($max_retries)，放弃重试。"
            exit 1
        fi
        echo "test.sh 执行失败 (退出码: $exit_code)。尝试第 $retry_count 次重试..."
        sleep 5
    fi
done
