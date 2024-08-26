#!/bin/bash

# 定义最大重试次数
max_retries=5
retry_count=0

while true; do
    # 执行 test.sh
    conda create -n MinerU python=3.10
    conda activate MinerU
    pip install pytest pytest-cov
    pip install magic-pdf[full]==0.7.0b1 --extra-index-url https://wheels.myhloli.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    git lfs install
    git lfs clone https://www.modelscope.cn/wanderkid/PDF-Extract-Kit.git

    # 获取 命令行 的返回值
    local exit_code=$?

    # 检查是否成功
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
        sleep 5  # 等待 5 秒后重试
    fi
done