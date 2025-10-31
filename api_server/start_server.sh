#!/bin/bash
# MinerU API 启动脚本

# 环境变量配置
export INPUT_DIR="/workspace/extracted_files/"
export OUTPUT_DIR="api_results/"
export PYTHONUNBUFFERED=1
export GPU_IDS="0"
export VRAM_SIZE_GB="20"
export WORKERS_PER_GPU="1"
export GPU_MEMORY_UTILIZATION="0.5"
export MAX_PAGES="1000"
export BATCH_SIZE="384"
export SHUFFLE="false"
export OFFSET="0"
export SPLIT_PDF_CHUNK_SIZE="10"
export BACKEND="vllm-engine"
export MINERU_MODEL_SOURCE="local"
export TORCHDYNAMO_VERBOSE="1"
export TORCH_LOGS="+dynamo"
export OMP_NUM_THREADS="3"
export MKL_NUM_THREADS="3"
export OPENBLAS_NUM_THREADS="3"
export MM_PROCESSOR_CACHE_GB="0"
export API_HOST="0.0.0.0"
export API_PORT="8001"
export MINERU_TOOLS_CONFIG_JSON="/data/MinerU/mineru.json"

# 创建必要目录
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "启动 MinerU API 服务器..."
cd "$(dirname "$0")"
python3 api_server.py > logs/server.log 2>&1