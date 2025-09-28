# 定义变量
CHUNK_NUM ?= 0000
MEMORY_LIMIT ?= 64G
NUM_PROCESSES ?= 4
PYTHON_PATH ?= /data/miniconda3/envs/py12/bin/python
MINERU_SCRIPT ?= /data/wxl_mineru/MinerU/demo/ocr_pdf_with_mineru.py
LOG_DIR = /data/ssd/logs/chunk_$(CHUNK_NUM)
INPUT_DIR = /data/ssd/chunks/chunk$(CHUNK_NUM)
OUTPUT_DIR = /data/ssd/processed/chunk_$(CHUNK_NUM)

# 代理和路径设置
HTTP_PROXY = http://10.20.32.201:8888
HTTPS_PROXY = http://10.20.32.201:8888
MODELSCOPE_CACHE = /data/ssd/modelscope_en/hub
CUDA_PATH = /data/apps/cuda/12.8
PATH := $(CUDA_PATH)/bin:$(PATH)
LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

# 线程设置
OMP_NUM_THREADS = 3
MKL_NUM_THREADS = 3
OPENBLAS_NUM_THREADS = 3

.PHONY: run run-simple run-simple-group clean help

run: $(LOG_DIR)
	@echo "Running MinerU processing for chunk $(CHUNK_NUM)"
	@echo "Memory limit: $(MEMORY_LIMIT), Processes: $(NUM_PROCESSES)"
	@http_proxy=$(HTTP_PROXY) https_proxy=$(HTTPS_PROXY) MODELSCOPE_CACHE=$(MODELSCOPE_CACHE) \
	PATH="$(PATH)" LD_LIBRARY_PATH="$(LD_LIBRARY_PATH)" \
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) MKL_NUM_THREADS=$(MKL_NUM_THREADS) OPENBLAS_NUM_THREADS=$(OPENBLAS_NUM_THREADS) \
	$(PYTHON_PATH) $(MINERU_SCRIPT) \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--vram-size-gb 8 \
		--num-processes $(NUM_PROCESSES) \
		--cuda-devices 0,1,2,3,4,5,6,7 \
		--max-tasks-per-worker 100 \
		--log-dir $(LOG_DIR) > $(LOG_DIR)/output.log 2>&1

# 简化版本，不使用cgroup
run-simple:
	@mkdir -p /data/ssd/logs/chunk_0000
	@MODELSCOPE_CACHE=/data/ssd/modelscope_en/hub \
	/data/miniconda3/envs/py12/bin/python /data/wxl_mineru/MinerU/demo/ocr_pdf_with_mineru.py \
		--input-dir /data/ssd/chunks/chunk0000 \
		--output-dir /data/ssd/processed/chunk_0000 \
		--vram-size-gb 8 \
		--num-processes 2 \
		--cuda-devices 0,1,2,3,4,5,6,7,8 \
		--max-tasks-per-worker 10000 \
		--log-dir /data/ssd/logs/chunk_0000 2>&1 | tee /data/ssd/logs/chunk_0000/output.log

# 使用cgroup限制内存到900GB的版本
run-simple-group:
	@mkdir -p /data/ssd/logs/chunk_0000
	@echo "Setting up cgroup v2 with 900GB memory limit..."
	@sudo mkdir -p /sys/fs/cgroup/mineru_group || true
	@sudo sh -c 'echo 966367641600 > /sys/fs/cgroup/mineru_group/memory.max'
	@echo "Starting MinerU with cgroup v2 memory limitation..."
	@sudo sh -c 'echo $$$$ > /sys/fs/cgroup/mineru_group/cgroup.procs'
	@sudo env MODELSCOPE_CACHE=/data/ssd/modelscope_en/hub \
	/data/miniconda3/envs/py12/bin/python /data/wxl_mineru/MinerU/demo/ocr_pdf_with_mineru.py \
		--input-dir /data/ssd/chunks/chunk0000 \
		--output-dir /data/ssd/processed/chunk_0000 \
		--vram-size-gb 8 \
		--num-processes 4 \
		--max-task-duration 1800 \
		--monitor-log-path /data/wxl_mineru/logs/process \
		--cuda-devices 0,1,2,3,4,5,6,7 \
		--log-dir /data/ssd/logs/chunk_0000 2>&1 | tee /data/ssd/logs/chunk_0000/output.log

# 调试版本，显示更多信息
run-debug: $(LOG_DIR)
	@echo "=== Debug Information ==="
	@echo "Python path: $(PYTHON_PATH)"
	@echo "Script: $(MINERU_SCRIPT)"
	@echo "Input dir: $(INPUT_DIR)"
	@echo "Output dir: $(OUTPUT_DIR)"
	@echo "Log dir: $(LOG_DIR)"
	@echo "Processes: $(NUM_PROCESSES)"
	@echo "=== Starting MinerU ==="
	@http_proxy=$(HTTP_PROXY) https_proxy=$(HTTPS_PROXY) MODELSCOPE_CACHE=$(MODELSCOPE_CACHE) \
	$(PYTHON_PATH) $(MINERU_SCRIPT) \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--vram-size-gb 8 \
		--num-processes $(NUM_PROCESSES) \
		--cuda-devices 0,1,2,3,4,5,6,7 \
		--max-tasks-per-worker 100 \
		--log-dir $(LOG_DIR)

$(LOG_DIR):
	@mkdir -p $(LOG_DIR)
	@echo "Created log directory: $(LOG_DIR)"

clean:
	@echo "Cleaning up..."
	@rm -rf $(LOG_DIR)
	@echo "Cleaned log directory: $(LOG_DIR)"

# 检查目录是否存在
check-dirs:
	@echo "=== Checking directories ==="
	@echo "Input dir: $(INPUT_DIR) - $(shell if [ -d "$(INPUT_DIR)" ]; then echo "EXISTS"; else echo "MISSING"; fi)"
	@echo "Output dir: $(OUTPUT_DIR) - $(shell if [ -d "$(OUTPUT_DIR)" ]; then echo "EXISTS"; else echo "MISSING"; fi)"
	@echo "Log dir: $(LOG_DIR) - $(shell if [ -d "$(LOG_DIR)" ]; then echo "EXISTS"; else echo "MISSING"; fi)"
	@echo "Python: $(PYTHON_PATH) - $(shell if [ -f "$(PYTHON_PATH)" ]; then echo "EXISTS"; else echo "MISSING"; fi)"
	@echo "Script: $(MINERU_SCRIPT) - $(shell if [ -f "$(MINERU_SCRIPT)" ]; then echo "EXISTS"; else echo "MISSING"; fi)"
kill-all:
	bash kill_all.sh
# 帮助信息
help:
	@echo "可用目标:"
	@echo "  run CHUNK_NUM=<n>        - 运行处理指定chunk的任务"
	@echo "  run-simple CHUNK_NUM=<n> - 简化版本（实时输出日志）"
	@echo "  run-simple-group         - 使用cgroup限制内存到900GB的简化版本"
	@echo "  run-debug CHUNK_NUM=<n>  - 调试版本（显示详细信息）"
	@echo "  check-dirs CHUNK_NUM=<n> - 检查所需目录和文件"
	@echo "  clean CHUNK_NUM=<n>      - 清理日志目录"
	@echo "  stats                    - 分析logs目录的统计数据"
	@echo "  stats-json               - 输出JSON格式的统计数据"
	@echo "  stats-dir CHUNK_NUM=<n>  - 分析指定chunk的日志统计"
	@echo "  cal                      - 运行sum_calculator.py"
	@echo ""
	@echo "可配置变量:"
	@echo "  CHUNK_NUM=<n>        - 设置chunk编号（默认: 0000）"
	@echo "  MEMORY_LIMIT=<size>  - 设置内存限制（默认: 64G）"
	@echo "  NUM_PROCESSES=<n>    - 设置进程数（默认: 4）"
cal:
	python sum_calculator.py

# 统计分析日志
stats:
	python sum_calcul.py

# 统计分析（JSON格式输出）
stats-json:
	python sum_calcul.py --output-format json

# 统计指定目录的日志
stats-dir:
	python sum_calcul.py --logs-dir $(LOG_DIR)