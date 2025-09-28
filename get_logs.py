#!/usr/bin/env python3
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess # 导入 subprocess 模块
import shlex      # 导入 shlex 模块，用于安全地引用 shell 命令参数

def count_files_in_directory_ls(directory_path):
    """
    使用 'ls | wc -l' 命令统计目录中的文件和子目录数量。
    
    Args:
        directory_path (str): 目录路径
        
    Returns:
        int: 目录中的条目数量（文件+子目录），如果发生错误则返回 -1。
    """
    try:
        path = Path(directory_path)
        
        # 目录存在性及类型检查
        if not path.exists():
            print(f"Error: Directory {directory_path} does not exist", file=sys.stderr)
            return -1
            
        if not path.is_dir():
            print(f"Error: {directory_path} is not a directory", file=sys.stderr)
            return -1
            
        # 构建 shell 命令：使用 ls 结合 wc -l 统计条目
        # shlex.quote 用于安全地引用路径，防止路径中包含空格或特殊字符导致命令解析错误
        # 注意：ls 默认会列出文件和子目录，wc -l 会统计行数。
        # 如果目录为空，ls 不会输出任何内容，wc -l 会返回 0。
        command = f"ls {shlex.quote(directory_path)} | wc -l"
        
        # 执行 shell 命令
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        
        # wc -l 的输出是行数，需要去除空白符并转换为整数
        file_count = int(result.stdout.strip())
        return file_count
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing shell command for {directory_path}: {e}", file=sys.stderr)
        print(f"STDOUT: {e.stdout.strip()}", file=sys.stderr)
        print(f"STDERR: {e.stderr.strip()}", file=sys.stderr)
        return -1
    except ValueError:
        print(f"Error parsing count from command output for {directory_path}: '{result.stdout.strip()}' is not a valid number", file=sys.stderr)
        return -1
    except Exception as e:
        print(f"Unexpected error counting items in {directory_path}: {e}", file=sys.stderr)
        return -1

def load_previous_data(data_file):
    """从 JSON 文件加载之前的数据（如果存在）"""
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading previous data: {e}", file=sys.stderr)
    return None

def save_current_data(data_file, current_data):
    """将当前数据保存到 JSON 文件"""
    try:
        with open(data_file, 'w') as f:
            json.dump(current_data, f)
    except Exception as e:
        print(f"Error saving current data: {e}", file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_logs.py <node_specific_value>", file=sys.stderr)
        sys.exit(1)
    
    node_specific_value = sys.argv[1]

    # Get data directories from environment variables with defaults
    temp_base = os.getenv('TEMP_BASE', '/tmp')
    source_base = os.getenv('SOURCE_BASE', '/ssd/chunks')
    processed_base = os.getenv('PROCESSED_BASE', '/ssd/processed')

    data_file = f"{temp_base}/chunk_{node_specific_value}_stats.json"

    # 定义要监控的目录
    source_dir = f"{source_base}/chunk{node_specific_value}/"
    processed_dir = f"{processed_base}/chunk_{node_specific_value}"
    
    # 使用 'ls | wc -l' 方法获取当前文件数量
    source_count = count_files_in_directory_ls(source_dir)
    processed_count = count_files_in_directory_ls(processed_dir)
    
    if source_count == -1 or processed_count == -1:
        sys.exit(1)
    
    # 获取当前时间
    current_time = time.time()
    current_datetime = datetime.now().isoformat()
    
    # 加载之前的数据
    previous_data = load_previous_data(data_file)
    
    # 准备要保存的当前数据
    current_data = {
        "source_count": source_count,
        "processed_count": processed_count,
        "timestamp": current_time,
        "datetime": current_datetime
    }
    
    # 如果有之前的数据，则计算指标
    if previous_data:
        time_diff = current_time - previous_data["timestamp"]
        processed_diff = processed_count - previous_data["processed_count"]
        
        if time_diff > 0:
            processing_rate = processed_diff / time_diff
        else:
            processing_rate = 0
    else:
        time_diff = 0
        processed_diff = 0
        processing_rate = 0
    
    # 保存当前数据以供下次运行
    save_current_data(data_file, current_data)
    
    # 准备结果
    result_data = {
        "chunk_name": f"chunk{node_specific_value}",
        "file_count": source_count, # 注意：这里现在是文件+子目录的总数
        "processed": processed_count, # 注意：这里现在是文件+子目录的总数
        "previous_processed": previous_data["processed_count"] if previous_data else 0,
        "rate": round(processing_rate, 2),
    }
    
    # 以 JSON 格式打印结果
    print(json.dumps(result_data))

if __name__ == "__main__":
    main()