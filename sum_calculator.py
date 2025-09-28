import datetime
import os

def calculate_sum_with_timestamp(data_filename, logfile="logs/sum_history.log"):
    """
    计算数据文件的总和，并记录时间差
    
    Args:
        data_filename: 包含数字的数据文件（如 logs/count_page.txt）
        logfile: 新的日志文件，用于记录时间戳和总和
    """
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(logfile) if os.path.dirname(logfile) else '.', exist_ok=True)
    
    # 读取数字文件并计算总和（支持两列格式：文件名\t页数）
    try:
        with open(data_filename, 'r', encoding='utf-8') as f:
            numbers = []
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 检查是否是两列格式（文件名\t页数）
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            page_count = int(parts[1])
                            numbers.append(page_count)
                        except ValueError:
                            print(f"警告: 无法解析页数 '{parts[1]}' 在行: {line}")
                            continue
                else:
                    # 兼容旧格式（只有数字）
                    try:
                        number = int(line)
                        numbers.append(number)
                    except ValueError:
                        print(f"警告: 无法解析数字 '{line}'")
                        continue

        current_sum = sum(numbers)
        file_count = len(numbers)
    except FileNotFoundError:
        print(f"错误: 数据文件 {data_filename} 不存在")
        return
    except Exception as e:
        print(f"读取数据文件错误: {e}")
        return
    
    current_time = datetime.datetime.now()
    
    # 读取上次记录（从新的日志文件）
    last_timestamp = None
    last_sum = 0
    
    try:
        with open(logfile, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line:
                    parts = last_line.split(',')
                    if len(parts) >= 2:
                        last_timestamp = datetime.datetime.fromisoformat(parts[0])
                        last_sum = int(parts[1])
    except FileNotFoundError:
        print("首次运行，创建新的日志文件")
    except Exception as e:
        print(f"读取日志文件错误: {e}")
    
    # 计算时间差
    if last_timestamp:
        time_diff = (current_time - last_timestamp).total_seconds()
        sum_increment = current_sum - last_sum
        increment_per_second = sum_increment / time_diff if time_diff > 0 else 0
        
        print(f"时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"处理文件数: {file_count}")
        print(f"当前总页数: {current_sum}")
        print(f"距离上次计算时间差: {time_diff:.2f}秒")
        print(f"页数增量: {sum_increment}")
        print(f"平均增速: {increment_per_second:.2f}页/秒")
    else:
        print(f"时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"处理文件数: {file_count}")
        print(f"当前总页数: {current_sum}")
        print("首次记录")
    
    # 记录本次结果到新的日志文件
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(f"{current_time.isoformat()},{current_sum}\n")
    
    print(f"记录已保存到: {logfile}")

# 使用
if __name__ == "__main__":
    # 第一个参数是数据文件（原有的 count_page.txt）
    # 第二个参数是新的日志文件（会自动创建）
    calculate_sum_with_timestamp("logs/count_page.txt", "logs/sum_history.log")