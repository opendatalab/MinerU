import os
import shutil

def move_pdfs(root_folder, destination_folder):
    # 遍历根目录及其子目录中的所有文件
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.pdf'):
                # 构建完整的文件路径
                src_path = os.path.join(root, file)
                # 构建目标路径
                dst_path = os.path.join(destination_folder, file)
                
                # 移动文件
                shutil.move(src_path, dst_path)
                print(f'Moved {file} to {destination_folder}')

# 使用方法
root_folder = r'D:\mineru\datasets\datasets'  # 源文件夹路径
destination_folder = r'D:\mineru\datasets\pdf'  # 目标文件夹路径

# 创建目标文件夹如果不存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

move_pdfs(root_folder, destination_folder)