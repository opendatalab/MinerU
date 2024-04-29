import subprocess
import os
def check_shell(cmd):
    res = os.system(cmd)
    assert res == 0

def count_folders_and_check_contents(directory):
    # 获取目录下的所有文件和文件夹
    contents = os.listdir(directory)
    folder_count = 0
    for item in contents:
        # 检查是否为文件夹
        if os.path.isdir(os.path.join(directory, item)):
            # 检查文件夹是否为空
            folder_path = os.path.join(directory, item)
            for folder in os.listdir(folder_path):
                folder_count = folder_count + 1
                assert os.listdir(folder_path) is not None
    print (folder_count)
    assert folder_count == 13


if __name__ == "__main__":
    count_folders_and_check_contents("/home/quyuan/code/Magic-PDF/Magic-PDF/Magic-PDF/ci") 