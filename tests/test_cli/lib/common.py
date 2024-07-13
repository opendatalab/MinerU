import subprocess
import os
def check_shell(cmd):
    res = os.system(cmd)
    assert res == 0

def count_folders_and_check_contents(file_path):
    # 获取目录下的所有文件和文件夹
    if os.path.exists(file_path):
        folder_count = os.path.getsize(file_path)
        assert folder_count > 0


if __name__ == "__main__":
    count_folders_and_check_contents("/home/quyuan/code/Magic-PDF/Magic-PDF/Magic-PDF/ci") 