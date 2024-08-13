import subprocess
import os


def check_shell(cmd):
    """
    shell successful
    """
    res = os.system(cmd)
    assert res == 0

def count_folders_and_check_contents(file_path):
    """"
    获取文件夹大小
    """
    check_path = os.path.join(file_path, "auto")
    if os.path.exists(file_path):
        folder_count = os.path.getsize(check_path)
        assert folder_count > 5

def delete_file(file_path):
    """
    删除文件
    """
    if os.path.exists(file_path):
        os.remove(file_path)

if __name__ == "__main__":
    count_folders_and_check_contents("/home/quyuan/code/Magic-PDF/Magic-PDF/Magic-PDF/ci") 