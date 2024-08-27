"""common definitions."""
import os


def check_shell(cmd):
    """shell successful."""
    res = os.system(cmd)
    assert res == 0


def cli_count_folders_and_check_contents(file_path, mode):
    """" count cli files."""
    check_path = os.path.join(file_path, mode)
    if os.path.exists(file_path):
        for files in os.listdir(check_path):
            folder_count = os.path.getsize(os.path.join(check_path, files))
            assert folder_count > 0


def sdk_count_folders_and_check_contents(file_path):
    """count folders."""
    if os.path.exists(file_path):
        file_count = os.path.getsize(file_path)
        assert file_count > 0
    else:
        exit(1)


def delete_file(file_path):
    """删除文件."""
    if os.path.exists(file_path):
        os.remove(file_path)
