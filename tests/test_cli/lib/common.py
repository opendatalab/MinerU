"""common definitions."""
import os
import shutil


def check_shell(cmd):
    """shell successful."""
    res = os.system(cmd)
    assert res == 0


def cli_count_folders_and_check_contents(file_path):
    """" count cli files."""
    if os.path.exists(file_path):
        for files in os.listdir(file_path):
            folder_count = os.path.getsize(os.path.join(file_path, files))
            assert folder_count > 0
    assert len(os.listdir(file_path)) > 5

def sdk_count_folders_and_check_contents(file_path):
    """count folders."""
    if os.path.exists(file_path):
        file_count = os.path.getsize(file_path)
        assert file_count > 0
    else:
        exit(1)


def delete_file(path):
    """delete file."""
    if not os.path.exists(path):
        if os.path.isfile(path):
            try:
                os.remove(path)
                print(f"File '{path}' deleted.")
            except TypeError as e:
                print(f"Error deleting file '{path}': {e}")
    elif os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Directory '{path}' and its contents deleted.")
        except TypeError as e:
            print(f"Error deleting directory '{path}': {e}")