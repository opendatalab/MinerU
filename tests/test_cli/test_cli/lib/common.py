import subprocess
def check_shell(cmd):
    res = subprocess.check_output(cmd, shell=True)
    assert res == 0

def count_folders_and_check_contents(directory):
    # 获取目录下的所有文件和文件夹
    contents = os.listdir(directory)
    folder_count = 0
    for item in contents:
        # 检查是否为文件夹
        if os.path.isdir(os.path.join(directory, item)):
            folder_count += 1
            # 检查文件夹是否为空
            folder_path = os.path.join(directory, item)
            assert os.listdir(folder_path) is not None
    assert folder_count == 3 
