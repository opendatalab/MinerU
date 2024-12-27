"""common definitions."""
import os
import shutil
import re
import json
import torch

def clear_gpu_memory():
    '''
    clear GPU memory
    '''
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

def check_shell(cmd):
    """shell successful."""
    res = os.system(cmd)
    assert res == 0

def update_config_file(file_path, key, value):
    """update config file."""
    with open(file_path, 'r', encoding="utf-8") as fr:
        config  = json.loads(fr.read())
    config[key] = value
        # 保存修改后的内容
    with open(file_path, 'w', encoding='utf-8') as fw:
        json.dump(config, fw, ensure_ascii=False, indent=4)

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

def check_latex_table_exists(file_path):
    """check latex table exists."""
    pattern = r'\\begin\{tabular\}.*?\\end\{tabular\}'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    matches = re.findall(pattern, content, re.DOTALL)
    return len(matches) > 0

def check_html_table_exists(file_path):
    """check html table exists."""
    pattern = r'<table.*?>.*?</table>'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    matches = re.findall(pattern, content, re.DOTALL)
    return len(matches) > 0

def check_close_tables(file_path):
    """delete no tables."""
    latex_pattern = r'\\begin\{tabular\}.*?\\end\{tabular\}'
    html_pattern = r'<table.*?>.*?</table>'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    latex_matches = re.findall(latex_pattern, content, re.DOTALL)
    html_matches = re.findall(html_pattern, content, re.DOTALL)
    if len(latex_matches) == 0 and len(html_matches) == 0:
        return True
    else:
        return False