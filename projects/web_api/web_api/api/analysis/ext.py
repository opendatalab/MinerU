import os

task_state_map = {
    0: "running",
    1: "done",
    2: "pending"
}


def find_file(file_key, file_dir):
    """
    查询文件
    :param file_key:  文件哈希
    :param file_dir:  文件目录
    :return:
    """
    pdf_path = ""
    for root, subDirs, files in os.walk(file_dir):
        for fileName in files:
            if fileName.startswith(file_key):
                pdf_path = os.path.join(root, fileName)
                break
        if pdf_path:
            break
    return pdf_path
