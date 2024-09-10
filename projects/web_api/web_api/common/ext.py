import hashlib
import mimetypes


def is_pdf(filename, file):
    """
    判断文件是否为PDF格式。

    :param filename: 文件名
    :param file: 文件对象
    :return: 如果文件是PDF格式，则返回True，否则返回False
    """
    # 检查文件扩展名  https://arxiv.org/pdf/2405.08702 pdf链接可能存在不带扩展名的情况，先注释
    # if not filename.endswith('.pdf'):
    #     return False

    # 检查MIME类型
    mime_type, _ = mimetypes.guess_type(filename)
    print(mime_type)
    if mime_type != 'application/pdf':
        return False

    # 可选：读取文件的前几KB内容并检查MIME类型
    # 这一步是可选的，用于更严格的检查
    # if not mimetypes.guess_type(filename, strict=False)[0] == 'application/pdf':
    #     return False

    # 检查文件内容
    file_start = file.read(5)
    file.seek(0)
    if not file_start.startswith(b'%PDF-'):
        return False

    return True


def url_is_pdf(file):
    """
    判断文件是否为PDF格式。

    :param file: 文件对象
    :return: 如果文件是PDF格式，则返回True，否则返回False
    """
    # 检查文件内容
    file_start = file.read(5)
    file.seek(0)
    if not file_start.startswith(b'%PDF-'):
        return False

    return True


def calculate_file_hash(file, algorithm='sha256'):
    """
    计算给定文件的哈希值。

    :param file: 文件对象
    :param algorithm: 哈希算法的名字，如:'sha256', 'md5', 'sha1'等
    :return: 文件的哈希值
    """
    hash_func = getattr(hashlib, algorithm)()
    block_size = 65536  # 64KB chunks
    # with open(file_path, 'rb') as file:
    buffer = file.read(block_size)
    while len(buffer) > 0:
        hash_func.update(buffer)
        buffer = file.read(block_size)
    file.seek(0)
    return hash_func.hexdigest()


def singleton_func(cls):
    instance = {}

    def _singleton(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return _singleton
