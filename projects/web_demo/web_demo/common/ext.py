import hashlib
import mimetypes
import urllib.parse


def is_pdf(filename, file):
    """
    判断文件是否为PDF格式，支持中文名和特殊字符。

    :param filename: 文件名
    :param file: 文件对象
    :return: 如果文件是PDF格式，则返回True，否则返回False
    """
    try:
        # 对文件名进行URL解码，处理特殊字符
        decoded_filename = urllib.parse.unquote(filename)
        
        # 检查MIME类型
        mime_type, _ = mimetypes.guess_type(decoded_filename)
        print(f"Detected MIME type: {mime_type}")
        
        # 某些情况下mime_type可能为None，需要特殊处理
        if mime_type is None:
            # 只检查文件内容的PDF标识
            file_start = file.read(5)
            file.seek(0)  # 重置文件指针
            return file_start.startswith(b'%PDF-')
            
        if mime_type != 'application/pdf':
            return False

        # 检查文件内容的PDF标识
        file_start = file.read(5)
        file.seek(0)  # 重置文件指针
        if not file_start.startswith(b'%PDF-'):
            return False

        return True
        
    except Exception as e:
        print(f"Error checking PDF format: {str(e)}")
        # 发生错误时，仍然尝试通过文件头判断
        try:
            file_start = file.read(5)
            file.seek(0)
            return file_start.startswith(b'%PDF-')
        except:
            return False


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
