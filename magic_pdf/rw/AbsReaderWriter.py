from abc import ABC, abstractmethod


class AbsReaderWriter(ABC):
    """
    同时支持二进制和文本读写的抽象类
    """
    MODE_TXT = "text"
    MODE_BIN = "binary"

    def __init__(self, parent_path):
        # 初始化代码可以在这里添加，如果需要的话
        self.parent_path = parent_path # 对于本地目录是父目录，对于s3是会写到这个path下。

    @abstractmethod
    def read(self, path: str, mode=MODE_TXT):
        """
        无论对于本地还是s3的路径，检查如果path是绝对路径，那么就不再 拼接parent_path, 如果是相对路径就拼接parent_path
        """
        raise NotImplementedError

    @abstractmethod
    def write(self, content: str, path: str, mode=MODE_TXT):
        """
        无论对于本地还是s3的路径，检查如果path是绝对路径，那么就不再 拼接parent_path, 如果是相对路径就拼接parent_path
        """
        raise NotImplementedError

    @abstractmethod
    def read_jsonl(self, path: str, byte_start=0, byte_end=None, encoding='utf-8'):
        """
        无论对于本地还是s3的路径，检查如果path是绝对路径，那么就不再 拼接parent_path, 如果是相对路径就拼接parent_path
        """
        raise NotImplementedError
