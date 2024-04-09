from abc import ABC, abstractmethod


class AbsReaderWriter(ABC):
    """
    同时支持二进制和文本读写的抽象类
    """

    def __init__(self):
        # 初始化代码可以在这里添加，如果需要的话
        pass

    @abstractmethod
    def read(self, path: str, mode="text"):
        pass

    @abstractmethod
    def write(self, content: str, path: str, mode="text"):
        pass



