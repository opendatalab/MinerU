
from abc import ABC, abstractmethod


class AbsReaderWriter(ABC):
    """
    同时支持二进制和文本读写的抽象类
    TODO
    """
    @abstractmethod
    def read(self, path: str):
        pass

    @abstractmethod
    def write(self, path: str, content: str):
        pass
    
    
    
    