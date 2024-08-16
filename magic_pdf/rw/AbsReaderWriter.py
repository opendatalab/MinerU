from abc import ABC, abstractmethod


class AbsReaderWriter(ABC):
    MODE_TXT = "text"
    MODE_BIN = "binary"
    @abstractmethod
    def read(self, path: str, mode=MODE_TXT):
        raise NotImplementedError

    @abstractmethod
    def write(self, content: str, path: str, mode=MODE_TXT):
        raise NotImplementedError

    @abstractmethod
    def read_offset(self, path: str, offset=0, limit=None) -> bytes:
        raise NotImplementedError
