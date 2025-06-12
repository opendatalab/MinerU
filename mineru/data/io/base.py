from abc import ABC, abstractmethod


class IOReader(ABC):
    @abstractmethod
    def read(self, path: str) -> bytes:
        """Read the file.

        Args:
            path (str): file path to read

        Returns:
            bytes: the content of the file
        """
        pass

    @abstractmethod
    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Read at offset and limit.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            offset (int, optional): the number of bytes skipped. Defaults to 0.
            limit (int, optional): the length of bytes want to read. Defaults to -1.

        Returns:
            bytes: the content of file
        """
        pass


class IOWriter(ABC):

    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write file with data.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            data (bytes): the data want to write
        """
        pass
