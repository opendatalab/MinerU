
from abc import ABC, abstractmethod


class DataReader(ABC):

    def read(self, path: str) -> bytes:
        """Read the file.

        Args:
            path (str): file path to read

        Returns:
            bytes: the content of the file
        """
        return self.read_at(path)

    @abstractmethod
    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Read the file at offset and limit.

        Args:
            path (str): the file path
            offset (int, optional): the number of bytes skipped. Defaults to 0.
            limit (int, optional): the length of bytes want to read. Defaults to -1.

        Returns:
            bytes: the content of the file
        """
        pass


class DataWriter(ABC):
    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write the data to the file.

        Args:
            path (str): the target file where to write
            data (bytes): the data want to write
        """
        pass

    def write_string(self, path: str, data: str) -> None:
        """Write the data to file, the data will be encoded to bytes.

        Args:
            path (str): the target file where to write
            data (str): the data want to write
        """

        def safe_encode(data: str, method: str):
            try:
                bit_data = data.encode(encoding=method, errors='replace')
                return bit_data, True
            except:  # noqa
                return None, False

        for method in ['utf-8', 'ascii']:
            bit_data, flag = safe_encode(data, method)
            if flag:
                self.write(path, bit_data)
                break
