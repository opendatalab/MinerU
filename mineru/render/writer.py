# Copyright (c) Opendatalab. All rights reserved.
import os
from abc import ABC, abstractmethod


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
                bit_data = data.encode(encoding=method, errors="replace")
                return bit_data, True
            except:  # noqa
                return None, False

        for method in ["utf-8", "ascii"]:
            bit_data, flag = safe_encode(data, method)
            if flag:
                self.write(path, bit_data)
                break


class FileBasedDataWriter(DataWriter):
    def __init__(self, parent_dir: str = "") -> None:
        """Initialized with parent_dir.

        Args:
            parent_dir (str, optional): the parent directory that may be used within methods. Defaults to ''.
        """
        self._parent_dir = parent_dir

    def write(self, path: str, data: bytes) -> None:
        """Write file with data.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            data (bytes): the data want to write
        """
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        if not os.path.exists(os.path.dirname(fn_path)) and os.path.dirname(fn_path) != "":
            os.makedirs(os.path.dirname(fn_path), exist_ok=True)

        with open(fn_path, "wb") as f:
            f.write(data)
