
import io

import requests

from magic_pdf.data.io.base import IOReader, IOWriter


class HttpReader(IOReader):

    def read(self, url: str) -> bytes:
        """Read the file.

        Args:
            path (str): file path to read

        Returns:
            bytes: the content of the file
        """
        return requests.get(url).content

    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Not Implemented."""
        raise NotImplementedError


class HttpWriter(IOWriter):
    def write(self, url: str, data: bytes) -> None:
        """Write file with data.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            data (bytes): the data want to write
        """
        files = {'file': io.BytesIO(data)}
        response = requests.post(url, files=files)
        assert 300 > response.status_code and response.status_code > 199
