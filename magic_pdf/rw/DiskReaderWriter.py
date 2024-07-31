import os
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from loguru import logger


class DiskReaderWriter(AbsReaderWriter):
    def __init__(self, parent_path, encoding="utf-8"):
        self.path = parent_path
        self.encoding = encoding

    def read(self, path, mode=AbsReaderWriter.MODE_TXT):
        if os.path.isabs(path):
            abspath = path
        else:
            abspath = os.path.join(self.path, path)
        if not os.path.exists(abspath):
            logger.error(f"file {abspath} not exists")
            raise Exception(f"file {abspath} no exists")
        if mode == AbsReaderWriter.MODE_TXT:
            with open(abspath, "r", encoding=self.encoding) as f:
                return f.read()
        elif mode == AbsReaderWriter.MODE_BIN:
            with open(abspath, "rb") as f:
                return f.read()
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")

    def write(self, content, path, mode=AbsReaderWriter.MODE_TXT):
        if os.path.isabs(path):
            abspath = path
        else:
            abspath = os.path.join(self.path, path)
        directory_path = os.path.dirname(abspath)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        if mode == AbsReaderWriter.MODE_TXT:
            with open(abspath, "w", encoding=self.encoding, errors="replace") as f:
                f.write(content)

        elif mode == AbsReaderWriter.MODE_BIN:
            with open(abspath, "wb") as f:
                f.write(content)
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")

    def read_offset(self, path: str, offset=0, limit=None):
        abspath = path
        if not os.path.isabs(path):
            abspath = os.path.join(self.path, path)
        with open(abspath, "rb") as f:
            f.seek(offset)
            return f.read(limit)


if __name__ == "__main__":
    if 0:
        file_path = "io/test/example.txt"
        drw = DiskReaderWriter("D:\projects\papayfork\Magic-PDF\magic_pdf")

        # 写入内容到文件
        drw.write(b"Hello, World!", path="io/test/example.txt", mode="binary")

        # 从文件读取内容
        content = drw.read(path=file_path)
        if content:
            logger.info(f"从 {file_path} 读取的内容: {content}")
    if 1:
        drw = DiskReaderWriter("/opt/data/pdf/resources/test/io/")
        content_bin = drw.read_offset("1.txt")
        assert content_bin == b"ABCD!"

        content_bin = drw.read_offset("1.txt", offset=1, limit=2)
        assert content_bin == b"BC"

