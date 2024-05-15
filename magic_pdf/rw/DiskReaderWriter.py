import os
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from loguru import logger


MODE_TXT = "text"
MODE_BIN = "binary"


class DiskReaderWriter(AbsReaderWriter):

    def __init__(self, parent_path, encoding="utf-8"):
        self.path = parent_path
        self.encoding = encoding

    def read(self, path, mode=MODE_TXT):
        if os.path.isabs(path):
            abspath = path
        else:
            abspath = os.path.join(self.path, path)
        if not os.path.exists(abspath):
            logger.error(f"文件 {abspath} 不存在")
            raise Exception(f"文件 {abspath} 不存在")
        if mode == MODE_TXT:
            with open(abspath, "r", encoding=self.encoding) as f:
                return f.read()
        elif mode == MODE_BIN:
            with open(abspath, "rb") as f:
                return f.read()
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")

    def write(self, content, path, mode=MODE_TXT):
        if os.path.isabs(path):
            abspath = path
        else:
            abspath = os.path.join(self.path, path)
        directory_path = os.path.dirname(abspath)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        if mode == MODE_TXT:
            with open(abspath, "w", encoding=self.encoding, errors="replace") as f:
                f.write(content)

        elif mode == MODE_BIN:
            with open(abspath, "wb") as f:
                f.write(content)
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")

    def read_jsonl(self, path: str, byte_start=0, byte_end=None, encoding="utf-8"):
        return self.read(path)


# 使用示例
if __name__ == "__main__":
    file_path = "io/test/example.txt"
    drw = DiskReaderWriter("D:\projects\papayfork\Magic-PDF\magic_pdf")

    # 写入内容到文件
    drw.write(b"Hello, World!", path="io/test/example.txt", mode="binary")

    # 从文件读取内容
    content = drw.read(path=file_path)
    if content:
        logger.info(f"从 {file_path} 读取的内容: {content}")
