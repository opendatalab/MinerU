import os
from magic_pdf.io.AbsReaderWriter import AbsReaderWriter
from loguru import logger
class DiskReaderWriter(AbsReaderWriter):
    def __init__(self, parent_path, encoding='utf-8'):
        self.path = parent_path
        self.encoding = encoding

    def read(self, mode="text"):
        if not os.path.exists(self.path):
            logger.error(f"文件 {self.path} 不存在")
            raise Exception(f"文件 {self.path} 不存在")
        if mode == "text":
            with open(self.path, 'r', encoding = self.encoding) as f:
                return f.read()
        elif mode == "binary":
            with open(self.path, 'rb') as f:
                return f.read()
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")

    def write(self, data, mode="text"):
        if mode == "text":
            with open(self.path, 'w', encoding=self.encoding) as f:
                f.write(data)
                logger.info(f"内容已成功写入 {self.path}")

        elif mode == "binary":
            with open(self.path, 'wb') as f:
                f.write(data)
                logger.info(f"内容已成功写入 {self.path}")
        else:
            raise ValueError("Invalid mode. Use 'text' or 'binary'.")


# 使用示例
if __name__ == "__main__":
    file_path = "example.txt"
    drw = DiskReaderWriter(file_path)

    # 写入内容到文件
    drw.write(b"Hello, World!", mode="binary")

    # 从文件读取内容
    content = drw.read()
    if content:
        logger.info(f"从 {file_path} 读取的内容: {content}")


