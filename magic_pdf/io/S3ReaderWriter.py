

from magic_pdf.io import AbsReaderWriter


class DiskReaderWriter(AbsReaderWriter):
    def __init__(self, parent_path, encoding='utf-8'):
        self.path = parent_path
        self.encoding = encoding

    def read(self):
        with open(self.path, 'rb') as f:
            return f.read()

    def write(self, data):
        with open(self.path, 'wb') as f:
            f.write(data)
            