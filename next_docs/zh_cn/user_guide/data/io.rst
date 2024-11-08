

IO
====

旨在从不同的媒介读取或写入字节。目前，我们提供了 S3Reader 和 S3Writer 用于兼容 AWS S3 的媒介，以及 HttpReader 和 HttpWriter 用于远程 HTTP 文件。如果 MinerU 没有提供合适的类，你可以实现新的类以满足个人场景的需求。实现新的类非常容易，唯一的要求是继承自 IOReader 或 IOWriter。

.. code:: python

    class SomeReader(IOReader):
        def read(self, path: str) -> bytes:
            pass

        def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
            pass


    class SomeWriter(IOWriter):
        def write(self, path: str, data: bytes) -> None:
            pass
        
