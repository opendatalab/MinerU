
IO
===

Aims for read or write bytes from different media, Currently We provide ``S3Reader``, ``S3Writer`` for AWS S3 compatible media 
and ``HttpReader``, ``HttpWriter`` for remote Http file. You can implement new classes to meet the needs of your personal scenarios 
if MinerU have not provide the suitable classes. It is easy to implement new classes, the only one requirement is to inherit from
``IOReader`` or ``IOWriter``

.. code:: python

    class SomeReader(IOReader):
        def read(self, path: str) -> bytes:
            pass

        def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
            pass


    class SomeWriter(IOWriter):
        def write(self, path: str, data: bytes) -> None:
            pass

Check :doc:`../../api/io` for more details

