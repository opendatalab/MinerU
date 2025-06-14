from .base import DataWriter


class DummyDataWriter(DataWriter):
    def write(self, path: str, data: bytes) -> None:
        """Dummy write method that does nothing."""
        pass

    def write_string(self, path: str, data: str) -> None:
        """Dummy write_string method that does nothing."""
        pass
