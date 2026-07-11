import pytest

from mineru.data.data_reader_writer.multi_bucket_s3 import MultiBucketS3DataReader, MultiBucketS3DataWriter
from mineru.data.utils.exceptions import InvalidParams
from mineru.data.utils.schemas import S3Config


class FakeS3Client:
    def __init__(self) -> None:
        self.read_calls: list[tuple[str, int, int]] = []
        self.write_calls: list[tuple[str, bytes]] = []

    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        self.read_calls.append((path, offset, limit))
        return b"data"

    def write(self, path: str, data: bytes) -> None:
        self.write_calls.append((path, data))


def make_config(bucket_name: str) -> S3Config:
    return S3Config(
        bucket_name=bucket_name,
        access_key="ak",
        secret_key="sk",
        endpoint_url="https://example.com",
    )


def test_reader_routes_s3a_uri_to_declared_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = MultiBucketS3DataReader(
        "default-bucket/default-prefix",
        [make_config("default-bucket"), make_config("target-bucket")],
    )
    default_client = FakeS3Client()
    target_client = FakeS3Client()
    clients = {"default-bucket": default_client, "target-bucket": target_client}

    def get_client(bucket_name: str) -> FakeS3Client:
        if bucket_name not in clients:
            raise InvalidParams(f"bucket name: {bucket_name} not found")
        return clients[bucket_name]

    monkeypatch.setattr(reader, "_MultiBucketS3DataReader__get_s3_client", get_client)

    assert reader.read("s3a://target-bucket/dir/file.pdf?bytes=5,7") == b"data"
    assert target_client.read_calls == [("dir/file.pdf", 5, 7)]
    assert default_client.read_calls == []


def test_writer_routes_s3a_uri_to_declared_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = MultiBucketS3DataWriter(
        "default-bucket/default-prefix",
        [make_config("default-bucket"), make_config("target-bucket")],
    )
    default_client = FakeS3Client()
    target_client = FakeS3Client()
    clients = {"default-bucket": default_client, "target-bucket": target_client}

    def get_client(bucket_name: str) -> FakeS3Client:
        if bucket_name not in clients:
            raise InvalidParams(f"bucket name: {bucket_name} not found")
        return clients[bucket_name]

    monkeypatch.setattr(writer, "_MultiBucketS3DataWriter__get_s3_client", get_client)

    writer.write("s3a://target-bucket/out/file.md", b"content")
    assert target_client.write_calls == [("out/file.md", b"content")]
    assert default_client.write_calls == []
