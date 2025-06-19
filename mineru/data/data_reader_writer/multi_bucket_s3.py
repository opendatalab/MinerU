
from ..utils.exceptions import InvalidConfig, InvalidParams
from .base import DataReader, DataWriter
from ..io.s3 import S3Reader, S3Writer
from ..utils.schemas import S3Config
from ..utils.path_utils import parse_s3_range_params, parse_s3path, remove_non_official_s3_args


class MultiS3Mixin:
    def __init__(self, default_prefix: str, s3_configs: list[S3Config]):
        """Initialized with multiple s3 configs.

        Args:
            default_prefix (str): the default prefix of the relative path. for example, {some_bucket}/{some_prefix} or {some_bucket}
            s3_configs (list[S3Config]): list of s3 configs, the bucket_name must be unique in the list.

        Raises:
            InvalidConfig: default bucket config not in s3_configs.
            InvalidConfig: bucket name not unique in s3_configs.
            InvalidConfig: default bucket must be provided.
        """
        if len(default_prefix) == 0:
            raise InvalidConfig('default_prefix must be provided')

        arr = default_prefix.strip('/').split('/')
        self.default_bucket = arr[0]
        self.default_prefix = '/'.join(arr[1:])

        found_default_bucket_config = False
        for conf in s3_configs:
            if conf.bucket_name == self.default_bucket:
                found_default_bucket_config = True
                break

        if not found_default_bucket_config:
            raise InvalidConfig(
                f'default_bucket: {self.default_bucket} config must be provided in s3_configs: {s3_configs}'
            )

        uniq_bucket = set([conf.bucket_name for conf in s3_configs])
        if len(uniq_bucket) != len(s3_configs):
            raise InvalidConfig(
                f'the bucket_name in s3_configs: {s3_configs} must be unique'
            )

        self.s3_configs = s3_configs
        self._s3_clients_h: dict = {}


class MultiBucketS3DataReader(DataReader, MultiS3Mixin):
    def read(self, path: str) -> bytes:
        """Read the path from s3, select diffect bucket client for each request
        based on the bucket, also support range read.

        Args:
            path (str): the s3 path of file, the path must be in the format of s3://bucket_name/path?offset,limit.
            for example: s3://bucket_name/path?0,100.

        Returns:
            bytes: the content of s3 file.
        """
        may_range_params = parse_s3_range_params(path)
        if may_range_params is None or 2 != len(may_range_params):
            byte_start, byte_len = 0, -1
        else:
            byte_start, byte_len = int(may_range_params[0]), int(may_range_params[1])
        path = remove_non_official_s3_args(path)
        return self.read_at(path, byte_start, byte_len)

    def __get_s3_client(self, bucket_name: str):
        if bucket_name not in set([conf.bucket_name for conf in self.s3_configs]):
            raise InvalidParams(
                f'bucket name: {bucket_name} not found in s3_configs: {self.s3_configs}'
            )
        if bucket_name not in self._s3_clients_h:
            conf = next(
                filter(lambda conf: conf.bucket_name == bucket_name, self.s3_configs)
            )
            self._s3_clients_h[bucket_name] = S3Reader(
                bucket_name,
                conf.access_key,
                conf.secret_key,
                conf.endpoint_url,
                conf.addressing_style,
            )
        return self._s3_clients_h[bucket_name]

    def read_at(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Read the file with offset and limit, select diffect bucket client
        for each request based on the bucket.

        Args:
            path (str): the file path.
            offset (int, optional): the number of bytes skipped. Defaults to 0.
            limit (int, optional): the number of bytes want to read. Defaults to -1 which means infinite.

        Returns:
            bytes: the file content.
        """
        if path.startswith('s3://'):
            bucket_name, path = parse_s3path(path)
            s3_reader = self.__get_s3_client(bucket_name)
        else:
            s3_reader = self.__get_s3_client(self.default_bucket)
            if self.default_prefix:
                path = self.default_prefix + '/' + path
        return s3_reader.read_at(path, offset, limit)


class MultiBucketS3DataWriter(DataWriter, MultiS3Mixin):
    def __get_s3_client(self, bucket_name: str):
        if bucket_name not in set([conf.bucket_name for conf in self.s3_configs]):
            raise InvalidParams(
                f'bucket name: {bucket_name} not found in s3_configs: {self.s3_configs}'
            )
        if bucket_name not in self._s3_clients_h:
            conf = next(
                filter(lambda conf: conf.bucket_name == bucket_name, self.s3_configs)
            )
            self._s3_clients_h[bucket_name] = S3Writer(
                bucket_name,
                conf.access_key,
                conf.secret_key,
                conf.endpoint_url,
                conf.addressing_style,
            )
        return self._s3_clients_h[bucket_name]

    def write(self, path: str, data: bytes) -> None:
        """Write file with data, also select diffect bucket client for each
        request based on the bucket.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            data (bytes): the data want to write.
        """
        if path.startswith('s3://'):
            bucket_name, path = parse_s3path(path)
            s3_writer = self.__get_s3_client(bucket_name)
        else:
            s3_writer = self.__get_s3_client(self.default_bucket)
            if self.default_prefix:
                path = self.default_prefix + '/' + path
        return s3_writer.write(path, data)
