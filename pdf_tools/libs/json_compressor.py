import json
import brotli
import base64

class JsonCompressor:

    @staticmethod
    def compress_json(data):
        """
        Compress a json object and encode it with base64
        """
        json_str = json.dumps(data)
        json_bytes = json_str.encode('utf-8')
        compressed = brotli.compress(json_bytes, quality=6)
        compressed_str = base64.b64encode(compressed).decode('utf-8')  # convert bytes to string
        return compressed_str

    @staticmethod
    def decompress_json(compressed_str):
        """
        Decode the base64 string and decompress the json object
        """
        compressed = base64.b64decode(compressed_str.encode('utf-8'))  # convert string to bytes
        decompressed_bytes = brotli.decompress(compressed)
        json_str = decompressed_bytes.decode('utf-8')
        data = json.loads(json_str)
        return data
