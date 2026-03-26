from mineru.utils.config_reader import (
    get_max_concurrent_requests,
    get_processing_window_size,
)

API_PROTOCOL_VERSION = 1
DEFAULT_MAX_CONCURRENT_REQUESTS = get_max_concurrent_requests(default=3)
DEFAULT_PROCESSING_WINDOW_SIZE = get_processing_window_size(default=64)
