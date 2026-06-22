import os
import tempfile
import json
from unittest import mock
import pytest

from mineru.utils.config_reader import get_api_url

def test_get_api_url_no_config_no_env():
    # Test when neither the env var nor mineru.json config file is set
    with mock.patch("os.getenv", return_value=None), \
         mock.patch("mineru.utils.config_reader.read_config", return_value=None):
        assert get_api_url() is None

def test_get_api_url_env_only():
    # Test when only the env var is set
    with mock.patch.dict(os.environ, {"MINERU_API_URL": "http://env-server:8000"}):
        assert get_api_url() == "http://env-server:8000"

def test_get_api_url_config_only():
    # Test when only the mineru.json config file is set
    mock_config = {"api_url": "http://config-server:8000"}
    with mock.patch("os.getenv", return_value=None), \
         mock.patch("mineru.utils.config_reader.read_config", return_value=mock_config):
        assert get_api_url() == "http://config-server:8000"

def test_get_api_url_config_hyphen_only():
    # Test when only the mineru.json config file is set with hyphen format "api-url"
    mock_config = {"api-url": "http://hyphen-server:8000"}
    with mock.patch("os.getenv", return_value=None), \
         mock.patch("mineru.utils.config_reader.read_config", return_value=mock_config):
        assert get_api_url() == "http://hyphen-server:8000"

def test_get_api_url_env_overrides_config():
    # Test that the env var overrides the mineru.json config file
    mock_config = {"api_url": "http://config-server:8000"}
    with mock.patch.dict(os.environ, {"MINERU_API_URL": "http://env-server:8000"}), \
         mock.patch("mineru.utils.config_reader.read_config", return_value=mock_config):
        assert get_api_url() == "http://env-server:8000"
