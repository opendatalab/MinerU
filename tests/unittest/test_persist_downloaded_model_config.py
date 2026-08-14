# Copyright (c) Opendatalab. All rights reserved.
import json
import os
import pytest
from unittest.mock import patch, MagicMock

from mineru.utils.models_download_utils import persist_downloaded_model_config


@pytest.fixture
def config_path(tmp_path):
    return str(tmp_path / "mineru.json")


def test_persist_success_writes_models_dir(config_path):
    """Happy path: config file is written with the correct models-dir entry."""
    with patch("mineru.utils.models_download_utils.get_tools_config_file_path", return_value=config_path), \
         patch("mineru.utils.models_download_utils.download_and_modify_json") as mock_dl:
        persist_downloaded_model_config("huggingface", "pipeline", "/tmp/models")
        mock_dl.assert_called_once()
        _, _, modifications = mock_dl.call_args[0]
        assert modifications["models-dir"]["pipeline"] == "/tmp/models"
        assert modifications["model-source"] == "huggingface"


def test_persist_failure_logs_actionable_warning(config_path, caplog):
    """When download_and_modify_json raises, warning must include model_root path and manual fix hint."""
    import logging
    with patch("mineru.utils.models_download_utils.get_tools_config_file_path", return_value=config_path), \
         patch("mineru.utils.models_download_utils.download_and_modify_json", side_effect=OSError("disk full")):
        # loguru does not integrate with caplog by default — capture via propagation
        import loguru
        import sys
        warning_messages = []

        def sink(message):
            warning_messages.append(str(message))

        handler_id = loguru.logger.add(sink, level="WARNING")
        try:
            persist_downloaded_model_config("modelscope", "pipeline", "/tmp/models")
        finally:
            loguru.logger.remove(handler_id)

    combined = " ".join(warning_messages)
    # Must tell the user the downloaded path so they can fix it manually
    assert "/tmp/models" in combined
    # Must explain that re-download will happen
    assert "re-download" in combined or "next run" in combined
    # Must show the manual fix snippet
    assert '"pipeline"' in combined


def test_persist_failure_does_not_raise(config_path):
    """A config-write failure must never propagate — the download already succeeded."""
    with patch("mineru.utils.models_download_utils.get_tools_config_file_path", return_value=config_path), \
         patch("mineru.utils.models_download_utils.download_and_modify_json", side_effect=PermissionError("no write")):
        # Should not raise
        persist_downloaded_model_config("huggingface", "vlm", "/tmp/vlm-models")


def test_persist_failure_for_vlm_mode(config_path):
    """Warning message must include the correct repo_mode (vlm) in the hint."""
    warning_messages = []
    import loguru

    handler_id = loguru.logger.add(warning_messages.append, level="WARNING")
    try:
        with patch("mineru.utils.models_download_utils.get_tools_config_file_path", return_value=config_path), \
             patch("mineru.utils.models_download_utils.download_and_modify_json", side_effect=OSError("network error")):
            persist_downloaded_model_config("huggingface", "vlm", "/tmp/vlm-root")
    finally:
        loguru.logger.remove(handler_id)

    combined = " ".join(str(m) for m in warning_messages)
    assert '"vlm"' in combined
    assert "/tmp/vlm-root" in combined


def test_persist_success_modelscope_pipeline(config_path):
    """modelscope + pipeline combination is correctly passed through."""
    with patch("mineru.utils.models_download_utils.get_tools_config_file_path", return_value=config_path), \
         patch("mineru.utils.models_download_utils.download_and_modify_json") as mock_dl:
        persist_downloaded_model_config("modelscope", "pipeline", "/ms/models")
        _, _, modifications = mock_dl.call_args[0]
        assert modifications["model-source"] == "modelscope"
        assert modifications["models-dir"]["pipeline"] == "/ms/models"
