import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the sys.path to find the rabbitmq_consumer module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rabbitmq_consumer import handle_message


class TestRabbitMQConsumer(unittest.TestCase):
    def setUp(self):
        self.mock_channel = MagicMock()
        self.mock_method = MagicMock()
        self.mock_method.delivery_tag = 12345
        self.mock_properties = MagicMock()

    @patch("rabbitmq_consumer.process_file")
    @patch("rabbitmq_consumer.init_writers")
    def test_handle_message_success(self, mock_init_writers, mock_process_file):
        # Arrange
        mock_writer = MagicMock()
        mock_image_writer = MagicMock()
        mock_pipe_result = MagicMock()
        mock_init_writers.return_value = (
            mock_writer,
            mock_image_writer,
            b"file_bytes",
            ".pdf",
        )
        mock_process_file.return_value = (MagicMock(), mock_pipe_result)

        file_path = "/path/to/test.pdf"
        message_body = {"file_path": file_path}
        body = json.dumps(message_body).encode("utf-8")

        # Act
        handle_message(self.mock_channel, self.mock_method, self.mock_properties, body)

        # Assert
        mock_init_writers.assert_called_once()
        mock_process_file.assert_called_once()
        self.mock_channel.basic_ack.assert_called_once_with(
            delivery_tag=self.mock_method.delivery_tag
        )
        self.mock_channel.basic_nack.assert_not_called()
        mock_pipe_result.dump_md.assert_called_once()
        mock_pipe_result.dump_content_list.assert_called_once()
        mock_pipe_result.dump_middle_json.assert_called_once()

    def test_handle_message_missing_filepath(self):
        # Arrange
        message_body = {"other_key": "some_value"}
        body = json.dumps(message_body).encode("utf-8")

        # Act
        handle_message(self.mock_channel, self.mock_method, self.mock_properties, body)

        # Assert
        self.mock_channel.basic_ack.assert_not_called()
        self.mock_channel.basic_nack.assert_called_once_with(
            delivery_tag=self.mock_method.delivery_tag, requeue=False
        )

    @patch("rabbitmq_consumer.process_file")
    @patch("rabbitmq_consumer.init_writers")
    def test_handle_message_processing_exception(
        self, mock_init_writers, mock_process_file
    ):
        # Arrange
        mock_init_writers.return_value = (
            MagicMock(),
            MagicMock(),
            b"file_bytes",
            ".pdf",
        )
        mock_process_file.side_effect = Exception("Processing failed")
        file_path = "/path/to/test.pdf"
        message_body = {"file_path": file_path}
        body = json.dumps(message_body).encode("utf-8")

        # Act
        handle_message(self.mock_channel, self.mock_method, self.mock_properties, body)

        # Assert
        self.mock_channel.basic_ack.assert_not_called()
        self.mock_channel.basic_nack.assert_called_once_with(
            delivery_tag=self.mock_method.delivery_tag, requeue=False
        )


if __name__ == "__main__":
    unittest.main() 