import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import base64

# Add the parent directory to the sys.path to find the rabbitmq_consumer module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rabbitmq_consumer import handle_message


class TestRabbitMQConsumer(unittest.TestCase):
    def setUp(self):
        self.mock_channel = MagicMock()
        self.mock_method = MagicMock()
        self.mock_method.delivery_tag = 12345
        self.mock_properties = MagicMock()

    @patch("rabbitmq_consumer.shutil.rmtree")
    @patch("rabbitmq_consumer.tempfile.mkdtemp")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("rabbitmq_consumer.base64.b64decode")
    @patch("rabbitmq_consumer.process_file")
    @patch("rabbitmq_consumer.init_writers")
    def test_handle_message_success_temp_dir(
        self,
        mock_init_writers,
        mock_process_file,
        mock_b64decode,
        mock_open,
        mock_mkdtemp,
        mock_rmtree,
    ):
        # Arrange
        temp_dir = "/fake/temp/dir"
        mock_mkdtemp.return_value = temp_dir
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

        file_name = "test.pdf"
        file_content_b64 = base64.b64encode(b"test content").decode("utf-8")
        message_body = {"file_name": file_name, "file_content_base64": file_content_b64}
        body = json.dumps(message_body).encode("utf-8")
        decoded_content = b"test content"
        mock_b64decode.return_value = decoded_content

        # Act
        handle_message(self.mock_channel, self.mock_method, self.mock_properties, body)

        # Assert
        mock_mkdtemp.assert_called_once()
        mock_b64decode.assert_called_once_with(file_content_b64)
        mock_open.assert_called_once_with(os.path.join(temp_dir, file_name), "wb")
        mock_open().write.assert_called_once_with(decoded_content)
        mock_init_writers.assert_called_once()
        mock_process_file.assert_called_once()
        self.mock_channel.basic_ack.assert_called_once_with(
            delivery_tag=self.mock_method.delivery_tag
        )
        self.mock_channel.basic_nack.assert_not_called()
        mock_pipe_result.dump_md.assert_called_once()
        mock_rmtree.assert_called_once_with(temp_dir)

    def test_handle_message_missing_info(self):
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
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("rabbitmq_consumer.base64.b64decode")
    def test_handle_message_processing_exception(
        self, mock_b64decode, mock_open, mock_init_writers, mock_process_file
    ):
        # Arrange
        mock_init_writers.return_value = (
            MagicMock(),
            MagicMock(),
            b"file_bytes",
            ".pdf",
        )
        mock_process_file.side_effect = Exception("Processing failed")
        file_name = "test.pdf"
        file_content_b64 = base64.b64encode(b"test content").decode("utf-8")
        message_body = {"file_name": file_name, "file_content_base64": file_content_b64}
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