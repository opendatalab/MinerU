import json
import os
import pika
import time
import unittest
import tempfile
import shutil
import base64
import uuid


class TestRabbitMQEndToEnd(unittest.TestCase):
    def setUp(self):
        self.input_file_path = r"D:\Coding\Web\MinerU_webapi\tests\unittest\test_data\assets\pdfs\test_01.pdf"

        self.host = os.getenv("RABBITMQ_HOST", "localhost")
        self.parse_queue = os.getenv("RABBITMQ_QUEUE", "mineru_parse_queue")
        self.result_queue = "mineru_results_queue"
        os.environ["RABBITMQ_RESULT_QUEUE"] = "mineru_results_queue"

    def tearDown(self):
        if "RABBITMQ_RESULT_QUEUE" in os.environ:
            del os.environ["RABBITMQ_RESULT_QUEUE"]

    def test_message_processing_e2e(self):
        # Get RabbitMQ connection details from environment variables
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host)
            )
            channel = connection.channel()
            channel.queue_declare(queue=self.parse_queue, durable=True)
            # Use exclusive queue for results, it will be deleted after connection closed
            channel.queue_declare(queue=self.result_queue, durable=True, exclusive=False)
            channel.queue_purge(queue=self.result_queue)
        except pika.exceptions.AMQPConnectionError as e:
            self.fail(
                f"Failed to connect to RabbitMQ: {e}. Please ensure RabbitMQ is running and accessible."
            )

        # Read file content and encode it in base64
        with open(self.input_file_path, "rb") as f:
            file_content = f.read()
        file_content_base64 = base64.b64encode(file_content).decode("utf-8")

        # Construct the message
        message = {
            "file_name": os.path.basename(self.input_file_path),
            "file_content_base64": file_content_base64,
        }
        body = json.dumps(message)

        # Publish the message to RabbitMQ
        channel.basic_publish(
            exchange="",
            routing_key=self.parse_queue,
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            ),
        )
        print(f" [x] Sent message to queue '{self.parse_queue}'")

        # Wait for the consumer to process the message and return a result
        print(f" [*] Waiting for result on queue '{self.result_queue}'...")
        result_body = None
        # Consume from the result queue with a timeout
        for method_frame, properties, body in channel.consume(
            self.result_queue, inactivity_timeout=60  # Increased timeout for PDF processing
        ):
            if method_frame:
                result_body = body
                channel.basic_ack(method_frame.delivery_tag)
                break

        channel.cancel()
        connection.close()

        self.assertIsNotNone(
            result_body,
            "Did not receive a result from the consumer within the timeout period.",
        )

        result_message = json.loads(result_body)

        self.assertEqual(
            result_message.get("file_name"), os.path.basename(self.input_file_path)
        )

        self.assertIn("md", result_message)
        self.assertIsInstance(result_message["md"], str)
        self.assertGreater(len(result_message["md"]), 0)

        self.assertIn("content_list", result_message)
        self.assertIsInstance(result_message["content_list"], list)
        self.assertGreater(len(result_message["content_list"]), 0)

        self.assertIn("middle_json", result_message)
        self.assertIsInstance(result_message["middle_json"], dict)
        self.assertGreater(len(result_message["middle_json"]), 0)

        print(" [+] Test passed: Received and verified result from queue.")


if __name__ == "__main__":
    # To run this test, ensure the consumer is running and RabbitMQ is accessible.
    # Set RABBITMQ_HOST and RABBITMQ_QUEUE environment variables if needed.
    unittest.main() 