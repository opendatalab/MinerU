import json
import os
import pika
import time
import unittest
import tempfile
import shutil


class TestRabbitMQEndToEnd(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.test_root_dir = tempfile.mkdtemp()
        self.input_file_path = os.path.join(self.test_root_dir, "test.txt")
        self.output_dir = os.path.join(self.test_root_dir, "output")

        # Create a dummy input file
        with open(self.input_file_path, "w") as f:
            f.write("This is a test file.")

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_root_dir)

    def test_message_processing_e2e(self):
        # Get RabbitMQ connection details from environment variables
        host = os.getenv("RABBITMQ_HOST", "localhost")
        queue = os.getenv("RABBITMQ_QUEUE", "parse_queue")

        # Construct the message
        message = {
            "file_path": self.input_file_path,
            "output_dir": self.output_dir
        }
        body = json.dumps(message)

        # Publish the message to RabbitMQ
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
            channel = connection.channel()
            channel.queue_declare(queue=queue, durable=True)
            channel.basic_publish(
                exchange="",
                routing_key=queue,
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                ),
            )
            connection.close()
            print(f" [x] Sent message to queue '{queue}': {body}")
        except pika.exceptions.AMQPConnectionError as e:
            self.fail(f"Failed to connect to RabbitMQ: {e}. Please ensure RabbitMQ is running and accessible.")

        # Wait for the consumer to process the message
        print(" [*] Waiting for consumer to process the message (10 seconds)...")
        time.sleep(10)

        # Check for the output files
        expected_output_path = os.path.join(self.output_dir, "test")
        self.assertTrue(
            os.path.isdir(expected_output_path),
            f"Output directory '{expected_output_path}' was not created."
        )

        expected_files = [
            "test.md",
            "test_content_list.json",
            "test_middle.json",
        ]

        for filename in expected_files:
            file_path = os.path.join(expected_output_path, filename)
            self.assertTrue(os.path.isfile(file_path), f"Output file '{file_path}' was not created.")
            self.assertGreater(os.path.getsize(file_path), 0, f"Output file '{file_path}' is empty.")
        
        print(" [+] Test passed: Output files created successfully.")


if __name__ == "__main__":
    # To run this test, ensure the consumer is running and RabbitMQ is accessible.
    # Set RABBITMQ_HOST and RABBITMQ_QUEUE environment variables if needed.
    unittest.main() 