"""
End-to-end test for the RabbitMQ consumer.

IMPORTANT:
This test suite requires the RabbitMQ consumer service to be running independently.
Before running this test, start the consumer in a separate terminal:
`python projects/web_api/rabbitmq_consumer.py`
"""
import base64
import json
import os
import time
import unittest
import uuid

import pika
from pika.exceptions import AMQPConnectionError


class TestRabbitMQE2E(unittest.TestCase):
    def setUp(self):
        """
        Connects to RabbitMQ and purges queues for a clean test state.
        Assumes the consumer service is running independently.
        """
        self.host = os.getenv("RABBITMQ_HOST", "localhost")
        self.parse_queue = os.getenv("RABBITMQ_QUEUE", "mineru_parse_queue")
        self.result_queue = os.getenv("RABBITMQ_RESULT_QUEUE", "mineru_results_queue")
        self.connection = None

        try:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
            channel = self.connection.channel()
            
            # Ensure queues exist before purging. This is idempotent.
            channel.queue_declare(queue=self.parse_queue, durable=True)
            channel.queue_declare(queue=self.result_queue, durable=True)
            
            # Purge queues
            channel.queue_purge(queue=self.parse_queue)
            channel.queue_purge(queue=self.result_queue)
        except AMQPConnectionError:
            self.fail(
                f"Could not connect to RabbitMQ at {self.host}. "
                "Please ensure RabbitMQ is running and the consumer service has been started."
            )

    def tearDown(self):
        """
        Closes the connection to RabbitMQ.
        """
        if self.connection and self.connection.is_open:
            self.connection.close()

    def test_pdf_processing_via_queue(self):
        """
        Tests the full-cycle processing of a PDF file sent via RabbitMQ.
        Publishes a task and waits for a result on the corresponding queues.
        """
        if not self.connection:
            self.fail("RabbitMQ connection not established in setUp.")

        channel = self.connection.channel()

        # 1. Prepare and publish the message
        pdf_path = os.path.join(os.path.dirname(__file__), "..", "test_data", "assets", "pdfs", "test_01.pdf")
        self.assertTrue(os.path.exists(pdf_path), f"Test PDF not found at {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        file_name = os.path.basename(pdf_path)
        file_content_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        correlation_id = str(uuid.uuid4())

        message = {
            "file_name": file_name,
            "file_content_base64": file_content_base64,
            "parse_method": "auto",
            "lang": "ch"
        }

        channel.basic_publish(
            exchange="",
            routing_key=self.parse_queue,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                correlation_id=correlation_id,
                delivery_mode=2,  # make message persistent
            )
        )
        print(f"\n [x] Sent PDF '{file_name}' to queue '{self.parse_queue}'")

        # 2. Wait for and retrieve the result
        result = None
        start_time = time.time()
        timeout = 120 # Increased timeout for potentially long processing
        print(f" [*] Waiting for result on queue '{self.result_queue}'...")

        while time.time() - start_time < timeout:
            method_frame, properties, body = channel.basic_get(queue=self.result_queue)
            if method_frame and properties and properties.correlation_id == correlation_id:
                channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                result = json.loads(body)
                print(f" [✔] Received result for correlation_id '{properties.correlation_id}'")
                break
            elif method_frame:
                # It's a message, but not the one we're looking for.
                # Put it back in the queue.
                print(f" [!] Received unexpected message, re-queueing.")
                channel.basic_nack(delivery_tag=method_frame.delivery_tag)
            
            time.sleep(1)

        # 3. Assert the result
        self.assertIsNotNone(result, f"Did not receive a result from the consumer within {timeout} seconds.")
        
        # This check helps the linter understand that `result` is not None below.
        if result is None:
            return

        self.assertEqual(result.get("file_name"), file_name)
        self.assertIn("md", result)
        self.assertIn("content_list", result)
        self.assertTrue(len(result.get("md", "")) > 0, "The 'md' content should not be empty.")
        print(" [✔] Result validation successful.")

if __name__ == '__main__':
    unittest.main() 