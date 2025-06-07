import json
import os

import pika

from app import init_writers, process_file


def handle_message(ch, method, properties, body):
    try:
        message = json.loads(body)
        file_path = message.get("file_path")
        parse_method = message.get("parse_method", "auto")
        output_dir = message.get("output_dir", "output")
        if not file_path:
            raise ValueError("`file_path` is required")

        file_name = os.path.basename(file_path).split(".")[0]
        output_path = os.path.join(output_dir, file_name)
        output_image_path = os.path.join(output_path, "images")

        writer, image_writer, file_bytes, file_extension = init_writers(
            file_path=file_path,
            file=None,
            output_path=output_path,
            output_image_path=output_image_path,
        )

        infer_result, pipe_result = process_file(
            file_bytes, file_extension, parse_method, image_writer
        )

        pipe_result.dump_md(writer, f"{file_name}.md", "images")
        pipe_result.dump_content_list(writer, f"{file_name}_content_list.json", "images")
        pipe_result.dump_middle_json(writer, f"{file_name}_middle.json")

        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"Failed to process message {body!r}: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    host = os.getenv("RABBITMQ_HOST", "localhost")
    queue = os.getenv("RABBITMQ_QUEUE", "parse_queue")

    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
    channel = connection.channel()
    channel.queue_declare(queue=queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue, on_message_callback=handle_message)

    print(f" [*] Waiting for messages in '{queue}'. To exit press CTRL+C")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
