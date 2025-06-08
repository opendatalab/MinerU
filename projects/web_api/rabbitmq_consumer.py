import json
import os
import base64
import shutil
import tempfile

import pika

from app import MemoryDataWriter, init_writers, process_file
import magic_pdf.model as model_config

model_config.__use_inside_model__ = True


def handle_message(ch, method, properties, body):
    temp_dir = tempfile.mkdtemp()
    try:
        message = json.loads(body)
        file_name = message.get("file_name")
        file_content_base64 = message.get("file_content_base64")

        if not file_name or not file_content_base64:
            raise ValueError("`file_name` and `file_content_base64` are required")

        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(file_content_base64))

        parse_method = message.get("parse_method", "auto")

        output_path = os.path.join(temp_dir, os.path.splitext(file_name)[0])
        output_image_path = os.path.join(output_path, "images")

        _, image_writer, file_bytes, file_extension = init_writers(
            file_path=file_path,
            file=None,
            output_path=output_path,
            output_image_path=output_image_path,
        )

        infer_result, pipe_result = process_file(
            file_bytes, file_extension, parse_method, image_writer
        )

        # Use MemoryDataWriter to get results in memory
        content_list_writer = MemoryDataWriter()
        md_content_writer = MemoryDataWriter()
        middle_json_writer = MemoryDataWriter()

        pipe_result.dump_content_list(content_list_writer, "", "images")
        pipe_result.dump_md(md_content_writer, "", "images")
        pipe_result.dump_middle_json(middle_json_writer, "")

        # Get content from memory writers
        content_list = json.loads(content_list_writer.get_value())
        md_content = md_content_writer.get_value()
        middle_json = json.loads(middle_json_writer.get_value())

        content_list_writer.close()
        md_content_writer.close()
        middle_json_writer.close()

        result_queue = os.getenv("RABBITMQ_RESULT_QUEUE", "mineru_results_queue")
        result_message = {
            "file_name": file_name,
            "md": md_content,
            "content_list": content_list,
            "middle_json": middle_json,
        }
        ch.queue_declare(queue=result_queue, durable=True)
        ch.basic_publish(
            exchange="",
            routing_key=result_queue,
            body=json.dumps(result_message, ensure_ascii=False),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            ),
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"Failed to process message {body!r}: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    finally:
        shutil.rmtree(temp_dir)


def main():
    host = os.getenv("RABBITMQ_HOST", "localhost")
    queue = os.getenv("RABBITMQ_QUEUE", "mineru_parse_queue")

    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
    channel = connection.channel()
    channel.queue_declare(queue=queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue, on_message_callback=handle_message)

    print(f" [*] Waiting for messages in '{queue}'. To exit press CTRL+C")
    while True:
        try:
            channel.start_consuming()
        except (KeyboardInterrupt, SystemExit):
            channel.stop_consuming()
            print(" [*] Exiting...")
            break
    connection.close()


if __name__ == "__main__":
    main()
