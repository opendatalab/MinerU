import json
import os
import base64
import shutil
import tempfile

import pika

from app import init_writers, process_file_mineru


def handle_message(ch, method, properties, body):
    temp_dir = tempfile.mkdtemp()
    try:
        message = json.loads(body)
        file_name = message.get("file_name")
        file_content_base64 = message.get("file_content_base64")
        correlation_id = properties.correlation_id

        if not file_name or not file_content_base64:
            raise ValueError("`file_name` and `file_content_base64` are required")

        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(file_content_base64))

        parse_method = message.get("parse_method", "auto")
        lang = message.get("lang", "ch")
        formula_enable = message.get("formula_enable", True)
        table_enable = message.get("table_enable", True)

        output_path = os.path.join(temp_dir, os.path.splitext(file_name)[0])
        output_image_path = os.path.join(output_path, "images")

        _, image_writer, file_bytes, file_extension = init_writers(
            file_path=file_path,
            file=None,
            output_path=output_path,
            output_image_path=output_image_path,
        )

        (
            model_json,
            middle_json,
            content_list,
            md_content,
            processed_bytes,
            pdf_info,
        ) = process_file_mineru(
            file_bytes,
            file_extension,
            image_writer,
            parse_method,
            lang,
            formula_enable,
            table_enable,
        )

        result_queue = os.getenv("RABBITMQ_RESULT_QUEUE", "mineru_results_queue")
        result_message = {
            "file_name": file_name,
            "md": md_content,
            "content_list": content_list,
            "middle_json": middle_json,
        }

        # Add correlation_id to result message if it was provided
        if correlation_id:
            result_message["correlation_id"] = correlation_id

        print(f"correlation_id: {correlation_id}")

        ch.queue_declare(queue=result_queue, durable=True)
        ch.basic_publish(
            exchange="",
            routing_key=result_queue,
            body=json.dumps(result_message, ensure_ascii=False),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
                correlation_id=correlation_id,  # Set correlation_id in message properties
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
    os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    main()
