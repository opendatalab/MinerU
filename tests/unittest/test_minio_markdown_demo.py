# Copyright (c) Opendatalab. All rights reserved.
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[2] / "demo" / "minio_markdown_demo.py"
MODULE_SPEC = importlib.util.spec_from_file_location("minio_markdown_demo", MODULE_PATH)
minio_markdown_demo = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(minio_markdown_demo)


class MinioMarkdownDemoTestCase(unittest.TestCase):
    def test_build_http_object_url_uses_path_style(self):
        object_url = minio_markdown_demo.build_http_object_url(
            "http://127.0.0.1:9000/",
            "mineru-bucket",
            "output",
            "task-123",
            "demo.pdf",
        )

        self.assertEqual(
            object_url,
            "http://127.0.0.1:9000/mineru-bucket/output/task-123/demo.pdf",
        )

    def test_rewrite_relative_image_paths_replaces_markdown_json_and_html_links(self):
        original_text = '\n'.join(
            [
                '![](images/page-1.png)',
                '"img_path": "images/page-2.png"',
                'src="images/page-3.png"',
                "src='./images/page-4.png'",
            ]
        )

        rewritten_text = minio_markdown_demo.rewrite_relative_image_paths(
            original_text,
            "ocr/demo.md",
            "task-123/demo",
            "http://127.0.0.1:9000",
            "mineru-bucket",
            "output",
        )

        expected_prefix = "http://127.0.0.1:9000/mineru-bucket/output/task-123/demo/ocr/images/"
        self.assertIn(f"![]({expected_prefix}page-1.png)", rewritten_text)
        self.assertIn(f'"img_path": "{expected_prefix}page-2.png"', rewritten_text)
        self.assertIn(f'src="{expected_prefix}page-3.png"', rewritten_text)
        self.assertIn(f"src='{expected_prefix}page-4.png'", rewritten_text)

    def test_resolve_minio_config_prefers_cli_over_env_and_config(self):
        args = SimpleNamespace(
            minio_url="http://cli-minio:9000",
            minio_ak="cli-ak",
            minio_sk="cli-sk",
        )

        with mock.patch.dict(
            os.environ,
            {
                "MINIO_URL": "http://env-minio:9000",
                "MINIO_AK": "env-ak",
                "MINIO_SK": "env-sk",
            },
            clear=False,
        ):
            with mock.patch.object(
                minio_markdown_demo,
                "get_s3_config",
                return_value=("config-ak", "config-sk", "http://config-minio:9000"),
            ):
                self.assertEqual(
                    minio_markdown_demo.resolve_minio_config("mineru-bucket", args),
                    ("cli-ak", "cli-sk", "http://cli-minio:9000"),
                )

    def test_resolve_minio_config_falls_back_to_config(self):
        args = SimpleNamespace(minio_url=None, minio_ak=None, minio_sk=None)

        with mock.patch.dict(os.environ, {}, clear=False):
            with mock.patch.object(
                minio_markdown_demo,
                "get_s3_config",
                return_value=("config-ak", "config-sk", "http://config-minio:9000"),
            ):
                self.assertEqual(
                    minio_markdown_demo.resolve_minio_config("mineru-bucket", args),
                    ("config-ak", "config-sk", "http://config-minio:9000"),
                )

    def test_extract_object_key_from_http_url(self):
        object_key = minio_markdown_demo.extract_object_key_from_http_url(
            "http://127.0.0.1:9000/mineru-bucket/output/task-123/demo/ocr/images/page%201.png",
            endpoint="http://127.0.0.1:9000",
            bucket_name="mineru-bucket",
            prefix="output",
        )

        self.assertEqual(object_key, "task-123/demo/ocr/images/page 1.png")

    def test_enrich_markdown_with_image_explanations_downloads_once_and_inserts_once_per_occurrence(self):
        image_url = (
            "http://127.0.0.1:9000/mineru-bucket/output/"
            "task-123/demo/ocr/images/page-1.png"
        )
        markdown_text = "\n".join(
            [
                "图 1 展示了整体流程。",
                f"![]({image_url})",
                "后文继续说明。",
                f"![]({image_url})",
            ]
        )
        fake_reader = mock.Mock()
        fake_reader.read.return_value = b"fake-image-bytes"
        vlm_config = {
            "api_key": "EMPTY",
            "base_url": "http://127.0.0.1:30000/v1",
            "model": "qwen2.5-vl",
            "prompt": "请解释图片",
            "timeout": 30,
            "max_tokens": 128,
        }

        with mock.patch.object(minio_markdown_demo, "OpenAI", return_value=object()):
            with mock.patch.object(
                minio_markdown_demo,
                "request_image_explanation_from_vlm",
                return_value="这是一张流程图，描述了解析与回填链路。",
            ) as request_mock:
                enriched = minio_markdown_demo.enrich_markdown_with_image_explanations(
                    markdown_text,
                    output_reader=fake_reader,
                    endpoint="http://127.0.0.1:9000",
                    bucket_name="mineru-bucket",
                    output_prefix="output",
                    language="ch",
                    vlm_config=vlm_config,
                )

        self.assertEqual(
            fake_reader.read.call_args_list,
            [mock.call("task-123/demo/ocr/images/page-1.png")],
        )
        self.assertEqual(request_mock.call_count, 1)
        self.assertEqual(enriched.count("> 图片解释：这是一张流程图，描述了解析与回填链路。"), 2)
        self.assertIn("图 1 展示了整体流程。", request_mock.call_args.kwargs["prompt"])
        self.assertIn("后文继续说明。", request_mock.call_args.kwargs["prompt"])


if __name__ == "__main__":
    unittest.main()
