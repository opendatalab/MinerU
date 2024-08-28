import tempfile
import os
import shutil
from click.testing import CliRunner

from magic_pdf.tools import cli_dev


def test_cli_pdf():
    # setup
    unitest_dir = "/tmp/magic_pdf/unittest/tools"
    filename = "cli_test_01"
    os.makedirs(unitest_dir, exist_ok=True)
    temp_output_dir = tempfile.mkdtemp(dir="/tmp/magic_pdf/unittest/tools")

    # run
    runner = CliRunner()
    result = runner.invoke(
        cli_dev.cli,
        [
            "pdf",
            "-p",
            "tests/test_tools/assets/cli/pdf/cli_test_01.pdf",
            "-j",
            "tests/test_tools/assets/cli_dev/cli_test_01.model.json",
            "-o",
            temp_output_dir,
        ],
    )

    # check
    assert result.exit_code == 0

    base_output_dir = os.path.join(temp_output_dir, "cli_test_01/auto")

    r = os.stat(os.path.join(base_output_dir, "content_list.json"))
    assert r.st_size > 5000

    r = os.stat(os.path.join(base_output_dir, f"{filename}.md"))
    assert r.st_size > 7000

    r = os.stat(os.path.join(base_output_dir, "middle.json"))
    assert r.st_size > 200000

    r = os.stat(os.path.join(base_output_dir, "model.json"))
    assert r.st_size > 15000

    r = os.stat(os.path.join(base_output_dir, "origin.pdf"))
    assert r.st_size > 500000

    r = os.stat(os.path.join(base_output_dir, "layout.pdf"))
    assert r.st_size > 500000

    r = os.stat(os.path.join(base_output_dir, "spans.pdf"))
    assert r.st_size > 500000

    assert os.path.exists(os.path.join(base_output_dir, "images")) is True
    assert os.path.isdir(os.path.join(base_output_dir, "images")) is True

    # teardown
    shutil.rmtree(temp_output_dir)


def test_cli_jsonl():
    # setup
    unitest_dir = "/tmp/magic_pdf/unittest/tools"
    filename = "cli_test_01"
    os.makedirs(unitest_dir, exist_ok=True)
    temp_output_dir = tempfile.mkdtemp(dir="/tmp/magic_pdf/unittest/tools")

    def mock_read_s3_path(s3path):
        with open(s3path, "rb") as f:
            return f.read()

    cli_dev.read_s3_path = mock_read_s3_path # mock

    # run
    runner = CliRunner()
    result = runner.invoke(
        cli_dev.cli,
        [
            "jsonl",
            "-j",
            "tests/test_tools/assets/cli_dev/cli_test_01.jsonl",
            "-o",
            temp_output_dir,
        ],
    )

    # check
    assert result.exit_code == 0

    base_output_dir = os.path.join(temp_output_dir, "cli_test_01/auto")

    r = os.stat(os.path.join(base_output_dir, "content_list.json"))
    assert r.st_size > 5000

    r = os.stat(os.path.join(base_output_dir, f"{filename}.md"))
    assert r.st_size > 7000

    r = os.stat(os.path.join(base_output_dir, "middle.json"))
    assert r.st_size > 200000

    r = os.stat(os.path.join(base_output_dir, "model.json"))
    assert r.st_size > 15000

    r = os.stat(os.path.join(base_output_dir, "origin.pdf"))
    assert r.st_size > 500000

    r = os.stat(os.path.join(base_output_dir, "layout.pdf"))
    assert r.st_size > 500000

    r = os.stat(os.path.join(base_output_dir, "spans.pdf"))
    assert r.st_size > 500000

    assert os.path.exists(os.path.join(base_output_dir, "images")) is True
    assert os.path.isdir(os.path.join(base_output_dir, "images")) is True

    # teardown
    shutil.rmtree(temp_output_dir)
