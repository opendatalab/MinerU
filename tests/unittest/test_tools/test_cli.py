import os
import shutil
import tempfile

from click.testing import CliRunner

from magic_pdf.tools.cli import cli


def test_cli_pdf():
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/tools'
    filename = 'cli_test_01'
    os.makedirs(unitest_dir, exist_ok=True)
    temp_output_dir = tempfile.mkdtemp(dir='/tmp/magic_pdf/unittest/tools')

    # run
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            '-p',
            'tests/unittest/test_tools/assets/cli/pdf/cli_test_01.pdf',
            '-o',
            temp_output_dir,
        ],
    )

    # check
    assert result.exit_code == 0

    base_output_dir = os.path.join(temp_output_dir, 'cli_test_01/auto')

    r = os.stat(os.path.join(base_output_dir, f'{filename}.md'))
    assert r.st_size > 7000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_middle.json'))
    assert r.st_size > 200000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_model.json'))
    assert r.st_size > 15000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_origin.pdf'))
    assert r.st_size > 400000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_layout.pdf'))
    assert r.st_size > 400000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_spans.pdf'))
    assert r.st_size > 400000

    assert os.path.exists(os.path.join(base_output_dir, 'images')) is True
    assert os.path.isdir(os.path.join(base_output_dir, 'images')) is True
    assert os.path.exists(os.path.join(base_output_dir, f'{filename}_content_list.json')) is True

    # teardown
    shutil.rmtree(temp_output_dir)


def test_cli_path():
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/tools'
    os.makedirs(unitest_dir, exist_ok=True)
    temp_output_dir = tempfile.mkdtemp(dir='/tmp/magic_pdf/unittest/tools')

    # run
    runner = CliRunner()
    result = runner.invoke(
        cli, ['-p', 'tests/unittest/test_tools/assets/cli/path', '-o', temp_output_dir]
    )

    # check
    assert result.exit_code == 0

    filename = 'cli_test_01'
    base_output_dir = os.path.join(temp_output_dir, 'cli_test_01/auto')

    r = os.stat(os.path.join(base_output_dir, f'{filename}.md'))
    assert r.st_size > 7000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_middle.json'))
    assert r.st_size > 200000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_model.json'))
    assert r.st_size > 15000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_origin.pdf'))
    assert r.st_size > 400000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_layout.pdf'))
    assert r.st_size > 400000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_spans.pdf'))
    assert r.st_size > 400000

    assert os.path.exists(os.path.join(base_output_dir, 'images')) is True
    assert os.path.isdir(os.path.join(base_output_dir, 'images')) is True
    assert os.path.exists(os.path.join(base_output_dir, f'{filename}_content_list.json')) is True

    base_output_dir = os.path.join(temp_output_dir, 'cli_test_02/auto')
    filename = 'cli_test_02'

    r = os.stat(os.path.join(base_output_dir, f'{filename}.md'))
    assert r.st_size > 5000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_middle.json'))
    assert r.st_size > 200000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_model.json'))
    assert r.st_size > 15000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_origin.pdf'))
    assert r.st_size > 400000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_layout.pdf'))
    assert r.st_size > 400000

    r = os.stat(os.path.join(base_output_dir, f'{filename}_spans.pdf'))
    assert r.st_size > 400000

    assert os.path.exists(os.path.join(base_output_dir, 'images')) is True
    assert os.path.isdir(os.path.join(base_output_dir, 'images')) is True
    assert os.path.exists(os.path.join(base_output_dir, f'{filename}_content_list.json')) is True

    # teardown
    shutil.rmtree(temp_output_dir)
