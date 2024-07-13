import os
conf = {
"code_path": os.environ.get('GITHUB_WORKSPACE'),
"pdf_dev_path" : os.environ.get('GITHUB_WORKSPACE') + "/tests/test_cli/pdf_dev",
"pdf_res_path": "/tmp/magic-pdf"
}

