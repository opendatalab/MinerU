import os
conf = {
"code_path": os.environ.get('GITHUB_WORKSPACE'),
"pdf_dev_path" : os.environ.get('GITHUB_WORKSPACE') + "/tests/test_cli/pdf_dev",
#"code_path": "/home/quyuan/ci/actions-runner/MinerU",
#"pdf_dev_path": "/home/quyuan/ci/actions-runner/MinerU/tests/test_cli/pdf_dev",
"pdf_res_path": "/tmp/magic-pdf",
"jsonl_path": "s3://llm-qatest-pnorm/mineru/test/line1.jsonl",
"s3_pdf_path": "s3://llm-qatest-pnorm/mineru/test/test_rearch_report.pdf"
}
