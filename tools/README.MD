# 工具脚本使用说明


### OCR Badcase Commands

- **Command without badcase output:**

  `python ocr_badcase.py pdf_json_label_0306.json ocr_dataset.json json_files.zip ocr_overall base_data_ocr.json`

- **Command with badcase output:**
  
  `python ocr_badcase.py pdf_json_label_0306.json ocr_dataset.json json_files.zip ocr_overall base_data_ocr.json --badcase_path ocr_badcase`

### Text Badcase Commands

- **Command without badcase output:**

    `python text_badcase.py pdf_json_label_0306.json pdf_json_label_0229.json json_files.zip text_overall base_data_text.json`



- **Command with badcase output:**

    ` python text_badcase.py pdf_json_label_0306.json pdf_json_label_0229.json json_files.zip text_overall base_data_text.json --badcase_path text_badcase`

- **Command with upload to s3:**

  -  add the following arguments to the command 

        `--s3_bucket_name llm-process-pperf --s3_file_directory qa-validate/pdf-datasets/badcase --AWS_ACCESS_KEY Your AK  --AWS_SECRET_KEY Your SK --END_POINT_URL Your Endpoint ` 

