# MinerU Reference Tiếng Việt

## Mục đích

Tài liệu này là bản reference ngắn bằng tiếng Việt, đặt trong `docs/en/reference` để tiện tra cứu nhanh khi đọc code hoặc vận hành repo.

Nó tóm tắt:

- MinerU dùng để làm gì
- các backend chính khác nhau ở đâu
- luồng xử lý một file
- các đầu ra chính
- module nào chịu trách nhiệm phần nào

## MinerU dùng để làm gì

MinerU là công cụ parse tài liệu để chuyển các đầu vào như:

- `PDF`
- ảnh tài liệu
- `DOCX`
- `PPTX`
- `XLSX`

thành các đầu ra có cấu trúc:

- Markdown
- JSON
- content list
- ảnh trích xuất

Nó phù hợp cho:

- RAG
- indexing tài liệu
- preprocessing cho LLM
- trích xuất nội dung có cấu trúc từ tài liệu phức tạp

## Các backend chính

### `pipeline`

Đây là pipeline OCR/layout truyền thống.

Đặc điểm:

- tương thích tốt
- hỗ trợ đa ngôn ngữ
- chạy được trên CPU
- thường ít hallucination hơn

Khi nào nên dùng:

- khi cần ổn định
- khi máy không có GPU
- khi muốn OCR truyền thống

Mã chính:

- `mineru/backend/pipeline/pipeline_analyze.py`

### `vlm-*`

Đây là backend dùng Vision-Language Model để hiểu trực tiếp từng trang tài liệu.

Đặc điểm:

- hiểu layout tốt hơn
- chính xác hơn với tài liệu phức tạp
- cần tài nguyên mạnh hơn

Biến thể:

- `vlm-auto-engine`: chạy model local
- `vlm-http-client`: gọi server VLM từ xa

Mã chính:

- `mineru/backend/vlm/vlm_analyze.py`

### `hybrid-*`

Đây là backend kết hợp VLM với OCR/formula refinement.

Đặc điểm:

- dùng VLM để hiểu bố cục
- dùng OCR để tăng độ chắc chắn ở vùng khó
- thường là lựa chọn mạnh nhất cho PDF thực tế

Biến thể:

- `hybrid-auto-engine`: local, cần GPU
- `hybrid-http-client`: client local + VLM server từ xa

Mã chính:

- `mineru/backend/hybrid/hybrid_analyze.py`

### `office`

Nhánh dành riêng cho `docx/pptx/xlsx`.

Đặc điểm:

- không đi qua pipeline PDF
- parse trực tiếp từ tài liệu Office

Mã chính:

- `mineru/backend/office/docx_analyze.py`
- `mineru/backend/office/pptx_analyze.py`
- `mineru/backend/office/xlsx_analyze.py`

## Luồng xử lý một file

Luồng tổng quát:

1. nhận request hoặc input path
2. lưu file upload / đọc file bytes
3. nhận diện loại file
4. nếu là Office thì đi thẳng vào office parser
5. nếu là ảnh thì chuyển sang PDF bytes
6. nếu là PDF/ảnh thì chọn backend `pipeline`, `vlm-*`, hoặc `hybrid-*`
7. backend sinh `middle_json` và model output
8. hệ thống render ra Markdown, JSON, content list, images

Các file điều phối chính:

- `mineru/cli/fast_api.py`
- `mineru/cli/common.py`

Hàm quan trọng:

- `do_parse()`
- `aio_do_parse()`

## Các đầu ra chính

Sau khi parse, MinerU có thể tạo:

- `{name}.md`
- `{name}_content_list.json`
- `{name}_content_list_v2.json`
- `{name}_middle.json`
- `{name}_model.json`
- thư mục `images/`
- file debug như `layout.pdf`, `span.pdf`

Ý nghĩa ngắn gọn:

- `.md`: nội dung dễ đọc cho người
- `content_list*.json`: cấu trúc gọn cho downstream
- `middle.json`: dữ liệu trung gian đầy đủ
- `model.json`: đầu ra gần với inference layer hơn
- `images/`: ảnh cắt ra từ tài liệu

## Module nào làm gì

### `mineru/cli`

Lớp entrypoint và orchestration.

Chứa logic:

- nhận request
- chọn backend
- khởi chạy API local tạm thời
- quản lý router, worker, output path

Nên đọc khi muốn hiểu:

- CLI hoạt động ra sao
- API route vào backend thế nào

### `mineru/backend`

Lớp parse thực tế.

Chứa:

- `pipeline`
- `vlm`
- `hybrid`
- `office`

Đây là nơi chịu trách nhiệm trích xuất nội dung từ file.

### `mineru/model`

Lớp model wrapper và converter.

Bao gồm:

- OCR
- layout
- table
- formula
- VLM server integration
- converter cho `docx/pptx/xlsx`

### `mineru/data`

Lớp storage và IO abstraction.

Phục vụ:

- local file IO
- S3 IO
- path/schema helpers

### `mineru/utils`

Lớp utility dùng chung toàn repo.

Ví dụ:

- PDF tools
- OCR helpers
- bbox utils
- config reader
- model download helpers
- visualization helpers

## Model được load ở đâu

Repo này hỗ trợ 3 kiểu model source:

- `huggingface`
- `modelscope`
- `local`

Hiện code đã được chỉnh để khi chạy từ source repo, model có thể được tải và load ngay trong repo.

Vị trí quan trọng:

- config: `MinerU/mineru.json`
- pipeline models: `MinerU/.mineru/models/pipeline`
- vlm models: `MinerU/.mineru/models/vlm`

Biến môi trường thường dùng:

- `MINERU_MODEL_SOURCE`
- `MINERU_TOOLS_CONFIG_JSON`
- `MINERU_DEVICE_MODE`

## CPU và GPU

Theo tài liệu repo:

- `pipeline`: chạy được trên CPU
- `vlm-auto-engine`: không hỗ trợ pure CPU
- `hybrid-auto-engine`: không hỗ trợ pure CPU
- `vlm-http-client`: chạy được trên CPU phía client
- `hybrid-http-client`: chạy được trên CPU phía client

Nói ngắn gọn:

- CPU only local: dùng `pipeline`
- muốn `hybrid` nhưng local chỉ có CPU: dùng `hybrid-http-client`

## Cách đọc code nhanh

Nếu mới vào repo, nên đọc theo thứ tự:

1. `mineru/cli/fast_api.py`
2. `mineru/cli/common.py`
3. `mineru/backend/hybrid/hybrid_analyze.py`
4. `mineru/backend/pipeline/pipeline_analyze.py`
5. `mineru/backend/office/*.py`
6. `mineru/utils/models_download_utils.py`
7. `mineru/utils/config_reader.py`

## Tài liệu nên đọc tiếp

Nếu muốn đi sâu hơn, xem thêm:

- `module_overview.md`
- `processing_flow.md`
- `output_files.md`
- `../usage/quick_usage.md`
- `../usage/cli_tools.md`
- `../usage/model_source.md`

## Ghi chú

Đây là một trang reference tiếng Việt đặt trong thư mục `docs/en/reference` theo đúng yêu cầu vị trí file.

Nếu cần, có thể tách tiếp thành:

- bản chuyên về backend
- bản chuyên về model/config
- bản chuyên về output format
