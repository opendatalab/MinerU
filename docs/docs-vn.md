# MinerU Tài Liệu Tiếng Việt

## Giới thiệu

MinerU là một công cụ parse tài liệu, dùng để chuyển:

- `PDF`
- ảnh tài liệu
- `DOCX`
- `PPTX`
- `XLSX`

thành các đầu ra có cấu trúc như:

- Markdown
- JSON
- content list
- ảnh trích xuất

Repo này hỗ trợ nhiều backend phân tích tài liệu, phù hợp cho các use case như:

- RAG
- document understanding
- knowledge extraction
- preprocessing cho LLM

## Cấu trúc tài liệu hiện có

Thư mục `docs/` của repo hiện chia làm 2 ngôn ngữ chính:

- `docs/en`: tài liệu tiếng Anh
- `docs/zh`: tài liệu tiếng Trung

Các nhóm nội dung chính gồm:

- `quick_start`: cài đặt và chạy nhanh
- `usage`: hướng dẫn dùng CLI, model source, tham số nâng cao
- `reference`: tài liệu tham chiếu như output format, flow xử lý, module overview
- `demo`: minh họa sử dụng
- `faq`: câu hỏi thường gặp

File này là bản tổng hợp tiếng Việt dựa trên các phần đó.

## MinerU xử lý được gì

MinerU dùng để trích xuất nội dung có cấu trúc từ tài liệu đầu vào:

- văn bản
- tiêu đề
- bảng
- hình ảnh
- biểu đồ
- công thức

Kết quả có thể dùng trực tiếp cho:

- đọc lại nội dung theo thứ tự hợp lý
- tạo Markdown dễ kiểm tra
- lấy JSON cho hệ thống downstream
- làm input cho indexing hoặc RAG pipeline

## Các backend chính

MinerU có 3 nhóm backend chính cho PDF/ảnh:

### `pipeline`

Đây là pipeline OCR/layout truyền thống.

Đặc điểm:

- tương thích tốt
- hỗ trợ đa ngôn ngữ
- ít hallucination hơn
- chạy được trên CPU

Phù hợp khi:

- cần tính ổn định
- cần chạy trên máy không có GPU
- ưu tiên OCR truyền thống

### `vlm-*`

Đây là backend dùng Vision-Language Model để hiểu trực tiếp trang tài liệu.

Đặc điểm:

- hiểu layout mạnh hơn
- thường cho chất lượng cao hơn với tài liệu phức tạp
- yêu cầu tài nguyên cao hơn

Biến thể:

- `vlm-auto-engine`: chạy local
- `vlm-http-client`: gọi server VLM từ xa

### `hybrid-*`

Đây là backend kết hợp VLM với OCR/formula refinement.

Đặc điểm:

- dùng VLM để hiểu layout
- dùng OCR/pipeline để tăng độ chắc chắn ở vùng khó
- thường là lựa chọn cân bằng tốt nhất cho PDF thực tế

Biến thể:

- `hybrid-auto-engine`: chạy local, cần GPU
- `hybrid-http-client`: client local + model VLM từ xa, chạy được trên CPU

## Luồng xử lý một file

Luồng tổng quát:

1. người dùng upload file hoặc truyền đường dẫn input
2. MinerU đọc bytes và nhận dạng loại file
3. nếu là `docx/pptx/xlsx` thì đi nhánh office
4. nếu là PDF/ảnh thì chuẩn hóa về PDF bytes
5. hệ thống chọn backend phù hợp
6. backend sinh `middle_json` và model output
7. MinerU render ra Markdown, JSON, ảnh, file debug

Các file mã nguồn quan trọng cho flow này:

- `mineru/cli/fast_api.py`
- `mineru/cli/common.py`
- `mineru/backend/pipeline/pipeline_analyze.py`
- `mineru/backend/vlm/vlm_analyze.py`
- `mineru/backend/hybrid/hybrid_analyze.py`
- `mineru/backend/office/*.py`

## Các loại đầu ra chính

Sau khi parse, MinerU có thể sinh ra:

- `{name}.md`
- `{name}_content_list.json`
- `{name}_content_list_v2.json`
- `{name}_middle.json`
- `{name}_model.json`
- thư mục `images/`
- file debug như `layout.pdf`, `span.pdf`

Ý nghĩa ngắn gọn:

- `.md`: bản nội dung dễ đọc
- `content_list*.json`: cấu trúc nội dung gọn hơn cho downstream
- `middle.json`: dữ liệu trung gian đầy đủ
- `model.json`: kết quả gần với đầu ra model hơn
- `images/`: ảnh crop ra từ tài liệu

## Các module chính trong repo

### `mineru/cli`

Lớp entrypoint và orchestration.

Nơi nên đọc nếu bạn muốn hiểu:

- CLI chạy như thế nào
- API nhận request ra sao
- backend được chọn như thế nào

File quan trọng:

- `mineru/cli/client.py`
- `mineru/cli/fast_api.py`
- `mineru/cli/common.py`
- `mineru/cli/router.py`

### `mineru/backend`

Lớp xử lý parse thực tế.

Gồm:

- `pipeline/`
- `vlm/`
- `hybrid/`
- `office/`

Đây là nơi chứa logic trích xuất nội dung từ file.

### `mineru/model`

Lớp model wrapper và converter cho:

- OCR
- layout
- formula
- table
- docx/pptx/xlsx
- VLM server integration

### `mineru/data`

Lớp IO và storage abstraction.

Dùng khi bạn cần:

- đọc/ghi file local
- ghi ra S3
- xử lý path/schema

### `mineru/utils`

Lớp utility dùng chung toàn repo.

Bao gồm:

- PDF tools
- OCR helpers
- config reader
- model download helpers
- bbox utilities
- visualization helpers

## Cấu hình model

MinerU hỗ trợ 3 kiểu model source:

- `huggingface`
- `modelscope`
- `local`

Trong repo này, code đã được chỉnh để model mặc định có thể tải và load trong chính repo nếu chạy từ source checkout.

Các vị trí quan trọng:

- config: `MinerU/mineru.json`
- model local:
  - `MinerU/.mineru/models/pipeline`
  - `MinerU/.mineru/models/vlm`

Biến môi trường hay dùng:

- `MINERU_MODEL_SOURCE`
- `MINERU_TOOLS_CONFIG_JSON`
- `MINERU_DEVICE_MODE`

## Cách chạy nhanh trong repo này

Repo hiện đã có thêm 2 script tiện dụng:

### `setup.sh`

Dùng để:

- tạo virtualenv
- cài package editable
- tạo `mineru.json`
- chuẩn bị thư mục model trong repo

Ví dụ:

```bash
./setup.sh
MINERU_DOWNLOAD_MODELS=1 ./setup.sh
```

### `run.sh`

Dùng để chạy MinerU nhanh với config local trong repo.

Ví dụ:

```bash
./run.sh parse demo/pdfs/demo1.pdf
./run.sh api --host 127.0.0.1 --port 8000
./run.sh gradio --server-name 0.0.0.0 --server-port 7860
```

## Khi nào nên dùng backend nào

- Dùng `pipeline` khi:
  - cần CPU
  - cần tính ổn định
  - tài liệu không quá phức tạp

- Dùng `hybrid-auto-engine` khi:
  - có GPU
  - muốn chất lượng tốt cho PDF thực tế

- Dùng `hybrid-http-client` khi:
  - máy local chỉ có CPU
  - nhưng bạn có VLM server từ xa

- Dùng `vlm-*` khi:
  - muốn đánh giá thuần năng lực VLM
  - cần parse tài liệu layout phức tạp

## Lộ trình đọc code đề xuất

Nếu bạn mới vào repo, nên đọc theo thứ tự:

1. `mineru/cli/fast_api.py`
2. `mineru/cli/common.py`
3. `mineru/backend/hybrid/hybrid_analyze.py`
4. `mineru/backend/pipeline/pipeline_analyze.py`
5. `mineru/backend/office/*.py`
6. `mineru/utils/models_download_utils.py`
7. `mineru/utils/config_reader.py`

## Tài liệu tham khảo nên đọc tiếp

Nếu bạn muốn đi sâu hơn, nên xem thêm:

- `docs/en/usage/quick_usage.md`
- `docs/en/usage/cli_tools.md`
- `docs/en/usage/model_source.md`
- `docs/en/reference/output_files.md`
- `docs/en/reference/processing_flow.md`
- `docs/en/reference/module_overview.md`
- `docs/zh/reference/processing_flow.md`
- `docs/zh/reference/module_overview.md`

## Ghi chú

File này là bản tổng hợp tiếng Việt để giúp onboard nhanh.

Nó không thay thế toàn bộ tài liệu chính thức trong `docs/en` và `docs/zh`, nhưng đủ để:

- hiểu repo đang làm gì
- biết backend nào nên dùng
- biết model nằm ở đâu
- biết bắt đầu đọc code từ đâu
