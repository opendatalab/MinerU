# 第一次 HTTP 请求 {#v1-http-api}

用一个 HTML 文件走完创建任务、上传、查询和下载，得到一个可打开的 ZIP。这个例子使用本地 V1 API 和原生 Flash 解析；如果你要直接上传单文件并等待附件，请用独立的 [file-parse 服务](file_parse.md)。

## 运行单文件示例 {#curl-example}

<span id="tasks"></span>

安装 MinerU 后，在第一个终端启动服务。

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier flash
```

将 [tasks-curl.sh](../../assets/examples/tasks-curl.sh) 下载到示例目录。在有 Bash、`curl` 和 `jq` 的第二个终端进入该目录。第一行创建输入文件。如果目录里已有[快速入门](../quick_start/index.md)创建的 `report.html`，直接运行第二行。

```bash
python -c "from pathlib import Path; Path('report.html').write_text('<h1>MinerU</h1><p>First conversion.</p>', encoding='utf-8')"
bash ./tasks-curl.sh report.html report.zip
```

打开新生成的 `report.zip`。`markdown.md` 中可以看到 `First conversion.`，同时还有 `middle_json.json` 和 `structured_content.json`；带图片的结果会包含 `images/` 素材。脚本要求输出路径尚未存在。要连接其他自部署服务，可设置 `MINERU_API_URL`；服务启用认证时设置 `MINERU_API_KEY`。

## 创建和上传 {#create-and-upload}

<span id="_1"></span>
脚本将 `file_name` 和真实文件大小提交给 `POST /v1/tasks`，保存 `task_id` 及各输入对应的 `file_id`。只有响应包含 `files[].upload` 时才向返回的地址 PUT 原始字节，采用该对象的请求头。上传成功即可查询。[Tasks 请求字段](tasks.md#create-request)

## 查询和下载 {#status-and-results}

<span id="_2"></span>
使用 `GET /v1/tasks/{task_id}` 查询。单文件状态为 `completed` 时，从该文件的 `result_url` 下载。多文件任务中，已完成的文件可以先下载；某个文件失败时，检查其 `error`。脚本会打印任务 ID 和回执目录，轮询预算耗尽时退出码为 `124`，可继续查询原任务。[状态与输出](tasks.md#outputs-and-downloads)

## 批量、恢复与迁移 {#client-and-migration}

<span id="_3"></span>
单文件示例只访问同一自部署 API 的上传与结果地址。需要 URL 输入、哈希预检、对象存储签名下载或中断恢复时，使用仓库中的[完整 Python 客户端](https://github.com/opendatalab/MinerU/blob/master/scripts/http_task_example.py)。下面的命令从仓库根目录处理两个文件。

```bash
MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_task_example.sh a.pdf b.pdf
```

指定原 `TASK_ID` 和相同输入顺序可恢复已有任务。完整选项和退出码见 [Tasks 客户端示例](tasks.md#client-examples)，旧 uploads/jobs 的替换方式见 [4.1 迁移指南](../reference/migration_4_1.md)。
