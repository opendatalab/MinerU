# Your first HTTP request {#v1-http-api}

Use an HTML file to create a task, upload, poll, and download a ZIP you can inspect. This example uses a local V1 API and native Flash parsing. To upload a single file and wait for an attachment directly, use the separate [file-parse service](file_parse.md).

## Run the single-file example {#curl-example}

<span id="tasks"></span>

With MinerU installed, start the service in the first terminal.

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier flash
```

Save [tasks-curl.sh](../../assets/examples/tasks-curl.sh) in your example directory. In a second terminal with Bash, `curl`, and `jq`, change into that directory. The first command creates the input file. If `report.html` from the [quick start](../quick_start/index.md) is already there, run just the second command.

```bash
python -c "from pathlib import Path; Path('report.html').write_text('<h1>MinerU</h1><p>First conversion.</p>', encoding='utf-8')"
bash ./tasks-curl.sh report.html report.zip
```

Open the new `report.zip`. Its `markdown.md` contains `First conversion.`, alongside `middle_json.json` and `structured_content.json`. Results with images include assets under `images/`. The output path must be unused. To connect to another self-hosted service, set `MINERU_API_URL`. For an authenticated service also set `MINERU_API_KEY`.

## Create and upload {#create-and-upload}

The script sends `file_name` and the actual file size to `POST /v1/tasks`. It saves the `task_id` and the `file_id` for each input. When a file has an `upload` object, PUT its raw bytes to the supplied URL with the supplied headers. After upload, poll the task. [Tasks request fields](tasks.md#create-request)

## Status and results {#status-and-results}

Poll `GET /v1/tasks/{task_id}`. When a file is `completed`, download its `result_url`. In a batch, completed files can be downloaded individually. Check each failed file's `error`. The example prints the task ID and receipt directory. Exit code `124` means its polling budget expired and you can still query the same task. [States and outputs](tasks.md#outputs-and-downloads)

## Batch processing, recovery, and migration {#client-and-migration}

The single-file example uses upload and result URLs on the same self-hosted API. For URL inputs, hash preflight, signed object-storage downloads, or recovery, use the repository's [full Python client](https://github.com/opendatalab/MinerU/blob/master/scripts/http_task_example.py). The following command processes two files from the repository root.

```bash
MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_task_example.sh a.pdf b.pdf
```

Supply the original `TASK_ID` and the same input order to resume. See the [Tasks client examples](tasks.md#client-examples) for options and exit codes, and the [4.1 migration guide](../reference/migration_4_1.md) for the transition from uploads/jobs.
