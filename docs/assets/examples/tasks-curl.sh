#!/usr/bin/env bash
# 单文件 curl 示例：仅连接同一自部署 API，SaaS/对象存储请使用完整 Python 客户端。
set -euo pipefail

input_file=${1:?Usage: tasks-curl.sh INPUT_FILE [RESULT_ZIP]}
result_file=${2:-result.zip}
api_base=${MINERU_API_URL:-http://127.0.0.1:8000}
api_base=${api_base%/}
[[ -f "$input_file" ]] || { echo "Input file does not exist" >&2; exit 1; }
[[ ! -e "$result_file" ]] || { echo "Result path already exists" >&2; exit 1; }
command -v jq >/dev/null
work_dir=$(mktemp -d)
printf 'Receipts: %s\n' "$work_dir"

api_headers=()
if [[ -n ${MINERU_API_KEY:-} ]]; then
  api_headers+=(-H "Authorization: Bearer $MINERU_API_KEY")
fi

# 真实文件大小进入 JSON，创建请求不自动重试；回执保留任务 ID 以便恢复。
file_bytes=$(wc -c < "$input_file" | tr -d '[:space:]')
jq -n --arg name "$(basename "$input_file")" --argjson bytes "$file_bytes" \
  '{files:[{file_name:$name,bytes:$bytes}],output_formats:["default"]}' > "$work_dir/request.json"
curl --fail-with-body --max-time 60 ${api_headers[@]+"${api_headers[@]}"} \
  -H 'Content-Type: application/json' --data-binary "@$work_dir/request.json" \
  "$api_base/v1/tasks" -o "$work_dir/create.json"
task_id=$(jq -er '.task_id | select(type == "string" and length > 0)' "$work_dir/create.json")
task_path=$(jq -nr --arg id "$task_id" '$id | @uri')
printf 'Task: %s\n' "$task_id"

# 仅在服务提供上传动作时 PUT；原样保留返回的上传头，不发送 complete 请求。
upload_url=$(jq -r '.files[0].upload.url // empty' "$work_dir/create.json")
if [[ -n "$upload_url" ]]; then
  case "$upload_url" in
    "$api_base"/v1/tasks/*) ;;
    *) echo 'This example requires a same-API upload URL; use the Python client for other deployments.' >&2; exit 1 ;;
  esac
  upload_headers=(${api_headers[@]+"${api_headers[@]}"})
  while IFS= read -r header; do
    upload_headers+=(-H "$header")
  done < <(jq -r '.files[0].upload.headers | to_entries[] | "\(.key): \(.value)"' "$work_dir/create.json")
  curl --fail-with-body --max-time 600 ${upload_headers[@]+"${upload_headers[@]}"} \
    -X PUT --upload-file "$input_file" "$upload_url" -o "$work_dir/upload.json"
fi

# 有界轮询不会取消上游任务；预算耗尽后凭已保存的 task_id 继续查询。
file_status=''
for ((attempt=0; attempt<300; attempt++)); do
  curl --fail-with-body --max-time 60 ${api_headers[@]+"${api_headers[@]}"} \
    "$api_base/v1/tasks/$task_path" -o "$work_dir/status.json"
  file_status=$(jq -er '.files[0].status' "$work_dir/status.json")
  case "$file_status" in
    completed) break ;;
    failed|canceled) cat "$work_dir/status.json" >&2; exit 1 ;;
  esac
  sleep 1
done
[[ "$file_status" == completed ]] || { echo "Poll budget exhausted; keep task $task_id" >&2; exit 124; }

# 本示例只向相同 API 的 Files 路由发送凭据，不跟随下载重定向。
result_url=$(jq -er '.files[0].result_url' "$work_dir/status.json")
case "$result_url" in
  "$api_base"/v1/files/*) ;;
  *) echo 'Use the Python client for signed or cross-origin result URLs.' >&2; exit 1 ;;
esac
http_status=$(curl --fail-with-body --max-time 600 ${api_headers[@]+"${api_headers[@]}"} \
  -w '%{http_code}' -o "$work_dir/result.zip" "$result_url")
[[ "$http_status" == 200 ]] || { echo "Unexpected download status: $http_status" >&2; exit 1; }
mv "$work_dir/result.zip" "$result_file"
printf 'Saved: %s\n' "$result_file"
