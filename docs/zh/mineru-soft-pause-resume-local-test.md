# MinerU 软暂停/继续本地验证命令

本文记录本地验证 MinerU API 软暂停、继续、取消，以及检查点文件的常用命令。测试目标是确认：

- `pause` 请求后，任务会从 `processing` 进入 `pause_requested`。
- 当前 processing window 完成后，任务进入 `paused`。
- `pause_resume/checkpoint.json` 已写入，并能看到下一次继续解析的页码。
- `resume` 后任务重新进入 `processing`。
- `cancel` 后任务进入 `cancelled`，新版本会清理该任务的 `pause_resume/` 检查点目录。

## 1. 准备环境

在本机 MinerU 源码目录执行：

```bash
cd /Users/sunmo/Downloads/MinerU-master
source .venv/bin/activate
```

建议先固定几个变量：

```bash
export API="http://127.0.0.1:8005"
export OUT="/tmp/mineru-pause-local"
export VLLM_SERVER="http://172.16.107.149:18085"

# 使用页数多一点的 PDF 更容易测出暂停效果。
export PDF="/Users/sunmo/Downloads/钢结构设计标准理解与应用-朱.pdf"

rm -rf "$OUT" /tmp/mineru-api-pause.log /tmp/mineru-submit.json /tmp/mineru-status.json
```

如果短 PDF 解析太快，可能还没来得及暂停就已经完成。此时换成长 PDF，或者继续保持 `MINERU_PROCESSING_WINDOW_SIZE=1`。

## 2. 启动 MinerU API

在第一个终端启动 API。这个终端保持运行，不要按 `Ctrl+C`：

```bash
MINERU_LOG_LEVEL=DEBUG \
MINERU_PROCESSING_WINDOW_SIZE=1 \
MINERU_API_OUTPUT_ROOT=/tmp/mineru-pause-local \
mineru-api --host 127.0.0.1 --port 8005 \
  2>&1 | tee -a /tmp/mineru-api-pause.log
```

如果没有激活 `.venv`，也可以直接使用：

```bash
MINERU_LOG_LEVEL=DEBUG \
MINERU_PROCESSING_WINDOW_SIZE=1 \
MINERU_API_OUTPUT_ROOT=/tmp/mineru-pause-local \
.venv/bin/mineru-api --host 127.0.0.1 --port 8005 \
  2>&1 | tee -a /tmp/mineru-api-pause.log
```

这里的关键配置是：

- `MINERU_PROCESSING_WINDOW_SIZE=1`：每个 window 只处理 1 页，暂停响应更容易观察。
- `MINERU_API_OUTPUT_ROOT=/tmp/mineru-pause-local`：任务输出和检查点都写到这个目录下。
- `/tmp/mineru-api-pause.log`：保留 API 日志，方便 grep 暂停和取消事件。

## 3. 检查 API 是否启动

在第二个终端执行：

```bash
curl -s "$API/health" | python3 -m json.tool
```

能返回健康状态后，再提交解析任务。

## 4. 提交解析任务

```bash
curl -s -X POST "$API/tasks" \
  -F "files=@${PDF};type=application/pdf" \
  -F "backend=vlm-http-client" \
  -F "server_url=${VLLM_SERVER}" \
  -F "return_middle_json=true" \
  -F "return_model_output=true" \
  -F "return_md=true" \
  -F "return_content_list=true" \
  | tee /tmp/mineru-submit.json
```

取出任务 ID：

```bash
export TASK_ID="$(python3 - <<'PY'
import json
print(json.load(open('/tmp/mineru-submit.json'))['task_id'])
PY
)"

echo "$TASK_ID"
```

## 5. 发送暂停命令

```bash
curl -s -X POST "$API/tasks/$TASK_ID/pause" | python3 -m json.tool
```

预期结果：

- 如果任务还在 `processing`，接口返回 `202`，任务状态变成 `pause_requested`。
- MinerU 不会打断当前正在跑的 window。
- 当前 window 完成、checkpoint 写完以后，任务才会进入 `paused`。

## 6. 查看任务状态

单次查看：

```bash
curl -s "$API/tasks/$TASK_ID" | python3 -m json.tool
```

持续查看：

```bash
while true; do
  date '+%H:%M:%S'
  curl -s "$API/tasks/$TASK_ID" \
    | python3 -m json.tool \
    | egrep '"status"|completed_until_page|next_start_page|page_count|checkpoint'
  sleep 1
done
```

判断暂停成功，主要看三点：

1. 状态从 `processing` 变成 `pause_requested`，再变成 `paused`。
2. `GET /tasks/$TASK_ID` 返回里能看到 `checkpoint`。
3. `checkpoint.status` 是 `paused`，`phase` 是 `after_window_completed`。

之前本地测试看到过类似结果：

```json
{
  "version": 1,
  "task_id": "052af4ea-420d-4846-94c7-8c72a7bbe4e2",
  "file_name": "钢结构设计标准理解与应用-朱",
  "backend": "vlm-http-client",
  "parse_method": "auto",
  "status": "paused",
  "phase": "after_window_completed",
  "page_count": 492,
  "window_size": 1,
  "completed_windows": 9,
  "completed_until_page": 9,
  "next_start_page": 10,
  "artifacts": {
    "middle_json_partial": "pause_resume/middle_json_partial.json",
    "model_output_partial": "pause_resume/model_output_partial.json",
    "latest_window": "pause_resume/windows/window-0008.json"
  }
}
```

这个结果表示：

- 任务已经暂停住了。
- 检查点阶段是 `after_window_completed`，也就是当前 window 已完成后暂停。
- 对外页码是 1-based，`completed_until_page=9` 表示前 9 页已经处理完。
- `next_start_page=10` 表示继续时从第 10 页开始。
- `latest_window=window-0008.json` 是最近一次完成的 window 结果。

## 7. 查看检查点文件

检查点目录由 API 启动时的 `MINERU_API_OUTPUT_ROOT` 和 `TASK_ID` 决定：

```bash
export CKPT_DIR="/tmp/mineru-pause-local/$TASK_ID/pause_resume"
echo "$CKPT_DIR"
```

本地测试时，一个实际路径类似：

```text
/tmp/mineru-pause-local/052af4ea-420d-4846-94c7-8c72a7bbe4e2/pause_resume/checkpoint.json
```

列出检查点目录：

```bash
find "$CKPT_DIR" -maxdepth 3 -type f -print
```

格式化查看 `checkpoint.json`：

```bash
cat "$CKPT_DIR/checkpoint.json" | python3 -m json.tool
```

打开检查点所在目录：

```bash
open "$CKPT_DIR"
```

正常会看到：

```text
pause_resume/
  checkpoint.json
  middle_json_partial.json
  model_output_partial.json
  windows/
    window-0000.json
    window-0001.json
    ...
```

如果没有 `checkpoint.json`，优先检查：

- `TASK_ID` 是否是当前任务的 ID。
- API 启动时是否设置了 `MINERU_API_OUTPUT_ROOT=/tmp/mineru-pause-local`。
- 任务是否已经进入 `paused`，而不是还停在 `pause_requested`。
- PDF 是否太短，导致任务已经 `completed`。

## 8. 继续解析

任务进入 `paused` 后，执行：

```bash
curl -s -X POST "$API/tasks/$TASK_ID/resume" | python3 -m json.tool
```

预期结果：

- 接口返回 `202`。
- 任务状态变回 `processing`。
- API 日志里能看到 `MINERU_PAUSE` 的 resume 记录。
- 解析从 checkpoint 的 `next_start_page` 继续往后跑。

查看 resume 日志：

```bash
grep "MINERU_PAUSE" /tmp/mineru-api-pause.log
```

持续看日志：

```bash
tail -f /tmp/mineru-api-pause.log | grep MINERU_PAUSE
```

## 9. 取消任务

取消命令：

```bash
curl -s -X POST "$API/tasks/$TASK_ID/cancel" | python3 -m json.tool
```

取消成功时，API 日志会出现类似：

```text
[MINERU_CANCEL] event=task_cancelled task_id=052af4ea-420d-4846-94c7-8c72a7bbe4e2 status=cancelled error=Task cancellation requested
```

这表示任务已经进入 `cancelled`。当前实现里，取消会清理该任务的 `pause_resume/` 检查点目录，避免同一个输出目录里的旧检查点影响后续排查。

如果是旧版本已经取消过的任务，或者想手动清掉检查点，可以执行：

```bash
rm -rf "/tmp/mineru-pause-local/$TASK_ID/pause_resume"
```

查看取消日志：

```bash
grep "MINERU_CANCEL" /tmp/mineru-api-pause.log
```

## 10. 观察暂停是否释放内存

软暂停只保证当前 window 完成后不再提交后续 window，不承诺立刻释放当前进程已经占用的内存或显存。可以用下面命令观察 MinerU API 进程内存：

```bash
export PID="$(pgrep -f 'mineru-api.*8005' | head -1)"
echo "$PID"
```

持续打印 RSS：

```bash
while true; do
  ts="$(date '+%H:%M:%S')"
  ps -o pid=,rss=,%mem=,command= -p "$PID" \
    | awk -v ts="$ts" '{printf "%s pid=%s rss=%.1fMB mem=%s%% cmd=%s\n", ts, $1, $2/1024, $3, $4}'
  sleep 1
done
```

如果要观察系统整体内存：

```bash
vm_stat 1
```

判断口径：

- 暂停成功的主要证据是任务状态和 checkpoint，不是内存下降。
- 软暂停一般不会明显释放进程内存。
- 如果目标是尽快释放资源，应该使用 `cancel`，或者在 Nova 侧使用独立 worker/process 隔离。

