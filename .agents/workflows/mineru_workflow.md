---
description: Automated PDF processing workflow for MinerU
---

# MinerU Background Workflow

This workflow automates the ingestion, processing, and organization of PDF documents using the MinerU API and a custom background scheduler.

## Prerequisites
- Python 3.10+
- MinerU API running at `http://localhost:8004`
- Access to `/media/cng420/Development/MinerU_IN` and `/media/cng420/Development/MinerU_OUT`

## Steps to Run

1. **Navigate to the workflow directory**
   ```bash
   cd mineru_workflow
   ```

2. **Ensure dependencies are installed**
   ```bash
   pip install -r requirements.txt
   ```

// turbo
3. **Start the background scheduler**
   ```bash
   python main.py
   ```
   *Note: This will start the `IntakeWorker` (scans every 5 mins) and `PollingWorker` (polls every 1 min).*

## Monitoring and Logs

- **Scheduler Logs**: Check the terminal output where `main.py` is running. It uses `loguru` for structured logging.
- **Task States**:
    - `SUBMITTED.json`: Tasks newly found in the IN directory.
    - `PROCESSING.json`: Tasks currently being processed by MinerU.
    - `COMPLETED.json`: Successfully processed tasks with human-readable titles.
    - `ERROR.json`: Tasks that failed or were cancelled.

## Error Handling

If a task enters the `ERROR` state:
1. Check `mineru_workflow/SUBMITTED/ERROR.json` for the error message.
2. Verify the MinerU API is healthy and reachable.
3. Manually move the file back to the root of `MinerU_IN` to re-trigger the intake if the issue is resolved.
