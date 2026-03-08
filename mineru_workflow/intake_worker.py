import time
import shutil
from pathlib import Path
from loguru import logger
from datetime import datetime, timezone
import traceback

from config import (
    BASE_DIR_IN, DIR_SUBMITTED, DIR_ERROR, 
    FILE_SUBMITTED, FILE_ERROR, FILE_PROCESSING
)
from file_utils import AtomicJsonFile
from mineru_client import MinerUClient

class IntakeWorker:
    def __init__(self):
        self.api_client = MinerUClient()

    def scan_and_submit(self):
        """Scan the input directory and submit any found PDFs sequentially."""
        try:
            # Find all .pdf files directly under BASE_DIR_IN (not recursing into SUBMITTED/PROCESSING/COMPLETED/ERROR)
            pdf_files = [p for p in BASE_DIR_IN.iterdir() if p.is_file() and p.suffix.lower() == '.pdf' and not p.name.startswith('.')]
            
            if not pdf_files:
                # No files found, exit cleanly
                return

            # Submit sequentially to avoid queue race conditions
            for pdf_path in pdf_files:
                self._submit_single_file(pdf_path)
                
        except Exception as e:
            logger.error(f"Error during scan and submit: {e}")
            logger.error(traceback.format_exc())

    def _submit_single_file(self, file_path: Path):
        logger.info(f"Submitting file to MinerU API: {file_path.name}")
        
        # Read the current processing tasks to avoid duplicates
        processing_tasks = AtomicJsonFile.read_json_dict(FILE_PROCESSING)
        
        # Check against existing tasks based on file path to prevent duplicate submission 
        # (This is basic idempotency, if same filename already pending)
        for t_info in processing_tasks.values():
            if t_info.get("original_filename") == file_path.name:
                logger.warning(f"File {file_path.name} already in processing queue (Task ID: {t_info.get('task_id')}). Skipping.")
                # We could potentially move it to a 'SKIPPED' dir, but let's leave it for now or move to Error.
                return

        response = self.api_client.submit_task(file_path)
        
        timestamp = datetime.now(timezone.utc).isoformat()

        
        log_record = {
            "original_filename": file_path.name,
            "original_full_path": str(file_path),
            "submitted_at": timestamp,
            "api_response": response["body"],
            "http_status_code": response["status_code"],
        }
        
        if response["success"]:
            # Task successfully submitted
            body = response["body"]
            task_id = body.get("task_id")
            log_record["task_id"] = task_id
            
            # Record securely in SUBMITTED.json log
            AtomicJsonFile.append_jsonl(FILE_SUBMITTED, log_record)
            
            # Add to PROCESSING.json registry
            processing_tasks[task_id] = log_record
            AtomicJsonFile.write_json_dict(FILE_PROCESSING, processing_tasks)
            
            # Move to SUBMITTED directory
            dest_path = DIR_SUBMITTED / file_path.name
            
            # Handle collision
            if dest_path.exists():
                dest_path = DIR_SUBMITTED / f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
            shutil.move(str(file_path), str(dest_path))
            
            logger.success(f"Successfully submitted {file_path.name} with Task ID {task_id}")
            
        else:
            # Task submission failed
            logger.error(f"Failed to submit {file_path.name}. API Response: {response['body']}")
            
            # Append failure log
            AtomicJsonFile.append_jsonl(FILE_ERROR, log_record)
            
            # Move to ERROR directory
            dest_path = DIR_ERROR / file_path.name
            if dest_path.exists():
                dest_path = DIR_ERROR / f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
            shutil.move(str(file_path), str(dest_path))

if __name__ == "__main__":
    from config import ensure_directories
    ensure_directories()
    worker = IntakeWorker()
    worker.scan_and_submit()
