import time
import shutil
import traceback
import subprocess
import os
from pathlib import Path
from loguru import logger
from datetime import datetime, timezone
from config import (
    BASE_DIR_OUT, FILE_PROCESSING, FILE_COMPLETED, FILE_ERROR
)
from file_utils import AtomicJsonFile, format_human_readable_title, safe_move_file
from mineru_client import MinerUClient

class PollingWorker:
    def __init__(self):
        self.api_client = MinerUClient()

    def poll_tasks(self):
        """Polls status for all submitted tasks currently in processing."""
        try:
            processing_tasks = AtomicJsonFile.read_json_dict(FILE_PROCESSING)
            if not processing_tasks:
                logger.info("No active tasks in processing queue.")
                return

            tasks_to_remove = []

            for task_id, task_data in processing_tasks.items():
                try:
                    self._check_single_task(task_id, task_data, tasks_to_remove)
                except Exception as e:
                    logger.error(f"Error polling task {task_id}: {e}")
                    logger.error(traceback.format_exc())
            
            # Remove completed/errored from processing
            if tasks_to_remove:
                # Re-read to prevent concurrent overwrite issues if intake worker modified it
                current_processing = AtomicJsonFile.read_json_dict(FILE_PROCESSING)
                for t in tasks_to_remove:
                    if t in current_processing:
                        del current_processing[t]
                AtomicJsonFile.write_json_dict(FILE_PROCESSING, current_processing)

        except Exception as e:
            logger.error(f"Error during polling: {e}")
            logger.error(traceback.format_exc())

    def _fix_folder_permissions(self, folder_path: Path):
        """Fix permissions of the docker output folder using a temporary container."""
        try:
            uid = os.getuid()
            gid = os.getgid()
            if uid == 0:
                return  # already running as root
            subprocess.run([
                "docker", "run", "--rm", 
                "-v", f"{folder_path.absolute()}:/target", 
                "alpine", "chown", "-R", f"{uid}:{gid}", "/target"
            ], check=True, capture_output=True)
            logger.info(f"Fixed ownership of {folder_path.name} to {uid}:{gid}")
        except Exception as e:
            logger.error(f"Failed to fix permissions for {folder_path.name}: {e}")

    def _check_single_task(self, task_id: str, task_data: dict, tasks_to_remove: list):
        logger.info(f"Checking status for Task ID: {task_id}")
        
        response = self.api_client.get_task_status(task_id)
        
        if not response["success"]:
            logger.warning(f"Failed to check status for {task_id}. Response: {response['body']}")
            return
            
        body = response["body"]
        status = body.get("status", "unknown")
        error_msg = body.get("error_message")
        
        # Consider a task terminal if it's completed, failed, cancelled, or has an error message
        is_completed = status == "completed"
        is_error = status in ["failed", "cancelled", "error"] or bool(error_msg)
        
        timestamp = datetime.now(timezone.utc).isoformat()

        log_record = {
            "task_id": task_id,
            "original_filename": task_data.get("original_filename"),
            "original_full_path": task_data.get("original_full_path"),
            "submitted_at": task_data.get("submitted_at"),
            "last_polled_at": timestamp,
            "api_response": body,
            "http_status_code": response["status_code"]
        }

        if is_error:
            logger.error(f"Task {task_id} failed or cancelled. Status: {status}, Error: {error_msg}")
            AtomicJsonFile.append_jsonl(FILE_ERROR, log_record)
            tasks_to_remove.append(task_id)
            
        elif is_completed:
            logger.success(f"Task {task_id} completed successfully.")
            
            # Request markdown content to get title
            task_data_response = self.api_client.get_task_data(task_id)
            md_content = ""
            if task_data_response["success"] and isinstance(task_data_response["body"], dict):
                md_info = task_data_response["body"].get("data", {}).get("markdown", {})
                if isinstance(md_info, dict):
                    md_content = md_info.get("content", "")
                    
            new_title = format_human_readable_title(
                raw_title="", 
                text_content=md_content, 
                original_filename=task_data.get("original_filename", "")
            )
            
            # Attach final data payload to log
            log_record["extracted_title"] = new_title
            log_record["api_data_response"] = task_data_response.get("body") if task_data_response["success"] else None
            
            AtomicJsonFile.append_jsonl(FILE_COMPLETED, log_record)
            
            # Rename output folder
            self._rename_output_folder(task_id, new_title)
            
            tasks_to_remove.append(task_id)
            
    def _rename_output_folder(self, task_id: str, new_title: str):
        """Finds the output folder in MinerU_OUT and renames it"""
        task_dir = BASE_DIR_OUT / task_id
        
        if not task_dir.exists():
            logger.warning(f"Output directory {task_dir} not found for task {task_id}.")
            return
            
        # Target folder
        dest_dir = BASE_DIR_OUT / new_title
        
        # Handle collision
        if dest_dir.exists():
            dest_dir = BASE_DIR_OUT / f"{new_title}_{task_id[:8]}"
            
        try:
            shutil.move(str(task_dir), str(dest_dir))
            logger.info(f"Renamed output folder to: {dest_dir.name}")
            self._fix_folder_permissions(dest_dir)
            # Now rename the files inside
            self._rename_output_files(dest_dir, new_title)
        except Exception as e:
            logger.error(f"Failed to rename output folder {task_dir}: {e}")

    def _rename_output_files(self, dest_dir: Path, new_title: str):
        """Renames the individual files inside the output folder and moves them out of nested subfolders."""
        try:
            # Find the 'auto' directory which contains the actual results
            auto_dirs = list(dest_dir.glob("**/auto"))
            if not auto_dirs:
                logger.warning(f"No 'auto' directory found in {dest_dir}")
                return
                
            auto_dir = auto_dirs[0]
            
            # 1. Rename and move the .md file
            for md_file in auto_dir.glob("*.md"):
                new_path = dest_dir / f"{new_title}.md"
                if not new_path.exists():
                    shutil.move(str(md_file), str(new_path))
                    logger.info(f"Moved/Renamed MD: {new_path.name}")

            # 2. Rename and move the origin PDF
            for pdf_file in auto_dir.glob("*_origin.pdf"):
                new_path = dest_dir / f"{new_title}.pdf"
                if not new_path.exists():
                    shutil.move(str(pdf_file), str(new_path))
                    logger.info(f"Moved/Renamed PDF: {new_path.name}")

            # 3. Rename and move other known artifacts
            suffixes = {
                "_content_list.json": ".content_list.json",
                "_content_list_v2.json": ".content_list_v2.json",
                "_middle.json": ".middle.json",
                "_model.json": ".model.json",
                "_layout.pdf": ".layout.pdf",
                "_span.pdf": ".span.pdf"
            }
            
            for f in auto_dir.glob("*"):
                for old_suffix, new_suffix in suffixes.items():
                    if f.name.endswith(old_suffix):
                        new_path = dest_dir / f"{new_title}{new_suffix}"
                        if not new_path.exists():
                            shutil.move(str(f), str(new_path))
                            logger.info(f"Moved/Renamed artifact: {new_path.name}")
                        break
            
            # 4. Move images directory if it exists
            images_dir = auto_dir / "images"
            if images_dir.exists():
                dest_images = dest_dir / "images"
                if not dest_images.exists():
                    shutil.move(str(images_dir), str(dest_images))
                    logger.info("Moved images directory to top level.")

        except Exception as e:
            logger.error(f"Error renaming/moving files in {dest_dir.name}: {e}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    from config import ensure_directories
    ensure_directories()
    worker = PollingWorker()
    worker.poll_tasks()
