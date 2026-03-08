import os
from pathlib import Path

# MinerU directories (using the fixed paths)
BASE_DIR_IN = Path("/media/cng420/Development/MinerU_IN")
BASE_DIR_OUT = Path("/media/cng420/Development/MinerU_OUT")

# State directories
DIR_SUBMITTED = BASE_DIR_IN / "SUBMITTED"
DIR_PROCESSING = BASE_DIR_IN / "PROCESSING"
DIR_COMPLETED = BASE_DIR_IN / "COMPLETED"
DIR_ERROR = BASE_DIR_IN / "ERROR"

# State files (JSON logs)
FILE_SUBMITTED = DIR_SUBMITTED / "SUBMITTED.json"
FILE_PROCESSING = DIR_PROCESSING / "PROCESSING.json"
FILE_COMPLETED = DIR_COMPLETED / "COMPLETED.json"
FILE_ERROR = DIR_ERROR / "ERROR.json"

# MinerU Tianshu API endpoints
API_BASE_URL = os.getenv("MINERU_API_BASE_URL", "http://localhost:8004")
API_SUBMIT = f"{API_BASE_URL}/api/v1/tasks/submit"
API_STATUS = f"{API_BASE_URL}/api/v1/tasks"  # /task_id appended later
API_DATA = f"{API_BASE_URL}/api/v1/tasks"    # /task_id/data appended later

# Polling and Scanning Intervals (in seconds)
# Speeding this up for more responsive feel per user feedback.
SCAN_INTERVAL_MINUTES = 5
POLL_INTERVAL_MINUTES = 1

SCAN_INTERVAL_SEC = SCAN_INTERVAL_MINUTES * 60
POLL_INTERVAL_SEC = POLL_INTERVAL_MINUTES * 60

def ensure_directories():
    """Ensure all required input, state, and output directories exist."""
    directories = [
        BASE_DIR_IN,
        DIR_SUBMITTED,
        DIR_PROCESSING,
        DIR_COMPLETED,
        DIR_ERROR,
        BASE_DIR_OUT
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
