import time
import schedule
import threading
from loguru import logger
from config import ensure_directories, SCAN_INTERVAL_MINUTES, POLL_INTERVAL_MINUTES
from intake_worker import IntakeWorker
from polling_worker import PollingWorker

def run_intake():
    logger.info("--- Starting scheduled Intake Scan ---")
    worker = IntakeWorker()
    worker.scan_and_submit()
    logger.info("--- Finished Intake Scan ---")

def run_polling():
    logger.info("--- Starting scheduled Status Polling ---")
    worker = PollingWorker()
    worker.poll_tasks()
    logger.info("--- Finished Status Polling ---")

def run_scheduler():
    logger.info("Starting MinerU Background Workflow Scheduler")
    ensure_directories()
    
    # Run immediately on start
    run_intake()
    run_polling()
    
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(run_intake)
    schedule.every(POLL_INTERVAL_MINUTES).minutes.do(run_polling)
    
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    # Create thread so it can run cleanly or be imported
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting workflow scheduler...")
