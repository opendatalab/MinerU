import time
import requests
from config import API_SUBMIT, API_STATUS, API_DATA
from pathlib import Path
from loguru import logger

class MinerUClient:
    def __init__(self, max_retries=3, backoff_factor=2):
        self.session = requests.Session()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _request_with_retry(self, method, url, **kwargs):
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()  # raise exception for 4xx/5xx
                return response
            except Exception as e:
                last_exception = e
                wait_time = self.backoff_factor ** attempt
                logger.warning(f"Request failed to {url} on attempt {attempt+1}. Retrying in {wait_time}s. Error: {e}")
                time.sleep(wait_time)
        logger.error(f"Max retries reached for {url}")
        raise last_exception

    def submit_task(self, file_path: Path):
        """Submit a file to the MinerU API."""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/pdf')}
                data = {
                    'backend': 'pipeline',
                    'lang': 'en',
                    'method': 'auto',
                    'formula_enable': 'true',
                    'table_enable': 'true',
                    'priority': 0
                }
                response = self._request_with_retry('POST', API_SUBMIT, files=files, data=data, timeout=30)
                
            status_code = response.status_code
            body = response.json() if response.ok else response.text
            
            return {
                "success": response.ok and isinstance(body, dict) and body.get("success", False),
                "status_code": status_code,
                "body": body,
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": 0,
                "body": f"Exception during request: {str(e)}"
            }

    def get_task_status(self, task_id: str):
        """Get status of an existing task."""
        try:
            url = f"{API_STATUS}/{task_id}"
            response = self._request_with_retry('GET', url, timeout=10)
            status_code = response.status_code
            body = response.json() if response.ok else response.text
            
            return {
                "success": response.ok and isinstance(body, dict) and body.get("success", False),
                "status_code": status_code,
                "body": body,
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": 0,
                "body": f"Exception during request: {str(e)}"
            }

    def get_task_data(self, task_id: str):
        """Get detailed data output for a completed task."""
        try:
            url = f"{API_DATA}/{task_id}/data?include_fields=md"
            response = self._request_with_retry('GET', url, timeout=10)
            status_code = response.status_code
            body = response.json() if response.ok else response.text
            
            return {
                "success": response.ok and isinstance(body, dict) and body.get("success", False),
                "status_code": status_code,
                "body": body,
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": 0,
                "body": f"Exception during request: {str(e)}"
            }

