import os
import time

from loguru import logger

from mineru.utils.table_merge import merge_table


def cross_page_table_merge(pdf_info: list[dict]):
    """Merge tables that span across multiple pages in a PDF document.

    Args:
        pdf_info (list[dict]): A list of dictionaries containing information about each page in the PDF.

    Returns:
        None
    """
    is_merge_table = os.getenv('MINERU_TABLE_MERGE_ENABLE', 'true')
    if is_merge_table.lower() in ['true', '1', 'yes']:
        merge_table(pdf_info)
    elif is_merge_table.lower() in ['false', '0', 'no']:
        pass
    else:
        logger.warning(f'unknown MINERU_TABLE_MERGE_ENABLE config: {is_merge_table}, pass')
        pass


def exclude_progress_bar_idle_time(progress_bar, idle_since: float | None, now: float | None = None):
    """Exclude non-processing idle time from a reused tqdm progress bar."""
    if progress_bar is None or idle_since is None:
        return

    if now is None:
        now = time.time()

    idle_duration = now - idle_since
    if idle_duration <= 0:
        return

    if hasattr(progress_bar, "start_t"):
        progress_bar.start_t += idle_duration
    if hasattr(progress_bar, "last_print_t"):
        progress_bar.last_print_t = now
    if hasattr(progress_bar, "last_print_n") and hasattr(progress_bar, "n"):
        progress_bar.last_print_n = progress_bar.n
