import os

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