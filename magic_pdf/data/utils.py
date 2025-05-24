
import multiprocessing as mp
import threading
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)

import fitz
import numpy as np
from loguru import logger



def fitz_doc_to_image(page, dpi=200) -> dict:
    """Convert fitz.Document to image, Then convert the image to numpy array.

    Args:
        page (_type_): pymudoc page
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img': numpy array, 'width': width, 'height': height }
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pm = page.get_pixmap(matrix=mat, alpha=False)

    # If the width or height exceeds 4500 after scaling, do not scale further.
    if pm.width > 4500 or pm.height > 4500:
        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    # Convert pixmap samples directly to numpy array
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)

    img_dict = {'img': img, 'width': pm.width, 'height': pm.height}

    return img_dict

def load_images_from_pdf(pdf_bytes: bytes, dpi=200, start_page_id=0, end_page_id=None) -> list:
    images = []
    with fitz.open('pdf', pdf_bytes) as doc:
        pdf_page_num = doc.page_count
        end_page_id = (
            end_page_id
            if end_page_id is not None and end_page_id >= 0
            else pdf_page_num - 1
        )
        if end_page_id > pdf_page_num - 1:
            logger.warning('end_page_id is out of range, use images length')
            end_page_id = pdf_page_num - 1

        for index in range(0, doc.page_count):
            if start_page_id <= index <= end_page_id:
                page = doc[index]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # If the width or height exceeds 4500 after scaling, do not scale further.
                if pm.width > 4500 or pm.height > 4500:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                # Convert pixmap samples directly to numpy array
                img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)

                img_dict = {'img': img, 'width': pm.width, 'height': pm.height}
            else:
                img_dict = {'img': [], 'width': 0, 'height': 0}

            images.append(img_dict)
    return images


def convert_page(bytes_page):
    pdfs = fitz.open('pdf', bytes_page)
    page = pdfs[0]
    return fitz_doc_to_image(page)

def parallel_process_pdf_safe(pages, num_workers=None, **kwargs):
    """Process PDF pages in parallel with serialization-safe approach."""
    if num_workers is None:
        num_workers = mp.cpu_count()


    # Process the extracted page data in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Process the page data
        results = list(
            executor.map(convert_page, pages)
        )

    return results


def threaded_process_pdf(pdf_path, num_threads=4, **kwargs):
    """Process all pages of a PDF using multiple threads.

    Parameters:
    -----------
    pdf_path : str
        Path to the PDF file
    num_threads : int
        Number of threads to use
    **kwargs :
        Additional arguments for fitz_doc_to_image

    Returns:
    --------
    images : list
        List of processed images, in page order
    """
    # Open the PDF
    doc = fitz.open(pdf_path)
    num_pages = len(doc)

    # Create a list to store results in the correct order
    results = [None] * num_pages

    # Create a thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {}
        for page_num in range(num_pages):
            page = doc[page_num]
            future = executor.submit(fitz_doc_to_image, page, **kwargs)
            futures[future] = page_num
        # Process results as they complete with progress bar
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                results[page_num] = future.result()
            except Exception as e:
                print(f'Error processing page {page_num}: {e}')
                results[page_num] = None

    # Close the document
    doc.close()

if __name__ == '__main__':
    pdf = fitz.open('/tmp/[MS-DOC].pdf')


    pdf_page = [fitz.open() for i in range(pdf.page_count)]
    [pdf_page[i].insert_pdf(pdf, from_page=i, to_page=i) for i in range(pdf.page_count)]

    pdf_page = [v.tobytes() for v in pdf_page]
    results = parallel_process_pdf_safe(pdf_page, num_workers=16)

    # threaded_process_pdf('/tmp/[MS-DOC].pdf', num_threads=16)

    """ benchmark results of multi-threaded processing (fitz page to image)
    total page nums: 578
    thread nums,    time cost
    1               7.351 sec
    2               6.334 sec
    4               5.968 sec
    8               6.728 sec
    16              8.085 sec
    """

    """ benchmark results of multi-processor processing (fitz page to image)
    total page nums: 578
    processor nums,    time cost
    1                  17.170 sec
    2                  10.170 sec
    4                  7.841 sec
    8                  7.900 sec
    16                 7.984 sec
    """
