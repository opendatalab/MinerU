import concurrent.futures

import fitz

from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.data.utils import fitz_doc_to_image  # PyMuPDF


def partition_array_greedy(arr, k):
    """Partition an array into k parts using a simple greedy approach.

    Parameters:
    -----------
    arr : list
        The input array of integers
    k : int
        Number of partitions to create

    Returns:
    --------
    partitions : list of lists
        The k partitions of the array
    """
    # Handle edge cases
    if k <= 0:
        raise ValueError('k must be a positive integer')
    if k > len(arr):
        k = len(arr)  # Adjust k if it's too large
    if k == 1:
        return [list(range(len(arr)))]
    if k == len(arr):
        return [[i] for i in range(len(arr))]

    # Sort the array in descending order
    sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i][1], reverse=True)

    # Initialize k empty partitions
    partitions = [[] for _ in range(k)]
    partition_sums = [0] * k

    # Assign each element to the partition with the smallest current sum
    for idx in sorted_indices:
        # Find the partition with the smallest sum
        min_sum_idx = partition_sums.index(min(partition_sums))

        # Add the element to this partition
        partitions[min_sum_idx].append(idx)  # Store the original index
        partition_sums[min_sum_idx] += arr[idx][1]

    return partitions


def process_pdf_batch(pdf_jobs, idx):
    """Process a batch of PDF pages using multiple threads.

    Parameters:
    -----------
    pdf_jobs : list of tuples
        List of (pdf_path, page_num) tuples
    output_dir : str or None
        Directory to save images to
    num_threads : int
        Number of threads to use
    **kwargs :
        Additional arguments for process_pdf_page

    Returns:
    --------
    images : list
        List of processed images
    """
    images = []

    for pdf_path, _ in pdf_jobs:
        doc = fitz.open(pdf_path)
        tmp = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            tmp.append(fitz_doc_to_image(page))
        images.append(tmp)
    return (idx, images)


def batch_build_dataset(pdf_paths, k, lang=None):
    """Process multiple PDFs by partitioning them into k balanced parts and
    processing each part in parallel.

    Parameters:
    -----------
    pdf_paths : list
        List of paths to PDF files
    k : int
        Number of partitions to create
    output_dir : str or None
        Directory to save images to
    threads_per_worker : int
        Number of threads to use per worker
    **kwargs :
        Additional arguments for process_pdf_page

    Returns:
    --------
    all_images : list
        List of all processed images
    """

    results = []
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        dataset = PymuDocDataset(pdf_bytes, lang=lang)
        results.append(dataset)
    return results


    #
    # # Get page counts for each PDF
    # pdf_info = []
    # total_pages = 0
    #
    # for pdf_path in pdf_paths:
    #     try:
    #         doc = fitz.open(pdf_path)
    #         num_pages = len(doc)
    #         pdf_info.append((pdf_path, num_pages))
    #         total_pages += num_pages
    #         doc.close()
    #     except Exception as e:
    #         print(f'Error opening {pdf_path}: {e}')
    #
    # # Partition the jobs based on page countEach job has 1 page
    # partitions = partition_array_greedy(pdf_info, k)
    #
    # # Process each partition in parallel
    # all_images_h = {}
    #
    # with concurrent.futures.ProcessPoolExecutor(max_workers=k) as executor:
    #     # Submit one task per partition
    #     futures = []
    #     for sn, partition in enumerate(partitions):
    #         # Get the jobs for this partition
    #         partition_jobs = [pdf_info[idx] for idx in partition]
    #
    #         # Submit the task
    #         future = executor.submit(
    #             process_pdf_batch,
    #             partition_jobs,
    #             sn
    #         )
    #         futures.append(future)
    #     # Process results as they complete
    #     for i, future in enumerate(concurrent.futures.as_completed(futures)):
    #         try:
    #             idx, images = future.result()
    #             all_images_h[idx] = images
    #         except Exception as e:
    #             print(f'Error processing partition: {e}')
    # results = [None] * len(pdf_paths)
    # for i in range(len(partitions)):
    #     partition = partitions[i]
    #     for j in range(len(partition)):
    #         with open(pdf_info[partition[j]][0], 'rb') as f:
    #             pdf_bytes = f.read()
    #         dataset = PymuDocDataset(pdf_bytes, lang=lang)
    #         dataset.set_images(all_images_h[i][j])
    #         results[partition[j]] = dataset
    # return results