
import fitz
import numpy as np
from loguru import logger

from magic_pdf.utils.annotations import ImportPIL


@ImportPIL
def fitz_doc_to_image(doc, dpi=200) -> dict:
    """Convert fitz.Document to image, Then convert the image to numpy array.

    Args:
        doc (_type_): pymudoc page
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img': numpy array, 'width': width, 'height': height }
    """
    from PIL import Image
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pm = doc.get_pixmap(matrix=mat, alpha=False)

    # If the width or height exceeds 4500 after scaling, do not scale further.
    if pm.width > 4500 or pm.height > 4500:
        pm = doc.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    img = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    img = np.array(img)

    img_dict = {'img': img, 'width': pm.width, 'height': pm.height}

    return img_dict

@ImportPIL
def load_images_from_pdf(pdf_bytes: bytes, dpi=200, start_page_id=0, end_page_id=None) -> list:
    from PIL import Image
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

                img = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
                img = np.array(img)
                img_dict = {'img': img, 'width': pm.width, 'height': pm.height}
            else:
                img_dict = {'img': [], 'width': 0, 'height': 0}

            images.append(img_dict)
    return images
