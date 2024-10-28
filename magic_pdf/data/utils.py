
import fitz
import numpy as np

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

    # If the width or height exceeds 9000 after scaling, do not scale further.
    if pm.width > 9000 or pm.height > 9000:
        pm = doc.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    img = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    img = np.array(img)

    img_dict = {'img': img, 'width': pm.width, 'height': pm.height}

    return img_dict
