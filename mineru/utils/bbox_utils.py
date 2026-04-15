# Copyright (c) Opendatalab. All rights reserved.
import math

import numpy as np


def normalize_to_int_bbox(box, image_size: tuple[int, int] | None = None) -> list[int] | None:
    if box is None:
        return None

    arr = np.asarray(box, dtype=np.float64)
    if arr.size == 0:
        return None

    if arr.ndim == 2 and arr.shape[-1] == 2:
        xs = arr[:, 0]
        ys = arr[:, 1]
        xmin = float(np.min(xs))
        ymin = float(np.min(ys))
        xmax = float(np.max(xs))
        ymax = float(np.max(ys))
    else:
        flat = arr.reshape(-1)
        if flat.size == 4:
            xmin, ymin, xmax, ymax = [float(v) for v in flat]
        elif flat.size >= 8:
            xs = flat[0::2]
            ys = flat[1::2]
            xmin = float(np.min(xs))
            ymin = float(np.min(ys))
            xmax = float(np.max(xs))
            ymax = float(np.max(ys))
        else:
            return None

    xmin = math.floor(xmin)
    ymin = math.floor(ymin)
    xmax = math.ceil(xmax)
    ymax = math.ceil(ymax)

    if image_size is not None:
        height, width = image_size
        xmin = max(0, min(int(width), xmin))
        ymin = max(0, min(int(height), ymin))
        xmax = max(0, min(int(width), xmax))
        ymax = max(0, min(int(height), ymax))

    if xmax <= xmin or ymax <= ymin:
        return None

    return [int(xmin), int(ymin), int(xmax), int(ymax)]
