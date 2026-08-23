# Copyright (c) Opendatalab. All rights reserved.
import cv2
import numpy as np

# Configurable constants for color extraction
EXPAND_MARGIN = 6
MIN_BLOCK_SIZE = 20
WHITE_DISTANCE_THRESHOLD = 25.0
COLOR_VARIANCE_MAX = 800.0
# When comparing the block's background to the surrounding page margin,
# if the distance is less than this, they are effectively the same background.
PAGE_SIMILARITY_THRESHOLD = 15.0

def _rgb_to_hex(color):
    """Convert RGB numpy array or tuple to hex string."""
    return "#{:02X}{:02X}{:02X}".format(int(color[0]), int(color[1]), int(color[2]))

def _euclidean_distance(c1, c2):
    """Calculate Euclidean distance between two colors."""
    return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))

def extract_block_style(np_img, bbox):
    """
    Extracts the background and text color of a layout block.
    
    Args:
        np_img (np.ndarray): The full page image (RGB format).
        bbox (list): [x0, y0, x1, y1]
        
    Returns:
        dict: Style metadata containing background_color, text_color, 
              has_colored_background, and color_confidence.
    """
    default_res = {
        "background_color": "#FFFFFF",
        "text_color": "#000000",
        "has_colored_background": False,
        "color_confidence": 0.0
    }
    
    if np_img is None or not bbox or len(bbox) != 4:
        return default_res

    img_h, img_w = np_img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    
    # Constrain to image bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)
    
    # Check minimum dimensions
    w, h = x1 - x0, y1 - y0
    if w < MIN_BLOCK_SIZE or h < MIN_BLOCK_SIZE:
        return default_res
        
    # Crop the inner block
    inner_crop = np_img[y0:y1, x0:x1]
    
    # Reshape for K-means
    pixels = inner_crop.reshape((-1, 3)).astype(np.float32)
    
    # If the image is entirely one color (or very close), K-means will fail or be meaningless
    if pixels.shape[0] == 0:
        return default_res
        
    # K-means clustering (K=2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    try:
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, flags)
    except Exception:
        return default_res
        
    labels = labels.flatten()
    
    # Determine which cluster is the background based on perimeter pixels
    # Perimeter of the inner crop
    top_labels = labels[:w]
    bottom_labels = labels[-w:] if h > 1 else np.array([])
    left_labels = labels[::w]
    right_labels = labels[w-1::w] if w > 1 else np.array([])
    
    perimeter_labels = np.concatenate([top_labels, bottom_labels, left_labels, right_labels])
    if len(perimeter_labels) == 0:
        return default_res
        
    # Background is the most frequent cluster on the perimeter
    bg_cluster_idx = int(np.bincount(perimeter_labels).argmax())
    fg_cluster_idx = 1 - bg_cluster_idx
    
    bg_color = centers[bg_cluster_idx]
    fg_color = centers[fg_cluster_idx]
    
    # Variance check
    bg_pixels = pixels[labels == bg_cluster_idx]
    fg_pixels = pixels[labels == fg_cluster_idx]
    
    bg_variance = np.var(bg_pixels) if len(bg_pixels) > 0 else 0
    # High variance in background implies it's not a uniform solid color (e.g., photograph)
    if bg_variance > COLOR_VARIANCE_MAX:
        # It might be an image or complex diagram
        return {
            "background_color": _rgb_to_hex(bg_color),
            "text_color": _rgb_to_hex(fg_color),
            "has_colored_background": False,
            "color_confidence": max(0.0, 1.0 - (bg_variance / (COLOR_VARIANCE_MAX * 2)))
        }

    # Distance from white
    dist_from_white = _euclidean_distance(bg_color, [255, 255, 255])
    is_white = dist_from_white < WHITE_DISTANCE_THRESHOLD
    
    # Check expanded margin to determine if this box is actually distinct from the page
    # Expanded bounding box
    ex0, ey0 = max(0, x0 - EXPAND_MARGIN), max(0, y0 - EXPAND_MARGIN)
    ex1, ey1 = min(img_w, x1 + EXPAND_MARGIN), min(img_h, y1 + EXPAND_MARGIN)
    
    # We want to sample the margin pixels (the pixels in the expanded box but outside the inner box)
    # A simple approximation is to take the top and bottom stripes of the expanded box
    margin_pixels = []
    if ey0 < y0:
        margin_pixels.append(np_img[ey0:y0, ex0:ex1].reshape(-1, 3))
    if ey1 > y1:
        margin_pixels.append(np_img[y1:ey1, ex0:ex1].reshape(-1, 3))
    if ex0 < x0:
        margin_pixels.append(np_img[y0:y1, ex0:x0].reshape(-1, 3))
    if ex1 > x1:
        margin_pixels.append(np_img[y0:y1, x1:ex1].reshape(-1, 3))
        
    margin_bg_color = [255, 255, 255] # Default to white page
    if margin_pixels:
        all_margin_pixels = np.concatenate(margin_pixels, axis=0)
        if len(all_margin_pixels) > 0:
            # Simple median of margin to find surrounding page color
            margin_bg_color = np.median(all_margin_pixels, axis=0)
            
    # If the block's background is very similar to the surrounding margin, it's just part of the page
    dist_from_margin = _euclidean_distance(bg_color, margin_bg_color)
    is_distinct_from_page = dist_from_margin > PAGE_SIMILARITY_THRESHOLD
    
    has_colored_background = (not is_white) and is_distinct_from_page
    
    # Compute confidence
    # 1.0 is highest confidence.
    confidence = 1.0
    
    # Penalize if background variance is somewhat high but not rejected
    if bg_variance > 100:
        confidence -= 0.2 * (bg_variance / COLOR_VARIANCE_MAX)
        
    # Penalize if background doesn't dominate perimeter strongly
    bg_perimeter_ratio = np.sum(perimeter_labels == bg_cluster_idx) / len(perimeter_labels)
    if bg_perimeter_ratio < 0.8:
        confidence -= 0.3 * (0.8 - bg_perimeter_ratio)
        
    # Ensure confidence is clamped
    confidence = max(0.1, min(1.0, confidence))
    
    # Edge case: If it's a white page and we correctly identified it's NOT a colored box,
    # then our confidence in the "has_colored_background = False" decision is very high.
    if is_white and not is_distinct_from_page:
        confidence = 0.95
        
    return {
        "background_color": _rgb_to_hex(bg_color),
        "text_color": _rgb_to_hex(fg_color),
        "has_colored_background": bool(has_colored_background),
        "color_confidence": round(float(confidence), 3)
    }
