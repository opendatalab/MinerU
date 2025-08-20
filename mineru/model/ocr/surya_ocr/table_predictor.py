
def _surya_table_recognition(self, img, mfd_res=None):
    """
    Use Surya OCR for Vietnamese table recognition
    Returns results in MinerU format (filter_boxes, filter_rec_res)
    """
    import time
    from PIL import Image
    
    try:
        ori_im = img.copy()
        
        # Use Surya detection for text detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Get text detection results from Surya
        det_predictions = self.surya_det_predictor([pil_img])
        
        if not det_predictions or not det_predictions[0].bboxes:
            logger.debug("No text detected by Surya")
            return None, None
        
        # Convert Surya detection results to MinerU format
        dt_boxes = []
        for bbox in det_predictions[0].bboxes:
            # Convert bbox format: [x1, y1, x2, y2] to polygon format
            x1, y1, x2, y2 = bbox.bbox
            # Create polygon points in the format expected by MinerU
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            dt_boxes.append(poly)
        
        if not dt_boxes:
            logger.debug("No valid detection boxes found")
            return None, None
        
        # Sort and process detection boxes
        dt_boxes = sorted_boxes(dt_boxes)
        dt_boxes = merge_det_boxes(dt_boxes)
        
        if mfd_res:
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)
        
        # Crop images for recognition
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        
        # Use Surya recognition for text recognition
        rec_res, elapse = self._surya_text_recognizer(img_crop_list)
        
        # Filter results based on confidence score
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        
        logger.debug(f"Surya table recognition completed: {len(filter_boxes)} boxes, elapsed: {elapse}")
        return filter_boxes, filter_rec_res
        
    except Exception as e:
        logger.error(f"Surya table recognition failed: {e}")
        # Fallback to original detection and recognition
        logger.info("Falling back to original OCR method")
        
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        
        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            return None, None
        
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        dt_boxes = merge_det_boxes(dt_boxes)
        
        if mfd_res:
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)
        
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        
        rec_res, elapse = self.text_recognizer(img_crop_list)
        
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        
        return filter_boxes, filter_rec_res