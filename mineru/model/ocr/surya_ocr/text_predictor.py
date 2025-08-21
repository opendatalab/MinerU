# Import Surya OCR for Vietnamese recognition
import os
import gc
import threading
from contextlib import contextmanager
import cv2
from loguru import logger
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor


class SuryaTextPredictor:
    def __init__(self):
        # Initialize foundation predictor first as it's required by recognition predictor
        self.surya_foundation_predictor = FoundationPredictor()
        self.surya_rec_predictor = RecognitionPredictor(self.surya_foundation_predictor)
        self.surya_det_predictor = DetectionPredictor()
        self._lock = threading.Lock()
        self._cleanup_needed = False
    def __call__(self, img_crop_list):
        """
        Use Surya OCR for Vietnamese text recognition
        """
        import time
        from PIL import Image
        
        start_time = time.time()
        rec_res = []
        
        # Use thread lock to prevent concurrent access issues
        with self._lock:
            try:
                # Convert OpenCV images to PIL Images for Surya
                pil_images = []
                for img_crop in img_crop_list:
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    pil_images.append(pil_img)
                
                # Run Surya OCR recognition using the new API with proper resource management
                predictions = self._run_ocr_with_cleanup(pil_images)
                
                # Convert Surya results to MinerU format
                for prediction in predictions:
                    if prediction.text_lines:
                        # Combine all text lines for this image
                        combined_text = ' '.join([line.text for line in prediction.text_lines])
                        # Calculate average confidence
                        avg_confidence = sum([line.confidence for line in prediction.text_lines]) / len(prediction.text_lines)
                        rec_res.append((combined_text, avg_confidence))
                    else:
                        # No text detected
                        rec_res.append(('', 0.0))
                        
            except Exception as e:
                logger.error(f"Surya OCR recognition failed: {e}")
                # Ensure cleanup on error
                self._cleanup_resources()
                # Fallback to empty results
                rec_res = [['', 0.0]] * len(img_crop_list)
            finally:
                # Force garbage collection to clean up any lingering resources
                gc.collect()
        
        elapse = time.time() - start_time
        return rec_res, elapse
    
    def _run_ocr_with_cleanup(self, pil_images):
        """
        Run OCR with proper resource cleanup to prevent semaphore leaks.
        """
        try:
            # Use 'ocr_with_boxes' task for Vietnamese text recognition
            predictions = self.surya_rec_predictor(
                pil_images, 
                ['ocr_with_boxes'] * len(pil_images),
                det_predictor=self.surya_det_predictor
            )
            return predictions
        finally:
            # Clean up any temporary resources
            self._cleanup_resources()
    
    def _cleanup_resources(self):
        """
        Clean up any multiprocessing resources that might cause semaphore leaks.
        """
        try:
            # Force garbage collection
            gc.collect()
            self._cleanup_needed = True
        except Exception:
            # Ignore cleanup errors
            pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup_resources()
        return False
    
    def cleanup(self):
        """Explicit cleanup method for manual resource management."""
        self._cleanup_resources()



