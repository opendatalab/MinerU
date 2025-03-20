from transformers.image_processing_utils import BaseImageProcessor
import numpy as np
import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2


# TODO: dereference cv2 if possible
class UnimerSwinImageProcessor(BaseImageProcessor):
    def __init__(
            self,
            image_size = (192, 672),
        ):
        self.input_size = [int(_) for _ in image_size]
        assert len(self.input_size) == 2
    
        self.transform = alb.Compose(
            [
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_input(item)
        return self.transform(image=image)['image'][:1]

    @staticmethod
    def crop_margin_numpy(img: np.ndarray) -> np.ndarray:
        """Crop margins of image using NumPy operations"""
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # Normalize and threshold
        if gray.max() == gray.min():
            return img

        normalized = (((gray - gray.min()) / (gray.max() - gray.min())) * 255).astype(np.uint8)
        binary = 255 * (normalized < 200).astype(np.uint8)

        # Find bounding box
        coords = cv2.findNonZero(binary)  # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box

        # Return cropped image
        return img[y:y + h, x:x + w]

    def prepare_input(self, img, random_padding: bool = False):
        """
        Convert PIL Image or numpy array to properly sized and padded image after:
            - crop margins
            - resize while maintaining aspect ratio
            - pad to target size
        """
        if img is None:
            return None

        try:
            img = self.crop_margin_numpy(img)
        except Exception:
            # might throw an error for broken files
            return None

        if img.shape[0] == 0 or img.shape[1] == 0:
            return None

        # Resize while preserving aspect ratio
        h, w = img.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Calculate padding
        pad_width, pad_height = self._get_padding_values(new_w, new_h, random_padding)

        # Create and apply padding
        channels = 3 if len(img.shape) == 3 else 1
        padded_img = np.full((self.input_size[0], self.input_size[1], channels), 255, dtype=np.uint8)
        padded_img[pad_height:pad_height + new_h, pad_width:pad_width + new_w] = resized_img

        return padded_img

    def _calculate_padding(self, new_w, new_h, random_padding):
        """Calculate padding values for PIL images"""
        delta_width = self.input_size[1] - new_w
        delta_height = self.input_size[0] - new_h

        pad_width, pad_height = self._get_padding_values(new_w, new_h, random_padding)

        return (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

    def _get_padding_values(self, new_w, new_h, random_padding):
        """Get padding values based on image dimensions and padding strategy"""
        delta_width = self.input_size[1] - new_w
        delta_height = self.input_size[0] - new_h

        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2

        return pad_width, pad_height
