from PIL import Image, ImageOps
from transformers.image_processing_utils import BaseImageProcessor
import numpy as np
import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import resize


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
                alb.ToGray(),
                alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_input(item)
        return self.transform(image=image)['image'][:1]

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

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

        # Handle numpy array
        elif isinstance(img, np.ndarray):
            try:
                img = self.crop_margin_numpy(img)
            except Exception:
                # might throw an error for broken files
                return None

            if img.shape[0] == 0 or img.shape[1] == 0:
                return None

            # Get current dimensions
            h, w = img.shape[:2]
            target_h, target_w = self.input_size

            # Calculate scale to preserve aspect ratio (equivalent to resize + thumbnail)
            scale = min(target_h / h, target_w / w)

            # Calculate new dimensions
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize the image while preserving aspect ratio
            resized_img = cv2.resize(img, (new_w, new_h))

            # Calculate padding values using the existing method
            delta_width = target_w - new_w
            delta_height = target_h - new_h

            pad_width, pad_height = self._get_padding_values(new_w, new_h, random_padding)

            # Apply padding (convert PIL padding format to OpenCV format)
            padding_color = [0, 0, 0] if len(img.shape) == 3 else [0]

            padded_img = cv2.copyMakeBorder(
                resized_img,
                pad_height,  # top
                delta_height - pad_height,  # bottom
                pad_width,  # left
                delta_width - pad_width,  # right
                cv2.BORDER_CONSTANT,
                value=padding_color
            )

            return padded_img

        # Handle PIL Image
        elif isinstance(img, Image.Image):
            try:
                img = self.crop_margin(img.convert("RGB"))
            except OSError:
                # might throw an error for broken files
                return None

            if img.height == 0 or img.width == 0:
                return None

            # Resize while preserving aspect ratio
            img = resize(img, min(self.input_size))
            img.thumbnail((self.input_size[1], self.input_size[0]))
            new_w, new_h = img.width, img.height

            # Calculate and apply padding
            padding = self._calculate_padding(new_w, new_h, random_padding)
            return np.array(ImageOps.expand(img, padding))

        else:
            return None

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
