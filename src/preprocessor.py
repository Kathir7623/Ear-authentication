import cv2
import numpy as np

class Preprocessor:
    @staticmethod
    def process(image, target_size=(224, 224)):
        """
        Preprocesses the ear image:
        1. Resize to target size.
        2. Convert to grayscale (optional, modern CNNs use RGB).
        3. Histogram Equalization (CLAHE) for lighting variation.
        4. Normalization.
        """
        # Resize
        img = cv2.resize(image, target_size)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # We apply it to the Luminance channel in LAB color space to preserve color
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Gaussian Blur to remove noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img

    @staticmethod
    def normalize_for_model(image):
        """
        Normalize image for CNN input (usually [0, 1] or mean/std normalization).
        """
        image = image.astype(np.float32) / 255.0
        # Ensure float32 for PyTorch consistency
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        return image.transpose(2, 0, 1) # HWC to CHW
