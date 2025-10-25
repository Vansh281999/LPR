import cv2
import numpy as np
from skimage import exposure, filters
import logging

class ImageEnhancer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def preprocess_image(self, image):
        """Advanced image preprocessing"""
        # Multiple enhancement techniques
        enhanced = self.enhance_contrast(image)
        enhanced = self.reduce_noise(enhanced)
        enhanced = self.sharpen_image(enhanced)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
        
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image) if len(image.shape) == 2 else clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
    def reduce_noise(self, image):
        """Reduce noise while preserving edges"""
        return cv2.bilateralFilter(image, 9, 75, 75)
        
    def sharpen_image(self, image):
        """Sharpen image using kernel convolution"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
        
    def enhance_for_fog(self, image):
        """Enhance image for foggy/misty conditions"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Dehazing using dark channel prior (simplified)
        enhanced = self.simplified_dehaze(enhanced)
        
        return enhanced
        
    def simplified_dehaze(self, image):
        """Simplified dehazing algorithm"""
        # Estimate atmospheric light
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        atmospheric_light = np.percentile(gray, 95)
        
        # Calculate transmission map
        transmission = 1 - 0.95 * (gray / atmospheric_light)
        transmission = np.clip(transmission, 0.1, 0.9)
        
        # Recover scene radiance
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            result[:,:,i] = (image[:,:,i].astype(np.float32) - atmospheric_light) / transmission + atmospheric_light
            
        return np.clip(result, 0, 255).astype(np.uint8)
        
    def preprocess_for_ocr(self, plate_image):
        """Preprocess license plate specifically for OCR"""
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
            
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50), interpolation=cv2.INTER_CUBIC)
            
        # Apply morphological operations to clean the text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary