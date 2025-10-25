import cv2
import numpy as np
import pytesseract
import sqlite3
import time
import logging
from datetime import datetime
from image_enhancement import ImageEnhancer

class LicensePlateRecognition:
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.setup_tesseract()
        self.setup_logging()
        
    def setup_tesseract(self):
        """Configure Tesseract OCR path"""
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        
    def setup_logging(self):
        """Setup logging for the system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('lpr_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def detect_license_plate(self, image_path, enhance_for_weather=False):
        """
        Enhanced license plate detection with weather adaptation
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None, None, None
                
            # Apply weather enhancement if needed
            if enhance_for_weather:
                image = self.enhancer.enhance_for_fog(image)
                image = self.enhancer.enhance_contrast(image)
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply advanced preprocessing
            processed = self.enhancer.preprocess_image(gray)
            
            # Multiple detection strategies
            plates = []
            
            # Strategy 1: Contour-based detection
            plates.extend(self._detect_by_contours(processed, image))
            
            # Strategy 2: Edge-based detection
            plates.extend(self._detect_by_edges(processed, image))
            
            # Strategy 3: Morphology-based detection
            plates.extend(self._detect_by_morphology(processed, image))
            
            # Strategy 4: Color/brightness-based rectangular plate (white plate fallback)
            if not plates:
                plates.extend(self._detect_by_white_plate(image))
            
            if not plates:
                self.logger.warning("No license plates detected")
                return image, None, None
                
            # Select the best candidate
            best_plate = self._select_best_plate(plates)
            
            return image, best_plate['plate'], best_plate['contour']
            
        except Exception as e:
            self.logger.error(f"Error in license plate detection: {str(e)}")
            return None, None, None
            
    def _detect_by_contours(self, processed, original):
        """Detect plates using contour analysis"""
        plates = []
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Flexible aspect ratio for different plate types
                if 2.0 <= aspect_ratio <= 6.0 and 1000 <= (w * h) <= 50000:
                    plate_region = original[y:y+h, x:x+w]
                    plates.append({
                        'plate': plate_region,
                        'contour': approx,
                        'confidence': self._calculate_confidence(plate_region),
                        'method': 'contour'
                    })
                    
        return plates
        
    def _detect_by_edges(self, processed, original):
        """Detect plates using edge detection"""
        plates = []
        
        # Enhanced edge detection
        edges = cv2.Canny(processed, 50, 150)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 <= area <= 50000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                if 2.0 <= aspect_ratio <= 6.0:
                    plate_region = original[y:y+h, x:x+w]
                    plates.append({
                        'plate': plate_region,
                        'contour': contour,
                        'confidence': self._calculate_confidence(plate_region),
                        'method': 'edge'
                    })
                    
        return plates
        
    def _detect_by_morphology(self, processed, original):
        """Detect plates using morphological operations"""
        plates = []
        
        # Create rectangular kernel for license plate shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        morph = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 <= area <= 40000:
                x, y, w, h = cv2.boundingRect(contour)
                plate_region = original[y:y+h, x:x+w]
                plates.append({
                    'plate': plate_region,
                    'contour': contour,
                    'confidence': self._calculate_confidence(plate_region),
                    'method': 'morphology'
                })
                
        return plates
        
    def _detect_by_white_plate(self, original):
        """Fallback detection targeting white reflective plates like UK/EU style"""
        plates = []
        try:
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            # White plate: low saturation, high value
            lower_white = np.array([0, 0, 180], dtype=np.uint8)
            upper_white = np.array([179, 60, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_white, upper_white)
            # Strengthen mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 2500 or area > 120000:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h)
                if aspect < 2.0 or aspect > 6.5:
                    continue
                # Check whiteness ratio inside box
                roi_mask = mask[y:y+h, x:x+w]
                white_ratio = np.mean(roi_mask) / 255.0
                if white_ratio < 0.35:
                    continue
                # Add a small margin for OCR
                mx = max(0, x - w // 20)
                my = max(0, y - h // 10)
                mw = min(original.shape[1], x + w + w // 20)
                mh = min(original.shape[0], y + h + h // 10)
                plate_region = original[my:mh, mx:mw]
                plates.append({
                    'plate': plate_region,
                    'contour': np.array([[mx,my],[mw,my],[mw,mh],[mx,mh]]),
                    'confidence': self._calculate_confidence(plate_region) + white_ratio * 50,
                    'method': 'white_fallback'
                })
        except Exception as e:
            self.logger.error(f"White-plate fallback error: {e}")
        return plates
        
    def _calculate_confidence(self, plate_region):
        """Calculate confidence score for plate detection"""
        if plate_region is None or plate_region.size == 0:
            return 0
            
        # Convert to grayscale if needed
        if len(plate_region.shape) == 3:
            gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_plate = plate_region
            
        # Calculate contrast score
        contrast = np.std(gray_plate)
        
        # Calculate edge density
        edges = cv2.Canny(gray_plate, 50, 150)
        edge_density = np.sum(edges) / (gray_plate.shape[0] * gray_plate.shape[1])
        
        # Combined confidence score
        confidence = (contrast / 50 + edge_density * 100) / 2
        return min(confidence, 100)
        
    def _select_best_plate(self, plates):
        """Select the best plate candidate based on confidence"""
        return max(plates, key=lambda x: x['confidence'])
        
    def extract_text_from_plate(self, license_plate_image):
        """
        Extract text from license plate with multiple OCR strategies
        """
        if license_plate_image is None:
            return "", 0
            
        try:
            # Preprocess plate for OCR
            processed_plate = self.enhancer.preprocess_for_ocr(license_plate_image)
            
            # Try multiple OCR configurations
            texts = []
            confidences = []
            
            # Strategy 1: Single line
            text1 = pytesseract.image_to_string(processed_plate, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            texts.append(text1.strip())
            
            # Strategy 2: Single word
            text2 = pytesseract.image_to_string(processed_plate, config='--psm 8')
            texts.append(text2.strip())
            
            # Strategy 3: Multiple lines
            text3 = pytesseract.image_to_string(processed_plate, config='--psm 7')
            texts.append(text3.strip())
            
            # Get confidence for the best result
            data = pytesseract.image_to_data(processed_plate, output_type=pytesseract.Output.DICT, config='--psm 8')
            if data['conf']:
                confidence = np.mean([conf for conf in data['conf'] if conf > 0])
            else:
                confidence = 0
                
            # Clean and select best text
            cleaned_texts = [self._clean_plate_text(text) for text in texts if text]
            best_text = max(cleaned_texts, key=len) if cleaned_texts else ""
            
            return best_text, confidence
            
        except Exception as e:
            self.logger.error(f"OCR Error: {str(e)}")
            return "", 0
            
    def _clean_plate_text(self, text):
        """Clean and validate license plate text"""
        import re
        
        # Remove unwanted characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # If pattern like AA00BBB, format as 'AA00 BBB'
        m = re.match(r'^([A-Z]{2})(\d{2})([A-Z]{3})$', cleaned)
        if m:
            return f"{m.group(1)}{m.group(2)} {m.group(3)}"
        
        # Basic validation
        if 4 <= len(cleaned) <= 10:  # Reasonable plate length
            return cleaned
        return ""
        
    def calculate_accuracy(self, detected_text, expected_text):
        """Calculate accuracy using Levenshtein distance"""
        if not detected_text or not expected_text:
            return 0
            
        # Simple character-based accuracy
        expected_upper = expected_text.upper().replace(' ', '')
        detected_upper = detected_text.upper().replace(' ', '')
        
        correct_chars = sum(1 for a, b in zip(detected_upper, expected_upper) if a == b)
        max_len = max(len(detected_upper), len(expected_upper))
        
        return (correct_chars / max_len) * 100 if max_len > 0 else 0