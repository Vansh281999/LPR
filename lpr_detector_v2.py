import os
import cv2
import numpy as np
import re
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except:
    PADDLE_AVAILABLE = False
    
import pytesseract

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global detector instance
ocr_reader = None
USE_PADDLE = False

def init_detector():
    """Initialize the OCR reader"""
    global ocr_reader, USE_PADDLE
    
    print("Initializing OCR reader...")
    if PADDLE_AVAILABLE:
        try:
            ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
            USE_PADDLE = True
            print("OCR reader (PaddleOCR) initialized successfully!")
        except Exception as e:
            print(f"PaddleOCR init failed: {e}. Falling back to Tesseract.")
            ocr_reader = None
            USE_PADDLE = False
    else:
        print("Using Tesseract OCR")
        USE_PADDLE = False

def preprocess_image(img):
    """Preprocess image for better plate detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Find edges
    edged = cv2.Canny(filtered, 30, 200)
    
    return gray, filtered, edged

def find_license_plate_contour(img):
    """Find license plate using contour detection"""
    gray, filtered, edged = preprocess_image(img)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    plate_contour = None
    plate_img = None
    x, y, w, h = 0, 0, 0, 0
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # License plates are typically rectangular (4 corners)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # License plate aspect ratio is typically between 2:1 and 5:1
            if 2.0 <= aspect_ratio <= 5.5:
                # Check if contour area is reasonable
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    plate_contour = approx
                    plate_img = img[y:y+h, x:x+w]
                    break
    
    return plate_contour, plate_img, (x, y, w, h)

def enhance_plate_image(plate_img):
    """Enhance plate image for better OCR in noisy environments"""
    if plate_img is None or plate_img.size == 0:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to larger size for better OCR (300% for very small plates)
    scale_percent = 300
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Aggressive denoising FIRST
    denoised = cv2.fastNlMeansDenoising(gray, None, h=20, templateWindowSize=7, searchWindowSize=21)
    
    # Apply bilateral filter to smooth while preserving edges
    bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(bilateral, -1, kernel)
    
    # Multiple thresholding methods
    # Method 1: Otsu's thresholding
    _, thresh_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive thresholding
    thresh_adaptive = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Combine both methods (use the one with more white pixels as it's likely better)
    if np.sum(thresh_otsu == 255) > np.sum(thresh_adaptive == 255):
        final = thresh_otsu
    else:
        final = thresh_adaptive
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
    
    return final

def clean_plate_text(text):
    """Clean and format detected plate text"""
    if not text:
        return None
    
    # Remove special characters and spaces
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Replace common OCR mistakes
    replacements = {
        'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'
    }
    
    # Indian plate format patterns (ordered by preference)
    patterns = [
        r'^[A-Z]{2}\d{2}[A-Z]\d{4}$',         # KL47B5099 (2 letters, 2 digits, 1 letter, 4 digits)
        r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',      # MH12AB1234 (2 letters,2 digits,2 letters,4 digits)
        r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$',  # more general fallback
    ]
    
    # Try direct match first
    for pattern in patterns:
        if re.match(pattern, cleaned):
            return cleaned
    
    # Try with replacements
    for old, new in replacements.items():
        test = cleaned.replace(old, new)
        for pattern in patterns:
            if re.match(pattern, test):
                return test
    
    # If length is reasonable, return anyway
    if 7 <= len(cleaned) <= 12:
        return cleaned
    
    return None


def try_autocorrect_plate(candidate):
    """Try simple autocorrections on a candidate string to match known plate patterns.
    This runs limited single- and two-character substitutions over alnum chars and
    returns the first correction that matches the expected patterns.
    """
    if not candidate:
        return None

    # Only consider uppercase alnum
    candidate = re.sub(r'[^A-Z0-9]', '', candidate.upper())
    if not candidate:
        return None

    # Prefer the most common formats first (2 digits after state code, then 1 or 2 letters)
    patterns = [
        re.compile(r'^[A-Z]{2}\d{2}[A-Z]?\d{4}$'),
        re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'),
        re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$'),
    ]

    # Quick check
    for p in patterns:
        if p.match(candidate):
            return candidate

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Try single substitution
    for i in range(len(candidate)):
        for ch in alphabet:
            s = candidate[:i] + ch + candidate[i+1:]
            for p in patterns:
                if p.match(s):
                    return s

    # Try two substitutions (limited) - stop early if too slow
    max_len_for_double = 12
    if len(candidate) <= max_len_for_double:
        for i in range(len(candidate)):
            for j in range(i+1, len(candidate)):
                for ch1 in alphabet:
                    for ch2 in alphabet:
                        s = list(candidate)
                        s[i] = ch1
                        s[j] = ch2
                        s = ''.join(s)
                        for p in patterns:
                            if p.match(s):
                                return s

    return None

def recognize_plate_text(plate_img):
    """Use OCR to recognize text from plate image"""
    global ocr_reader, USE_PADDLE
    
    if plate_img is None or plate_img.size == 0:
        return None, 0.0
    
    try:
        # Enhance plate image
        enhanced = enhance_plate_image(plate_img)
        
        detected_texts = []
        
        # Try PaddleOCR first
        if USE_PADDLE and ocr_reader is not None:
            try:
                # Try original
                result = ocr_reader.ocr(plate_img, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        detected_texts.append((text, confidence))
                
                # Try enhanced
                if enhanced is not None:
                    result = ocr_reader.ocr(enhanced, cls=True)
                    if result and result[0]:
                        for line in result[0]:
                            text = line[1][0]
                            confidence = line[1][1]
                            detected_texts.append((text, confidence))
            except Exception as e:
                print(f"PaddleOCR error: {e}")
        
        # Try Tesseract with optimized configurations
        if not detected_texts or len(detected_texts) < 2:
            try:
                # Multiple Tesseract configurations optimized for license plates
                configs = [
                    # PSM 7: Single line, best for license plates
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    # PSM 8: Single word
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    # PSM 13: Raw line for very clean images
                    r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    # PSM 6: Single uniform block
                    r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ]
                
                # Try enhanced image first (usually better)
                if enhanced is not None:
                    for i, config in enumerate(configs):
                        text = pytesseract.image_to_string(enhanced, config=config)
                        if text.strip():
                            # Higher confidence for enhanced with best config
                            conf = 0.85 if i == 0 else (0.80 if i == 1 else 0.75)
                            detected_texts.append((text.strip(), conf))
                
                # Then try original
                for i, config in enumerate(configs):
                    text = pytesseract.image_to_string(plate_img, config=config)
                    if text.strip():
                        detected_texts.append((text.strip(), 0.70))
                        
            except Exception as e:
                print(f"Tesseract error: {e}")
        
        if not detected_texts:
            return None, 0.0
        
        # Sort by confidence
        detected_texts.sort(key=lambda x: x[1], reverse=True)

        # QUICK PASS: if any raw detected text already matches the strict/common plate pattern,
        # return it immediately (this helps when original OCR produced the correct string but
        # other heuristics/cleaning pick a looser candidate).
        strict_pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z]?\d{4}$')
        for raw_text, conf in detected_texts:
            cleaned_raw = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
            if strict_pattern.match(cleaned_raw):
                return cleaned_raw, conf
        
        # Try to clean and validate each detected text
        for text, conf in detected_texts:
            cleaned = clean_plate_text(text)
            if cleaned and len(cleaned) >= 7:
                return cleaned, conf
        
        # If no valid format found, return best guess if reasonable
        best_text = detected_texts[0][0]
        cleaned = re.sub(r'[^A-Z0-9]', '', best_text.upper())
        # Try returning cleaned best guess
        if 7 <= len(cleaned) <= 12:
            return cleaned, detected_texts[0][1]

        # As a fallback, attempt autocorrection (single/two char substitutions)
        autocorr = try_autocorrect_plate(best_text)
        if autocorr:
            return autocorr, detected_texts[0][1]

        return None, 0.0
        
    except Exception as e:
        print(f"OCR error: {e}")
        return None, 0.0

def detect_and_recognize_plate(img):
    """
    Main function to detect and recognize license plate
    Returns dict with success status, plate_number, confidence, and annotated_image
    """
    try:
        annotated_img = img.copy()
        
        # Method 1: Contour-based detection
        plate_contour, plate_img, (x, y, w, h) = find_license_plate_contour(img)
        
        if plate_img is not None and plate_img.size > 0:
            # Draw rectangle
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (51, 181, 155), 3)
            
            # Recognize text
            plate_number, confidence = recognize_plate_text(plate_img)
            
            if plate_number and len(plate_number) >= 7:
                # Add text to image
                cv2.putText(
                    annotated_img, 
                    plate_number,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (51, 181, 155),
                    2,
                    cv2.LINE_AA
                )
                
                return {
                    'success': True,
                    'plate_number': plate_number,
                    'confidence': confidence,
                    'annotated_image': annotated_img
                }
        
        # Method 2: Try multiple regions of interest
        h, w = img.shape[:2]
        regions = [
            img[int(h*0.5):int(h*0.9), int(w*0.1):int(w*0.9)],  # Lower center
            img[int(h*0.4):int(h*0.8), int(w*0.2):int(w*0.8)],  # Middle center
            img[int(h*0.6):h, int(w*0.2):int(w*0.8)],           # Bottom
        ]
        
        for region in regions:
            if region.size == 0:
                continue
            plate_contour, plate_img, (rx, ry, rw, rh) = find_license_plate_contour(region)
            if plate_img is not None and plate_img.size > 0:
                plate_number, confidence = recognize_plate_text(plate_img)
                if plate_number and len(plate_number) >= 7:
                    cv2.rectangle(annotated_img, (rx, ry), (rx+rw, ry+rh), (51, 181, 155), 3)
                    cv2.putText(
                        annotated_img, plate_number, (rx, ry - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (51, 181, 155), 2, cv2.LINE_AA
                    )
                    return {
                        'success': True,
                        'plate_number': plate_number,
                        'confidence': confidence,
                        'annotated_image': annotated_img
                    }
        
        return {
            'success': False,
            'error': 'No license plate detected in image',
            'annotated_image': annotated_img
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'annotated_image': img
        }

if __name__ == "__main__":
    # Test the detector
    init_detector()
    
    test_image_path = "test_images/sample.jpg"
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        result = detect_and_recognize_plate(img)
        print(f"Detection result: {result}")
