import cv2
import os
from lpr_detector_v2 import init_detector, detect_and_recognize_plate

# Initialize
print("Initializing detector...")
init_detector()

# Get latest upload
uploads_dir = "uploads"
if os.path.exists(uploads_dir):
    files = [f for f in os.listdir(uploads_dir) if f.endswith(('.jpg', '.png'))]
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)), reverse=True)
        latest = os.path.join(uploads_dir, files[0])
        
        print(f"\nTesting with: {latest}")
        print("=" * 60)
        
        # Read image
        img = cv2.imread(latest)
        if img is None:
            print("ERROR: Could not read image")
        else:
            print(f"Image size: {img.shape}")
            
            # Test detection
            result = detect_and_recognize_plate(img)
            
            print("\nResult:")
            print(f"  Success: {result['success']}")
            if result['success']:
                print(f"  Plate Number: {result['plate_number']}")
                print(f"  Confidence: {result['confidence']:.2f}")
            else:
                print(f"  Error: {result.get('error', 'Unknown')}")
            
            # Save annotated image
            output_path = "test_output.jpg"
            cv2.imwrite(output_path, result['annotated_image'])
            print(f"\nAnnotated image saved to: {output_path}")
    else:
        print("No images found in uploads/")
else:
    print("uploads/ directory not found")

# Also test with a simple rectangular detection
print("\n" + "=" * 60)
print("Testing contour detection details...")
print("=" * 60)

if os.path.exists(uploads_dir) and files:
    img = cv2.imread(os.path.join(uploads_dir, files[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    print(f"\nFound {len(contours)} contours")
    print("\nTop 10 contours by area:")
    
    plate_candidates = 0
    for i, contour in enumerate(contours[:10]):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        is_rect = len(approx) == 4
        is_plate_ratio = 2.0 <= aspect_ratio <= 5.5
        is_big_enough = area > 1000
        
        status = "✓ PLATE?" if (is_rect and is_plate_ratio and is_big_enough) else "✗"
        
        print(f"\n  {i+1}. {status}")
        print(f"     Area: {area:.0f}, Corners: {len(approx)}, Ratio: {aspect_ratio:.2f}")
        print(f"     Rect: {is_rect}, Good ratio: {is_plate_ratio}, Big enough: {is_big_enough}")
        
        if is_rect and is_plate_ratio and is_big_enough:
            plate_candidates += 1
    
    print(f"\n✓ Found {plate_candidates} potential license plate(s)")
    
    # Save debug images
    cv2.imwrite("debug_edged.jpg", edged)
    cv2.imwrite("debug_filtered.jpg", filtered)
    print("\nDebug images saved: debug_edged.jpg, debug_filtered.jpg")
