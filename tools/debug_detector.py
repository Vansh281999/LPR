import os
import sys
import pathlib
import cv2
import numpy as np
import argparse
from datetime import datetime

# Ensure project root is on sys.path so relative imports work when script is run directly
project_root = str(pathlib.Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lpr_detector_v2 import init_detector, find_license_plate_contour, enhance_plate_image, recognize_plate_text, USE_PADDLE, PADDLE_AVAILABLE
import pytesseract

UPLOADS = 'uploads'

def pick_most_recent_image(folder):
    imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    if not imgs:
        return None
    imgs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return imgs[0]


def print_header(title):
    print('\n' + '='*8 + ' ' + title + ' ' + '='*8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', help='Path to image to debug (optional)')
    args = parser.parse_args()

    print_header('ENV CHECK')
    print('Paddle available (build-time):', PADDLE_AVAILABLE)
    print('Using Paddle at runtime:', USE_PADDLE)
    print('pytesseract.tesseract_cmd =', pytesseract.pytesseract.tesseract_cmd)
    print('Tesseract binary exists:', os.path.exists(pytesseract.pytesseract.tesseract_cmd))

    # pick image
    img_path = args.image
    if not img_path:
        img_path = pick_most_recent_image(UPLOADS)

    if not img_path or not os.path.exists(img_path):
        print('No image provided and no images found in uploads/. Please pass --image <path>')
        raise SystemExit(1)

    print_header('IMAGE SELECTED')
    print('Image path:', img_path)

    img = cv2.imread(img_path)
    if img is None:
        print('Failed to read image (cv2.imread returned None)')
        raise SystemExit(1)

    print('Image shape:', img.shape)

    # initialize detector
    print_header('INIT DETECTOR')
    init_detector()

    # find contour
    print_header('FIND CONTOUR')
    plate_contour, plate_img, bbox = find_license_plate_contour(img)
    print('Contour found:', plate_contour is not None)
    print('BBox:', bbox)

    timestamp = int(datetime.utcnow().timestamp())
    base_out = os.path.join(UPLOADS, f'debug_{timestamp}')

    annotated = img.copy()

    if plate_contour is not None and plate_img is not None and plate_img.size>0:
        x,y,w,h = bbox
        # draw rectangle on annotated for visual debug
        cv2.rectangle(annotated, (x,y), (x+w, y+h), (0,255,0), 3)
        crop_path = base_out + '_plate_crop.jpg'
        cv2.imwrite(crop_path, plate_img)
        print('Saved plate crop to', crop_path)

        # try enhancement
        enhanced = enhance_plate_image(plate_img)
        if enhanced is not None:
            enh_path = base_out + '_plate_enhanced.jpg'
            cv2.imwrite(enh_path, enhanced)
            print('Saved enhanced plate to', enh_path)
        else:
            print('Enhancement returned None')

        # run recognition
        print_header('OCR RESULTS (recognize_plate_text)')
        text, conf = recognize_plate_text(plate_img)
        print('recognize_plate_text ->', text, conf)

        # raw pytesseract outputs on enhanced + original
        print_header('RAW TESSERACT OUTPUTS')
        try:
            configs = [
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            for cfg in configs:
                if enhanced is not None:
                    raw = pytesseract.image_to_string(enhanced, config=cfg)
                    print('enhanced config', cfg, '->', repr(raw))
                raw2 = pytesseract.image_to_string(plate_img, config=cfg)
                print('original config', cfg, '->', repr(raw2))
        except Exception as e:
            print('pytesseract error:', e)

        # put text on annotated
        if text:
            cv2.putText(annotated, text, (x, max(10,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    else:
        print('No plate contour detected. Will try ROI-based fallback scans.')
        h,w = img.shape[:2]
        regions = [
            img[int(h*0.5):int(h*0.9), int(w*0.1):int(w*0.9)],
            img[int(h*0.4):int(h*0.8), int(w*0.2):int(w*0.8)],
            img[int(h*0.6):h, int(w*0.2):int(w*0.8)],
        ]
        found = False
        for idx, region in enumerate(regions):
            if region.size==0:
                continue
            pcont, pcrop, pbbox = find_license_plate_contour(region)
            print(f'Region {idx} contour found:', pcont is not None)
            if pcrop is not None and pcrop.size>0:
                found = True
                rpath = base_out + f'_region{idx}_crop.jpg'
                cv2.imwrite(rpath, pcrop)
                print('Saved region crop to', rpath)
                # enhancement
                enhanced = enhance_plate_image(pcrop)
                if enhanced is not None:
                    epath = base_out + f'_region{idx}_enhanced.jpg'
                    cv2.imwrite(epath, enhanced)
                    print('Saved region enhanced to', epath)
                t, c = recognize_plate_text(pcrop)
                print('Region OCR ->', t, c)
        if not found:
            print('No plate found in any ROI regions.')

    annotated_path = base_out + '_annotated.jpg'
    cv2.imwrite(annotated_path, annotated)
    print('\nSaved annotated image to', annotated_path)

    print_header('SUGGESTIONS')
    print('- If contour detection failed: check camera framing, plate size in pixels, plate rotation, heavy blur or extreme glare. Reduce Canny thresholds or lower minimum area threshold in find_license_plate_contour.')
    print('- If OCR failed despite crop/ enhancement: verify Tesseract path, try increasing image contrast or using PaddleOCR (install paddleocr).')
    print('- If results are inconsistent: collect several failing crops and share them for tuning whitelist/psm and training a plate-specific model.')
