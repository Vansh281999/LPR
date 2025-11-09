# Smart LPR — Project Overview and Important Code

Last updated: 2025-11-09

This document explains the entire project: what it contains, the technologies used and why, the runtime/workflow, and the most important code snippets to understand/maintain the system.

## 1 — High level summary

- Purpose: A small smart license-plate recognition (LPR) system that detects plates from camera frames or uploaded images, recognizes the plate text (OCR), logs detections in a local SQLite registry, and notifies registered owners by email/WhatsApp.
- Location: repository root contains backend (Flask) and frontend assets.

Directory layout (important files/folders):

- `app.py` — Flask backend, API endpoints, DB initialization and notification wiring.
- `lpr_detector_v2.py` — Plate localization + enhancement + OCR pipeline (OpenCV + Tesseract/PaddleOCR fallbacks).
- `tools/` — small utilities and diagnostics (e.g., `audit_fuzzy_registry.py`).
- `uploads/` — saved camera frames, uploaded images, and debug artifacts.
- `reports/` — generated audit reports (CSV).
- `society_vehicles.db` — SQLite database (owners registry, detections, notifications).
- `frontend/` — static frontend files (index.html, app.js, styles.css).

## 2 — Technologies used and their roles

- Python (3.13 in this project environment) — language for backend logic and tools.
- Flask — lightweight web framework exposing REST endpoints used by front-end and camera feed.
- OpenCV (`cv2`) — image processing: grayscale conversion, filtering, Canny edges, contour detection, resizing and morphological ops used to find and clean plate regions.
- Tesseract (`pytesseract`) — primary OCR engine used for character recognition. Windows path is set to `C:\Program Files\Tesseract-OCR\tesseract.exe` in the code.
- PaddleOCR (optional) — an alternative OCR engine; the project tries to initialize it but falls back to Tesseract if initialization fails in the runtime environment.
- SQLite — stores the registry of allowed plates (`owners`), detection logs (`detections`), and notification history (`notifications`). The DB file is `society_vehicles.db` in the project root.
- JavaScript / Frontend — small UI to interact with the API (in `frontend/`).
- (Optional) Email and WhatsApp integrations — functions in `lpr.py` (already integrated in `app.py`) to notify owners when their vehicle is detected.

## 3 — Dataflow / runtime workflow

Typical request (camera feed or upload):

1. Image arrives at the backend via `/api/detect` (file upload) or `/api/camera_detect` (base64 payload).
2. Backend saves the input frame to `uploads/` and calls `detect_and_recognize_plate(img)` from `lpr_detector_v2.py`.
3. `detect_and_recognize_plate` does contour-based localization to find a plate region, enhances the plate crop, and tries OCR using PaddleOCR (if available) or Tesseract with several configs.
4. Detected strings are cleaned and validated via regex heuristics. If the recognized string doesn't match known plate patterns, autocorrect heuristics attempt single- or two-character fixes.
5. After a plate candidate is produced, `app.py` checks the `owners` table for a registry match. If no exact match is found, it runs a Levenshtein-based fuzzy match (`fuzzy_match_plate`) and, if within threshold, auto-corrects to the matched registry plate.
6. Detection is logged to `detections` table. If the vehicle is registered, notification(s) are sent and recorded in `notifications` table.

Important behavior notes:
- The system favors returning strict-format matches quickly (a 'quick pass') to avoid over-processing correct OCR outputs.
- Enhancement sometimes helps or hurts OCR; both original and enhanced crops are tried.
- Registry fuzzy-matching is a pragmatic safety net for real-world OCR errors (small edit-distance mistakes).

## 4 — Database schema (as created by `app.py`)

- `owners` table:
  - `id` INTEGER PRIMARY KEY AUTOINCREMENT
  - `plate_number` TEXT UNIQUE
  - `owner_name` TEXT
  - `email` TEXT
  - `phone` / `whatsapp` TEXT
  - `registered_at` TEXT

- `detections` table:
  - `id` INTEGER PRIMARY KEY AUTOINCREMENT
  - `plate_number` TEXT NOT NULL
  - `image_path` TEXT
  - `detected_at` TEXT NOT NULL

- `notifications` table:
  - `id` INTEGER PRIMARY KEY AUTOINCREMENT
  - `detection_id` INTEGER (FK -> detections.id)
  - `channel` TEXT
  - `status` TEXT
  - `details` TEXT
  - `sent_at` TEXT NOT NULL

## 5 — Key configuration & running the app

1. Create or activate the virtual environment and install dependencies from `backend/requirements.txt` if present. This project runs under a `.venv` created in the repo in earlier runs.

2. Ensure system Tesseract is installed and reachable at `C:\Program Files\Tesseract-OCR\tesseract.exe` (Windows) or adjust `pytesseract.pytesseract.tesseract_cmd` in `lpr_detector_v2.py`.

3. Start the Flask app (example using the venv python):

```powershell
& 'C:\Users\vansh\Documents\lpr\.venv\Scripts\python.exe' app.py
```

4. Endpoints to know:
- `GET /api/registry` — list registered vehicles.
- `POST /api/registry` — add a vehicle.
- `POST /api/detect` — multipart file upload to detect a plate.
- `POST /api/camera_detect` — JSON `{ "image": "data:image/jpeg;base64,..." }`.
- `GET /api/detections` — recent detections.

## 6 — Troubleshooting & tips

- If OCR fails on many images:
  - Inspect intermediate crops in `uploads/` (the debug tool writes annotated crops and enhancements).
  - Try disabling enhancement or tuning the enhancement parameters (resize scale, denoising strength).
  - Consider enabling or fixing PaddleOCR initialization if GPU/CPU compatibility is resolved (Paddle is optional and can be more accurate in some cases).

- If fuzzy-matching is too aggressive:
  - Tune the `max_distance` parameter in `fuzzy_match_plate` (in `app.py`) or add a second-best-gap threshold.
  - Persist the original OCR candidate in the DB for auditability before auto-correction.

## 7 — Important code snippets

Below are the most important functions and code excerpts you will likely need to read or modify. They are copied verbatim from the codebase so you can refer to them quickly.

### 7.1 Levenshtein + fuzzy match (from `app.py`)

```python
def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings (pure Python)."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # use two-row DP for memory efficiency
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


def fuzzy_match_plate(candidate: str, max_distance: int = 3):
    """Try to match a recognized plate to a registered plate using fuzzy matching.
    Returns (matched_plate, distance) or (None, None).
    """
    if not candidate:
        return None, None
    cand = re.sub(r'[^A-Z0-9]', '', candidate.upper())
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('SELECT plate_number FROM owners')
        rows = [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    best = (None, 999)
    for reg in rows:
        reg_clean = re.sub(r'[^A-Z0-9]', '', str(reg).upper())
        d = levenshtein(cand, reg_clean)
        if d < best[1]:
            best = (reg_clean, d)

    if best[0] is not None and best[1] <= max_distance:
        return best[0], best[1]
    return None, None
```

### 7.2 Plate detection & recognition flow (from `lpr_detector_v2.py`)

This is the main high-level function used by the Flask app. It performs contour-based localization, OCR (Paddle or Tesseract), post-processing and returns an annotated image.

```python
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
        regions = [ ... ]
        # (the real code tries a few ROI slices and runs the same detect/recognize flow)

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
```

If you want the complete `detect_and_recognize_plate` body copied instead of the shortened version above, let me know and I'll paste it in full — it's already present in `lpr_detector_v2.py`.

### 7.3 Important OCR helper (recognize_plate_text) — summary

- Enhances the plate crop via `enhance_plate_image` (resize + denoise + threshold + morphology).
- Tries PaddleOCR (if available) on original & enhanced crops.
- Tries Tesseract with multiple configurations (PSM 7,8,13,6) on enhanced then original.
- Cleans and validates candidate strings using `clean_plate_text` (pattern matching + replacements), and falls back to `try_autocorrect_plate` which attempts limited single/two-character substitutions.

## 8 — Recommended next improvements (practical, low-risk)

1. Persist the raw OCR candidate in `detections` (add `original_plate_text`) so you can audit corrections later.
2. Add a configurable fuzzy-match threshold and a second-best gap check to reduce accidental mis-corrections when registry grows.
3. Expand the audit tool (`tools/audit_fuzzy_registry.py`) to include two-character and insertion variants and run it periodically (or as a CI check) when registry changes.
4. Consider storing normalized plate strings in the `owners` table (additional column `plate_normalized`) to avoid repeated cleaning at lookup time.
5. Optionally, replace contour-based localization with a small YOLO/SSD detection model for plates to improve robustness across viewpoints and occlusions.

## 9 — Where to find things in the code (quick pointers)

- Flask endpoints: `app.py` (start here to understand API flows and DB wiring).
- Detection logic and OCR stack: `lpr_detector_v2.py` (core image logic).
- Notification functions: `lpr.py` (email/WhatsApp helpers used by `app.py`).
- Diagnostics & audit: `tools/debug_detector.py`, `tools/audit_fuzzy_registry.py`.

---

If you'd like, I can:

- Add the complete `recognize_plate_text` and `try_autocorrect_plate` functions verbatim at the end of this document.
- Create a short `docs/README_QUICK.md` for maintainers with exact commands for common tasks (start app, run audit, add owner, run detector on an image).
- Add a DB migration to persist `original_plate_text` for future detections.

Which of these would you like next?

## 10 — Detailed step-by-step pipeline (end-to-end)

Below is an expanded, explicit workflow that traces an input image from ingestion to notification, and enumerates all subsystems and third-party components involved.

1) Image ingestion
    - Source: camera feed (base64 frames posted to `/api/camera_detect`) or user upload (multipart POST to `/api/detect`).
    - Action: backend saves the received image to `uploads/` with a timestamped filename for audit/debug.

2) Plate localization — two available approaches in the codebase
    - Contour-based (primary in `lpr_detector_v2.py`, used by `app.py`):
      - Convert to grayscale and apply bilateral filter to smooth while preserving edges.
      - Run Canny edge detection and find contours with OpenCV.
      - Approximate contours with polygons and select a 4-corner contour whose bounding box has an aspect ratio typically between ~2:1 and ~5.5:1 and a minimum area threshold.
      - Crop the plate region from the input image.

    - Haar cascade + segmentation + CNN (implemented in `lpr.py`, optional):
      - Load a cascade classifier (XML) via OpenCV's `CascadeClassifier` (file expected at `data/indian_license_plate.xml`).
      - Use `detectMultiScale` to find plate ROIs.
      - For a found ROI, perform binarization and morphological ops, then run a contour-based character segmentation routine (`segment_characters`) to extract individual character images.
      - Use a trained TensorFlow/Keras model (`build_model()` + `show_results`) to classify each segmented character into alphanumerics and join them into a plate string.

3) Enhancement & preprocessing (applies to contour-based path)
    - Resize the plate crop (e.g., up to 300%) to help OCR.
    - Aggressive denoising (Non-local Means), bilateral filtering to keep edges, and sharpening (unsharp mask / kernel) to emphasize character strokes.
    - Thresholding: Otsu's and adaptive Gaussian threshold are both generated; the pipeline picks the one with more white pixels heuristically.
    - Morphological close/open to remove speckle noise and join character strokes.

4) OCR / character recognition
    - PaddleOCR (optional): attempted first if the `paddleocr` package is importable and initializes successfully. Paddle can provide bounding boxes and confidences for lines/words.
    - Tesseract (`pytesseract`): used as the robust fallback and primary OCR in many environments. Multiple Tesseract modes are tried:
      - PSM 7 (single text line), PSM 8 (single word), PSM 13 (raw line), PSM 6 (single block) with a whitelist of A-Z0-9.
      - Both enhanced and original plate crops are tried with these configs to produce multiple candidate strings with associated heuristic confidences.
    - Character-CNN path (from `lpr.py`): uses segmented characters and a trained Keras model to classify each character. This path is useful when a model is available and segmentation works well.

5) Post-processing & validation
    - Cleaning: remove non-alphanumeric chars, upper-case normalization, and replacement of common OCR confusions (e.g., O↔0, I↔1, Z↔2, S↔5, B↔8).
    - Strict-format quick-pass: if any raw OCR candidate already matches expected regional plate regex (e.g., `^[A-Z]{2}\d{2}[A-Z]?\d{4}$` for many Indian plates), accept it immediately.
    - Heuristics / autocorrect: if no strict match, `try_autocorrect_plate()` runs limited single- and two-character substitutions trying to fit common formats.

6) Registry lookup and fuzzy correction
    - Exact match: `app.py` queries the `owners` table for `plate_number`.
    - Fuzzy match: if no exact match, `fuzzy_match_plate()` computes Levenshtein distance between the candidate (normalized) and every registered plate, returning the best match if its distance <= `max_distance` (configurable; code uses 3 by default and 4 in some camera flows).
    - Safety: consider adding a second-best gap check (require best_distance + gap <= second_best_distance) before auto-correcting in production.

7) Logging & persistence
    - The detection is recorded in the `detections` table with `plate_number` (the corrected plate if fuzzy-matching applied), `image_path` and `detected_at` timestamp.
    - Notification attempts (email/WhatsApp) and their results (sent/skipped/error) are recorded in the `notifications` table with timestamps.

8) Notifications & third-party systems
    - Email notifications:
      - Uses Python's `smtplib` and `email.mime.text.MIMEText` via `send_email_notification()` in `lpr.py`.
      - SMTP configuration is read from `config.json` (if present) or environment variables: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM.
      - Any SMTP-compatible provider can be used (Gmail/Workspace via App passwords, SendGrid, Mailgun, your own SMTP server).

    - WhatsApp notifications:
      - Uses Twilio's WhatsApp Business API via the `twilio` Python package in `send_whatsapp_message()`.
      - Configuration: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM (or via `config.json` under `twilio`).
      - If Twilio is not configured or the `twilio` package is missing, the call returns `skipped` or `error` accordingly.

    - External runtime dependencies:
      - Tesseract OCR binary must be installed on the host and reachable by `pytesseract` (Windows default path is used in `lpr_detector_v2.py`).
      - (Optional) PaddleOCR Python package if used.
      - (Optional) Twilio account and number for WhatsApp delivery.

9) Final behavior exposed to clients
    - The API response includes `plate_number`, `confidence`, `registered` (bool), `notification_sent`, `gate_opening` (app-specific flag), and `auto_corrected`/`corrected_from` when fuzzy-matching corrected the candidate.

10) Auditability & diagnostics
     - Raw uploads and debug crops are saved in `uploads/` so you can manually inspect problematic frames.
     - The `tools/debug_detector.py` and `tools/audit_fuzzy_registry.py` helpers produce annotated images and CSVs for offline analysis.

## 11 — Quick references (where the code lives)

- Contour-based detector + OCR: `lpr_detector_v2.py` (used by `app.py`).
- Haar cascade + segmentation + CNN: `lpr.py` (alternate path, includes model training utilities).
- Flask API, DB wiring, and fuzzy-matching: `app.py`.
- Notification helpers and config: `lpr.py` (contains `send_email_notification`, `send_whatsapp_message`, `load_config`).
- Audit & debug tools: `tools/debug_detector.py`, `tools/audit_fuzzy_registry.py`.

---

If you want, I can now:
- Append the full text of `recognize_plate_text` and `try_autocorrect_plate` into this document for copy-paste convenience.
- Add a short sequence diagram image or ASCII flowchart for visual clarity.
- Implement the `original_plate_text` DB column and a migration script so raw OCR results are preserved.

Tell me which of the three you'd like next and I'll proceed. 
