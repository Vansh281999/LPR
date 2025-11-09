# Quick Setup Guide

## Fix Camera Access Issue

**Problem:** "Cannot read properties of undefined (reading 'getUserMedia')"

**Solution:** Access the app via `http://localhost:5000` instead of the IP address.

Browsers require either:
- `localhost` or `127.0.0.1` for HTTP camera access
- OR HTTPS connection

**Steps:**
1. Open your browser
2. Go to: **http://localhost:5000** (not http://172.x.x.x:5000)
3. Click "Start Camera" when prompted
4. Allow camera permissions

---

## Install Tesseract OCR (Required for text recognition)

### Option 1: Automated Install (Recommended)
Run this command:
```powershell
powershell -ExecutionPolicy Bypass -File install_tesseract.ps1
```

### Option 2: Manual Install
1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. Download file: `tesseract-ocr-w64-setup-5.5.0.20241111.exe`
3. Run installer
4. **IMPORTANT:** During install, check "Add to PATH"
5. Install to: `C:\Program Files\Tesseract-OCR`
6. Restart terminal
7. Verify: `tesseract --version`

### After Installing Tesseract:
1. Close the app (Ctrl+C)
2. Restart terminal
3. Run: `python app.py`
4. The OCR will now work!

---

## Using the System

### With Camera (Localhost only):
1. Go to http://localhost:5000
2. Click "Start Camera"
3. Allow camera permissions
4. Point at license plate
5. Click "Capture & Detect"

### With File Upload (Works on any URL):
1. Go to http://localhost:5000 or http://172.16.144.27:5000
2. Click upload zone or drag image
3. Click "Detect License Plate"

---

## Troubleshooting

### Camera not working?
- ✅ Use http://localhost:5000 (not IP address)
- ✅ Use Chrome, Firefox, or Edge (latest version)
- ✅ Check webcam is connected
- ✅ Allow camera permissions when prompted
- ✅ Close other apps using camera (Zoom, Teams, etc.)

### "Tesseract not found" error?
- ✅ Install Tesseract (see above)
- ✅ Make sure it's added to PATH
- ✅ Restart terminal after installation
- ✅ Run: `tesseract --version` to verify

### Detection not working?
- ✅ Use clear, well-lit images
- ✅ Plate should be visible and not at extreme angle
- ✅ Avoid glare or shadows on plate
- ✅ Try different images if one doesn't work

---

## Quick Commands

Start the app:
```bash
python app.py
```

Install dependencies:
```bash
pip install flask flask-cors opencv-python paddleocr pytesseract numpy
```

Check Tesseract:
```bash
tesseract --version
```
