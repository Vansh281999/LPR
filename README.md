# Smart LPR System

License Plate Recognition System for Smart Housing Communities

## Quick Start

1. **Install Tesseract OCR** (Required for text recognition):
   - Download: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to `C:\Program Files\Tesseract-OCR\`
   - Add to PATH: `C:\Program Files\Tesseract-OCR\`

2. **Run the application**:
   ```bash
   python app.py
   ```
   Or double-click: `start.bat`

3. **Open browser**:
   Navigate to http://localhost:5000

## Features

- ✅ **Automatic License Plate Detection** using computer vision
- ✅ **OCR Text Recognition** with PaddleOCR or Tesseract
- ✅ **Vehicle Registry** management
- ✅ **Detection History** with database storage
- ✅ **Modern Web Interface** with drag-and-drop upload
- ✅ **Real-time Statistics** dashboard

## System Requirements

- Python 3.8+
- Tesseract OCR
- OpenCV
- Flask

## Installation

### Windows (Tesseract)

1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (tesseract-ocr-w64-setup-5.x.x.exe)
3. Add to PATH or set environment variable:
   ```
   TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
   ```

### Verify Installation

```bash
tesseract --version
```

## Usage

1. **Upload Image**: Drag and drop or click to upload a vehicle image
2. **Detect Plate**: Click the "Detect License Plate" button
3. **View Results**: See detected plate number, confidence score, and registration status

## API Endpoints

- `GET /api/stats` - Dashboard statistics
- `POST /api/detect` - Detect license plate from uploaded image
- `GET /api/detections` - Get detection history
- `GET /api/registry` - Get registered vehicles
- `POST /api/registry` - Register a new vehicle
- `GET /api/notifications` - Get notification history

## Troubleshooting

### "Tesseract not found" error

Install Tesseract OCR and add it to your PATH:
```bash
# Windows
set PATH=%PATH%;C:\Program Files\Tesseract-OCR
```

### Low detection accuracy

- Use high-quality images with good lighting
- Ensure license plate is clearly visible
- Avoid glare or shadows on the plate
- Use images with plates at a reasonable angle

### Port already in use

Change the port in `app.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
```

## License

MIT License
