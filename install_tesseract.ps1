# Install Tesseract OCR for Windows
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Tesseract OCR Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Tesseract is already installed
$tesseractPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (Test-Path $tesseractPath) {
    Write-Host "Tesseract is already installed at: $tesseractPath" -ForegroundColor Green
    & $tesseractPath --version
    Write-Host ""
    Write-Host "Checking PATH..." -ForegroundColor Yellow
    
    $envPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    if ($envPath -like "*Tesseract-OCR*") {
        Write-Host "Tesseract is in PATH" -ForegroundColor Green
    } else {
        Write-Host "Tesseract is NOT in PATH. Adding now..." -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable(
            "Path",
            $envPath + ";C:\Program Files\Tesseract-OCR",
            "Machine"
        )
        Write-Host "Added to PATH. Please restart your terminal." -ForegroundColor Green
    }
    exit
}

Write-Host "Tesseract not found. Installing..." -ForegroundColor Yellow
Write-Host ""

# Download URL for latest Tesseract
$downloadUrl = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.5.0.20241111.exe"
$installerPath = "$env:TEMP\tesseract-installer.exe"

Write-Host "Downloading Tesseract..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "Download complete!" -ForegroundColor Green
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download manually from:" -ForegroundColor Yellow
    Write-Host "https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Cyan
    exit 1
}

Write-Host ""
Write-Host "Starting installer..." -ForegroundColor Yellow
Write-Host "IMPORTANT: During installation, make sure to:" -ForegroundColor Red
Write-Host "  1. Install to: C:\Program Files\Tesseract-OCR" -ForegroundColor Yellow
Write-Host "  2. Check the box: 'Add to PATH'" -ForegroundColor Yellow
Write-Host ""

Start-Process -FilePath $installerPath -Wait

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "Please restart your terminal and run: tesseract --version" -ForegroundColor Cyan
Write-Host ""
Write-Host "Then restart the LPR app: python app.py" -ForegroundColor Cyan
