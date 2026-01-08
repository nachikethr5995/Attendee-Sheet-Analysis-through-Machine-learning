# Install PP-OCRv4 Models for PaddleOCR (PowerShell version)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Installing PP-OCRv4 Models" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if virtual environment exists
if (-not (Test-Path "venv_wsl")) {
    Write-Host "❌ ERROR: venv_wsl not found!" -ForegroundColor Red
    Write-Host "   Please run install_dependencies.sh first in WSL" -ForegroundColor Yellow
    exit 1
}

Write-Host "Note: This script should ideally be run in WSL for best compatibility." -ForegroundColor Yellow
Write-Host "However, we'll try to run it with the system Python..." -ForegroundColor Yellow
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python or use WSL." -ForegroundColor Red
    exit 1
}

# Check if PaddleOCR is installed
Write-Host "Checking PaddleOCR installation..." -ForegroundColor Cyan
try {
    python -c "from paddleocr import PaddleOCR" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PaddleOCR is installed" -ForegroundColor Green
    } else {
        throw
    }
} catch {
    Write-Host "❌ PaddleOCR not found. Installing PaddleOCR..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install paddlepaddle paddleocr
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ PaddleOCR installation failed. Please install manually or use WSL." -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ PaddleOCR installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Downloading and Installing PP-OCRv4 Models" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will download ~200MB of model files." -ForegroundColor Yellow
Write-Host "Models will be installed to: ~/.paddleocr/" -ForegroundColor Yellow
Write-Host ""

# Run the installation script
python install_ppocrv4.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "✅ PP-OCRv4 Installation Complete!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "The models are now ready to use." -ForegroundColor Green
    Write-Host "PaddleOCR will automatically use PP-OCRv4 when initialized with the upgraded configuration." -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "❌ PP-OCRv4 Installation Failed" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above." -ForegroundColor Yellow
    Write-Host "For best results, run this in WSL: bash INSTALL_PPOCRV4.sh" -ForegroundColor Yellow
    exit 1
}







