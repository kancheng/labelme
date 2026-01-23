# Labelme Standalone Executable Build Script (using UV environment)
# This script uses pyinstaller to package labelme into a standalone executable

Write-Host "Starting to build Labelme standalone executable..." -ForegroundColor Green

# Set path variables (using absolute paths)
$LABELME_PATH = Resolve-Path ".\labelme"
$OSAM_PATH = uv run python -c "import os, osam; print(os.path.dirname(osam.__file__))"

Write-Host "LABELME_PATH: $LABELME_PATH" -ForegroundColor Cyan
Write-Host "OSAM_PATH: $OSAM_PATH" -ForegroundColor Cyan

# Check if required files exist
if (-not (Test-Path "$LABELME_PATH\__main__.py")) {
    Write-Host "Error: Cannot find labelme/__main__.py" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$OSAM_PATH\_models\yoloworld\clip\bpe_simple_vocab_16e6.txt.gz")) {
    Write-Host "Warning: Cannot find OSAM model file, build may fail" -ForegroundColor Yellow
}

# Build command (using absolute paths)
$mainScript = "$LABELME_PATH\__main__.py"
$osamData = "$OSAM_PATH\_models\yoloworld\clip\bpe_simple_vocab_16e6.txt.gz"
$configFile = "$LABELME_PATH\config\default_config.yaml"
$iconsPath = "$LABELME_PATH\icons"
$translatePath = "$LABELME_PATH\translate"
$iconFile = "$LABELME_PATH\icons\icon-256.png"

Write-Host "`nInstalling PyInstaller..." -ForegroundColor Yellow
uv pip install pyinstaller

Write-Host "`nRunning pyinstaller..." -ForegroundColor Yellow

# Check if spec file exists, if so use it, otherwise use command line
$specFile = "build\Labelme.spec"
if (Test-Path $specFile) {
    Write-Host "Using existing spec file: $specFile" -ForegroundColor Cyan
    & uv run python -m PyInstaller `
        $specFile `
        --noconfirm
} else {
    Write-Host "Using command line arguments (spec file not found)" -ForegroundColor Yellow
    $hooksPath = Resolve-Path ".\hooks" -ErrorAction SilentlyContinue
    $hookArgs = if ($hooksPath) { "--additional-hooks-dir=$hooksPath" } else { "" }
    
    # Use uv run to execute pyinstaller (using python -m method)
    & uv run python -m PyInstaller `
        $mainScript `
        --name=Labelme `
        --windowed `
        --noconfirm `
        --specpath=build `
        --add-data="$osamData;osam\_models\yoloworld\clip" `
        --add-data="$configFile;labelme\config" `
        --add-data="$iconsPath\*;labelme\icons" `
        --add-data="$translatePath\*;translate" `
        --icon="$iconFile" `
        $hookArgs `
        --onedir
}


