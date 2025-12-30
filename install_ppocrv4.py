"""Install PP-OCRv4 models for PaddleOCR.

This script downloads and installs PP-OCRv4 models (detection, recognition, classification)
for English language OCR. Models are downloaded to the PaddleOCR model cache directory.
"""

import sys
import os
from pathlib import Path
import urllib.request
import tarfile
import shutil

# Add Back_end to path
sys.path.insert(0, str(Path(__file__).parent))

from core.logging import log

# PP-OCRv4 model URLs (English models)
# Note: If manual download fails, PaddleOCR will auto-download models on first use
# when ocr_version='PP-OCRv4' is specified
PP_OCR_V4_MODELS = {
    'det': {
        'urls': [
            'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar',
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_PP-OCRv4_det_infer.tar',  # Alternative
        ],
        'dir_name': 'en_PP-OCRv4_det_infer',
        'target_dir': 'PP-OCRv4_det'
    },
    'rec': {
        'urls': [
            'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar',
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_PP-OCRv4_rec_infer.tar',  # Alternative
        ],
        'dir_name': 'en_PP-OCRv4_rec_infer',
        'target_dir': 'PP-OCRv4_rec'
    },
    'cls': {
        'urls': [
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/cls/ch_ppocr_mobile_v2.0_cls_infer.tar',
            'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar',  # Alternative (may not be exact match)
        ],
        'dir_name': 'ch_ppocr_mobile_v2.0_cls_infer',
        'target_dir': 'ch_ppocr_mobile_v2.0_cls'
    }
}

# PaddleOCR model cache directory (default location)
# On Linux/WSL: ~/.paddleocr/
# On Windows: C:\Users\<username>\.paddleocr\
def get_paddleocr_cache_dir():
    """Get PaddleOCR model cache directory."""
    home = Path.home()
    cache_dir = home / '.paddleocr'
    
    # Also check for PaddleX cache (newer versions)
    paddlex_cache = home / '.paddlex' / 'official_models'
    if paddlex_cache.exists():
        log.info(f"Found PaddleX cache: {paddlex_cache}")
    
    return cache_dir


def download_file(url: str, dest_path: Path, description: str):
    """Download a file from URL to destination path.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        description: Description of what's being downloaded
    """
    log.info(f"Downloading {description}...")
    log.info(f"  URL: {url}")
    log.info(f"  Destination: {dest_path}")
    
    try:
        # Create parent directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\r  Progress: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end='', flush=True)
        
        urllib.request.urlretrieve(url, dest_path, reporthook=show_progress)
        print()  # New line after progress
        log.info(f"✅ Downloaded {description}")
        return True
    except Exception as e:
        log.error(f"❌ Failed to download {description}: {str(e)}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial download
        return False


def extract_tar(tar_path: Path, extract_dir: Path, description: str):
    """Extract a tar file to a directory.
    
    Args:
        tar_path: Path to tar file
        extract_dir: Directory to extract to
        description: Description of what's being extracted
    """
    log.info(f"Extracting {description}...")
    log.info(f"  Archive: {tar_path}")
    log.info(f"  Destination: {extract_dir}")
    
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_dir)
        
        log.info(f"✅ Extracted {description}")
        return True
    except Exception as e:
        log.error(f"❌ Failed to extract {description}: {str(e)}")
        return False


def install_ppocrv4_model(model_type: str, model_info: dict, cache_dir: Path):
    """Install a PP-OCRv4 model.
    
    Args:
        model_type: Type of model ('det', 'rec', 'cls')
        model_info: Model information dict with urls (list), dir_name, target_dir
        cache_dir: PaddleOCR cache directory
    """
    log.info(f"\n{'='*60}")
    log.info(f"Installing PP-OCRv4 {model_type.upper()} model")
    log.info(f"{'='*60}")
    
    # Create temp directory for downloads
    temp_dir = Path(__file__).parent / 'temp_ppocrv4_downloads'
    temp_dir.mkdir(exist_ok=True)
    
    tar_file = temp_dir / f"{model_info['dir_name']}.tar"
    
    # Step 1: Download (try multiple URLs)
    if not tar_file.exists():
        # Get URLs list (support both 'url' and 'urls' keys for backward compatibility)
        urls = model_info.get('urls', [])
        if not urls and 'url' in model_info:
            urls = [model_info['url']]
        
        if not urls:
            log.error(f"No URLs provided for {model_type} model")
            return False
        
        download_success = False
        for i, url in enumerate(urls, 1):
            if url:
                log.info(f"Trying URL {i}/{len(urls)}: {url}")
                if download_file(url, tar_file, f"PP-OCRv4 {model_type} model"):
                    download_success = True
                    break
                else:
                    if i < len(urls):
                        log.warning(f"Failed to download from URL {i}, trying next URL...")
        
        if not download_success:
            log.warning(f"⚠️  Could not download {model_type} model manually from any URL.")
            log.info(f"   PaddleOCR will auto-download it on first use when ocr_version='PP-OCRv4' is specified.")
            return False
    else:
        log.info(f"✅ Tar file already exists: {tar_file}")
    
    # Step 2: Extract
    extract_dir = temp_dir / model_info['dir_name']
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        if not extract_tar(tar_file, temp_dir, f"PP-OCRv4 {model_type} model"):
            return False
    else:
        log.info(f"✅ Already extracted: {extract_dir}")
    
    # Step 3: Copy to PaddleOCR cache
    source_dir = extract_dir
    target_dir = cache_dir / model_info['target_dir']
    
    log.info(f"Copying model to PaddleOCR cache...")
    log.info(f"  Source: {source_dir}")
    log.info(f"  Target: {target_dir}")
    
    try:
        # Remove existing target if it exists
        if target_dir.exists():
            log.info(f"  Removing existing model at {target_dir}")
            shutil.rmtree(target_dir)
        
        # Copy model directory
        shutil.copytree(source_dir, target_dir)
        log.info(f"✅ PP-OCRv4 {model_type} model installed to: {target_dir}")
        return True
    except Exception as e:
        log.error(f"❌ Failed to copy model: {str(e)}")
        return False


def main():
    """Main installation function."""
    log.info("="*60)
    log.info("PP-OCRv4 Model Installation Script")
    log.info("="*60)
    log.info("")
    log.info("This script will download and install PP-OCRv4 models for English OCR.")
    log.info("Models will be installed to PaddleOCR's cache directory.")
    log.info("")
    
    # Get PaddleOCR cache directory
    cache_dir = get_paddleocr_cache_dir()
    log.info(f"PaddleOCR cache directory: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("")
    
    # Install each model
    success_count = 0
    total_models = len(PP_OCR_V4_MODELS)
    
    for model_type, model_info in PP_OCR_V4_MODELS.items():
        if install_ppocrv4_model(model_type, model_info, cache_dir):
            success_count += 1
        else:
            log.error(f"❌ Failed to install {model_type} model")
    
    # Cleanup temp directory
    temp_dir = Path(__file__).parent / 'temp_ppocrv4_downloads'
    if temp_dir.exists():
        log.info(f"\nCleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
            log.info("✅ Cleanup complete")
        except Exception as e:
            log.warning(f"⚠️  Failed to cleanup temp directory: {str(e)}")
    
    # Summary
    log.info("")
    log.info("="*60)
    log.info("Installation Summary")
    log.info("="*60)
    log.info(f"Successfully installed: {success_count}/{total_models} models")
    
    if success_count == total_models:
        log.info("✅ All PP-OCRv4 models installed successfully!")
        log.info("")
        log.info("The models are now available at:")
        for model_type, model_info in PP_OCR_V4_MODELS.items():
            model_path = cache_dir / model_info['target_dir']
            log.info(f"  {model_type.upper()}: {model_path}")
        log.info("")
        log.info("PaddleOCR will automatically use these models when initialized with:")
        log.info("  ocr_version='PP-OCRv4'")
        return 0
    else:
        failed_count = total_models - success_count
        log.warning(f"⚠️  Installation incomplete: {failed_count} model(s) failed to download manually")
        log.info("")
        log.info("This is OK! PaddleOCR will automatically download missing models on first use")
        log.info("when you initialize it with ocr_version='PP-OCRv4'.")
        log.info("")
        if success_count > 0:
            log.info(f"✅ {success_count} model(s) installed successfully:")
            for model_type, model_info in PP_OCR_V4_MODELS.items():
                model_path = cache_dir / model_info['target_dir']
                if model_path.exists():
                    log.info(f"  {model_type.upper()}: {model_path}")
        log.info("")
        log.info("To use PP-OCRv4, your code should initialize PaddleOCR with:")
        log.info("  PaddleOCR(ocr_version='PP-OCRv4', use_angle_cls=True, lang='en')")
        log.info("")
        log.info("Missing models will be downloaded automatically on first use.")
        return 0  # Return success since auto-download will handle the rest


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log.info("\n\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"\n\nUnexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

