#!/usr/bin/env python3
"""
Setup script for Safe Flight environment.

This script handles:
- Environment verification (conda, Python version)
- Model downloads (MiDaS depth estimation)
- Directory structure creation
- Webots path detection and validation

Usage:
    python scripts/setup.py              # Full setup
    python scripts/setup.py --models     # Download models only
    python scripts/setup.py --verify     # Verify setup only
    python scripts/setup.py --clean      # Remove downloaded models

Typically called via:
    make setup
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


# Model definitions
MODELS = {
    'midas_v21_small': {
        'filename': 'midas_v21_small.onnx',
        'url': 'https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small-256x256.onnx',
        'size_mb': 63.7,
        'description': 'MiDaS v2.1 Small - Monocular depth estimation (256x256)',
    },
}


def print_header(text: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def print_step(step: int, total: int, text: str):
    """Print a step indicator."""
    print(f"\n[{step}/{total}] {text}")


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"


def check_python_version() -> bool:
    """Check Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"  {Colors.RED}✗{Colors.NC} Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"  {Colors.GREEN}✓{Colors.NC} Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_conda_env() -> bool:
    """Check if running in conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env:
        if conda_env == 'safe-flight':
            print(f"  {Colors.GREEN}✓{Colors.NC} Conda environment: {conda_env}")
            return True
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.NC} Conda environment: {conda_env} (expected: safe-flight)")
            return True
    else:
        print(f"  {Colors.YELLOW}⚠{Colors.NC} Not running in conda environment")
        print(f"      Run: conda activate safe-flight")
        return False


def check_dependencies() -> dict:
    """Check for required Python packages."""
    dependencies = {
        'onnxruntime': False,
        'opencv-python': False,
        'numpy': False,
        'yaml': False,
        'loguru': False,
    }
    
    # Map package names to import names
    import_names = {
        'opencv-python': 'cv2',
        'yaml': 'yaml',
    }
    
    for pkg in dependencies:
        import_name = import_names.get(pkg, pkg.replace('-', '_'))
        try:
            __import__(import_name)
            dependencies[pkg] = True
            print(f"  {Colors.GREEN}✓{Colors.NC} {pkg}")
        except ImportError:
            print(f"  {Colors.RED}✗{Colors.NC} {pkg}")
    
    return dependencies


def check_webots() -> bool:
    """Check for Webots installation."""
    webots_paths = [
        '/Applications/Webots.app',  # macOS
        '/usr/local/webots',         # Linux
        'C:\\Program Files\\Webots', # Windows
    ]
    
    for path in webots_paths:
        if os.path.exists(path):
            print(f"  {Colors.GREEN}✓{Colors.NC} Webots found: {path}")
            return True
    
    # Check WEBOTS_HOME environment variable
    webots_home = os.environ.get('WEBOTS_HOME', '')
    if webots_home and os.path.exists(webots_home):
        print(f"  {Colors.GREEN}✓{Colors.NC} Webots found: {webots_home}")
        return True
    
    print(f"  {Colors.YELLOW}⚠{Colors.NC} Webots not found (required for SITL simulation)")
    print(f"      Download from: https://cyberbotics.com/")
    return False


def create_directories():
    """Create required directory structure."""
    directories = [
        MODELS_DIR,
        DATA_DIR / 'raw',
        DATA_DIR / 'processed',
        DATA_DIR / 'visualization',
        PROJECT_ROOT / 'sim' / 'webots' / 'logs',
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True)
            print(f"  Created: {directory.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  Exists:  {directory.relative_to(PROJECT_ROOT)}")


def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"  Downloading {description}...")
        print(f"    URL: {url}")
        
        # Create a simple progress reporter
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                bar_len = 40
                filled_len = int(bar_len * percent / 100)
                bar = '█' * filled_len + '░' * (bar_len - filled_len)
                print(f"\r    [{bar}] {percent:.1f}% ({format_size(downloaded)})", end='', flush=True)
        
        urllib.request.urlretrieve(url, dest, reporthook=report_progress)
        print()  # New line after progress bar
        
        print(f"  {Colors.GREEN}✓{Colors.NC} Downloaded: {dest.name} ({format_size(dest.stat().st_size)})")
        return True
        
    except Exception as e:
        print(f"\n  {Colors.RED}✗{Colors.NC} Download failed: {e}")
        return False


def download_models(force: bool = False) -> bool:
    """Download required ML models."""
    print_header("Downloading Models")
    
    all_success = True
    
    for model_id, model_info in MODELS.items():
        dest_path = MODELS_DIR / model_info['filename']
        
        print(f"\n{model_info['description']}")
        print(f"  Size: ~{model_info['size_mb']} MB")
        
        if dest_path.exists() and not force:
            print(f"  {Colors.GREEN}✓{Colors.NC} Already downloaded: {model_info['filename']}")
            continue
        
        success = download_file(
            url=model_info['url'],
            dest=dest_path,
            description=model_info['filename']
        )
        
        if not success:
            all_success = False
    
    return all_success


def verify_models() -> bool:
    """Verify all required models are present."""
    all_present = True
    
    for model_id, model_info in MODELS.items():
        model_path = MODELS_DIR / model_info['filename']
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  {Colors.GREEN}✓{Colors.NC} {model_info['filename']} ({size_mb:.1f} MB)")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} {model_info['filename']} (missing)")
            all_present = False
    
    return all_present


def clean_models():
    """Remove downloaded models."""
    print_header("Cleaning Models")
    
    removed = 0
    for model_id, model_info in MODELS.items():
        model_path = MODELS_DIR / model_info['filename']
        if model_path.exists():
            model_path.unlink()
            print(f"  Removed: {model_info['filename']}")
            removed += 1
    
    if removed == 0:
        print("  No models to remove")
    else:
        print(f"\n  Removed {removed} model(s)")


def run_verification():
    """Run full setup verification."""
    print_header("Environment Verification")
    
    total_checks = 5
    passed = 0
    
    print_step(1, total_checks, "Python version")
    if check_python_version():
        passed += 1
    
    print_step(2, total_checks, "Conda environment")
    if check_conda_env():
        passed += 1
    
    print_step(3, total_checks, "Python dependencies")
    deps = check_dependencies()
    if all(deps.values()):
        passed += 1
    
    print_step(4, total_checks, "Webots installation")
    if check_webots():
        passed += 1
    
    print_step(5, total_checks, "ML models")
    if verify_models():
        passed += 1
    
    print()
    print("=" * 60)
    if passed == total_checks:
        print(f"{Colors.GREEN}All {total_checks} checks passed!{Colors.NC}")
    else:
        print(f"{Colors.YELLOW}{passed}/{total_checks} checks passed{Colors.NC}")
    print("=" * 60)
    
    return passed == total_checks


def main():
    parser = argparse.ArgumentParser(
        description='Setup Safe Flight environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/setup.py              # Full setup
    python scripts/setup.py --models     # Download models only
    python scripts/setup.py --verify     # Verify setup only
    python scripts/setup.py --clean      # Remove downloaded models
"""
    )
    
    parser.add_argument('--models', action='store_true',
                        help='Download models only')
    parser.add_argument('--verify', action='store_true',
                        help='Verify setup only (no downloads)')
    parser.add_argument('--clean', action='store_true',
                        help='Remove downloaded models')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download of models')
    
    args = parser.parse_args()
    
    print_header("Safe Flight Setup")
    print(f"Project root: {PROJECT_ROOT}")
    
    if args.clean:
        clean_models()
        return
    
    if args.verify:
        success = run_verification()
        sys.exit(0 if success else 1)
    
    if args.models:
        # Models only
        success = download_models(force=args.force)
        sys.exit(0 if success else 1)
    
    # Full setup
    print_step(1, 4, "Creating directory structure")
    create_directories()
    
    print_step(2, 4, "Checking environment")
    check_python_version()
    check_conda_env()
    
    print_step(3, 4, "Checking dependencies")
    deps = check_dependencies()
    missing = [k for k, v in deps.items() if not v]
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install {' '.join(missing)}")
    
    print_step(4, 4, "Downloading models")
    download_models(force=args.force)
    
    # Final verification
    print()
    run_verification()


if __name__ == '__main__':
    main()
