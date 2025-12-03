#!/usr/bin/env python3
"""
Unified model downloader for Safe Flight vision models.

Downloads ONNX models for object detection, semantic segmentation, and depth estimation.
Consolidates functionality from download_vision_model.py, download_midas.py, 
download_segmentation_models.py, and download_onnx_models.py.
"""

import sys
import urllib.request
import argparse
from pathlib import Path
from loguru import logger

# Check for PyTorch (optional, for model conversion)
try:
    import torch
    import torch.onnx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def download_with_progress(url: str, destination: Path):
    """Download file with progress bar."""
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, int(count * block_size * 100 / total_size))
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            size_mb = total_size / 1024 / 1024
            downloaded_mb = count * block_size / 1024 / 1024
            sys.stdout.write(f'\r  [{bar}] {percent}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)')
            sys.stdout.flush()
    
    urllib.request.urlretrieve(url, str(destination), progress_hook)
    print()  # New line after progress


def setup_models_directory() -> Path:
    """Create models directory if it doesn't exist."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir


def download_vision_detection_models(models_dir: Path):
    """Download object detection models (YOLOv5, SSD)."""
    logger.info("\n" + "=" * 70)
    logger.info("Object Detection Models")
    logger.info("=" * 70)
    
    models = {
        "yolov5n": {
            "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
            "filename": "yolov5n.onnx",
            "size": "~8 MB",
            "description": "YOLOv5-Nano - Ultra-lightweight detection (RECOMMENDED)"
        },
        "ssd_mobilenet": {
            "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx",
            "filename": "ssd_mobilenet_v1.onnx",
            "size": "~27 MB",
            "description": "SSD-MobileNetV1 - General object detection"
        }
    }
    
    for key, model_info in models.items():
        destination = models_dir / model_info["filename"]
        
        if destination.exists():
            logger.info(f"\n✓ Already exists: {model_info['filename']}")
            continue
        
        logger.info(f"\nDownloading {model_info['description']}...")
        logger.info(f"  Size: {model_info['size']}")
        
        try:
            download_with_progress(model_info["url"], destination)
            file_size_mb = destination.stat().st_size / 1024 / 1024
            logger.success(f"✓ Downloaded: {model_info['filename']} ({file_size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"✗ Download failed: {e}")
            if destination.exists():
                destination.unlink()


def download_depth_models(models_dir: Path):
    """Download MiDaS depth estimation models."""
    logger.info("\n" + "=" * 70)
    logger.info("Depth Estimation Models")
    logger.info("=" * 70)
    
    models = {
        "midas_small": {
            "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
            "filename": "midas_v21_small.onnx",
            "size": "~64 MB",
            "speed": "Fast (~60 FPS on CPU)",
            "description": "MiDaS v2.1 Small - Recommended for real-time (RECOMMENDED)"
        },
        "midas_v21": {
            "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model.onnx",
            "filename": "midas_v21.onnx",
            "size": "~100 MB",
            "speed": "Medium (~25 FPS on CPU)",
            "description": "MiDaS v2.1 - Higher quality depth estimation"
        }
    }
    
    for key, model_info in models.items():
        destination = models_dir / model_info["filename"]
        
        if destination.exists():
            logger.info(f"\n✓ Already exists: {model_info['filename']}")
            continue
        
        logger.info(f"\nDownloading {model_info['description']}...")
        logger.info(f"  Size: {model_info['size']}")
        logger.info(f"  Speed: {model_info['speed']}")
        
        try:
            download_with_progress(model_info["url"], destination)
            file_size_mb = destination.stat().st_size / 1024 / 1024
            logger.success(f"✓ Downloaded: {model_info['filename']} ({file_size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"✗ Download failed: {e}")
            if destination.exists():
                destination.unlink()


def download_segmentation_models(models_dir: Path, use_pytorch: bool = False):
    """Download semantic segmentation models."""
    logger.info("\n" + "=" * 70)
    logger.info("Semantic Segmentation Models")
    logger.info("=" * 70)
    
    if use_pytorch and TORCH_AVAILABLE:
        logger.info("Using PyTorch to download and convert models...")
        download_segmentation_with_pytorch(models_dir)
    else:
        logger.info("Downloading pre-converted ONNX models...")
        download_segmentation_preconverted(models_dir)


def download_segmentation_preconverted(models_dir: Path):
    """Download pre-converted segmentation models."""
    models = {
        "deeplabv3_mobilenet": {
            "url": "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/deeplabv3/model/deeplabv3_mnv2_pascal_train_aug_2018_01_29.onnx",
            "filename": "deeplabv3_mobilenet_v2.onnx",
            "size": "~16 MB",
            "description": "DeepLabV3-MobileNetV2 - Lightweight segmentation"
        }
    }
    
    for key, model_info in models.items():
        destination = models_dir / model_info["filename"]
        
        if destination.exists():
            logger.info(f"\n✓ Already exists: {model_info['filename']}")
            continue
        
        logger.info(f"\nDownloading {model_info['description']}...")
        logger.info(f"  Size: {model_info['size']}")
        
        try:
            download_with_progress(model_info["url"], destination)
            file_size_mb = destination.stat().st_size / 1024 / 1024
            logger.success(f"✓ Downloaded: {model_info['filename']} ({file_size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"✗ Download failed: {e}")
            if destination.exists():
                destination.unlink()


def download_segmentation_with_pytorch(models_dir: Path):
    """Download and convert segmentation models using PyTorch."""
    variants = [
        ("resnet50", "DeepLabV3-ResNet50 - High quality"),
        ("mobilenet_v3_large", "DeepLabV3-MobileNetV3 - Lightweight")
    ]
    
    for variant, description in variants:
        model_name = f"deeplabv3_{variant}"
        output_path = models_dir / f"{model_name}.onnx"
        
        if output_path.exists():
            logger.info(f"\n✓ Already exists: {model_name}.onnx")
            continue
        
        logger.info(f"\nDownloading and converting {description}...")
        
        try:
            # Load pre-trained model
            model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                model_name,
                pretrained=True
            )
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 512, 512)
            
            # Export to ONNX
            logger.info(f"  Converting to ONNX...")
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch', 2: 'height', 3: 'width'}
                },
                opset_version=11
            )
            
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.success(f"✓ Converted: {model_name}.onnx ({file_size_mb:.1f} MB)")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")


def list_downloaded_models(models_dir: Path):
    """List all downloaded models."""
    logger.info("\n" + "=" * 70)
    logger.info("Downloaded Models")
    logger.info("=" * 70)
    
    model_files = list(models_dir.glob("*.onnx"))
    
    if not model_files:
        logger.warning("No models found!")
        return
    
    for model_file in sorted(model_files):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ {model_file.name} ({size_mb:.1f} MB)")


def print_usage_instructions():
    """Print instructions for using downloaded models."""
    logger.info("\n" + "=" * 70)
    logger.info("Usage Instructions")
    logger.info("=" * 70)
    
    logger.info("\nUpdate config/sim.yaml or config/hardware.yaml with your model paths:")
    logger.info("\n  For object detection mode:")
    logger.info("    vision:")
    logger.info("      mode: 'detection'")
    logger.info("      detection:")
    logger.info("        model_path: 'models/yolov5n.onnx'")
    
    logger.info("\n  For segmentation mode:")
    logger.info("    vision:")
    logger.info("      mode: 'segmentation'")
    logger.info("      segmentation:")
    logger.info("        segmentation_model_path: 'models/deeplabv3_resnet50.onnx'")
    logger.info("        depth_model_path: 'models/midas_v21_small.onnx'")
    
    logger.info("\nNext steps:")
    logger.info("  1. Run tests: python scripts/test.py --vision")
    logger.info("  2. Launch SITL: python launch_sim.py")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download vision models for Safe Flight obstacle avoidance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--detection',
        action='store_true',
        help='Download object detection models (YOLOv5, SSD)'
    )
    parser.add_argument(
        '--depth',
        action='store_true',
        help='Download depth estimation models (MiDaS)'
    )
    parser.add_argument(
        '--segmentation',
        action='store_true',
        help='Download semantic segmentation models (DeepLabV3)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all models'
    )
    parser.add_argument(
        '--pytorch',
        action='store_true',
        help='Use PyTorch to convert segmentation models (requires torch)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List downloaded models'
    )
    
    args = parser.parse_args()
    
    # Setup models directory
    models_dir = setup_models_directory()
    
    logger.info("=" * 70)
    logger.info("Safe Flight Model Downloader")
    logger.info("=" * 70)
    logger.info(f"Models directory: {models_dir.absolute()}")
    
    # If --list, just show what's downloaded
    if args.list:
        list_downloaded_models(models_dir)
        return
    
    # If no specific flags, download recommended defaults
    if not (args.detection or args.depth or args.segmentation or args.all):
        logger.info("\nNo specific models selected, downloading recommended defaults...")
        logger.info("(Use --all to download everything, or specify --detection, --depth, --segmentation)")
        args.detection = True
        args.depth = True
    
    # Download selected models
    if args.all:
        download_vision_detection_models(models_dir)
        download_depth_models(models_dir)
        download_segmentation_models(models_dir, use_pytorch=args.pytorch)
    else:
        if args.detection:
            download_vision_detection_models(models_dir)
        if args.depth:
            download_depth_models(models_dir)
        if args.segmentation:
            if args.pytorch and not TORCH_AVAILABLE:
                logger.warning("\nPyTorch not available! Install with: pip install torch torchvision")
                logger.info("Falling back to pre-converted ONNX models...")
            download_segmentation_models(models_dir, use_pytorch=args.pytorch)
    
    # Show summary
    list_downloaded_models(models_dir)
    print_usage_instructions()
    
    logger.info("\n" + "=" * 70)
    logger.success("Model Download Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nDownload cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
