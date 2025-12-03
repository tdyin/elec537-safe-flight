#!/usr/bin/env python3
"""
Clean up logs and visualization data from Safe Flight SITL runs.

Usage:
    python scripts/cleanup.py                    # Interactive mode
    python scripts/cleanup.py --all              # Clean everything
    python scripts/cleanup.py --logs             # Clean logs only
    python scripts/cleanup.py --viz              # Clean visualizations only
    python scripts/cleanup.py --keep-latest 5    # Keep 5 most recent of each type
    python scripts/cleanup.py --older-than 7     # Clean files older than 7 days
    python scripts/cleanup.py --dry-run          # Show what would be deleted
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / 'sim' / 'webots' / 'logs'
VIZ_DIR = PROJECT_ROOT / 'data' / 'visualization'


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    units = ['B', 'KB', 'MB', 'GB']
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} {units[-1]}"


def get_file_age_days(file_path: Path) -> int:
    """Get the age of a file in days."""
    if not file_path.exists():
        return 0
    mtime = file_path.stat().st_mtime
    age_seconds = time.time() - mtime
    return int(age_seconds / 86400)


def get_file_size(path: Path) -> int:
    """Get size of a file or directory in bytes."""
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total
    return 0


def count_files(directory: Path, pattern: str) -> Tuple[int, int]:
    """Count files matching pattern and return (count, total_size)."""
    if not directory.exists():
        return 0, 0
    
    count = 0
    total_size = 0
    for file in directory.glob(pattern):
        if file.is_file():
            count += 1
            total_size += file.stat().st_size
    return count, total_size


def find_files_by_extensions(directory: Path, extensions: List[str]) -> List[Path]:
    """Find all files with given extensions in directory."""
    if not directory.exists():
        return []
    
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*.{ext}'))
    return [f for f in files if f.is_file()]


def show_status():
    """Show storage status for logs, visualizations, and cache."""
    print()
    print("=" * 80)
    print("SAFE FLIGHT - Storage Status")
    print("=" * 80)
    
    # Logs
    log_count, log_size = count_files(LOG_DIR, '*.log')
    print()
    print(f"{Colors.BLUE}📋 Logs{Colors.NC} ({LOG_DIR}):")
    print(f"   Files: {log_count}")
    print(f"   Size:  {format_size(log_size)}")
    
    if log_count > 0:
        log_files = sorted(LOG_DIR.glob('*.log'), key=lambda f: f.stat().st_mtime, reverse=True)
        if log_files:
            oldest_age = get_file_age_days(log_files[-1])
            newest_age = get_file_age_days(log_files[0])
            print(f"   Age:   {newest_age}-{oldest_age} days old")
    
    # Visualizations
    viz_count = 0
    viz_size = 0
    viz_extensions = ['png', 'jpg', 'jpeg', 'pdf', 'html']
    
    for ext in viz_extensions:
        c, s = count_files(VIZ_DIR, f'*.{ext}')
        viz_count += c
        viz_size += s
    
    print()
    print(f"{Colors.BLUE}📊 Visualizations{Colors.NC} ({VIZ_DIR}):")
    print(f"   Files: {viz_count}")
    print(f"   Size:  {format_size(viz_size)}")
    
    if viz_count > 0:
        viz_files = find_files_by_extensions(VIZ_DIR, viz_extensions)
        if viz_files:
            viz_files_sorted = sorted(viz_files, key=lambda f: f.stat().st_mtime, reverse=True)
            oldest_age = get_file_age_days(viz_files_sorted[-1])
            newest_age = get_file_age_days(viz_files_sorted[0])
            print(f"   Age:   {newest_age}-{oldest_age} days old")
    
    # Cache
    cache_count = 0
    cache_size = 0
    
    # Find all __pycache__ directories
    for pycache in PROJECT_ROOT.rglob('__pycache__'):
        if pycache.is_dir():
            cache_count += 1
            cache_size += get_file_size(pycache)
    
    # Find all .pyc and .pyo files
    for pattern in ['*.pyc', '*.pyo']:
        for file in PROJECT_ROOT.rglob(pattern):
            if file.is_file():
                cache_count += 1
                cache_size += file.stat().st_size
    
    # Find cache directories
    cache_dirs = ['.pytest_cache', '.mypy_cache', '.ruff_cache', '.tox']
    for cache_name in cache_dirs:
        for cache_dir in PROJECT_ROOT.rglob(cache_name):
            if cache_dir.is_dir():
                cache_count += 1
                cache_size += get_file_size(cache_dir)
    
    # Find .coverage files
    for coverage_file in PROJECT_ROOT.rglob('.coverage*'):
        if coverage_file.is_file():
            cache_count += 1
            cache_size += coverage_file.stat().st_size
    
    print()
    print(f"{Colors.BLUE}🗂️  Cache{Colors.NC} ({PROJECT_ROOT}):")
    print(f"   Items: {cache_count}")
    print(f"   Size:  {format_size(cache_size)}")
    
    total_size = log_size + viz_size + cache_size
    print()
    print(f"{Colors.GREEN}💾 Total Storage:{Colors.NC} {format_size(total_size)}")
    print("=" * 80)
    print()


def collect_files_to_delete(
    clean_logs: bool,
    clean_viz: bool,
    clean_cache: bool,
    keep_latest: Optional[int],
    older_than: Optional[int]
) -> List[Path]:
    """Collect files to delete based on options."""
    files_to_delete = []
    
    if clean_logs and LOG_DIR.exists():
        log_files = sorted(
            [f for f in LOG_DIR.glob('*.log') if f.is_file()],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        if keep_latest is not None:
            log_files = log_files[keep_latest:]
        elif older_than is not None:
            log_files = [f for f in log_files if get_file_age_days(f) > older_than]
        
        files_to_delete.extend(log_files)
    
    if clean_viz and VIZ_DIR.exists():
        viz_extensions = ['png', 'jpg', 'jpeg', 'pdf', 'html']
        viz_files = find_files_by_extensions(VIZ_DIR, viz_extensions)
        viz_files = sorted(viz_files, key=lambda f: f.stat().st_mtime, reverse=True)
        
        if keep_latest is not None:
            viz_files = viz_files[keep_latest:]
        elif older_than is not None:
            viz_files = [f for f in viz_files if get_file_age_days(f) > older_than]
        
        files_to_delete.extend(viz_files)
    
    if clean_cache:
        # __pycache__ directories
        for pycache in PROJECT_ROOT.rglob('__pycache__'):
            if pycache.is_dir():
                files_to_delete.append(pycache)
        
        # .pyc and .pyo files
        for pattern in ['*.pyc', '*.pyo']:
            for file in PROJECT_ROOT.rglob(pattern):
                if file.is_file():
                    files_to_delete.append(file)
        
        # Cache directories
        cache_dirs = ['.pytest_cache', '.mypy_cache', '.ruff_cache', '.tox']
        for cache_name in cache_dirs:
            for cache_dir in PROJECT_ROOT.rglob(cache_name):
                if cache_dir.is_dir():
                    files_to_delete.append(cache_dir)
        
        # .coverage files
        for coverage_file in PROJECT_ROOT.rglob('.coverage*'):
            if coverage_file.is_file():
                files_to_delete.append(coverage_file)
    
    return files_to_delete


def categorize_files(files: List[Path]) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """Categorize files into logs, images, cache, and others."""
    logs = []
    images = []
    cache_items = []
    others = []
    
    cache_patterns = ['__pycache__', '.pyc', '.pyo', '.pytest_cache', 
                      '.mypy_cache', '.ruff_cache', '.coverage', '.tox']
    
    for file in files:
        file_str = str(file)
        if file.suffix == '.log':
            logs.append(file)
        elif file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            images.append(file)
        elif any(pattern in file_str for pattern in cache_patterns):
            cache_items.append(file)
        else:
            others.append(file)
    
    return logs, images, cache_items, others


def show_file_list(files: List[Path], category: str, emoji: str, max_show: int = 5):
    """Show a list of files with their sizes and ages."""
    if not files:
        return
    
    print()
    print(f"{Colors.BLUE}{emoji} {category}{Colors.NC} ({len(files)} {'files' if len(files) != 1 else 'file'}):")
    
    for i, file in enumerate(files[:max_show]):
        size = get_file_size(file)
        if file.is_dir():
            print(f"  - {file.name}/ ({format_size(size)})")
        else:
            age = get_file_age_days(file)
            print(f"  - {file.name} ({format_size(size)}, {age} days old)")
    
    if len(files) > max_show:
        print(f"  ... and {len(files) - max_show} more")


def interactive_mode() -> Tuple[bool, bool, bool]:
    """Run interactive mode to select what to clean."""
    print("What would you like to clean?")
    print("  1. Logs only")
    print("  2. Visualizations only")
    print("  3. Cache only")
    print("  4. Logs and Visualizations")
    print("  5. Everything (logs, viz, cache)")
    print("  6. Cancel")
    print()
    
    try:
        choice = input("Choice [1-6]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled")
        sys.exit(0)
    
    if choice == '1':
        return True, False, False
    elif choice == '2':
        return False, True, False
    elif choice == '3':
        return False, False, True
    elif choice == '4':
        return True, True, False
    elif choice == '5':
        return True, True, True
    else:
        print("Cancelled")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Clean up Safe Flight logs and visualization data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/cleanup.py                      # Interactive mode
    python scripts/cleanup.py --all                # Clean everything
    python scripts/cleanup.py --logs --keep-latest 5   # Keep 5 most recent logs
    python scripts/cleanup.py --older-than 7       # Clean files older than 7 days
    python scripts/cleanup.py --dry-run --all      # Preview what would be deleted
"""
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Clean logs, visualizations, and cache')
    parser.add_argument('--logs', action='store_true',
                        help='Clean log files only')
    parser.add_argument('--viz', '--visualizations', action='store_true',
                        help='Clean visualization files only')
    parser.add_argument('--cache', action='store_true',
                        help='Clean Python cache files only')
    parser.add_argument('--keep-latest', type=int, metavar='N',
                        help='Keep N most recent files of each type')
    parser.add_argument('--older-than', type=int, metavar='DAYS',
                        help='Delete files older than DAYS days')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without deleting')
    parser.add_argument('--status', action='store_true',
                        help='Show storage status only')
    
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        clean_logs = True
        clean_viz = True
        clean_cache = True
    else:
        clean_logs = args.logs
        clean_viz = args.viz
        clean_cache = args.cache
    
    # Show status if requested or no action specified
    if args.status or not (clean_logs or clean_viz or clean_cache):
        show_status()
        if args.status:
            return
    
    # Interactive mode if nothing specified
    if not (clean_logs or clean_viz or clean_cache):
        clean_logs, clean_viz, clean_cache = interactive_mode()
    
    # Collect files to delete
    files_to_delete = collect_files_to_delete(
        clean_logs=clean_logs,
        clean_viz=clean_viz,
        clean_cache=clean_cache,
        keep_latest=args.keep_latest,
        older_than=args.older_than
    )
    
    # Check if anything to delete
    if not files_to_delete:
        print(f"{Colors.GREEN}✓ No files to delete{Colors.NC}")
        return
    
    # Calculate total size
    total_size = sum(get_file_size(f) for f in files_to_delete)
    
    # Show what will be deleted
    print()
    if args.dry_run:
        print(f"{Colors.YELLOW}[DRY RUN]{Colors.NC} Files to delete ({len(files_to_delete)} files, {format_size(total_size)}):")
    else:
        print(f"Files to delete ({len(files_to_delete)} files, {format_size(total_size)}):")
    print("-" * 80)
    
    # Categorize and show files
    logs, images, cache_items, others = categorize_files(files_to_delete)
    
    show_file_list(logs, "Logs", "📋")
    show_file_list(images, "Visualizations", "📊")
    show_file_list(cache_items, "Cache", "🗂️")
    show_file_list(others, "Other files", "📄")
    
    print()
    
    # Dry run - don't delete
    if args.dry_run:
        print(f"{Colors.YELLOW}[DRY RUN]{Colors.NC} No files were actually deleted")
        return
    
    # Confirm deletion
    print(f"{Colors.YELLOW}⚠️  Total: {len(files_to_delete)} files, {format_size(total_size)}{Colors.NC}")
    
    try:
        confirm = input("Delete these files? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled - no files deleted")
        return
    
    if confirm not in ['y', 'yes']:
        print("Cancelled - no files deleted")
        return
    
    # Delete files and directories
    success = 0
    errors = 0
    
    for item in files_to_delete:
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            success += 1
        except Exception as e:
            print(f"{Colors.RED}Error deleting {item.name}: {e}{Colors.NC}")
            errors += 1
    
    print()
    print(f"{Colors.GREEN}✓ Deleted {success} files{Colors.NC}")
    if errors > 0:
        print(f"{Colors.RED}✗ {errors} errors{Colors.NC}")
    
    # Show new status
    show_status()


if __name__ == '__main__':
    main()
