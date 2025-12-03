#!/usr/bin/env bash
#
# Clean up logs and visualization data from Safe Flight SITL runs.
#
# Usage:
#   ./scripts/cleanup.sh                    # Interactive mode
#   ./scripts/cleanup.sh --all              # Clean everything
#   ./scripts/cleanup.sh --logs             # Clean logs only
#   ./scripts/cleanup.sh --viz              # Clean visualizations only
#   ./scripts/cleanup.sh --keep-latest 5    # Keep 5 most recent of each type
#   ./scripts/cleanup.sh --older-than 7     # Clean files older than 7 days
#   ./scripts/cleanup.sh --dry-run          # Show what would be deleted

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/sim/webots/logs"
VIZ_DIR="$PROJECT_ROOT/data/visualization"

# Options
DRY_RUN=false
CLEAN_LOGS=false
CLEAN_VIZ=false
CLEAN_CACHE=false
KEEP_LATEST=""
OLDER_THAN=""
STATUS_ONLY=false

# Helper functions
format_size() {
    local size=$1
    if command -v numfmt &> /dev/null; then
        numfmt --to=iec-i --suffix=B "$size"
    else
        # Fallback for systems without numfmt
        awk -v size="$size" 'BEGIN {
            units[0]="B"; units[1]="KB"; units[2]="MB"; units[3]="GB"
            for(i=0; size>=1024 && i<3; i++) size/=1024
            printf "%.1f %s", size, units[i]
        }'
    fi
}

get_file_age_days() {
    local file=$1
    if [[ ! -e "$file" ]]; then echo "0"; return; fi
    
    local now=$(date +%s)
    local mtime
    if [[ "$OSTYPE" == "darwin"* ]]; then
        mtime=$(stat -f %m "$file")
    else
        mtime=$(stat -c %Y "$file")
    fi
    echo $(( (now - mtime) / 86400 ))
}

count_files() {
    local pattern=$1
    if [[ -d "$(dirname "$pattern")" ]]; then
        # Use find to count files, handle no matches gracefully
        find "$(dirname "$pattern")" -maxdepth 1 -name "$(basename "$pattern")" -type f 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

get_total_size() {
    local pattern=$1
    local total=0
    
    if [[ -d "$(dirname "$pattern")" ]]; then
        while IFS= read -r file; do
            if [[ -f "$file" ]]; then
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    size=$(stat -f %z "$file")
                else
                    size=$(stat -c %s "$file")
                fi
                total=$((total + size))
            fi
        done < <(find "$(dirname "$pattern")" -maxdepth 1 -name "$(basename "$pattern")" -type f 2>/dev/null)
    fi
    
    echo "$total"
}

show_status() {
    echo ""
    echo "================================================================================"
    echo "SAFE FLIGHT - Storage Status"
    echo "================================================================================"
    
    # Logs
    local log_count=$(count_files "$LOG_DIR/*.log")
    local log_size=$(get_total_size "$LOG_DIR/*.log")
    
    echo ""
    echo -e "${BLUE}📋 Logs${NC} ($LOG_DIR):"
    echo "   Files: $log_count"
    echo "   Size:  $(format_size $log_size)"
    
    if [[ $log_count -gt 0 ]]; then
        local oldest=$(find "$LOG_DIR" -name "*.log" -type f -print0 2>/dev/null | xargs -0 ls -t | tail -1)
        local newest=$(find "$LOG_DIR" -name "*.log" -type f -print0 2>/dev/null | xargs -0 ls -t | head -1)
        if [[ -n "$oldest" && -n "$newest" ]]; then
            local oldest_age=$(get_file_age_days "$oldest")
            local newest_age=$(get_file_age_days "$newest")
            echo "   Age:   $newest_age-$oldest_age days old"
        fi
    fi
    
    # Visualizations
    local viz_count=0
    local viz_size=0
    
    for ext in png jpg jpeg pdf html; do
        viz_count=$((viz_count + $(count_files "$VIZ_DIR/*.$ext")))
        viz_size=$((viz_size + $(get_total_size "$VIZ_DIR/*.$ext")))
    done
    
    echo ""
    echo -e "${BLUE}📊 Visualizations${NC} ($VIZ_DIR):"
    echo "   Files: $viz_count"
    echo "   Size:  $(format_size $viz_size)"
    
    if [[ $viz_count -gt 0 ]]; then
        local oldest=$(find "$VIZ_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.pdf" -o -name "*.html" \) -print0 2>/dev/null | xargs -0 ls -t | tail -1)
        local newest=$(find "$VIZ_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.pdf" -o -name "*.html" \) -print0 2>/dev/null | xargs -0 ls -t | head -1)
        if [[ -n "$oldest" && -n "$newest" ]]; then
            local oldest_age=$(get_file_age_days "$oldest")
            local newest_age=$(get_file_age_days "$newest")
            echo "   Age:   $newest_age-$oldest_age days old"
        fi
    fi
    
    # Cache
    local cache_count=0
    local cache_size=0
    
    # Recursively count all __pycache__ directories
    while IFS= read -r dir; do
        if [[ -d "$dir" ]]; then
            ((cache_count++))
            if [[ "$OSTYPE" == "darwin"* ]]; then
                dir_size=$(du -sk "$dir" 2>/dev/null | cut -f1)
                cache_size=$((cache_size + dir_size * 1024))
            else
                dir_size=$(du -sb "$dir" 2>/dev/null | cut -f1)
                cache_size=$((cache_size + dir_size))
            fi
        fi
    done < <(find "$PROJECT_ROOT" -type d -name "__pycache__" 2>/dev/null)
    
    # Recursively count all .pyc and .pyo files
    while IFS= read -r file; do
        if [[ -f "$file" ]]; then
            ((cache_count++))
            if [[ "$OSTYPE" == "darwin"* ]]; then
                size=$(stat -f %z "$file")
            else
                size=$(stat -c %s "$file")
            fi
            cache_size=$((cache_size + size))
        fi
    done < <(find "$PROJECT_ROOT" -type f \( -name "*.pyc" -o -name "*.pyo" \) 2>/dev/null)
    
    # Recursively count all cache directories (.pytest_cache, .mypy_cache, .ruff_cache, .tox)
    for cache_name in ".pytest_cache" ".mypy_cache" ".ruff_cache" ".tox"; do
        while IFS= read -r dir; do
            if [[ -d "$dir" ]]; then
                ((cache_count++))
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    dir_size=$(du -sk "$dir" 2>/dev/null | cut -f1)
                    cache_size=$((cache_size + dir_size * 1024))
                else
                    dir_size=$(du -sb "$dir" 2>/dev/null | cut -f1)
                    cache_size=$((cache_size + dir_size))
                fi
            fi
        done < <(find "$PROJECT_ROOT" -type d -name "$cache_name" 2>/dev/null)
    done
    
    # Recursively count all .coverage files
    while IFS= read -r file; do
        if [[ -f "$file" ]]; then
            ((cache_count++))
            if [[ "$OSTYPE" == "darwin"* ]]; then
                size=$(stat -f %z "$file")
            else
                size=$(stat -c %s "$file")
            fi
            cache_size=$((cache_size + size))
        fi
    done < <(find "$PROJECT_ROOT" -type f -name ".coverage*" 2>/dev/null)
    
    echo ""
    echo -e "${BLUE}🗂️  Cache${NC} ($PROJECT_ROOT):"
    echo "   Items: $cache_count"
    echo "   Size:  $(format_size $cache_size)"
    
    local total_size=$((log_size + viz_size + cache_size))
    echo ""
    echo -e "${GREEN}💾 Total Storage:${NC} $(format_size $total_size)"
    echo "================================================================================"
    echo ""
}

show_help() {
    cat << EOF
Clean up Safe Flight logs and visualization data

Usage: $(basename "$0") [OPTIONS]

Options:
    --all               Clean logs, visualizations, and cache
    --logs              Clean log files only
    --viz               Clean visualization files only
    --cache             Clean Python cache files only
    --keep-latest N     Keep N most recent files of each type
    --older-than DAYS   Delete files older than DAYS days
    --dry-run           Show what would be deleted without deleting
    --status            Show storage status only
    -h, --help          Show this help message

Examples:
    $(basename "$0")                      # Interactive mode
    $(basename "$0") --all                # Clean everything
    $(basename "$0") --logs --keep-latest 5   # Keep 5 most recent logs
    $(basename "$0") --older-than 7       # Clean files older than 7 days
    $(basename "$0") --dry-run --all      # Preview what would be deleted
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            CLEAN_LOGS=true
            CLEAN_VIZ=true
            CLEAN_CACHE=true
            shift
            ;;
        --logs)
            CLEAN_LOGS=true
            shift
            ;;
        --viz|--visualizations)
            CLEAN_VIZ=true
            shift
            ;;
        --cache)
            CLEAN_CACHE=true
            shift
            ;;
        --keep-latest)
            KEEP_LATEST="$2"
            shift 2
            ;;
        --older-than)
            OLDER_THAN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Show status if requested or no action specified
if [[ "$STATUS_ONLY" == true ]] || [[ "$CLEAN_LOGS" == false && "$CLEAN_VIZ" == false && "$CLEAN_CACHE" == false ]]; then
    show_status
    if [[ "$STATUS_ONLY" == true ]]; then
        exit 0
    fi
fi

# Interactive mode if nothing specified
if [[ "$CLEAN_LOGS" == false && "$CLEAN_VIZ" == false && "$CLEAN_CACHE" == false ]]; then
    echo "What would you like to clean?"
    echo "  1. Logs only"
    echo "  2. Visualizations only"
    echo "  3. Cache only"
    echo "  4. Logs and Visualizations"
    echo "  5. Everything (logs, viz, cache)"
    echo "  6. Cancel"
    echo ""
    read -p "Choice [1-6]: " choice
    
    case $choice in
        1)
            CLEAN_LOGS=true
            ;;
        2)
            CLEAN_VIZ=true
            ;;
        3)
            CLEAN_CACHE=true
            ;;
        4)
            CLEAN_LOGS=true
            CLEAN_VIZ=true
            ;;
        5)
            CLEAN_LOGS=true
            CLEAN_VIZ=true
            CLEAN_CACHE=true
            ;;
        *)
            echo "Cancelled"
            exit 0
            ;;
    esac
fi

# Build file list
files_to_delete=()

if [[ "$CLEAN_LOGS" == true ]]; then
    if [[ -d "$LOG_DIR" ]]; then
        log_files=()
        while IFS= read -r file; do
            [[ -n "$file" ]] && log_files+=("$file")
        done < <(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | sort -r)
        
        # Filter by criteria
        if [[ -n "$KEEP_LATEST" ]]; then
            filtered=()
            count=0
            for file in "${log_files[@]}"; do
                if [[ $count -ge $KEEP_LATEST ]]; then
                    filtered+=("$file")
                fi
                ((count++))
            done
            log_files=("${filtered[@]}")
        elif [[ -n "$OLDER_THAN" ]]; then
            filtered=()
            for file in "${log_files[@]}"; do
                age=$(get_file_age_days "$file")
                if [[ $age -gt $OLDER_THAN ]]; then
                    filtered+=("$file")
                fi
            done
            log_files=("${filtered[@]}")
        fi
        
        files_to_delete+=("${log_files[@]}")
    fi
fi

if [[ "$CLEAN_VIZ" == true ]]; then
    if [[ -d "$VIZ_DIR" ]]; then
        viz_files=()
        while IFS= read -r file; do
            [[ -n "$file" ]] && viz_files+=("$file")
        done < <(find "$VIZ_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.pdf" -o -name "*.html" \) 2>/dev/null | sort -r)
        
        # Filter by criteria
        if [[ -n "$KEEP_LATEST" ]]; then
            filtered=()
            count=0
            for file in "${viz_files[@]}"; do
                if [[ $count -ge $KEEP_LATEST ]]; then
                    filtered+=("$file")
                fi
                ((count++))
            done
            viz_files=("${filtered[@]}")
        elif [[ -n "$OLDER_THAN" ]]; then
            filtered=()
            for file in "${viz_files[@]}"; do
                age=$(get_file_age_days "$file")
                if [[ $age -gt $OLDER_THAN ]]; then
                    filtered+=("$file")
                fi
            done
            viz_files=("${filtered[@]}")
        fi
        
        files_to_delete+=("${viz_files[@]}")
    fi
fi

if [[ "$CLEAN_CACHE" == true ]]; then
    cache_items=()
    
    # Recursively find all __pycache__ directories
    while IFS= read -r dir; do
        [[ -n "$dir" ]] && cache_items+=("$dir")
    done < <(find "$PROJECT_ROOT" -type d -name "__pycache__" 2>/dev/null)
    
    # Recursively find all .pyc and .pyo files
    while IFS= read -r file; do
        [[ -n "$file" ]] && cache_items+=("$file")
    done < <(find "$PROJECT_ROOT" -type f \( -name "*.pyc" -o -name "*.pyo" \) 2>/dev/null)
    
    # Recursively find all .pytest_cache directories
    while IFS= read -r dir; do
        [[ -n "$dir" ]] && cache_items+=("$dir")
    done < <(find "$PROJECT_ROOT" -type d -name ".pytest_cache" 2>/dev/null)
    
    # Recursively find all .mypy_cache directories
    while IFS= read -r dir; do
        [[ -n "$dir" ]] && cache_items+=("$dir")
    done < <(find "$PROJECT_ROOT" -type d -name ".mypy_cache" 2>/dev/null)
    
    # Recursively find all .ruff_cache directories
    while IFS= read -r dir; do
        [[ -n "$dir" ]] && cache_items+=("$dir")
    done < <(find "$PROJECT_ROOT" -type d -name ".ruff_cache" 2>/dev/null)
    
    # Recursively find all .coverage files
    while IFS= read -r file; do
        [[ -n "$file" ]] && cache_items+=("$file")
    done < <(find "$PROJECT_ROOT" -type f -name ".coverage*" 2>/dev/null)
    
    # Recursively find all .tox directories
    while IFS= read -r dir; do
        [[ -n "$dir" ]] && cache_items+=("$dir")
    done < <(find "$PROJECT_ROOT" -type d -name ".tox" 2>/dev/null)
    
    # Note: --keep-latest and --older-than don't apply to cache (cache should be fully regeneratable)
    files_to_delete+=("${cache_items[@]}")
fi

# Check if anything to delete
if [[ ${#files_to_delete[@]} -eq 0 ]]; then
    echo -e "${GREEN}✓ No files to delete${NC}"
    exit 0
fi

# Calculate total size
total_size=0
for file in "${files_to_delete[@]}"; do
    if [[ -f "$file" ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            size=$(stat -f %z "$file")
        else
            size=$(stat -c %s "$file")
        fi
        total_size=$((total_size + size))
    elif [[ -d "$file" ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            dir_size=$(du -sk "$file" 2>/dev/null | cut -f1)
            total_size=$((total_size + dir_size * 1024))
        else
            dir_size=$(du -sb "$file" 2>/dev/null | cut -f1)
            total_size=$((total_size + dir_size))
        fi
    fi
done

# Show what will be deleted
echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}[DRY RUN]${NC} Files to delete (${#files_to_delete[@]} files, $(format_size $total_size)):"
else
    echo "Files to delete (${#files_to_delete[@]} files, $(format_size $total_size)):"
fi
echo "--------------------------------------------------------------------------------"

# Group by type
logs=()
images=()
cache_items=()
others=()

for file in "${files_to_delete[@]}"; do
    case "$file" in
        *.log)
            logs+=("$file")
            ;;
        *.png|*.jpg|*.jpeg)
            images+=("$file")
            ;;
        *__pycache__*|*.pyc|*.pyo|*.pytest_cache*|*.mypy_cache*|*.ruff_cache*|*.coverage*|*.tox*)
            cache_items+=("$file")
            ;;
        *)
            others+=("$file")
            ;;
    esac
done

# Show logs
if [[ ${#logs[@]} -gt 0 ]]; then
    echo ""
    echo -e "${BLUE}📋 Logs${NC} (${#logs[@]} files):"
    count=0
    for file in "${logs[@]}"; do
        if [[ $count -lt 5 ]]; then
            age=$(get_file_age_days "$file")
            if [[ "$OSTYPE" == "darwin"* ]]; then
                size=$(stat -f %z "$file")
            else
                size=$(stat -c %s "$file")
            fi
            echo "  - $(basename "$file") ($(format_size $size), $age days old)"
        fi
        ((count++))
    done
    if [[ ${#logs[@]} -gt 5 ]]; then
        echo "  ... and $((${#logs[@]} - 5)) more"
    fi
fi

# Show images
if [[ ${#images[@]} -gt 0 ]]; then
    echo ""
    echo -e "${BLUE}📊 Visualizations${NC} (${#images[@]} files):"
    count=0
    for file in "${images[@]}"; do
        if [[ $count -lt 5 ]]; then
            age=$(get_file_age_days "$file")
            if [[ "$OSTYPE" == "darwin"* ]]; then
                size=$(stat -f %z "$file")
            else
                size=$(stat -c %s "$file")
            fi
            echo "  - $(basename "$file") ($(format_size $size), $age days old)"
        fi
        ((count++))
    done
    if [[ ${#images[@]} -gt 5 ]]; then
        echo "  ... and $((${#images[@]} - 5)) more"
    fi
fi

# Show cache
if [[ ${#cache_items[@]} -gt 0 ]]; then
    echo ""
    echo -e "${BLUE}🗂️  Cache${NC} (${#cache_items[@]} items):"
    count=0
    for item in "${cache_items[@]}"; do
        if [[ $count -lt 5 ]]; then
            if [[ -d "$item" ]]; then
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    size=$(du -sk "$item" 2>/dev/null | cut -f1)
                    size=$((size * 1024))
                else
                    size=$(du -sb "$item" 2>/dev/null | cut -f1)
                fi
                echo "  - $(basename "$item")/ ($(format_size $size))"
            else
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    size=$(stat -f %z "$item")
                else
                    size=$(stat -c %s "$item")
                fi
                echo "  - $(basename "$item") ($(format_size $size))"
            fi
        fi
        ((count++))
    done
    if [[ ${#cache_items[@]} -gt 5 ]]; then
        echo "  ... and $((${#cache_items[@]} - 5)) more"
    fi
fi

# Show others
if [[ ${#others[@]} -gt 0 ]]; then
    echo ""
    echo -e "${BLUE}📄 Other files${NC} (${#others[@]} files):"
    count=0
    for file in "${others[@]}"; do
        if [[ $count -lt 5 ]]; then
            age=$(get_file_age_days "$file")
            if [[ "$OSTYPE" == "darwin"* ]]; then
                size=$(stat -f %z "$file")
            else
                size=$(stat -c %s "$file")
            fi
            echo "  - $(basename "$file") ($(format_size $size), $age days old)"
        fi
        ((count++))
    done
    if [[ ${#others[@]} -gt 5 ]]; then
        echo "  ... and $((${#others[@]} - 5)) more"
    fi
fi

echo ""

# Dry run - don't delete
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}[DRY RUN]${NC} No files were actually deleted"
    exit 0
fi

# Confirm deletion
echo -e "${YELLOW}⚠️  Total: ${#files_to_delete[@]} files, $(format_size $total_size)${NC}"
read -p "Delete these files? [y/N]: " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled - no files deleted"
    exit 0
fi

# Delete files and directories
success=0
errors=0

for item in "${files_to_delete[@]}"; do
    if [[ -d "$item" ]]; then
        if rm -rf "$item" 2>/dev/null; then
            ((success++))
        else
            echo -e "${RED}Error deleting $(basename "$item")/${NC}"
            ((errors++))
        fi
    else
        if rm -f "$item" 2>/dev/null; then
            ((success++))
        else
            echo -e "${RED}Error deleting $(basename "$item")${NC}"
            ((errors++))
        fi
    fi
done

echo ""
echo -e "${GREEN}✓ Deleted $success files${NC}"
if [[ $errors -gt 0 ]]; then
    echo -e "${RED}✗ $errors errors${NC}"
fi

# Show new status
show_status
