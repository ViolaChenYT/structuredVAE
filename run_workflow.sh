#!/bin/bash
# Script to run the combined scVAE training and plotting workflow

# Set default values
CORES=8
MEMORY="32G"
MAX_SCVAE_JOBS=4
MAX_PLOTTING_JOBS=8
DRY_RUN=false
FORCE_RUN=false
VERBOSE=false
CLEANUP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cores)
            CORES="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --max-scvae-jobs)
            MAX_SCVAE_JOBS="$2"
            shift 2
            ;;
        # --max-plotting-jobs)
        #     MAX_PLOTTING_JOBS="$2"
        #     shift 2
        #     ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cores N              Total number of cores to use (default: 8)"
            echo "  --memory SIZE          Total memory available (default: 32G)"
            echo "  --max-scvae-jobs N     Max scVAE training jobs in parallel (default: 4)"
            # echo "  --max-plotting-jobs N  Max plotting jobs in parallel (default: 8)"
            echo "  --dry-run              Show what would be done without executing"
            echo "  --force                Force execution even if outputs exist"
            echo "  --verbose              Enable verbose output"
            echo "  --cleanup              Clean up intermediate files after completion"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p logs
mkdir -p results/plots
mkdir -p data

# Print configuration
echo "=== Combined scVAE Workflow Configuration ==="
echo "Total cores: $CORES"
echo "Total memory: $MEMORY"
echo "Max scVAE jobs: $MAX_SCVAE_JOBS"
# echo "Max plotting jobs: $MAX_PLOTTING_JOBS"
echo "Dry run: $DRY_RUN"
echo "Force: $FORCE_RUN"
echo "Verbose: $VERBOSE"
echo "Cleanup: $CLEANUP"
echo "============================================="
echo

# Check if Snakemake is available
if ! command -v snakemake &> /dev/null; then
    echo "Error: Snakemake is not installed or not in PATH"
    echo "Please install Snakemake: pip install snakemake"
    exit 1
fi

# Check if required files exist
if [ ! -f "Snakefile" ]; then
    echo "Error: Snakefile not found in current directory"
    exit 1
fi

if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found in current directory"
    exit 1
fi

if [ ! -f "scripts/train_single_path.py" ]; then
    echo "Error: scripts/train_single_path.py not found"
    exit 1
fi

# Check if input data files exist
if [ ! -f "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict1.json.gz" ]; then
    echo "Error: paths_dict.json.gz not found"
    echo "Expected location: /n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict.json.gz"
    exit 1
fi

if [ ! -f "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/original_to_merged1.csv" ]; then
    echo "Error: original_to_merged.csv not found"
    echo "Expected location: /n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/original_to_merged1.csv"
    exit 1
fi

if [ ! -f "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/data/fuzzy_lineage_mapping.json" ]; then
    echo "Error: fuzzy_lineage_mapping.json not found"
    echo "Expected location: /n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/data/fuzzy_lineage_mapping.json"
    exit 1
fi

if [ ! -f "data/packer2019_preprocessed.h5ad" ]; then
    echo "Error: packer2019_preprocessed.h5ad not found"
    echo "Expected location: data/packer2019_preprocessed.h5ad"
    exit 1
fi

# Build Snakemake command
SNAKEMAKE_CMD="snakemake --configfile config.yaml --use-conda --conda-frontend mamba"

# Add cores
SNAKEMAKE_CMD="$SNAKEMAKE_CMD --cores $CORES"

# Add resource limits
SNAKEMAKE_CMD="$SNAKEMAKE_CMD --resources scvae_jobs=$MAX_SCVAE_JOBS"

# Add memory limit
MEMORY_MB=$(( $(echo $MEMORY | sed 's/G//') * 1024 ))
SNAKEMAKE_CMD="$SNAKEMAKE_CMD --resources mem_mb=$MEMORY_MB"

# Add dry run if requested
if [ "$DRY_RUN" = true ]; then
    SNAKEMAKE_CMD="$SNAKEMAKE_CMD --dry-run"
fi

# Add force if requested
if [ "$FORCE_RUN" = true ]; then
    SNAKEMAKE_CMD="$SNAKEMAKE_CMD --forceall"
fi

# Add verbose if requested
if [ "$VERBOSE" = true ]; then
    SNAKEMAKE_CMD="$SNAKEMAKE_CMD --verbose"
fi

# Note: Logging is handled by individual job logs in logs/ directory

# Add error handling
SNAKEMAKE_CMD="$SNAKEMAKE_CMD --keep-going"

# Add latency wait for file system
SNAKEMAKE_CMD="$SNAKEMAKE_CMD --latency-wait 30"

# Add performance optimizations
SNAKEMAKE_CMD="$SNAKEMAKE_CMD --rerun-incomplete"
SNAKEMAKE_CMD="$SNAKEMAKE_CMD --local-cores $CORES"

# Print the command that will be executed
echo "Running Snakemake workflow..."
echo "Command: $SNAKEMAKE_CMD"
echo

# Run Snakemake and capture output
echo "Starting Snakemake execution..."
eval $SNAKEMAKE_CMD 2>&1 | tee logs/snakemake.log

# Check exit status
if [ $? -eq 0 ]; then
    echo
    echo "=== scVAE Training Workflow completed successfully! ==="
    echo "scVAE training results: data/scvae_path_*/"
    echo "Filtered data files: data/*.loom"
    echo "Logs: logs/"
    
    if [ -d "data" ]; then
        SCVAE_COUNT=$(find data -name "scvae_training_complete.txt" | wc -l)
        echo "Completed $SCVAE_COUNT scVAE training jobs"
    fi
    
    # Cleanup if requested
    if [ "$CLEANUP" = true ]; then
        echo "Cleaning up intermediate files..."
        find logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
        echo "Cleanup completed"
    fi
else
    echo
    echo "=== scVAE Training Workflow failed! ==="
    echo "Check the logs for error details:"
    echo "  - Snakemake log: logs/snakemake.log"
    echo "  - Individual job logs: logs/*_scvae_training.log"
    echo
    echo "To resume from where it left off, run:"
    echo "  $0 --cores $CORES --memory $MEMORY"
    exit 1
fi
