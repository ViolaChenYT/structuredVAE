# Snakemake Workflow for scVAE Plots Generation

This Snakemake workflow automates the generation of plots for all lineage paths using the `scvae-plots.py` script.

## Overview

The workflow processes all available lineage paths by:
1. Extracting path names from `.loom` files in the `data/` directory
2. Running `scvae-plots.py` for each path with the correct input and loom file paths
3. Collecting all generated plots in the `results/plots/` directory

## Files Structure

```
structuredVAE/
├── Snakefile                    # Main Snakemake workflow
├── config.yaml                  # Configuration file
├── run_snakemake.sh            # Execution script
├── scripts/
│   ├── scvae-plots.py          # Original plotting script
│   └── get_paths.py            # Helper script to extract path names
├── data/                       # Input data directory
│   ├── *.loom                  # Loom files for each path
│   └── scvae_path_*/           # scVAE results directories
├── results/                    # Output directory
│   └── plots/                  # Generated plots
└── logs/                       # Log files
```

## Usage

### Quick Start

```bash
# Run the workflow with default settings
./run_snakemake.sh

# Run with custom settings
./run_snakemake.sh --cores 8 --memory 16G --verbose
```

### Manual Snakemake Execution

```bash
# Dry run to see what would be executed
snakemake --dry-run

# Run with 4 cores
snakemake --cores 4

# Run with specific target
snakemake results/plots/AB_ABalaaaalal+{1}1_1dlatent_frequency_distribution.png
```

### Check Available Paths

```bash
# Get all path names
python scripts/get_paths.py

# Check which paths have scVAE results
python scripts/get_paths.py --check_scvae

# Save path information to file
python scripts/get_paths.py --check_scvae --output paths_info.json
```

## Configuration

Edit `config.yaml` to customize:
- Data directory paths
- Output directory structure
- Default parameters for scVAE
- Snakemake execution settings

## Workflow Rules

### Main Rules

1. **`all`**: Default target rule that generates all plots
2. **`generate_plots`**: Runs `scvae-plots.py` for each path
3. **`check_scvae_results`**: Verifies scVAE results exist
4. **`generate_summary`**: Creates a summary report

### Input Requirements

For each path, the workflow expects:
- Loom file: `data/{path_name}.loom`
- scVAE results in: `data/scvae_path_{path_name}/{path_name}/no_split/no_preprocessing/GMVAE/gaussian_mixture-c_6/zero_inflated_negative_binomial-l_1-h_64_64-mc_1-iw_1-bn-wu_200/e_500-mc_1-iw_1/full/`
- Required files: `latent_values-y.tsv.gz`, `latent_values-z.tsv.gz`

### Output Files

For each path, generates:
- `results/plots/{path_name}_1dlatent_frequency_distribution.png`
- `results/plots/{path_name}_1dlatent_frequency_distribution_by_lineage.png`

## Command Line Options

### run_snakemake.sh Options

- `--cores N`: Number of cores to use (default: 4)
- `--memory SIZE`: Memory limit per job (default: 8G)
- `--timeout SEC`: Timeout in seconds (default: 3600)
- `--dry-run`: Show what would be done without executing
- `--force`: Force execution even if outputs exist
- `--verbose`: Enable verbose output

### Snakemake Options

- `--cores N`: Number of cores for parallel execution
- `--dry-run`: Show execution plan without running
- `--forceall`: Force execution of all rules
- `--keep-going`: Continue execution even if some jobs fail
- `--rerun-incomplete`: Rerun incomplete jobs

## Troubleshooting

### Common Issues

1. **Missing scVAE results**: Ensure the scVAE training has completed for all paths
2. **Permission errors**: Check file permissions in the data directory
3. **Memory issues**: Reduce `--cores` or increase `--memory`
4. **Timeout errors**: Increase `--timeout` value

### Log Files

- `logs/snakemake.log`: Main Snakemake execution log
- `logs/{path_name}_scvae_plots.log`: Individual job logs
- `results/summary_report.txt`: Summary of processed paths

### Debugging

```bash
# Run with verbose output
./run_snakemake.sh --verbose

# Check specific path
snakemake --dry-run results/plots/{path_name}_1dlatent_frequency_distribution.png

# Force rerun of failed jobs
snakemake --rerun-incomplete --cores 1
```

## Performance Tips

1. **Parallel execution**: Use `--cores` to run multiple paths simultaneously
2. **Memory management**: Adjust `--memory` based on available system memory
3. **Incremental runs**: Snakemake only reruns jobs when inputs change
4. **Resource limits**: Use `--resources` to limit resource usage per job

## Example Commands

```bash
# Check what paths are available
python scripts/get_paths.py --check_scvae --available_only

# Run workflow for all available paths
./run_snakemake.sh --cores 8 --memory 16G

# Dry run to see execution plan
./run_snakemake.sh --dry-run --verbose

# Force rerun everything
./run_snakemake.sh --force --cores 4
```
