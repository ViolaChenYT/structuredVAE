# Combined scVAE Training and Plotting Workflow

This Snakemake workflow **parallelizes** the slow `scvae_run.py` process and then runs plotting on the generated results.

## 🎯 **Workflow Overview**

```
1. Load paths from paths_dict.json.gz
2. For each path in parallel:
   ├── Train scVAE (slow, but parallelized!)
   ├── Generate latent files  
   └── Create completion marker
3. For each completed scVAE training:
   ├── Run scvae-plots.py
   └── Generate plots
4. Generate summary report
```

## 🚀 **Key Benefits**

- **Parallelizes scVAE training** (the bottleneck in your original workflow)
- **Runs plotting automatically** after scVAE training completes
- **Resource management** - controls memory and CPU usage per job
- **Fault tolerance** - continues even if some jobs fail
- **Resumable** - can restart from where it left off
- **Uses existing conda environment** - leverages your existing `scvae` environment

## 📁 **Files**

- **`Snakefile`** - Main workflow definition
- **`config.yaml`** - Configuration file
- **`run_workflow.sh`** - Easy execution script
- **`scripts/train_single_path.py`** - Python script for single path training
- **`README.md`** - This documentation

## 🚀 **Quick Start**

### **Basic Usage:**
```bash
# Run with default settings (8 cores, 32GB memory)
./run_workflow.sh

# Run with custom settings
./run_workflow.sh --cores 16 --memory 64G --max-scvae-jobs 8
```

### **Check What Would Be Done:**
```bash
# Dry run to see execution plan
./run_workflow.sh --dry-run --verbose
```

## ⚙️ **Configuration Options**

### **Resource Management:**
```bash
# Total system resources
--cores 16                    # Total CPU cores
--memory 64G               # Total memory

# Job concurrency limits  
--max-scvae-jobs 8           # Max scVAE training jobs in parallel
--max-plotting-jobs 16       # Max plotting jobs in parallel
```

### **Execution Control:**
```bash
--dry-run                    # Show what would be done
--force                      # Force rerun everything
--verbose                    # Detailed output
--cleanup                    # Clean up intermediate files
```

## 📈 **Performance Optimization**

### **Memory Allocation:**
- **scVAE training**: 8GB per job (configurable)
- **Plotting**: 2GB per job (configurable)
- **Total memory**: Should be `max_scvae_jobs × 8GB + max_plotting_jobs × 2GB`

### **Example Configurations:**

**Small System (16GB RAM, 8 cores):**
```bash
./run_workflow.sh --cores 8 --memory 16G --max-scvae-jobs 2 --max-plotting-jobs 4
```

**Large System (64GB RAM, 16 cores):**
```bash
./run_workflow.sh --cores 16 --memory 64G --max-scvae-jobs 8 --max-plotting-jobs 16
```

## 📊 **Expected Output**

```
data/
├── {path_name}.loom                           # Generated loom files
└── scvae_path_{path_name}/
    └── {path_name}/
        ├── no_split/no_preprocessing/GMVAE/.../full/
        │   ├── latent_values-y.tsv.gz
        │   └── latent_values-z.tsv.gz
        └── scvae_training_complete.txt        # Completion marker

results/plots/
├── {path_name}_1dlatent_frequency_distribution.png
└── {path_name}_1dlatent_frequency_distribution_by_lineage.png
```

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **Out of Memory:**
   ```bash
   # Reduce concurrent jobs
   ./run_workflow.sh --max-scvae-jobs 2 --max-plotting-jobs 4
   ```

2. **Conda Environment Issues:**
   ```bash
   # Check if your scvae environment exists
   conda env list | grep scvae
   
   # If it doesn't exist, create it with the packages you need
   conda create -n scvae python=3.9 pandas numpy scipy matplotlib seaborn scanpy anndata scvi-tools pytorch
   pip install scvae
   ```

3. **Jobs Timing Out:**
   - Check `config.yaml` for timeout settings
   - Increase memory allocation per job

4. **Missing Input Files:**
   - Verify `paths_dict.json.gz` exists
   - Check `packer2019_preprocessed.h5ad` is in `data/`

### **Resume Failed Jobs:**
```bash
# Resume from where it left off
./run_workflow.sh --cores 8 --memory 32G

# Force rerun everything
./run_workflow.sh --force --cores 8 --memory 32G
```

### **Debug Specific Path:**
```bash
# Run just one path
snakemake --cores 1 data/scvae_path_{path_name}/{path_name}/scvae_training_complete.txt
```

## 📊 **Monitoring Progress**

### **Check Job Status:**
```bash
# See running jobs
snakemake --dry-run

# Check completion markers
find data -name "scvae_training_complete.txt" | wc -l

# Check generated plots
find results/plots -name "*.png" | wc -l
```

### **View Logs:**
```bash
# Main workflow log
tail -f logs/snakemake.log

# Specific job logs
tail -f logs/{path_name}_scvae_training.log
tail -f logs/{path_name}_scvae_plots.log
```

## 🎯 **Expected Performance**

**Original `scvae_run.py`**: Sequential execution
- Time: `N_paths × scVAE_training_time`
- Memory: Single job memory usage

**Parallel Workflow**: 
- Time: `max(scVAE_training_time, plotting_time)` (much faster!)
- Memory: `max_scvae_jobs × 8GB` (controlled)

**Example**: 100 paths, 30min each
- **Sequential**: 50 hours
- **Parallel (8 jobs)**: ~6.25 hours

## 🚀 **Advanced Usage**

### **Custom Resource Limits:**
Edit `config.yaml`:
```yaml
scvae_training:
  threads: 4
  memory_mb: 16384  # 16GB
  timeout: 10800    # 3 hours
```

### **Manual Conda Environment Setup:**
```bash
# Check if your scvae environment exists
conda env list | grep scvae

# If it doesn't exist, create it
conda create -n scvae python=3.9 pandas numpy scipy matplotlib seaborn scanpy anndata scvi-tools pytorch
pip install scvae

# Activate and test
conda activate scvae
python scripts/train_single_path.py --help
```

This workflow should dramatically speed up your scVAE training and plotting process by running multiple paths in parallel!
