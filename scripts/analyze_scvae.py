import json
import gzip
import pandas as pd
import numpy as np
import subprocess
import sys
import os
from pathlib import Path
import glob

def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)

def find_scvae_latent_files(base_dir, paths_dict, analyze_all=False):
    """
    Find latent_values-z.tsv.gz files for scVAE paths.
    
    If analyze_all=False: Only processes paths defined in paths_dict
    If analyze_all=True: Finds all scvae_path_* directories and processes them regardless of paths_dict
    
    Returns a list of tuples: (path_name, latent_file_path)
    """
    results = []
    
    if analyze_all:
        # Find all scvae_path_* directories
        scvae_pattern = os.path.join(base_dir, "scvae_path_*")
        scvae_dirs = glob.glob(scvae_pattern)
        
        for scvae_dir in scvae_dirs:
            # Extract path name from directory name
            dir_name = os.path.basename(scvae_dir)
            if dir_name.startswith("scvae_path_"):
                path_name = dir_name.replace("scvae_path_", "")
                
                # Search recursively for latent_values-z.tsv.gz files within this path's directory
                pattern = os.path.join(scvae_dir, "**", "latent_values-z.tsv.gz")
                latent_files = glob.glob(pattern, recursive=True)
                
                if latent_files:
                    # Take the first found file (assuming there should be only one per path)
                    latent_file_path = latent_files[0]
                    results.append((path_name, latent_file_path))
                    print(f"Found latent file for path {path_name}: {latent_file_path}")
                else:
                    print(f"Warning: No latent file found for path {path_name} in directory: {scvae_dir}")
    else:
        # Original behavior: only process paths in paths_dict
        for path_name in paths_dict.keys():
            # Construct the base directory path for this path
            path_base_dir = os.path.join(base_dir, f"scvae_path_{path_name}", path_name)
            
            if not os.path.exists(path_base_dir):
                print(f"Warning: Base directory not found for path {path_name}: {path_base_dir}")
                continue
            
            # Search recursively for latent_values-z.tsv.gz files within this path's directory
            pattern = os.path.join(path_base_dir, "**", "latent_values-z.tsv.gz")
            latent_files = glob.glob(pattern, recursive=True)
            
            if latent_files:
                # Take the first found file (assuming there should be only one per path)
                latent_file_path = latent_files[0]
                results.append((path_name, latent_file_path))
                print(f"Found latent file for path {path_name}: {latent_file_path}")
            else:
                print(f"Warning: No latent file found for path {path_name} in directory: {path_base_dir}")
    
    return results

def load_latent_z(latent_file_path):
    """
    Load latent Z values from tsv.gz file.
    Returns DataFrame with latent embeddings.
    """
    try:
        Z_df = pd.read_csv(latent_file_path, sep="\t", compression="infer")
        
        # Handle potential index column (like in your extraction script)
        if Z_df.dtypes.iloc[0] == "object":
            Z_df = Z_df.set_index(Z_df.columns[0])
        
        # Select only numeric columns
        Z_df = Z_df.select_dtypes(include=[np.number])
        
        return Z_df
        
    except Exception as e:
        print(f"Error loading latent file {latent_file_path}: {e}")
        return None

def align_to_adata(adata, df_or_arr, name):
    """Align DataFrame or array to adata object"""
    if isinstance(df_or_arr, pd.DataFrame):
        if df_or_arr.index.dtype == object and len(set(df_or_arr.index) & set(adata.obs_names)) > 0:
            arr = df_or_arr.reindex(adata.obs_names).to_numpy()
        else:
            arr = df_or_arr.to_numpy()
    elif isinstance(df_or_arr, pd.Series):
        if df_or_arr.index.dtype == object and len(set(df_or_arr.index) & set(adata.obs_names)) > 0:
            ser = df_or_arr.reindex(adata.obs_names)
        else:
            ser = df_or_arr
        adata.obs[name] = ser.astype(str).values
        return
    else:
        arr = np.asarray(df_or_arr)
    
    assert arr.shape[0] == adata.n_obs, f"{name}: row count {arr.shape[0]} != adata {adata.n_obs}"
    adata.obsm[name] = arr.astype(np.float32)

def load_y_latent(latent_file_path):
    """
    Load y latent values (cluster assignments) from tsv.gz file.
    Returns DataFrame with cluster assignments.
    """
    try:
        # Replace 'z' with 'y' in the file path
        y_file_path = latent_file_path.replace('latent_values-z.tsv.gz', 'latent_values-y.tsv.gz')
        
        if not os.path.exists(y_file_path):
            print(f"Warning: Y latent file not found: {y_file_path}")
            return None
            
        Y_df = pd.read_csv(y_file_path, sep="\t", compression="infer")
        
        # Handle potential index column
        if Y_df.dtypes.iloc[0] == "object":
            Y_df = Y_df.set_index(Y_df.columns[0])
        
        # Select only numeric columns
        Y_df = Y_df.select_dtypes(include=[np.number])
        
        return Y_df
        
    except Exception as e:
        print(f"Error loading Y latent file {y_file_path}: {e}")
        return None

def calculate_ari(cluster_labels, lineage_labels):
    """
    Calculate Adjusted Rand Index between cluster labels and lineage labels.
    """
    try:
        from sklearn.metrics import adjusted_rand_score
        
        # Convert to numpy arrays and handle any missing values
        cluster_array = np.array(cluster_labels)
        lineage_array = np.array(lineage_labels)
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(cluster_array) | np.isnan(lineage_array))
        if not np.any(valid_mask):
            return np.nan
            
        cluster_clean = cluster_array[valid_mask]
        lineage_clean = lineage_array[valid_mask]
        
        # Calculate ARI
        ari = adjusted_rand_score(cluster_clean, lineage_clean)
        return ari
        
    except Exception as e:
        print(f"Error calculating ARI: {e}")
        return np.nan

def analyze_single_scvae_path(path_name, latent_file_path, base_dir, paths_dict):
    """
    Analyze a single scVAE path using latent Z embeddings.
    Returns correlation, p-value, and ARI, or None, None, None if analysis fails.
    """
    try:
        from scipy.stats import spearmanr
        
        # Load latent Z values
        Z_df = load_latent_z(latent_file_path)
        if Z_df is None:
            return None, None, None
        
        # Load Y latent values (cluster assignments)
        Y_df = load_y_latent(latent_file_path)
        
        # Get path length from paths dictionary
        path_length = len(paths_dict.get(path_name, []))
        
        # Try to load the corresponding loom file
        loom_file = os.path.join(base_dir, f"{path_name}.loom")
        if not os.path.exists(loom_file):
            print(f"Warning: Loom file not found for path {path_name}: {loom_file}")
            return None, None, None
        
        # Load the loom file using anndata
        import anndata
        try:
            adata = anndata.read_loom(loom_file)
        except Exception as e:
            print(f"Error loading loom file {loom_file}: {e}")
            return None, None, None
        
        # Create lineage category (same logic as analyze_scvi.py)
        new_key = "lineage_category"
        if 'lineage' in adata.obs.columns:
            adata.obs[new_key] = adata.obs['lineage'].apply(
                lambda x: len(x.split("/")[0]) if "/" in x else len(x)
            )
        else:
            print(f"Warning: No lineage column found in {loom_file}")
            return None, None, None
        
        # Use your alignment approach for Z
        align_to_adata(adata, Z_df, "X_scvae_Z")
        
        # Now we can calculate correlation
        lineage_numeric = pd.to_numeric(adata.obs['lineage_category'])
        embedding_1d = adata.obsm['X_scvae_Z'].flatten()
        
        # Calculate Spearman correlation using scipy
        correlation, p_value = spearmanr(lineage_numeric, embedding_1d)
        
        # Calculate ARI if Y latent is available
        ari = np.nan
        if Y_df is not None:
            # Align Y latent with adata
            align_to_adata(adata, Y_df, "X_scvae_Y")
            
            # Get cluster assignments (assuming single column for hard assignments)
            if adata.obsm['X_scvae_Y'].shape[1] == 1:
                cluster_labels = adata.obsm['X_scvae_Y'].flatten()
            else:
                # If multiple columns, take argmax for hard assignments
                cluster_labels = np.argmax(adata.obsm['X_scvae_Y'], axis=1)
            
            # Calculate ARI between cluster assignments and lineage labels
            ari = calculate_ari(cluster_labels, lineage_numeric)
        
        # Return absolute correlation (like in analyze_scvi.py)
        return abs(correlation), p_value, ari
        
    except Exception as e:
        print(f"Error analyzing {path_name}: {e}")
        return None, None, None

def main():
    # Configuration
    ANALYZE_ALL_SCVAE_PATHS = False  # Set to True to analyze all scVAE paths, False to only analyze paths in paths_dict
    
    # Paths
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    paths_dict_file = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree.json.gz"
    adata_file = "/n/fs/ragr-data/users/viola/structuredVAE/data/packer2019_preprocessed.h5ad"
    
    # Load paths dictionary
    print("Loading paths dictionary...")
    paths_dict = load_paths_dict(paths_dict_file)
    print(f"Found {len(paths_dict)} paths in dictionary.")
    
    # Find all scVAE latent files
    print("Finding scVAE latent files...")
    if ANALYZE_ALL_SCVAE_PATHS:
        print("Mode: Analyzing ALL scVAE paths (regardless of paths_dict)")
    else:
        print("Mode: Analyzing only paths defined in paths_dict")
    
    latent_files = find_scvae_latent_files(base_dir, paths_dict, analyze_all=ANALYZE_ALL_SCVAE_PATHS)
    print(f"Found {len(latent_files)} scVAE latent files.")
    
    # Prepare results list
    results = []
    
    # Analyze each scVAE path
    for path_name, latent_file_path in latent_files:
        print(f"Analyzing scVAE path: {path_name}")
        
        # Check if path exists in paths_dict
        if path_name in paths_dict:
            path_length = len(paths_dict[path_name])
        else:
            path_length = 0  # Unknown path length if not in paths_dict
            print(f"Warning: Path {path_name} not found in paths_dict, using path_length=0")
        
        # Analyze the path
        correlation, p_value, ari = analyze_single_scvae_path(
            path_name, latent_file_path, base_dir, paths_dict
        )
        
        results.append({
            'path_name': path_name,
            'path_length': path_length,
            'correlation': correlation,
            'p_value': p_value,
            'ari': ari
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    if ANALYZE_ALL_SCVAE_PATHS:
        output_file = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvae_analysis_results_all.csv"
    else:
        output_file = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvae_analysis_results.csv"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Calculate summary statistics
    print("\n" + "="*60)
    print("SCVAE ANALYSIS SUMMARY")
    print("="*60)
    
    # Handle empty results
    if len(df) == 0:
        print("No scVAE results found.")
        return
    
    # Filter out NaN values for statistics
    valid_correlations = df['correlation'].dropna()
    valid_p_values = df['p_value'].dropna()
    valid_aris = df['ari'].dropna()
    
    if len(valid_correlations) > 0:
        print(f"\nCORRELATION STATISTICS (n={len(valid_correlations)}):")
        print(f"  Mean: {valid_correlations.mean():.4f}")
        print(f"  Median: {valid_correlations.median():.4f}")
        print(f"  25th percentile: {valid_correlations.quantile(0.25):.4f}")
        print(f"  75th percentile: {valid_correlations.quantile(0.75):.4f}")
        print(f"  Standard deviation: {valid_correlations.std():.4f}")
        print(f"  Min: {valid_correlations.min():.4f}")
        print(f"  Max: {valid_correlations.max():.4f}")
    
    if len(valid_p_values) > 0:
        print(f"\nP-VALUE STATISTICS (n={len(valid_p_values)}):")
        print(f"  Mean: {valid_p_values.mean():.4e}")
        print(f"  Median: {valid_p_values.median():.4e}")
        print(f"  25th percentile: {valid_p_values.quantile(0.25):.4e}")
        print(f"  75th percentile: {valid_p_values.quantile(0.75):.4e}")
        print(f"  Standard deviation: {valid_p_values.std():.4e}")
        print(f"  Min: {valid_p_values.min():.4e}")
        print(f"  Max: {valid_p_values.max():.4e}")
    
    if len(valid_aris) > 0:
        print(f"\nARI STATISTICS (n={len(valid_aris)}):")
        print(f"  Mean: {valid_aris.mean():.4f}")
        print(f"  Median: {valid_aris.median():.4f}")
        print(f"  25th percentile: {valid_aris.quantile(0.25):.4f}")
        print(f"  75th percentile: {valid_aris.quantile(0.75):.4f}")
        print(f"  Standard deviation: {valid_aris.std():.4f}")
        print(f"  Min: {valid_aris.min():.4f}")
        print(f"  Max: {valid_aris.max():.4f}")
    
    print(f"\nTotal scVAE paths analyzed: {len(df)}")
    print(f"Successful analyses: {len(valid_correlations)}")
    print(f"Failed analyses: {len(df) - len(valid_correlations)}")
    print(f"ARI analyses: {len(valid_aris)}")
    
    # Create comparison with scVI if scVI results exist
    scvi_results_file = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_analysis_results.csv"
    if os.path.exists(scvi_results_file):
        print("\n" + "="*60)
        print("CREATING SCVAE vs SCVI COMPARISON")
        print("="*60)
        
        # Load scVI results
        scvi_df = pd.read_csv(scvi_results_file)
        print(f"Loaded {len(scvi_df)} scVI results")
        
        # Create comparison
        comparison_results = []
        for _, scvae_row in df.iterrows():
            path_name = scvae_row['path_name']
            
            # Find corresponding scVI result
            scvi_match = scvi_df[scvi_df['path_name'] == path_name]
            
            if len(scvi_match) > 0:
                scvi_row = scvi_match.iloc[0]
                comparison_results.append({
                    'path_name': path_name,
                    'path_length': scvae_row['path_length'],
                    'scvae_correlation': scvae_row['correlation'],
                    'scvae_p_value': scvae_row['p_value'],
                    'scvae_ari': scvae_row['ari'],
                    'scvi_correlation': scvi_row['correlation'],
                    'scvi_p_value': scvi_row['p_value']
                })
        
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            
            # Save comparison
            if ANALYZE_ALL_SCVAE_PATHS:
                comparison_file = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvae_vs_scvi_comparison_all.csv"
            else:
                comparison_file = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvae_vs_scvi_comparison.csv"
            
            comparison_df.to_csv(comparison_file, index=False)
            print(f"Comparison saved to: {comparison_file}")
            
            # Calculate comparison statistics
            valid_comparison = comparison_df.dropna(subset=['scvae_correlation', 'scvi_correlation'])
            if len(valid_comparison) > 0:
                print(f"\nCOMPARISON STATISTICS (n={len(valid_comparison)}):")
                print(f"  scVAE mean correlation: {valid_comparison['scvae_correlation'].mean():.4f}")
                print(f"  scVI mean correlation: {valid_comparison['scvi_correlation'].mean():.4f}")
                print(f"  Mean difference (scVAE - scVI): {(valid_comparison['scvae_correlation'] - valid_comparison['scvi_correlation']).mean():.4f}")
                
                # Paired t-test
                from scipy.stats import ttest_rel
                t_stat, p_val = ttest_rel(valid_comparison['scvae_correlation'], valid_comparison['scvi_correlation'])
                print(f"  Paired t-test p-value: {p_val:.4e}")
                
                # Count where scVAE > scVI
                scvae_better = (valid_comparison['scvae_correlation'] > valid_comparison['scvi_correlation']).sum()
                print(f"  Paths where scVAE > scVI: {scvae_better}/{len(valid_comparison)}")
        else:
            print("No overlapping paths found between scVAE and scVI results")
    else:
        print(f"\nscVI results file not found: {scvi_results_file}")
        print("Skipping comparison analysis")

if __name__ == '__main__':
    main()
