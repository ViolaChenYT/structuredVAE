import json
import gzip
import pandas as pd
import numpy as np
import subprocess
import sys
import os
import argparse
from pathlib import Path

def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)

def analyze_single_path(adata_path, obs_key='lineage', obsm_key='X_scVI'):
    """
    Analyze a single path using the analyze_order.py script.
    Returns correlation and p-value, or None, None if analysis fails.
    """
    try:
        import anndata
        from scipy.stats import spearmanr
        
        # Load data
        adata = anndata.read_h5ad(adata_path)
        if obs_key not in adata.obs.columns:
            raise KeyError(f"Column '{obs_key}' not found in adata.obs.")
        if obsm_key not in adata.obsm:
            raise KeyError(f"Key '{obsm_key}' not found in adata.obsm.")
        new_key = "lineage_category"
        # If there's a "/" in the string, get length of first part, otherwise get length of whole string
        adata.obs[new_key] = adata.obs[obs_key].apply(
            lambda x: len(x.split("/")[0]) if "/" in x else len(x)
        )
        lineage_nodes = adata.obs[new_key]
        embedding_1d = adata.obsm[obsm_key].flatten()
        # Convert to numeric
        lineage_nodes_numeric = pd.to_numeric(lineage_nodes)
        
        # Calculate correlation
        correlation, p_value = spearmanr(lineage_nodes_numeric, embedding_1d)
        
        # Return absolute correlation (like in analyze_order.py)
        return abs(correlation), p_value
        
    except Exception as e:
        print(f"Error analyzing {adata_path}: {e}")
        return None, None

def print_summary_statistics(df):
    """Print summary statistics for the results DataFrame."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Filter out NaN values for statistics
    valid_correlations = df['correlation'].dropna()
    valid_p_values = df['p_value'].dropna()
    
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
    
    print(f"\nTotal paths analyzed: {len(df)}")
    print(f"Successful analyses: {len(valid_correlations)}")
    print(f"Failed analyses: {len(df) - len(valid_correlations)}")

def run_analysis():
    """Run the full analysis."""
    # Paths
    paths_dict_file = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree.json.gz"
    
    # Load paths dictionary
    print("Loading paths dictionary...")
    paths_dict = load_paths_dict(paths_dict_file)
    print(f"Found {len(paths_dict)} paths to analyze.")
    
    # Prepare results list
    results = []
    
    # Analyze each path
    for path_name, path_nodes in paths_dict.items():
        print(f"Analyzing path: {path_name} of length: {len(path_nodes)}")
        
        # For now, we'll need to construct the adata path
        # This assumes the adata files follow a naming pattern
        # You may need to adjust this based on your actual file structure
        adata_path = f"/n/fs/ragr-data/users/viola/structuredVAE/data/scvi_path_{path_name}/trained.h5ad"
        
        # Check if file exists
        if not os.path.exists(adata_path):
            print(f"Warning: {adata_path} not found, skipping...")
            results.append({
                'path_name': path_name,
                'path_length': len(path_nodes),
                'correlation': np.nan,
                'p_value': np.nan
            })
            continue
        
        # Analyze the path
        correlation, p_value = analyze_single_path(adata_path)
        
        results.append({
            'path_name': path_name,
            'path_length': len(path_nodes),
            'correlation': correlation,
            'p_value': p_value
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_analysis_results.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Analyze scVI results')
    parser.add_argument('--rerun', action='store_true', 
                       help='Rerun the analysis even if results file exists')
    args = parser.parse_args()
    
    output_file = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_analysis_results.csv"
    
    # Check if results file exists and if we should rerun
    if os.path.exists(output_file) and not args.rerun:
        print(f"Results file exists: {output_file}")
        print("Loading existing results...")
        df = pd.read_csv(output_file)
        print(f"Loaded {len(df)} existing results")
        print_summary_statistics(df)
    else:
        if args.rerun:
            print("--rerun flag set, rerunning analysis...")
        else:
            print("No existing results found, running analysis...")
        
        df = run_analysis()
        print_summary_statistics(df)

if __name__ == '__main__':
    main()
