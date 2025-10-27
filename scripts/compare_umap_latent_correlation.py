#!/usr/bin/env python3
"""
Compare 1D UMAP and 1D PCA from raw expression data vs learned latent coordinates from scVI and scVAE.

This script:
1. Loads existing correlation results from scVI and scVAE analysis CSV files
2. For each path, computes 1D UMAP and 1D PCA from raw expression data (adata.X) 
3. Calculates Spearman correlations between 1D UMAP/1D PCA and lineage
4. Compares all four methods (scVI, scVAE, 1D UMAP, 1D PCA) in terms of mean correlation and distribution
5. Generates comparison plots and summary statistics
"""

import os
import json
import gzip
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import these only when needed to avoid early import errors
try:
    import anndata
    import scipy.sparse as sp
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_AVAILABLE = False

def spearman_correlation(x, y):
    """
    Calculate Spearman correlation coefficient manually to avoid scipy import issues.
    
    Parameters:
    -----------
    x, y : array-like
        Input arrays
    
    Returns:
    --------
    tuple
        (correlation, p_value)
    """
    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return np.nan, np.nan
    
    # Calculate ranks
    x_ranks = np.argsort(np.argsort(x_clean))
    y_ranks = np.argsort(np.argsort(y_clean))
    
    # Calculate Pearson correlation of ranks
    n = len(x_ranks)
    x_mean = np.mean(x_ranks)
    y_mean = np.mean(y_ranks)
    
    numerator = np.sum((x_ranks - x_mean) * (y_ranks - y_mean))
    x_var = np.sum((x_ranks - x_mean) ** 2)
    y_var = np.sum((y_ranks - y_mean) ** 2)
    
    if x_var == 0 or y_var == 0:
        return np.nan, np.nan
    
    correlation = numerator / np.sqrt(x_var * y_var)
    
    # Simple p-value approximation (not exact but reasonable for large n)
    if n > 3:
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        # This is a rough approximation
        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + 1))
    else:
        p_value = 1.0
    
    return correlation, p_value

def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)


def compute_1d_umap_from_raw(adata, n_pca_components=50, random_state=42):
    """
    Compute 1D UMAP from raw expression data.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with expression data
    n_pca_components : int
        Number of PCA components to use
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    np.array
        1D UMAP coordinates
    """
    # Get expression data
    if adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X
    
    # Convert to dense if sparse
    if sp.issparse(X):
        X = X.toarray()
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    # Perform 1D UMAP
    umap_1d = umap.UMAP(n_components=1, random_state=random_state, n_neighbors=15, min_dist=0.1)
    X_umap_1d = umap_1d.fit_transform(X_pca)
    
    return X_umap_1d.flatten()


def compute_1d_pca_from_raw(adata, n_pca_components=50, random_state=42):
    """
    Compute 1D PCA from raw expression data.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with expression data
    n_pca_components : int
        Number of PCA components to use for preprocessing
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    np.array
        1D PCA coordinates (first principal component)
    """
    # Get expression data
    if adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X
    
    # Convert to dense if sparse
    if sp.issparse(X):
        X = X.toarray()
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA and get first component
    pca = PCA(n_components=1, random_state=random_state)
    X_pca_1d = pca.fit_transform(X_scaled)
    
    return X_pca_1d.flatten()


def load_existing_results(scvi_results_file, scvae_results_file):
    """
    Load existing correlation results from scVI and scVAE analysis files.
    
    Parameters:
    -----------
    scvi_results_file : str
        Path to scVI results CSV file
    scvae_results_file : str
        Path to scVAE results CSV file
    
    Returns:
    --------
    tuple
        (scvi_df, scvae_df, common_paths)
    """
    print("Loading existing correlation results...")
    
    # Load scVI results
    if os.path.exists(scvi_results_file):
        scvi_df = pd.read_csv(scvi_results_file)
        print(f"Loaded {len(scvi_df)} scVI results")
    else:
        print(f"Warning: scVI results file not found: {scvi_results_file}")
        scvi_df = pd.DataFrame()
    
    # Load scVAE results
    if os.path.exists(scvae_results_file):
        scvae_df = pd.read_csv(scvae_results_file)
        print(f"Loaded {len(scvae_df)} scVAE results")
    else:
        print(f"Warning: scVAE results file not found: {scvae_results_file}")
        scvae_df = pd.DataFrame()
    
    # Find common paths
    if len(scvi_df) > 0 and len(scvae_df) > 0:
        scvi_paths = set(scvi_df['path_name'])
        scvae_paths = set(scvae_df['path_name'])
        common_paths = list(scvi_paths.intersection(scvae_paths))
        print(f"Found {len(common_paths)} common paths between scVI and scVAE")
    else:
        common_paths = []
        print("No common paths found")
    
    return scvi_df, scvae_df, common_paths

def analyze_umap_for_path(path_name, base_dir, paths_dict):
    """
    Analyze 1D UMAP correlation for a single path.
    
    Parameters:
    -----------
    path_name : str
        Name of the path
    base_dir : str
        Base directory containing the data
    paths_dict : dict
        Paths dictionary
    
    Returns:
    --------
    tuple
        (correlation, p_value) or (None, None) if analysis fails
    """
    try:
        from scipy.stats import spearmanr
        
        # Load the loom file for this path
        loom_file = os.path.join(base_dir, f"{path_name}.loom")
        if not os.path.exists(loom_file):
            print(f"Warning: Loom file not found for path {path_name}: {loom_file}")
            return None, None
        
        # Load the loom file using anndata
        import anndata
        try:
            adata = anndata.read_loom(loom_file)
        except Exception as e:
            print(f"Error loading loom file {loom_file}: {e}")
            return None, None
        
        # Create lineage category (same logic as analyze_scvae.py)
        new_key = "lineage_category"
        if 'lineage' in adata.obs.columns:
            adata.obs[new_key] = adata.obs['lineage'].apply(
                lambda x: len(x.split("/")[0]) if "/" in x else len(x)
            )
        else:
            print(f"Warning: No lineage column found in {loom_file}")
            return None, None
        
        # Compute 1D UMAP from raw expression data
        print(f"Computing 1D UMAP for {path_name}...")
        umap_coords = compute_1d_umap_from_raw(adata)
        
        # Calculate Spearman correlation between UMAP and lineage
        lineage_numeric = pd.to_numeric(adata.obs['lineage_category'])
        correlation, p_value = spearmanr(lineage_numeric, umap_coords)
        
        # Return absolute correlation (like in analyze_scvae.py)
        return abs(correlation), p_value
        
    except Exception as e:
        print(f"Error analyzing UMAP for {path_name}: {e}")
        return None, None


def analyze_pca_for_path(path_name, base_dir, paths_dict):
    """
    Analyze 1D PCA correlation for a single path.
    
    Parameters:
    -----------
    path_name : str
        Name of the path
    base_dir : str
        Base directory containing the data
    paths_dict : dict
        Paths dictionary
    
    Returns:
    --------
    tuple
        (correlation, p_value) or (None, None) if analysis fails
    """
    try:
        from scipy.stats import spearmanr
        
        # Load the loom file for this path
        loom_file = os.path.join(base_dir, f"{path_name}.loom")
        if not os.path.exists(loom_file):
            print(f"Warning: Loom file not found for path {path_name}: {loom_file}")
            return None, None
        
        # Load the loom file using anndata
        import anndata
        try:
            adata = anndata.read_loom(loom_file)
        except Exception as e:
            print(f"Error loading loom file {loom_file}: {e}")
            return None, None
        
        # Create lineage category (same logic as analyze_scvae.py)
        new_key = "lineage_category"
        if 'lineage' in adata.obs.columns:
            adata.obs[new_key] = adata.obs['lineage'].apply(
                lambda x: len(x.split("/")[0]) if "/" in x else len(x)
            )
        else:
            print(f"Warning: No lineage column found in {loom_file}")
            return None, None
        
        # Compute 1D PCA from raw expression data
        print(f"Computing 1D PCA for {path_name}...")
        pca_coords = compute_1d_pca_from_raw(adata)
        
        # Calculate Spearman correlation between PCA and lineage
        lineage_numeric = pd.to_numeric(adata.obs['lineage_category'])
        correlation, p_value = spearmanr(lineage_numeric, pca_coords)
        
        # Return absolute correlation (like in analyze_scvae.py)
        return abs(correlation), p_value
        
    except Exception as e:
        print(f"Error analyzing PCA for {path_name}: {e}")
        return None, None

def create_four_method_comparison_plot(results_df, output_dir):
    """
    Create comparison plots for all four methods (scVI, scVAE, 1D UMAP, 1D PCA).
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with all four methods
    output_dir : str
        Output directory for plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    methods = ['scvi_correlation', 'scvae_correlation', 'umap_correlation', 'pca_correlation']
    method_names = ['scVI', 'scVAE', '1D UMAP', '1D PCA']
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        corrs = results_df[method].dropna()
        if len(corrs) > 0:
            axes[i].hist(corrs, bins=20, alpha=0.7, color=color, edgecolor='black')
            axes[i].axvline(corrs.mean(), color='red', linestyle='--', 
                           label=f'Mean: {corrs.mean():.3f}')
            axes[i].set_xlabel('Spearman Correlation')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{name}: Correlation with Lineage')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/four_method_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Box plot comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    for method, name in zip(methods, method_names):
        corrs = results_df[method].dropna()
        if len(corrs) > 0:
            box_data.append(corrs)
            box_labels.append(f'{name}\n(n={len(corrs)})')
    
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors_box = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Method Comparison: Correlation with Lineage')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/four_method_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Scatter plot comparisons
    if len(results_df) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        comparisons = [
            ('scvi_correlation', 'scvae_correlation', 'scVI vs scVAE'),
            ('scvi_correlation', 'umap_correlation', 'scVI vs 1D UMAP'),
            ('scvi_correlation', 'pca_correlation', 'scVI vs 1D PCA'),
            ('scvae_correlation', 'umap_correlation', 'scVAE vs 1D UMAP'),
            ('scvae_correlation', 'pca_correlation', 'scVAE vs 1D PCA'),
            ('umap_correlation', 'pca_correlation', '1D UMAP vs 1D PCA')
        ]
        
        for i, (method1, method2, title) in enumerate(comparisons):
            valid_data = results_df[[method1, method2]].dropna()
            if len(valid_data) > 0:
                axes[i].scatter(valid_data[method1], valid_data[method2], 
                              alpha=0.7, s=100)
                min_val = min(valid_data[method1].min(), valid_data[method2].min())
                max_val = max(valid_data[method1].max(), valid_data[method2].max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                axes[i].set_xlabel(method1.replace('_correlation', '').replace('_', ' ').title())
                axes[i].set_ylabel(method2.replace('_correlation', '').replace('_', ' ').title())
                axes[i].set_title(title)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/four_method_scatter_comparisons.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main analysis function."""
    if not IMPORTS_AVAILABLE:
        print("Error: Required imports are not available. Please check your environment.")
        return None
    
    # Configuration
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    output_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/four_method_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to existing results
    scvi_results_file = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_analysis_results.csv"
    scvae_results_file = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvae_analysis_results.csv"
    paths_dict_file = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree.json.gz"
    
    # Load paths dictionary
    print("Loading paths dictionary...")
    paths_dict = load_paths_dict(paths_dict_file)
    print(f"Found {len(paths_dict)} paths in dictionary")
    
    # Load existing results
    scvi_df, scvae_df, common_paths = load_existing_results(scvi_results_file, scvae_results_file)
    
    if len(common_paths) == 0:
        print("No common paths found between scVI and scVAE results. Exiting.")
        return
    
    # Limit to a reasonable number of paths for analysis
    import random
    random.seed(42)
    random_sample = 1000
    if len(common_paths) > random_sample:
        selected_paths = random.sample(common_paths, random_sample)
        print(f"Randomly selected {len(selected_paths)} paths from {len(common_paths)} common paths")
    else:
        selected_paths = common_paths
        print(f"Using all {len(selected_paths)} common paths")
    
    # Prepare results list
    all_results = []
    
    # Analyze each selected path
    for i, path_name in enumerate(selected_paths):
        print(f"\n[{i+1}/{len(selected_paths)}] Processing {path_name}...")
        
        # Get existing scVI and scVAE results
        scvi_row = scvi_df[scvi_df['path_name'] == path_name].iloc[0] if len(scvi_df[scvi_df['path_name'] == path_name]) > 0 else None
        scvae_row = scvae_df[scvae_df['path_name'] == path_name].iloc[0] if len(scvae_df[scvae_df['path_name'] == path_name]) > 0 else None
        
        # Analyze 1D UMAP
        umap_correlation, umap_p_value = analyze_umap_for_path(path_name, base_dir, paths_dict)
        
        # Analyze 1D PCA
        pca_correlation, pca_p_value = analyze_pca_for_path(path_name, base_dir, paths_dict)
        
        # Compile results
        result = {
            'path_name': path_name,
            'path_length': len(paths_dict.get(path_name, [])),
            'scvi_correlation': scvi_row['correlation'] if scvi_row is not None else np.nan,
            'scvi_p_value': scvi_row['p_value'] if scvi_row is not None else np.nan,
            'scvae_correlation': scvae_row['correlation'] if scvae_row is not None else np.nan,
            'scvae_p_value': scvae_row['p_value'] if scvae_row is not None else np.nan,
            'scvae_ari': scvae_row['ari'] if scvae_row is not None and 'ari' in scvae_row else np.nan,
            'umap_correlation': umap_correlation,
            'umap_p_value': umap_p_value,
            'pca_correlation': pca_correlation,
            'pca_p_value': pca_p_value
        }
        
        all_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_file = f"{output_dir}/four_method_comparison_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Print comprehensive summary statistics
    print(f"\n{'='*80}")
    print("FOUR-METHOD COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    methods = ['scvi_correlation', 'scvae_correlation', 'umap_correlation', 'pca_correlation']
    method_names = ['scVI', 'scVAE', '1D UMAP', '1D PCA']
    
    for method, name in zip(methods, method_names):
        corrs = results_df[method].dropna()
        if len(corrs) > 0:
            print(f"\n{name.upper()} CORRELATION STATISTICS (n={len(corrs)}):")
            print(f"  Mean: {corrs.mean():.4f}")
            print(f"  Median: {corrs.median():.4f}")
            print(f"  25th percentile: {corrs.quantile(0.25):.4f}")
            print(f"  75th percentile: {corrs.quantile(0.75):.4f}")
            print(f"  Standard deviation: {corrs.std():.4f}")
            print(f"  Min: {corrs.min():.4f}")
            print(f"  Max: {corrs.max():.4f}")
        else:
            print(f"\n{name.upper()}: No valid correlations found")
    
    # Statistical comparisons
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISONS")
    print(f"{'='*60}")
    
    # Paired t-tests between methods
    from scipy.stats import ttest_rel
    
    comparisons = [
        ('scvi_correlation', 'scvae_correlation', 'scVI vs scVAE'),
        ('scvi_correlation', 'umap_correlation', 'scVI vs 1D UMAP'),
        ('scvi_correlation', 'pca_correlation', 'scVI vs 1D PCA'),
        ('scvae_correlation', 'umap_correlation', 'scVAE vs 1D UMAP'),
        ('scvae_correlation', 'pca_correlation', 'scVAE vs 1D PCA'),
        ('umap_correlation', 'pca_correlation', '1D UMAP vs 1D PCA')
    ]
    
    for method1, method2, comparison_name in comparisons:
        valid_data = results_df[[method1, method2]].dropna()
        if len(valid_data) > 1:
            try:
                t_stat, p_val = ttest_rel(valid_data[method1], valid_data[method2])
                mean_diff = (valid_data[method1] - valid_data[method2]).mean()
                print(f"\n{comparison_name} (n={len(valid_data)}):")
                print(f"  Mean difference: {mean_diff:.4f}")
                print(f"  Paired t-test p-value: {p_val:.4e}")
                
                # Count where method1 > method2
                method1_better = (valid_data[method1] > valid_data[method2]).sum()
                print(f"  Paths where {method1.split('_')[0]} > {method2.split('_')[0]}: {method1_better}/{len(valid_data)}")
            except Exception as e:
                print(f"\n{comparison_name}: Error in statistical test: {e}")
        else:
            print(f"\n{comparison_name}: Insufficient data for comparison")
    
    # Create plots
    print(f"\nCreating comparison plots...")
    create_four_method_comparison_plot(results_df, output_dir)
    print(f"Plots saved to: {output_dir}")
    
    return results_df

if __name__ == "__main__":
    results_df = main()
