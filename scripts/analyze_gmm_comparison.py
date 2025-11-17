#!/usr/bin/env python3
"""
GMM Analysis Script for scVAE vs scVI vs 1D UMAP vs 1D PCA Comparison

This script performs Gaussian Mixture Model (GMM) fitting on learned latent variables
from scVAE, scVI models, and 1D UMAP/PCA from raw expression data, and compares their clustering performance.

Features:
- Fits GMM with number of components equal to path_length
- Calculates entropy and distribution of entropy for each data point
- Evaluates goodness of fit using BIC, AIC, silhouette score, and ARI
- Calculates max probability to cluster for each data point
- Generates separate CSV files for GMM metrics
- Compares all four methods: scVI, scVAE, 1D UMAP, 1D PCA
"""

import json
import gzip
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import anndata
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Import UMAP with error handling
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("Warning: UMAP not available. 1D UMAP analysis will be skipped.")
    UMAP_AVAILABLE = False

def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)

def load_latent_z(latent_file_path):
    """
    Load latent Z values from tsv.gz file.
    Returns DataFrame with latent embeddings.
    """
    try:
        Z_df = pd.read_csv(latent_file_path, sep="\t", compression="infer")
        
        # Handle potential index column
        if Z_df.dtypes.iloc[0] == "object":
            Z_df = Z_df.set_index(Z_df.columns[0])
        
        # Select only numeric columns
        Z_df = Z_df.select_dtypes(include=[np.number])
        
        return Z_df
        
    except Exception as e:
        print(f"Error loading latent file {latent_file_path}: {e}")
        return None

def load_adata_for_lineage(base_dir, path_name):
    """
    Load the corresponding loom file to get lineage information.
    """
    try:
        loom_file = os.path.join(base_dir, f"{path_name}.loom")
        if not os.path.exists(loom_file):
            print(f"Warning: Loom file not found for path {path_name}: {loom_file}")
            return None
        
        adata = anndata.read_loom(loom_file)
        
        # Create lineage category
        if 'lineage' in adata.obs.columns:
            adata.obs['lineage_category'] = adata.obs['lineage'].apply(
                lambda x: len(x.split("/")[0]) if "/" in x else len(x)
            )
            return adata
        else:
            print(f"Warning: No lineage column found in {loom_file}")
            return None
            
    except Exception as e:
        print(f"Error loading loom file for {path_name}: {e}")
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
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not available")
    
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

def calculate_mixture_proportion_comparison(gmm, cluster_assignments, lineage_labels, n_components):
    """
    Compare GMM mixture proportions with actual lineage proportions.
    
    Parameters:
    - gmm: Fitted GMM model
    - cluster_assignments: GMM cluster assignments
    - lineage_labels: True lineage labels
    - n_components: Number of GMM components
    
    Returns:
    - dict with mixture proportion comparison metrics
    """
    try:
        # Get GMM mixture proportions (weights)
        gmm_proportions = gmm.weights_
        
        # Calculate actual lineage proportions
        lineage_counts = np.bincount(lineage_labels.astype(int))
        lineage_proportions = lineage_counts / np.sum(lineage_counts)
        
        # Pad lineage_proportions if it has fewer components than GMM
        if len(lineage_proportions) < n_components:
            padded_lineage = np.zeros(n_components)
            padded_lineage[:len(lineage_proportions)] = lineage_proportions
            lineage_proportions = padded_lineage
        elif len(lineage_proportions) > n_components:
            # Truncate if lineage has more components
            lineage_proportions = lineage_proportions[:n_components]
            lineage_proportions = lineage_proportions / np.sum(lineage_proportions)
        
        # Calculate Jensen-Shannon Divergence (symmetric version of KL divergence)
        from scipy.spatial.distance import jensenshannon
        js_divergence = jensenshannon(gmm_proportions, lineage_proportions)
        
        # Calculate KL Divergence (GMM -> Lineage)
        from scipy.stats import entropy
        kl_divergence = entropy(gmm_proportions, lineage_proportions)
        
        # Calculate KL Divergence (Lineage -> GMM) for symmetry
        kl_divergence_reverse = entropy(lineage_proportions, gmm_proportions)
        
        # Calculate correlation between proportions
        correlation = np.corrcoef(gmm_proportions, lineage_proportions)[0, 1]
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(gmm_proportions - lineage_proportions))
        
        # Calculate root mean square error
        rmse = np.sqrt(np.mean((gmm_proportions - lineage_proportions) ** 2))
        
        return {
            'js_divergence': js_divergence,
            'kl_divergence_gmm_to_lineage': kl_divergence,
            'kl_divergence_lineage_to_gmm': kl_divergence_reverse,
            'proportion_correlation': correlation,
            'proportion_mae': mae,
            'proportion_rmse': rmse,
            'gmm_proportions': gmm_proportions.tolist(),
            'lineage_proportions': lineage_proportions.tolist()
        }
        
    except Exception as e:
        print(f"Error calculating mixture proportion comparison: {e}")
        return {
            'js_divergence': np.nan,
            'kl_divergence_gmm_to_lineage': np.nan,
            'kl_divergence_lineage_to_gmm': np.nan,
            'proportion_correlation': np.nan,
            'proportion_mae': np.nan,
            'proportion_rmse': np.nan,
            'gmm_proportions': [],
            'lineage_proportions': []
        }

def fit_gmm_and_analyze(latent_data, n_components, adata=None, model_name="unknown"):
    """
    Fit GMM to latent data and calculate various metrics.
    
    Parameters:
    - latent_data: numpy array of latent embeddings
    - n_components: number of GMM components (path_length)
    - adata: AnnData object with lineage information (optional)
    - model_name: name of the model for logging
    
    Returns:
    - dict with GMM metrics
    """
    try:
        # Ensure we have enough data points for the number of components
        if len(latent_data) < n_components:
            print(f"Warning: Not enough data points ({len(latent_data)}) for {n_components} components in {model_name}")
            return None     
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=200)
        gmm.fit(latent_data)
        
        # Get probabilities for each data point
        probabilities = gmm.predict_proba(latent_data)  # shape: (n_samples, n_components)
        
        # Calculate entropy for each data point
        point_entropies = [entropy(probs) for probs in probabilities]
        
        # Calculate max probability to cluster for each data point
        max_probabilities = np.max(probabilities, axis=1)
        
        # Calculate goodness of fit metrics
        bic = gmm.bic(latent_data)
        aic = gmm.aic(latent_data)
        
        # Calculate additional goodness of fit metrics
        log_likelihood = gmm.score(latent_data)
        perplexity = np.exp(-log_likelihood / len(latent_data))
        
        # Get cluster assignments
        cluster_assignments = gmm.predict(latent_data)
        
        # Calculate silhouette score
        if len(set(cluster_assignments)) > 1:  # Need at least 2 clusters for silhouette
            silhouette = silhouette_score(latent_data, cluster_assignments)
        else:
            silhouette = np.nan
        
        # Calculate Calinski-Harabasz index
        if len(set(cluster_assignments)) > 1:
            calinski_harabasz = calinski_harabasz_score(latent_data, cluster_assignments)
        else:
            calinski_harabasz = np.nan
        
        # Calculate Davies-Bouldin index
        if len(set(cluster_assignments)) > 1:
            davies_bouldin = davies_bouldin_score(latent_data, cluster_assignments)
        else:
            davies_bouldin = np.nan
        
        # Calculate ARI with lineage if adata is provided
        ari_with_lineage = np.nan
        mixture_proportion_metrics = {}
        if adata is not None and 'lineage_category' in adata.obs.columns:
            try:
                lineage_labels = pd.to_numeric(adata.obs['lineage_category']).values
                ari_with_lineage = adjusted_rand_score(cluster_assignments, lineage_labels)
                
                # Calculate mixture proportion comparison
                mixture_proportion_metrics = calculate_mixture_proportion_comparison(
                    gmm, cluster_assignments, lineage_labels, n_components
                )
            except Exception as e:
                print(f"Warning: Could not calculate ARI with lineage for {model_name}: {e}")
                mixture_proportion_metrics = {
                    'js_divergence': np.nan,
                    'kl_divergence_gmm_to_lineage': np.nan,
                    'kl_divergence_lineage_to_gmm': np.nan,
                    'proportion_correlation': np.nan,
                    'proportion_mae': np.nan,
                    'proportion_rmse': np.nan
                }
        
        # Calculate distribution statistics for entropy
        entropy_stats = {
            'mean_entropy': np.mean(point_entropies),
            'std_entropy': np.std(point_entropies),
            'min_entropy': np.min(point_entropies),
            'max_entropy': np.max(point_entropies),
            'median_entropy': np.median(point_entropies)
        }
        
        # Calculate distribution statistics for max probabilities
        max_prob_stats = {
            'mean_max_prob': np.mean(max_probabilities),
            'std_max_prob': np.std(max_probabilities),
            'min_max_prob': np.min(max_probabilities),
            'max_max_prob': np.max(max_probabilities),
            'median_max_prob': np.median(max_probabilities)
        }
        
        return {
            'n_components': n_components,
            'n_samples': len(latent_data),
            'bic': bic,
            'aic': aic,
            'log_likelihood': log_likelihood,
            'perplexity': perplexity,
            'silhouette_score': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin,
            'ari_with_lineage': ari_with_lineage,
            'converged': gmm.converged_,
            'n_iter': gmm.n_iter_,
            **entropy_stats,
            **max_prob_stats,
            **mixture_proportion_metrics
        }
        
    except Exception as e:
        print(f"Error fitting GMM for {model_name}: {e}")
        return None

def analyze_scvae_gmm(path_name, latent_file_path, base_dir, paths_dict):
    """
    Analyze scVAE latent data with GMM.
    """
    try:
        # Load latent Z values
        Z_df = load_latent_z(latent_file_path)
        if Z_df is None:
            return None
        
        # Get path length from paths dictionary
        path_length = len(paths_dict.get(path_name, []))
        
        # Load adata for lineage information
        adata = load_adata_for_lineage(base_dir, path_name)
        
        # Align latent data with adata if available
        if adata is not None:
            align_to_adata(adata, Z_df, "X_scvae_Z")
            latent_data = adata.obsm['X_scvae_Z']
        else:
            latent_data = Z_df.to_numpy()
        
        # Fit GMM and analyze
        gmm_results = fit_gmm_and_analyze(
            latent_data, 
            path_length, 
            adata, 
            f"scVAE_{path_name}"
        )
        
        if gmm_results is None:
            return None
        
        # Add path information
        gmm_results.update({
            'path_name': path_name,
            'path_length': path_length,
            'model': 'scVAE'
        })
        
        return gmm_results
        
    except Exception as e:
        print(f"Error analyzing scVAE GMM for {path_name}: {e}")
        return None

def analyze_scvi_gmm(path_name, base_dir, paths_dict):
    """
    Analyze scVI latent data with GMM.
    """
    try:
        # Construct the adata path for scVI
        adata_path = os.path.join(base_dir, f"scvi_path_{path_name}", "trained.h5ad")
        
        if not os.path.exists(adata_path):
            print(f"Warning: scVI adata file not found: {adata_path}")
            return None
        
        # Load scVI adata
        adata = anndata.read_h5ad(adata_path)
        
        if 'X_scVI' not in adata.obsm:
            print(f"Warning: X_scVI not found in adata for {path_name}")
            return None
        
        # Get path length
        path_length = len(paths_dict.get(path_name, []))
        
        # Create lineage category if not exists
        if 'lineage' in adata.obs.columns and 'lineage_category' not in adata.obs.columns:
            adata.obs['lineage_category'] = adata.obs['lineage'].apply(
                lambda x: len(x.split("/")[0]) if "/" in x else len(x)
            )
        
        # Get latent data
        latent_data = adata.obsm['X_scVI']
        
        # Fit GMM and analyze
        gmm_results = fit_gmm_and_analyze(
            latent_data, 
            path_length, 
            adata, 
            f"scVI_{path_name}"
        )
        
        if gmm_results is None:
            return None
        
        # Add path information
        gmm_results.update({
            'path_name': path_name,
            'path_length': path_length,
            'model': 'scVI'
        })
        
        return gmm_results
        
    except Exception as e:
        print(f"Error analyzing scVI GMM for {path_name}: {e}")
        return None

def save_coordinates_to_file(coordinates, path_name, method, output_dir):
    """
    Save coordinates to a CSV file.
    
    Parameters:
    -----------
    coordinates : np.array
        The coordinates to save
    path_name : str
        Name of the path
    method : str
        Method name (UMAP, PCA, etc.)
    output_dir : str
        Output directory
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        coords_file = os.path.join(output_dir, f"{method.lower()}_coordinates.csv")
        
        # Create DataFrame with coordinates
        coords_df = pd.DataFrame({
            'cell_id': range(len(coordinates)),
            'coordinates': coordinates
        })
        
        # Add path_name column
        coords_df['path_name'] = path_name
        
        # Append to file (create if doesn't exist)
        if os.path.exists(coords_file):
            coords_df.to_csv(coords_file, mode='a', header=False, index=False)
        else:
            coords_df.to_csv(coords_file, index=False)
            
    except Exception as e:
        print(f"Warning: Could not save coordinates for {path_name}: {e}")

def save_correlations_to_file(correlations_data, output_dir):
    """
    Save correlation results to a CSV file.
    
    Parameters:
    -----------
    correlations_data : list
        List of dictionaries with correlation data
    output_dir : str
        Output directory
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        corr_file = os.path.join(output_dir, "path_correlations.csv")
        
        # Create DataFrame
        corr_df = pd.DataFrame(correlations_data)
        
        # Save to file
        corr_df.to_csv(corr_file, index=False)
        print(f"Correlations saved to: {corr_file}")
        
    except Exception as e:
        print(f"Warning: Could not save correlations: {e}")

def analyze_umap_gmm(path_name, base_dir, paths_dict, save_coords=True, output_dir=None):
    """
    Analyze 1D UMAP data with GMM.
    """
    try:
        # Load the loom file for this path
        loom_file = os.path.join(base_dir, f"{path_name}.loom")
        if not os.path.exists(loom_file):
            print(f"Warning: Loom file not found for path {path_name}: {loom_file}")
            return None
        
        # Load the loom file
        adata = anndata.read_loom(loom_file)
        
        # Create lineage category
        if 'lineage' in adata.obs.columns:
            adata.obs['lineage_category'] = adata.obs['lineage'].apply(
                lambda x: len(x.split("/")[0]) if "/" in x else len(x)
            )
        else:
            print(f"Warning: No lineage column found in {loom_file}")
            return None
        
        # Get path length
        path_length = len(paths_dict.get(path_name, []))
        
        # Compute 1D UMAP from raw expression data
        print(f"Computing 1D UMAP for {path_name}...")
        umap_coords = compute_1d_umap_from_raw(adata)
        
        # Store UMAP coordinates in adata.obs
        adata.obs['umap_1d'] = umap_coords
        
        # Save coordinates if requested (legacy CSV method)
        if save_coords and output_dir is not None:
            save_coordinates_to_file(umap_coords, path_name, 'UMAP', output_dir)
        
        # Save the updated adata back to file
        print(f"Saving updated adata with UMAP coordinates for {path_name}...")
        adata.write_loom(loom_file)
        
        # Reshape to 2D for GMM (add a dummy dimension)
        latent_data = umap_coords.reshape(-1, 1)
        
        # Fit GMM and analyze
        gmm_results = fit_gmm_and_analyze(
            latent_data, 
            path_length, 
            adata, 
            f"UMAP_{path_name}"
        )
        
        if gmm_results is None:
            return None
        
        # Add path information
        gmm_results.update({
            'path_name': path_name,
            'path_length': path_length,
            'model': 'UMAP'
        })
        
        return gmm_results
        
    except Exception as e:
        print(f"Error analyzing UMAP GMM for {path_name}: {e}")
        return None

def analyze_pca_gmm(path_name, base_dir, paths_dict, save_coords=True, output_dir=None):
    """
    Analyze 1D PCA data with GMM.
    """
    try:
        # Load the loom file for this path
        loom_file = os.path.join(base_dir, f"{path_name}.loom")
        if not os.path.exists(loom_file):
            print(f"Warning: Loom file not found for path {path_name}: {loom_file}")
            return None
        
        # Load the loom file
        adata = anndata.read_loom(loom_file)
        
        # Create lineage category
        if 'lineage' in adata.obs.columns:
            adata.obs['lineage_category'] = adata.obs['lineage'].apply(
                lambda x: len(x.split("/")[0]) if "/" in x else len(x)
            )
        else:
            print(f"Warning: No lineage column found in {loom_file}")
            return None
        
        # Get path length
        path_length = len(paths_dict.get(path_name, []))
        
        # Compute 1D PCA from raw expression data
        print(f"Computing 1D PCA for {path_name}...")
        pca_coords = compute_1d_pca_from_raw(adata)
        
        # Store PCA coordinates in adata.obs
        adata.obs['pca_1d'] = pca_coords
        
        # Save coordinates if requested (legacy CSV method)
        if save_coords and output_dir is not None:
            save_coordinates_to_file(pca_coords, path_name, 'PCA', output_dir)
        
        # Save the updated adata back to file
        print(f"Saving updated adata with PCA coordinates for {path_name}...")
        adata.write_loom(loom_file)
        
        # Reshape to 2D for GMM (add a dummy dimension)
        latent_data = pca_coords.reshape(-1, 1)
        
        # Fit GMM and analyze
        gmm_results = fit_gmm_and_analyze(
            latent_data, 
            path_length, 
            adata, 
            f"PCA_{path_name}"
        )
        
        if gmm_results is None:
            return None
        
        # Add path information
        gmm_results.update({
            'path_name': path_name,
            'path_length': path_length,
            'model': 'PCA'
        })
        
        return gmm_results
        
    except Exception as e:
        print(f"Error analyzing PCA GMM for {path_name}: {e}")
        return None

def visualize_coordinates_from_adata(path_name, base_dir, method='pca', figsize=(10, 6), bins=50, alpha=0.7):
    """
    Visualize 1D coordinates from AnnData object with lineage coloring.
    
    Parameters:
    -----------
    path_name : str
        Name of the path
    base_dir : str
        Base directory containing loom files
    method : str
        Either 'pca' or 'umap' to specify which coordinates to visualize
    figsize : tuple
        Figure size (width, height)
    bins : int
        Number of bins for histogram
    alpha : float
        Transparency for overlapping histograms
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load the loom file for this path
        loom_file = os.path.join(base_dir, f"{path_name}.loom")
        if not os.path.exists(loom_file):
            print(f"Warning: Loom file not found for path {path_name}: {loom_file}")
            return None
        
        # Load the loom file
        adata = anndata.read_loom(loom_file)
        
        # Check if coordinates exist
        coord_col = f'{method}_1d'
        if coord_col not in adata.obs.columns:
            print(f"Warning: {coord_col} not found in adata for {path_name}")
            return None
        
        # Get coordinates and lineage information
        coords = adata.obs[coord_col].values
        lineage_labels = adata.obs['lineage'].values if 'lineage' in adata.obs.columns else None
        
        if lineage_labels is None:
            print(f"Warning: No lineage information found for {path_name}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique lineage labels and assign colors
        unique_lineages = sorted(list(set(lineage_labels)))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_lineages)))
        
        # Plot histograms for each lineage
        for i, lineage in enumerate(unique_lineages):
            lineage_coords = coords[lineage_labels == lineage]
            if len(lineage_coords) > 0:
                ax.hist(lineage_coords, bins=bins, alpha=alpha, 
                       label=lineage, color=colors[i], density=True)
        
        ax.set_title(f'{method.upper()} Coordinates Distribution by Lineage - {path_name}')
        ax.set_xlabel(f'{method.upper()} Coordinate')
        ax.set_ylabel('Normalized Frequency')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error visualizing coordinates for {path_name}: {e}")
        return None

def visualize_multiple_paths_coordinates(base_dir, path_names, method='pca', n_cols=5, figsize=(20, 16)):
    """
    Visualize coordinates for multiple paths in a grid layout.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing loom files
    path_names : list
        List of path names to visualize
    method : str
        Either 'pca' or 'umap' to specify which coordinates to visualize
    n_cols : int
        Number of columns in the grid
    figsize : tuple
        Figure size (width, height)
    """
    try:
        import matplotlib.pyplot as plt
        
        n_paths = len(path_names)
        n_rows = (n_paths + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Create visualizations for each path
        for i, path_name in enumerate(path_names):
            if i >= len(axes_flat):
                break
                
            try:
                # Load the loom file for this path
                loom_file = os.path.join(base_dir, f"{path_name}.loom")
                if not os.path.exists(loom_file):
                    axes_flat[i].text(0.5, 0.5, f'File not found\n{path_name}', 
                                    ha='center', va='center', transform=axes_flat[i].transAxes)
                    axes_flat[i].set_title(f'Path: {path_name}')
                    continue
                
                # Load the loom file
                adata = anndata.read_loom(loom_file)
                
                # Check if coordinates exist
                coord_col = f'{method}_1d'
                if coord_col not in adata.obs.columns:
                    axes_flat[i].text(0.5, 0.5, f'No {method.upper()} coords\n{path_name}', 
                                    ha='center', va='center', transform=axes_flat[i].transAxes)
                    axes_flat[i].set_title(f'Path: {path_name}')
                    continue
                
                # Get coordinates and lineage information
                coords = adata.obs[coord_col].values
                lineage_labels = adata.obs['lineage'].values if 'lineage' in adata.obs.columns else None
                
                if lineage_labels is None:
                    axes_flat[i].text(0.5, 0.5, f'No lineage info\n{path_name}', 
                                    ha='center', va='center', transform=axes_flat[i].transAxes)
                    axes_flat[i].set_title(f'Path: {path_name}')
                    continue
                
                # Get unique lineage labels and assign colors
                unique_lineages = sorted(list(set(lineage_labels)))
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_lineages)))
                
                # Plot histograms for each lineage
                for j, lineage in enumerate(unique_lineages):
                    lineage_coords = coords[lineage_labels == lineage]
                    if len(lineage_coords) > 0:
                        axes_flat[i].hist(lineage_coords, bins=50, alpha=0.7, 
                                        label=lineage, color=colors[j], density=True)
                
                axes_flat[i].set_title(f'{path_name}')
                axes_flat[i].set_xlabel(f'{method.upper()} Coordinate')
                axes_flat[i].set_ylabel('Normalized Frequency')
                axes_flat[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes_flat[i].grid(True, alpha=0.3)
                
            except Exception as e:
                axes_flat[i].text(0.5, 0.5, f'Error\n{path_name}\n{str(e)[:50]}...', 
                                ha='center', va='center', transform=axes_flat[i].transAxes)
                axes_flat[i].set_title(f'Path: {path_name}')
        
        # Hide unused subplots
        for i in range(len(path_names), len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating multi-path visualization: {e}")
        return None

def compute_correlations_for_path(path_name, base_dir, paths_dict):
    """
    Compute correlations between 1D UMAP/PCA and lineage for a single path.
    
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
    dict
        Dictionary with correlation results
    """
    try:
        from scipy.stats import spearmanr
        
        # Load the loom file for this path
        loom_file = os.path.join(base_dir, f"{path_name}.loom")
        if not os.path.exists(loom_file):
            print(f"Warning: Loom file not found for path {path_name}: {loom_file}")
            return None
        
        # Load the loom file
        adata = anndata.read_loom(loom_file)
        
        # Create lineage category
        if 'lineage' in adata.obs.columns:
            adata.obs['lineage_category'] = adata.obs['lineage'].apply(
                lambda x: len(x.split("/")[0]) if "/" in x else len(x)
            )
        else:
            print(f"Warning: No lineage column found in {loom_file}")
            return None
        
        # Get lineage data
        lineage_numeric = pd.to_numeric(adata.obs['lineage_category'])
        
        result = {
            'path_name': path_name,
            'path_length': len(paths_dict.get(path_name, [])),
            'n_cells': len(adata)
        }
        
        # Compute 1D UMAP correlation
        if UMAP_AVAILABLE:
            try:
                print(f"Computing UMAP correlation for {path_name}...")
                umap_coords = compute_1d_umap_from_raw(adata)
                umap_corr, umap_p = spearmanr(lineage_numeric, umap_coords)
                result.update({
                    'umap_correlation': abs(umap_corr),
                    'umap_p_value': umap_p
                })
            except Exception as e:
                print(f"Error computing UMAP correlation for {path_name}: {e}")
                result.update({
                    'umap_correlation': np.nan,
                    'umap_p_value': np.nan
                })
        else:
            result.update({
                'umap_correlation': np.nan,
                'umap_p_value': np.nan
            })
        
        # Compute 1D PCA correlation
        try:
            print(f"Computing PCA correlation for {path_name}...")
            pca_coords = compute_1d_pca_from_raw(adata)
            pca_corr, pca_p = spearmanr(lineage_numeric, pca_coords)
            result.update({
                'pca_correlation': abs(pca_corr),
                'pca_p_value': pca_p
            })
        except Exception as e:
            print(f"Error computing PCA correlation for {path_name}: {e}")
            result.update({
                'pca_correlation': np.nan,
                'pca_p_value': np.nan
            })
        
        return result
        
    except Exception as e:
        print(f"Error computing correlations for {path_name}: {e}")
        return None

def find_scvae_latent_files(base_dir, paths_dict, analyze_all=False):
    """
    Find latent_values-z.tsv.gz files for scVAE paths.
    """
    results = []
    
    if analyze_all:
        # Find all scvae_path_* directories
        scvae_pattern = os.path.join(base_dir, "scvae_path_*")
        scvae_dirs = glob.glob(scvae_pattern)
        
        for scvae_dir in scvae_dirs:
            dir_name = os.path.basename(scvae_dir)
            if dir_name.startswith("scvae_path_"):
                path_name = dir_name.replace("scvae_path_", "")
                
                pattern = os.path.join(scvae_dir, "**", "latent_values-z.tsv.gz")
                latent_files = glob.glob(pattern, recursive=True)
                
                if latent_files:
                    latent_file_path = latent_files[0]
                    results.append((path_name, latent_file_path))
                    print(f"Found latent file for path {path_name}: {latent_file_path}")
                else:
                    print(f"Warning: No latent file found for path {path_name}")
    else:
        # Only process paths in paths_dict
        for path_name in paths_dict.keys():
            path_base_dir = os.path.join(base_dir, f"scvae_path_{path_name}", path_name)
            
            if not os.path.exists(path_base_dir):
                print(f"Warning: Base directory not found for path {path_name}: {path_base_dir}")
                continue
            
            pattern = os.path.join(path_base_dir, "**", "latent_values-z.tsv.gz")
            latent_files = glob.glob(pattern, recursive=True)
            
            if latent_files:
                latent_file_path = latent_files[0]
                results.append((path_name, latent_file_path))
                print(f"Found latent file for path {path_name}: {latent_file_path}")
            else:
                print(f"Warning: No latent file found for path {path_name}")
    
    return results

def main():
    """Main analysis function."""
    # Configuration
    ANALYZE_ALL_SCVAE_PATHS = False  # Set to True to analyze all scVAE paths
    MAX_PATHS_TO_ANALYZE = None  # Limit number of paths for UMAP/PCA analysis (can be slow). Set to None for all paths.
    
    # Paths
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    paths_dict_file = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree.json.gz"
    
    # Load paths dictionary
    print("Loading paths dictionary...")
    paths_dict = load_paths_dict(paths_dict_file)
    print(f"Found {len(paths_dict)} paths in dictionary.")
    
    # Find scVAE latent files
    print("Finding scVAE latent files...")
    latent_files = find_scvae_latent_files(base_dir, paths_dict, analyze_all=ANALYZE_ALL_SCVAE_PATHS)
    print(f"Found {len(latent_files)} scVAE latent files.")
    
    # Prepare results
    scvae_results = []
    scvi_results = []
    umap_results = []
    pca_results = []
    
    # Analyze scVAE paths
    print("\nAnalyzing scVAE GMM...")
    for path_name, latent_file_path in latent_files:
        print(f"Analyzing scVAE GMM for path: {path_name}")
        
        result = analyze_scvae_gmm(path_name, latent_file_path, base_dir, paths_dict)
        if result is not None:
            scvae_results.append(result)
        else:
            print(f"Failed to analyze scVAE GMM for {path_name}")
    
    # Analyze scVI paths
    print("\nAnalyzing scVI GMM...")
    for path_name in paths_dict.keys():
        print(f"Analyzing scVI GMM for path: {path_name}")
        
        result = analyze_scvi_gmm(path_name, base_dir, paths_dict)
        if result is not None:
            scvi_results.append(result)
        else:
            print(f"Failed to analyze scVI GMM for {path_name}")
    
    # Create coordinates output directory
    coords_output_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/coordinates"
    
    # Analyze UMAP paths (limited number due to computational cost)
    print("\nAnalyzing 1D UMAP GMM...")
    if UMAP_AVAILABLE:
        # Get common paths between scVAE and scVI for UMAP analysis
        scvae_paths = set([r['path_name'] for r in scvae_results])
        scvi_paths = set([r['path_name'] for r in scvi_results])
        common_paths = list(scvae_paths.intersection(scvi_paths))
        
        # Limit to MAX_PATHS_TO_ANALYZE paths if specified
        if MAX_PATHS_TO_ANALYZE is not None and len(common_paths) > MAX_PATHS_TO_ANALYZE:
            import random
            random.seed(42)
            common_paths = random.sample(common_paths, MAX_PATHS_TO_ANALYZE)
            print(f"Limited UMAP analysis to {MAX_PATHS_TO_ANALYZE} paths")
        
        for i, path_name in enumerate(common_paths):
            print(f"Analyzing UMAP GMM for path {i+1}/{len(common_paths)}: {path_name}")
            
            result = analyze_umap_gmm(path_name, base_dir, paths_dict, save_coords=True, output_dir=coords_output_dir)
            if result is not None:
                umap_results.append(result)
            else:
                print(f"Failed to analyze UMAP GMM for {path_name}")
    else:
        print("UMAP not available, skipping UMAP analysis")
    
    # Analyze PCA paths (limited number due to computational cost)
    print("\nAnalyzing 1D PCA GMM...")
    # Use same common paths as UMAP
    if len(common_paths) > 0:
        for i, path_name in enumerate(common_paths):
            print(f"Analyzing PCA GMM for path {i+1}/{len(common_paths)}: {path_name}")
            
            result = analyze_pca_gmm(path_name, base_dir, paths_dict, save_coords=True, output_dir=coords_output_dir)
            if result is not None:
                pca_results.append(result)
            else:
                print(f"Failed to analyze PCA GMM for {path_name}")
    
    # Compute correlations for all paths
    print("\nComputing correlations for all paths...")
    correlation_results = []
    
    # Get all paths that have loom files
    all_paths = []
    for path_name in paths_dict.keys():
        loom_file = os.path.join(base_dir, f"{path_name}.loom")
        if os.path.exists(loom_file):
            all_paths.append(path_name)
    
    # Limit to MAX_PATHS_TO_ANALYZE for correlation computation if specified
    if MAX_PATHS_TO_ANALYZE is not None and len(all_paths) > MAX_PATHS_TO_ANALYZE:
        import random
        random.seed(42)
        all_paths = random.sample(all_paths, MAX_PATHS_TO_ANALYZE)
        print(f"Limited correlation analysis to {MAX_PATHS_TO_ANALYZE} paths")
    
    for i, path_name in enumerate(all_paths):
        print(f"Computing correlations for path {i+1}/{len(all_paths)}: {path_name}")
        
        result = compute_correlations_for_path(path_name, base_dir, paths_dict)
        if result is not None:
            correlation_results.append(result)
        else:
            print(f"Failed to compute correlations for {path_name}")
    
    # Save correlations
    if correlation_results:
        save_correlations_to_file(correlation_results, coords_output_dir)
    
    # Create DataFrames
    scvae_df = pd.DataFrame(scvae_results)
    scvi_df = pd.DataFrame(scvi_results)
    umap_df = pd.DataFrame(umap_results)
    pca_df = pd.DataFrame(pca_results)
    correlation_df = pd.DataFrame(correlation_results)
    
    # Save results
    os.makedirs("/n/fs/ragr-data/users/viola/structuredVAE/results", exist_ok=True)
    
    if ANALYZE_ALL_SCVAE_PATHS:
        scvae_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvae_gmm_analysis_all.csv"
        scvi_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_gmm_analysis_all.csv"
        umap_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/umap_gmm_analysis_all.csv"
        pca_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/pca_gmm_analysis_all.csv"
        comparison_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/four_method_gmm_comparison_all.csv"
    else:
        scvae_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvae_gmm_analysis.csv"
        scvi_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_gmm_analysis.csv"
        umap_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/umap_gmm_analysis.csv"
        pca_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/pca_gmm_analysis.csv"
        comparison_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/four_method_gmm_comparison.csv"
    
    scvae_df.to_csv(scvae_output, index=False)
    scvi_df.to_csv(scvi_output, index=False)
    umap_df.to_csv(umap_output, index=False)
    pca_df.to_csv(pca_output, index=False)
    
    # Save correlation results
    if len(correlation_df) > 0:
        corr_output = "/n/fs/ragr-data/users/viola/structuredVAE/results/umap_pca_correlations.csv"
        correlation_df.to_csv(corr_output, index=False)
        print(f"UMAP/PCA correlations saved to: {corr_output}")
    
    print(f"\nscVAE GMM results saved to: {scvae_output}")
    print(f"scVI GMM results saved to: {scvi_output}")
    print(f"UMAP GMM results saved to: {umap_output}")
    print(f"PCA GMM results saved to: {pca_output}")
    
    # Create four-method comparison
    all_results = [scvae_df, scvi_df, umap_df, pca_df]
    method_names = ['scVAE', 'scVI', 'UMAP', 'PCA']
    
    # Find paths that exist in all methods
    all_paths = []
    for df in all_results:
        if len(df) > 0:
            all_paths.append(set(df['path_name']))
    
    if len(all_paths) > 0:
        common_paths_all = set.intersection(*all_paths)
        print(f"\nFound {len(common_paths_all)} common paths across all methods")
        
        if len(common_paths_all) > 0:
            comparison_results = []
            
            for path_name in common_paths_all:
                result = {'path_name': path_name}
                
                # Get results for each method
                for df, method in zip(all_results, method_names):
                    if len(df) > 0:
                        method_rows = df[df['path_name'] == path_name]
                        if len(method_rows) > 0:
                            row = method_rows.iloc[0]
                            result.update({
                                f'{method.lower()}_path_length': row['path_length'],
                                f'{method.lower()}_bic': row['bic'],
                                f'{method.lower()}_aic': row['aic'],
                                f'{method.lower()}_log_likelihood': row['log_likelihood'],
                                f'{method.lower()}_perplexity': row['perplexity'],
                                f'{method.lower()}_silhouette': row['silhouette_score'],
                                f'{method.lower()}_calinski_harabasz': row['calinski_harabasz'],
                                f'{method.lower()}_davies_bouldin': row['davies_bouldin'],
                                f'{method.lower()}_ari_lineage': row['ari_with_lineage'],
                                f'{method.lower()}_mean_entropy': row['mean_entropy'],
                                f'{method.lower()}_mean_max_prob': row['mean_max_prob'],
                                f'{method.lower()}_js_divergence': row.get('js_divergence', np.nan),
                                f'{method.lower()}_proportion_correlation': row.get('proportion_correlation', np.nan),
                                f'{method.lower()}_proportion_mae': row.get('proportion_mae', np.nan)
                            })
                
                comparison_results.append(result)
            
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)
                comparison_df.to_csv(comparison_output, index=False)
                print(f"Four-method GMM comparison saved to: {comparison_output}")
                
                # Print comprehensive summary statistics
                print("\n" + "="*100)
                print("FOUR-METHOD GMM ANALYSIS SUMMARY")
                print("="*100)
                
                for df, method in zip(all_results, method_names):
                    if len(df) > 0:
                        print(f"\n{method.upper()} GMM Analysis (n={len(df)}):")
                        print(f"  Mean BIC: {df['bic'].mean():.2f}")
                        print(f"  Mean AIC: {df['aic'].mean():.2f}")
                        print(f"  Mean Log-likelihood: {df['log_likelihood'].mean():.2f}")
                        print(f"  Mean Perplexity: {df['perplexity'].mean():.2f}")
                        print(f"  Mean Silhouette: {df['silhouette_score'].mean():.4f}")
                        print(f"  Mean Calinski-Harabasz: {df['calinski_harabasz'].mean():.2f}")
                        print(f"  Mean Davies-Bouldin: {df['davies_bouldin'].mean():.4f}")
                        print(f"  Mean ARI with lineage: {df['ari_with_lineage'].mean():.4f}")
                        print(f"  Mean entropy: {df['mean_entropy'].mean():.4f}")
                        print(f"  Mean max probability: {df['mean_max_prob'].mean():.4f}")
                        if 'js_divergence' in df.columns:
                            print(f"  Mean JS Divergence: {df['js_divergence'].mean():.4f}")
                            print(f"  Mean Proportion Correlation: {df['proportion_correlation'].mean():.4f}")
                
                # Statistical comparisons between methods
                if len(comparison_df) > 0:
                    print(f"\n{'='*60}")
                    print("STATISTICAL COMPARISONS")
                    print(f"{'='*60}")
                    
                    # Compare BIC across methods
                    print(f"\nBIC Comparison (n={len(comparison_df)}):")
                    for method in method_names:
                        if f'{method.lower()}_bic' in comparison_df.columns:
                            bic_values = comparison_df[f'{method.lower()}_bic'].dropna()
                            if len(bic_values) > 0:
                                print(f"  {method}: {bic_values.mean():.2f} ± {bic_values.std():.2f}")
                    
                    # Compare ARI with lineage across methods
                    print(f"\nARI with Lineage Comparison (n={len(comparison_df)}):")
                    for method in method_names:
                        if f'{method.lower()}_ari_lineage' in comparison_df.columns:
                            ari_values = comparison_df[f'{method.lower()}_ari_lineage'].dropna()
                            if len(ari_values) > 0:
                                print(f"  {method}: {ari_values.mean():.4f} ± {ari_values.std():.4f}")
                    
                    # Compare Silhouette scores across methods
                    print(f"\nSilhouette Score Comparison (n={len(comparison_df)}):")
                    for method in method_names:
                        if f'{method.lower()}_silhouette' in comparison_df.columns:
                            sil_values = comparison_df[f'{method.lower()}_silhouette'].dropna()
                            if len(sil_values) > 0:
                                print(f"  {method}: {sil_values.mean():.4f} ± {sil_values.std():.4f}")
        else:
            print("No common paths found across all methods")
    else:
        print("Insufficient data for four-method comparison")

def create_coordinate_visualizations(base_dir, n_paths=50, method='pca', output_dir=None):
    """
    Create visualizations of 1D coordinates stored in AnnData objects.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing loom files
    n_paths : int
        Number of random paths to visualize
    method : str
        Either 'pca' or 'umap' to specify which coordinates to visualize
    output_dir : str
        Output directory for saving visualizations
    """
    try:
        import matplotlib.pyplot as plt
        import random
        
        # Create output directory
        if output_dir is None:
            output_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all loom files
        loom_files = glob.glob(os.path.join(base_dir, "*.loom"))
        path_names = [os.path.basename(f).replace('.loom', '') for f in loom_files]
        
        print(f"Found {len(path_names)} loom files")
        
        # Randomly select paths
        random.seed(42)
        if len(path_names) > n_paths:
            selected_paths = random.sample(path_names, n_paths)
        else:
            selected_paths = path_names
            n_paths = len(selected_paths)
        
        print(f"Selected {len(selected_paths)} paths for {method.upper()} visualization")
        
        # Create multi-path visualization
        print(f"Creating {method.upper()} coordinate visualizations...")
        fig = visualize_multiple_paths_coordinates(
            base_dir, selected_paths, method=method, n_cols=5, figsize=(20, 16)
        )
        
        if fig is not None:
            # Save the visualization
            output_file = os.path.join(output_dir, f"{method}_coordinates_histograms.png")
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"{method.upper()} coordinate visualizations saved to: {output_file}")
            
            # Also create individual visualizations for a few paths
            print("Creating individual path visualizations...")
            for i, path_name in enumerate(selected_paths[:10]):  # First 10 paths
                try:
                    fig_individual = visualize_coordinates_from_adata(
                        path_name, base_dir, method=method, figsize=(10, 6)
                    )
                    if fig_individual is not None:
                        individual_output = os.path.join(
                            output_dir, f"{method}_coordinates_{path_name}.png"
                        )
                        fig_individual.savefig(individual_output, dpi=300, bbox_inches='tight')
                        plt.close(fig_individual)
                        print(f"Individual {method.upper()} visualization for {path_name} saved")
                except Exception as e:
                    print(f"Error creating individual visualization for {path_name}: {e}")
            
            plt.close(fig)
        
        print(f"{method.upper()} coordinate visualization complete!")
        
    except Exception as e:
        print(f"Error creating coordinate visualizations: {e}")

if __name__ == '__main__':
    main()
