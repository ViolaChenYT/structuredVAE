"""
Train PHATE on all C. elegans lineage paths.

This script loads pre-trained scVI data and computes PHATE 1D embeddings.
Reports: abs Spearman rho, abs Kendall tau, NMI, and ARI with lineage.

Requirements:
- Pre-trained scVI models in data/scvi_path_*/trained.h5ad
- phate library installed
"""

import os
import json
import gzip
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import phate
import scanpy as sc
import anndata
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from itertools import product


def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)


def expand_choices(items, sep="/", strip=True, dedup=False):
    """
    items: list[str] possibly containing 'a/b' meaning choose one
    returns: list[list[str]] of all expanded lists
    """
    # turn each item into a list of options
    opts = []
    for s in items:
        parts = s.split(sep)
        if strip:
            parts = [p.strip() for p in parts]
        # if no sep, keep as single option; if sep, use all parts
        opts.append(parts if len(parts) > 1 else [parts[0]])
    # Cartesian product over positions
    combos = [list(choice) for choice in product(*opts)]
    if dedup:  # remove duplicate lists if options had duplicates
        seen = set()
        uniq = []
        for c in combos:
            t = tuple(c)
            if t not in seen:
                seen.add(t); uniq.append(c)
        return uniq
    return combos


def is_prefix_match(prefix: str, s: str, wildcard: str = "x") -> bool:
    """Return True iff prefix (with 'x' as single-char wildcard) matches the start of s."""
    if len(prefix) > len(s):
        return False
    for a, b in zip(prefix, s):
        if a != wildcard and b!=wildcard and a != b:
            return False
    return True


def is_prefix_chain_wild(L, wildcard: str = "x") -> bool:
    """Check L[i] is a wildcard-prefix of L[i+1] for all i."""
    return all(is_prefix_match(L[i], L[i+1], wildcard) for i in range(len(L) - 1))


def label_order_index(label_list, sep="/", wildcard="x"):
    # Find the first expanded path whose length-lex order forms a valid wildcard-prefix chain
    path = next(
        (sorted(p, key=lambda s: (len(s), s))
         for p in expand_choices(label_list, sep=sep)
         if is_prefix_chain_wild(sorted(p, key=lambda s: (len(s), s)), wildcard)),
        None
    )
    if path is None:
        return {}, 0
    label_idx = {lab: len(lab) for lab in path}
    missing_node = int(len(label_idx) < len(path[-1])-len(path[0])+1)  # keep original rule
    # Map each original token like "a/b" to the first option that appears in label_idx
    origin = {}
    for lab in label_list:
        val = next((label_idx[o] for o in lab.split(sep) if o in label_idx), None)
        if val is not None:
            origin[lab] = val
    return origin, missing_node


def compute_phate_embedding(path_name, base_dir, result_dir):
    """
    Compute PHATE 1D embedding for one lineage path.
    
    Args:
        path_name: Name of the path
        base_dir: Base directory for data
        result_dir: Directory to save results
        
    Returns:
        data: AnnData object with PHATE embeddings
        phate_1d: PHATE 1D embeddings array
    """
    # Load pre-trained scVI data
    print(f"Loading data for {path_name}...")
    scvi_data_path = os.path.join(base_dir, f"scvi_path_{path_name}", "trained.h5ad")
    
    if not os.path.exists(scvi_data_path):
        raise FileNotFoundError(
            f"Pre-trained scVI file not found: {scvi_data_path}\n"
            f"Please run scVI training first (e.g., scvi_celegan_path.py)"
        )
    
    data = anndata.read_h5ad(scvi_data_path)
    print(f"  Loaded {len(data)} cells")
    
    # Preprocess data
    print(f"  Preprocessing data...")
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.scale(data, max_value=10)
    sc.tl.pca(data, svd_solver="arpack")
    
    # Compute PHATE 1D embedding
    print(f"  Computing PHATE 1D embedding...")
    phate_operator = phate.PHATE(n_components=1)
    phate_1d = phate_operator.fit_transform(data.obsm['X_pca'])
    
    # Store embeddings
    data.obsm["X_phate"] = phate_1d
    
    # Save 1D coordinate
    save_path = os.path.join(result_dir, f"phate_{path_name}")
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'phate_1d.npy'), phate_1d)
    
    return data, phate_1d


def compute_correlation(adata, latent_key):
    """Compute correlation between latent embeddings and lineage depth."""
    new_key = "lineage_category"
    if 'lineage' in adata.obs.columns:
        label_idx_dict, if_missing = label_order_index(adata.obs["lineage"].unique().tolist())
        adata.obs[new_key] = adata.obs['lineage'].apply(
            lambda x: label_idx_dict[x]
        )
    else:
        print(f"Warning: No lineage column found in adata")
        return None, None, None, adata
    
    lineage_numeric = pd.to_numeric(adata.obs[new_key]).values
    
    # Flatten latent embeddings to 1D for correlation
    latent_1d = adata.obsm[latent_key].flatten()
    
    # Check for degenerate cases
    if len(np.unique(lineage_numeric)) < 2 or len(np.unique(latent_1d)) < 2:
        print(f"Warning: Insufficient variance for correlation (unique lineages: {len(np.unique(lineage_numeric))}, unique latent values: {len(np.unique(latent_1d))})")
        return np.nan, np.nan, np.nan, adata
    
    try:
        correlation, p_value = spearmanr(lineage_numeric, latent_1d)
        tau, _ = kendalltau(lineage_numeric, latent_1d)
        return abs(correlation), abs(tau), p_value, adata
    except Exception as e:
        print(f"Warning: Error computing correlation: {e}")
        return np.nan, np.nan, np.nan, adata


def compute_clustering_metrics(adata, latent_key, path=None):
    """Fit GMM to latent embeddings and compute ARI and NMI with lineage."""
    lineage_labels = pd.to_numeric(adata.obs['lineage_category']).values
    n_components = len(set(lineage_labels))
    
    # Handle edge case: insufficient components
    if n_components < 2:
        path_msg = f" (path: {path})" if path is not None else ""
        print(f"Warning: Only {n_components} unique lineage(s){path_msg}, skipping clustering metrics")
        return np.nan, np.nan
    
    latent_data = adata.obsm[latent_key].reshape(-1, 1)
    
    # Check for degenerate data
    if len(latent_data) < n_components:
        path_msg = f" (path: {path})" if path is not None else ""
        print(f"Warning: Insufficient data points ({len(latent_data)}) for {n_components} components{path_msg}")
        return np.nan, np.nan
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(latent_data)
    cluster_assignments = gmm.predict(latent_data)
    
    # Compute ARI and NMI
    ari = adjusted_rand_score(cluster_assignments, lineage_labels)
    try:
        nmi = normalized_mutual_info_score(lineage_labels, cluster_assignments, average_method='arithmetic')
    except Exception as e:
        path_msg = f" (path: {path})" if path is not None else ""
        print(f"Warning: Could not calculate NMI{path_msg}: {e}")
        nmi = np.nan
    
    return ari, nmi


def evaluate_phate(data, path_name, result_dir, latent_key="X_phate"):
    """
    Evaluate PHATE results and compute metrics.
    
    Args:
        data: AnnData object with embeddings
        path_name: Name of the path
        result_dir: Directory to save results
        latent_key: Key in obsm for latent embeddings
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Compute correlation with lineage
    correlation, tau, p_value, data = compute_correlation(data, latent_key)
    
    # Compute clustering metrics (ARI and NMI)
    ari, nmi = compute_clustering_metrics(data, latent_key, path=path_name)
    
    # Save annotated data
    save_path = os.path.join(result_dir, f"phate_{path_name}")
    os.makedirs(save_path, exist_ok=True)
    data.write_h5ad(os.path.join(save_path, 'trained.h5ad'))
    
    # Return only requested metrics
    metrics = {
        'path_name': path_name,
        'abs_spearman': correlation,
        'abs_kendall_tau': tau,
        'ari': ari,
        'nmi': nmi,
    }
    
    return metrics


def main():
    """Main function to run PHATE on all paths."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PHATE on all lineage paths")
    parser.add_argument("--remake-plots-only", action="store_true",
                       help="If set, skip PHATE computation and only remake evaluations from existing results")
    args = parser.parse_args()
    
    # Configuration
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    result_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/phate"
    paths_dict_file = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree_new.json.gz"
    
    os.makedirs(result_dir, exist_ok=True)
    
    # Load paths
    paths_dict = load_paths_dict(paths_dict_file)
    print(f"Found {len(paths_dict)} paths to process")
    
    if args.remake_plots_only:
        print("Mode: Remake evaluations only (skipping PHATE computation)")
    
    # Process each path
    rows = []
    for i, path_name in enumerate(sorted(paths_dict.keys())):
        # Check if already processed
        coord_path = os.path.join(result_dir, f"phate_{path_name}", 'phate_1d.npy')
        if not args.remake_plots_only and os.path.exists(coord_path):
            print(f"[{i+1}/{len(paths_dict)}] Skipping {path_name} (already processed)")
            continue
        
        print(f"\n[{i+1}/{len(paths_dict)}] Processing {path_name}...")
        
        try:
            if args.remake_plots_only:
                # Load existing data and remake evaluations
                data_path = os.path.join(result_dir, f"phate_{path_name}", 'trained.h5ad')
                if not os.path.exists(data_path):
                    print(f"  Warning: {data_path} not found, skipping")
                    continue
                
                data = anndata.read_h5ad(data_path)
                if 'X_phate' not in data.obsm:
                    print(f"  Warning: No 'X_phate' embeddings found in {data_path}, skipping")
                    continue
                
                print(f"  Remaking evaluations for {path_name}...")
                # Evaluate without recomputing PHATE
                metrics = evaluate_phate(data, path_name, result_dir, latent_key="X_phate")
            else:
                # Compute PHATE embedding (saves 1D coordinate)
                data, phate_1d = compute_phate_embedding(
                    path_name=path_name,
                    base_dir=base_dir,
                    result_dir=result_dir
                )
                
                # Evaluate
                metrics = evaluate_phate(data, path_name, result_dir)
            
            print(f"Results for {path_name}:")
            print(f"  Abs Spearman: {metrics['abs_spearman']:.4f}")
            print(f"  Abs Kendall's tau: {metrics['abs_kendall_tau']:.4f}")
            print(f"  ARI: {metrics['ari']:.4f}")
            print(f"  NMI: {metrics['nmi']:.4f}")
            
            # Save per-path metrics
            per_path_dir = os.path.join(result_dir, f"phate_{path_name}")
            pd.DataFrame([metrics]).to_csv(os.path.join(per_path_dir, 'metrics.csv'), index=False)
            
            rows.append(metrics)
            
        except Exception as e:
            print(f"ERROR processing {path_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save aggregated results
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(result_dir, 'phate_results.csv'), index=False)
        print(f"\nSaved aggregated results to {result_dir}/phate_results.csv")
        print(f"Processed {len(rows)} paths successfully")
    else:
        print("\nNo paths were processed")


if __name__ == "__main__":
    main()
