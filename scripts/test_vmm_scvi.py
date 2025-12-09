"""
Train VampPrior Mixture Model (VMM) scVI on all C. elegans lineage paths.

This script loads pre-trained scVI embeddings and trains VMM on top of them.
It follows vmm/Celegan-path.py but:
1. Loads pre-existing scVI embeddings from scvi_path_{path}/trained.h5ad
2. Trains only the VMM (VampPrior Mixture) model with TensorFlow
3. Avoids retraining scVI (faster, more consistent with other analyses)

Similar to test_log_norm_gauss.py approach.

Requirements:
- TF_USE_LEGACY_KERAS=1 environment variable must be set
- Pre-trained scVI models in data/scvi_path_*/trained.h5ad
- vmm/priors.py must exist with select_prior() function
- scvae conda environment with TensorFlow dependencies
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import json
import gzip
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add vmm directory to path for imports
vmm_dir = os.path.join(os.path.dirname(__file__), 'vmm')
sys.path.insert(0, vmm_dir)

import numpy as np
import pandas as pd
import anndata
import scipy.sparse as sp
import scanpy
import tensorflow as tf

# Import from vmm directory
import priors
import single_cell_models as sc
from callbacks import PerformanceMonitor

from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import entropy, spearmanr, kendalltau
from scipy.spatial.distance import jensenshannon
from sklearn.mixture import GaussianMixture as GaussianMixture_sklearn
import matplotlib.pyplot as plt
from itertools import product

# Verify TensorFlow Keras setup
print("tf.keras path:", tf.keras.__file__)
assert "tf_keras" in tf.keras.__file__, "TF_USE_LEGACY_KERAS must be set!"

# Check TensorFlow GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow: {len(gpus)} GPU(s) available")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("TensorFlow: Using CPU")


def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)


def prepare_path_data(path_name, base_dir):
    """
    Load and prepare data for one lineage path.
    
    Args:
        path_name: Name of the path (e.g., "MSxap_MSxappppx")
        base_dir: Base directory containing .loom files
        
    Returns:
        data: AnnData object with the filtered cells
        counts: TensorFlow tensor of raw counts
        batch_id: One-hot encoded batch tensor
        library_log_mean: Mean log library size per batch
        library_log_var: Variance log library size per batch
        n_components: Number of lineage components (for num_clusters)
    """
    # Parse path name to get start and end lineages
    parts = path_name.split('_')
    if len(parts) != 2:
        raise ValueError(f"Path name {path_name} should have format 'start_end'")
    
    start, end = parts
    
    # Load full dataset
    adata = scanpy.read(f"/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/data/packer2019_preprocessed.h5ad")
    
    # Create lineage path (all intermediate lineages from start to end)
    lineage_path = [end[:len(start)+i] for i in range(0, len(end)-len(start)+1)]
    
    # Filter cells in this lineage path
    data = adata[adata.obs["lineage"].isin(lineage_path)]
    
    # Sanity check: ensure we have cells
    if len(data) == 0:
        raise ValueError(f"No cells found for path {path_name} with lineages {lineage_path}")
    
    # Get number of unique lineages (components)
    n_components = len(set(data.obs['lineage']))
    
    print(f"  Found {len(data)} cells across {n_components} unique lineages: {sorted(set(data.obs['lineage']))}")
    
    # Sanity check: ensure reasonable number of components
    if n_components < 2:
        print(f"  Warning: Only {n_components} unique lineage(s) found, results may be degenerate")
    
    # Convert to dense counts (raw counts, no normalization)
    counts = tf.convert_to_tensor(data.X.toarray(), dtype=tf.float32)
    
    # Create one-hot batch encoding
    b = data.obs["batch"].astype("category")
    cats = list(b.cat.categories)
    codes = b.cat.codes.to_numpy()
    K = len(cats)
    onehot = np.eye(K, dtype=np.int8)[codes]
    batch_id = tf.convert_to_tensor(onehot, dtype=tf.float32)
    
    # Compute library size statistics per batch
    log_counts_batch = np.ma.log(tf.einsum('ij,ik->ik', tf.cast(counts, tf.float32), batch_id))
    library_log_mean = np.mean(log_counts_batch, axis=0)
    library_log_var = np.var(log_counts_batch, axis=0)
    
    return data, counts, batch_id, library_log_mean, library_log_var, n_components


def train_vmm_scvi(path_name, base_dir, result_dir, max_clusters=5, 
                   max_epochs=10000, patience=100, batch_size=128,
                   trial_seed=42, device=1, use_gpu=True):
    """
    Train VMM scVI model on one lineage path.
    
    This follows the exact training procedure from vmm/Celegan-path.py but loads
    pre-trained scVI embeddings from existing trained.h5ad files:
    1. Load pre-trained scVI embeddings from scvi_path_{path_name}/trained.h5ad
    2. Set up VampPrior pseudo-inputs
    3. Train custom scVI with VampPriorMixture prior
    
    Args:
        path_name: Name of the path
        base_dir: Base directory for data
        result_dir: Directory to save results
        max_clusters: Maximum number of clusters for mixture
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        batch_size: Training batch size
        trial_seed: Random seed
        device: GPU device ID (unused if loading pre-trained)
        
    Returns:
        model: Trained VMM scVI model
        embeddings: Latent embeddings
        cluster_probs: Cluster assignment probabilities
        data: AnnData object with results
        history: Training history
    """
    # Load pre-trained scVI data (like test_log_norm_gauss.py does)
    print(f"Loading pre-trained scVI data for {path_name}...")
    scvi_data_path = os.path.join(base_dir, f"scvi_path_{path_name}", "trained.h5ad")
    
    if not os.path.exists(scvi_data_path):
        raise FileNotFoundError(
            f"Pre-trained scVI file not found: {scvi_data_path}\n"
            f"Please run scVI training first (e.g., scvi_celegan_path.py)"
        )
    
    data = anndata.read_h5ad(scvi_data_path)
    
    # Check for scVI embeddings
    if 'X_scVI' not in data.obsm:
        raise ValueError(f"No X_scVI embeddings found in {scvi_data_path}")
    
    print(f"  Loaded {len(data)} cells with pre-trained scVI embeddings")
    
    # Extract raw counts and batch info for VMM training
    # Note: data.X should still be raw counts in the trained.h5ad
    counts = tf.convert_to_tensor(data.X.toarray() if sp.issparse(data.X) else data.X, dtype=tf.float32)
    
    # Get batch encoding
    b = data.obs["batch"].astype("category")
    cats = list(b.cat.categories)
    codes = b.cat.codes.to_numpy()
    K = len(cats)
    onehot = np.eye(K, dtype=np.int8)[codes]
    batch_id = tf.convert_to_tensor(onehot, dtype=tf.float32)
    
    # Compute library size statistics per batch
    log_counts_batch = np.ma.log(tf.einsum('ij,ik->ik', tf.cast(counts, tf.float32), batch_id))
    library_log_mean = np.mean(log_counts_batch, axis=0)
    library_log_var = np.var(log_counts_batch, axis=0)
    
    # Get number of unique lineages
    n_components = len(set(data.obs['lineage']))
    print(f"  Found {n_components} unique lineages")
    
    # Model configuration
    model_config = {
        'n_layers': 2,
        'hidden_dim': 64,
        'latent_dim': 1,
        'likelihood': 'zinb',
        'dropout_rate': 0.2,
        'learning_rate': 1e-3
    }
    
    # Create save directory
    save_path = os.path.join(result_dir, f"vmm_scvi_{path_name}")
    os.makedirs(save_path, exist_ok=True)
    
    # Get scVI embeddings (already computed)
    embeddings_scvi = data.obsm['X_scVI']
    np.save(os.path.join(save_path, 'scvi_embeddings.npy'), embeddings_scvi)
    
    # Create train/validation split (90/10 like original)
    train_ratio = 0.9
    n_cells = len(data)
    n_train = int(n_cells * train_ratio)
    
    # Set seed for reproducible split
    np.random.seed(trial_seed)
    indices = np.random.permutation(n_cells)
    i_train = indices[:n_train]
    i_valid = indices[n_train:]
    
    # Save indices
    np.save(os.path.join(save_path, 'train_indices.npy'), i_train)
    np.save(os.path.join(save_path, 'valid_indices.npy'), i_valid)
    
    print(f"  Train/valid split: {len(i_train)}/{len(i_valid)} cells")
    
    # ========== Train VMM scVI with VampPrior ==========
    print(f"Training VMM scVI with VampPrior for {path_name}...")
    print(f"  TensorFlow will use GPU if available")
    
    # Prepare train/validation data
    train_data = dict(x=tf.gather(counts, i_train), s=tf.gather(batch_id, i_train))
    valid_data = dict(x=tf.gather(counts, i_valid), s=tf.gather(batch_id, i_valid))
    
    # Set TensorFlow random seed
    tf.keras.utils.set_random_seed(trial_seed)
    tf.config.experimental.enable_op_determinism()
    
    # Create VampPrior pseudo-inputs
    # Note: labels is not used (use_labels=False in config)
    u = sc.vamp_prior_pseudo_inputs(
        count_matrix=counts,
        one_hot_batch_id=batch_id,
        num_clusters=max_clusters,
        cell_labels=None  # use_labels=False
    )
    
    # Configure prior
    prior_config = dict(
        inference='MAP-DP',
        prior_learning_ratio=1.0,
        use_labels=False
    )
    
    # Select and configure latent prior
    latent_prior = priors.select_prior(
        'VampPriorMixture',
        **prior_config,
        latent_dim=model_config['latent_dim'],
        num_clusters=max_clusters,
        u=u,
        learning_rate=model_config['learning_rate'] * prior_config['prior_learning_ratio']
    )
    
    # Build custom scVI model with VampPrior
    model = sc.scVI(
        n_genes=counts.shape[1],
        n_batches=batch_id.shape[1],
        prior=latent_prior,
        use_observed_library_size=True,
        library_log_loc=library_log_mean,
        library_log_scale=library_log_var ** 0.5,
        **model_config
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate']))
    
    # Train VMM scVI
    hist = model.fit(
        x=train_data,
        validation_data=valid_data,
        batch_size=batch_size,
        epochs=max_epochs,
        verbose=False,
        callbacks=[PerformanceMonitor(patience=patience)]
    )
    
    # Save model weights and history
    model.save_weights(os.path.join(save_path, 'vmmscvi_best_checkpoint'))
    with open(os.path.join(save_path, 'vmmscvi_history.pkl'), 'wb') as f:
        pickle.dump(hist.history, f)
    
    # Get final embeddings and cluster probabilities
    tf.keras.utils.set_random_seed(trial_seed)
    vmmscvi_embeddings = model.predict(dict(x=counts, s=batch_id), batch_size=batch_size)
    np.save(os.path.join(save_path, 'vmmscvi_embeddings.npy'), vmmscvi_embeddings)
    
    cluster_probs = model.cluster_probabilities(vmmscvi_embeddings)
    if cluster_probs.shape[1] > 1:
        np.save(os.path.join(save_path, 'vmmscvi_cluster_probs.npy'), cluster_probs.numpy())
    
    data.obsm["vmmscvi"] = vmmscvi_embeddings
    
    return model, vmmscvi_embeddings, cluster_probs, data, hist


def plot_histogram(adata, latent_key, path, result_dir):
    """Plot histogram of latent embeddings by lineage."""
    labels = adata.obs["lineage"]
    unique_items = set(labels)
    unique_lineages = sorted(unique_items, key=lambda s: len(s.split('/')[0]))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_lineages)))
    
    # Get all coordinates to determine global range for shared binning
    all_coords = adata.obsm[latent_key].flatten()
    global_min = np.min(all_coords)
    global_max = np.max(all_coords)
    
    # Create shared bin edges (201 edges = 200 bins)
    # Using more bins since we're now binning across all lineages, not per lineage
    bin_edges = np.linspace(global_min, global_max, 201)
    
    plt.figure(figsize=(10, 6))
    for i, lineage in enumerate(unique_lineages):
        lineage_coords = adata.obsm[latent_key][adata.obs["lineage"] == lineage]
        if len(lineage_coords) > 0:
            # Add cell count to legend label
            n_cells = len(lineage_coords)
            label = f"{lineage} (n={n_cells})"
            plt.hist(lineage_coords, bins=bin_edges, alpha=0.5, label=label, 
                    color=colors[i], density=True)
    
    plt.xlabel("Z")
    plt.ylabel("Density")
    plt.title(f"Histogram of VMM scVI Z by lineage: {path}")
    plt.legend(title="lineage", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_dir = os.path.join(result_dir, path)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'vmm_scvi_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()

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
    #print(label_idx)
    #print(path)
    missing_node = int(len(label_idx) < len(path[-1])-len(path[0])+1)  # keep original rule
    # Map each original token like "a/b" to the first option that appears in label_idx
    origin = {}
    for lab in label_list:
        val = next((label_idx[o] for o in lab.split(sep) if o in label_idx), None)
        if val is not None:
            origin[lab] = val
    return origin, missing_node

def compute_correlation(adata, latent_key):
    """Compute correlation between latent embeddings and lineage depth."""
    new_key = "lineage_category"
    if 'lineage' in adata.obs.columns:
        label_idx_dict, if_missing = label_order_index(adata.obs["lineage"].unique().tolist())
        adata.obs[new_key] = adata.obs['lineage'].apply(
            # lambda x: len(x.split("/")[0]) if "/" in x else len(x)
            lambda x: label_idx_dict[x]
        )
    else:
        print(f"Warning: No lineage column found in adata")
        return None, None, None, adata
    
    lineage_numeric = pd.to_numeric(adata.obs[new_key]).values
    
    # Fix: Flatten latent embeddings to 1D for correlation
    latent_1d = adata.obsm[latent_key].flatten()
    
    # Check for degenerate cases
    if len(np.unique(lineage_numeric)) < 2 or len(np.unique(latent_1d)) < 2:
        print(f"Warning: Insufficient variance for correlation (unique lineages: {len(np.unique(lineage_numeric))}, unique latent values: {len(np.unique(latent_1d))})")
        return np.nan, np.nan, np.nan, adata
    
    try:
        correlation, p_value = spearmanr(lineage_numeric, latent_1d)
        tau, _ = kendalltau(lineage_numeric, latent_1d)
        return abs(correlation), tau, p_value, adata
    except Exception as e:
        print(f"Warning: Error computing correlation: {e}")
        return np.nan, np.nan, np.nan, adata


def fit_gmm_and_analyze(adata, latent_key, path=None):
    """Fit GMM to latent embeddings and compute metrics."""
    lineage_labels = pd.to_numeric(adata.obs['lineage_category']).values
    n_components = len(set(lineage_labels))
    
    # Handle edge case: insufficient components
    if n_components < 2:
        path_msg = f" (path: {path})" if path is not None else ""
        print(f"Warning: Only {n_components} unique lineage(s){path_msg}, skipping GMM analysis")
        return None, None
    
    latent_data = adata.obsm[latent_key].reshape(-1, 1)
    
    # Check for degenerate data
    if len(latent_data) < n_components:
        path_msg = f" (path: {path})" if path is not None else ""
        print(f"Warning: Insufficient data points ({len(latent_data)}) for {n_components} components{path_msg}")
        return None, None
    
    gmm = GaussianMixture_sklearn(n_components=n_components, random_state=42)
    gmm.fit(latent_data)
    
    # Compute entropy
    probabilities = gmm.predict_proba(latent_data)
    point_entropies = [entropy(probs) for probs in probabilities]
    entropy_value = np.mean(point_entropies)
    
    # Cluster assignments
    cluster_assignments = gmm.predict(latent_data)
    
    # ARI with lineage
    ari_with_lineage = adjusted_rand_score(cluster_assignments, lineage_labels)
    
    # Mixture proportion metrics
    gmm_proportions = gmm.weights_
    lineage_counts = np.bincount(lineage_labels.astype(int))
    lineage_proportions = lineage_counts / np.sum(lineage_counts)
    
    # Pad or truncate to match dimensions
    if len(lineage_proportions) < n_components:
        padded_lineage = np.zeros(n_components)
        padded_lineage[:len(lineage_proportions)] = lineage_proportions
        lineage_proportions = padded_lineage
    elif len(lineage_proportions) > n_components:
        lineage_proportions = lineage_proportions[:n_components]
        lineage_proportions = lineage_proportions / np.sum(lineage_proportions)
    
    js_divergence = jensenshannon(gmm_proportions, lineage_proportions)
    kl_divergence = entropy(gmm_proportions, lineage_proportions)
    lin_correlation = np.corrcoef(gmm_proportions, lineage_proportions)[0, 1]
    mae = np.mean(np.abs(gmm_proportions - lineage_proportions))
    rmse = np.sqrt(np.mean((gmm_proportions - lineage_proportions) ** 2))
    
    mixture_proportion_metrics = {
        'js_divergence': js_divergence,
        'kl_divergence': kl_divergence,
        'correlation': lin_correlation,
        'mae': mae,
        'rmse': rmse
    }
    
    # Calculate silhouette score
    try:
        silhouette_val = silhouette_score(latent_data, cluster_assignments)
    except ValueError as e:
        path_msg = f" (path: {path})" if path is not None else ""
        print(f"Warning: Could not calculate silhouette score{path_msg}: {e}")
        silhouette_val = np.nan
    
    gmm_metrics = {
        'entropy': entropy_value,
        'bic': gmm.bic(latent_data),
        'aic': gmm.aic(latent_data),
        'log_likelihood': gmm.score(latent_data),
        'perplexity': np.exp(-gmm.score(latent_data) / len(latent_data)),
        'silhouette': silhouette_val,
        'ari_with_lineage': ari_with_lineage,
    }
    
    return gmm_metrics, mixture_proportion_metrics


def evaluate_vmm_scvi(data, path_name, result_dir, latent_key="vmmscvi"):
    """
    Evaluate VMM scVI results and compute metrics.
    
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
    
    # Plot histogram
    plot_histogram(data, latent_key, path_name, result_dir)
    
    # Fit GMM and analyze
    gmm_metrics, mixture_metrics = fit_gmm_and_analyze(data, latent_key, path=path_name)
    
    # Save annotated data
    save_path = os.path.join(result_dir, f"vmm_scvi_{path_name}")
    data.write_h5ad(os.path.join(save_path, 'trained.h5ad'))
    
    # Combine metrics
    metrics = {
        'path_name': path_name,
        'correlation': correlation,
        'kendall_tau': tau,
        'p_value': p_value,
    }
    
    # Handle None returns from GMM analysis (edge cases)
    if gmm_metrics is not None:
        metrics.update({k: (v.item() if hasattr(v, 'item') else v) for k, v in gmm_metrics.items()})
    else:
        metrics.update({
            'entropy': np.nan,
            'bic': np.nan,
            'aic': np.nan,
            'log_likelihood': np.nan,
            'perplexity': np.nan,
            'silhouette': np.nan,
            'ari_with_lineage': np.nan,
        })
    
    if mixture_metrics is not None:
        metrics.update({k: (v.item() if hasattr(v, 'item') else v) for k, v in mixture_metrics.items()})
    else:
        metrics.update({
            'js_divergence': np.nan,
            'kl_divergence': np.nan,
            'mixture_correlation': np.nan,
            'mae': np.nan,
            'rmse': np.nan
        })
    
    return metrics


def main():
    """Main function to run VMM scVI on all paths."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VMM scVI on all lineage paths")
    parser.add_argument("--remake-plots-only", action="store_true",
                       help="If set, skip training and only remake plots/evaluations from existing results")
    args = parser.parse_args()
    
    # Configuration
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    result_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/vmm_scvi"
    paths_dict_file = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree_small.json.gz"
    
    os.makedirs(result_dir, exist_ok=True)
    
    # Load paths
    paths_dict = load_paths_dict(paths_dict_file)
    print(f"Found {len(paths_dict)} paths to process")
    
    if args.remake_plots_only:
        print("Mode: Remake plots only (skipping training)")
    
    # Training configuration
    max_clusters = 5
    max_epochs = 10000
    patience = 100
    batch_size = 128
    trial_seed = 42
    device = 1  # GPU device ID
    
    # Process each path
    rows = []
    for i, path_name in enumerate(sorted(paths_dict.keys())):
        # Check if already processed
        plot_path = os.path.join(result_dir, path_name, 'vmm_scvi_histogram.png')
        if not args.remake_plots_only and os.path.exists(plot_path):
            print(f"[{i+1}/{len(paths_dict)}] Skipping {path_name} (already processed)")
            continue
        
        print(f"\n[{i+1}/{len(paths_dict)}] Processing {path_name}...")
        
        try:
            if args.remake_plots_only:
                # Load existing data and remake plots/evaluations
                data_path = os.path.join(result_dir, path_name, 'trained.h5ad')
                if not os.path.exists(data_path):
                    print(f"  Warning: {data_path} not found, skipping")
                    continue
                
                data = anndata.read_h5ad(data_path)
                if 'vmmscvi' not in data.obsm:
                    print(f"  Warning: No 'vmmscvi' embeddings found in {data_path}, skipping")
                    continue
                
                print(f"  Remaking plots/evaluations for {path_name}...")
                # Evaluate without training
                metrics = evaluate_vmm_scvi(data, path_name, result_dir)
            else:
                # Train model
                model, embeddings, cluster_probs, data, history = train_vmm_scvi(
                    path_name=path_name,
                    base_dir=base_dir,
                    result_dir=result_dir,
                    max_clusters=max_clusters,
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    trial_seed=trial_seed,
                    device=device,
                    use_gpu=True
                )
                
                # Evaluate
                metrics = evaluate_vmm_scvi(data, path_name, result_dir)
            
            print(f"Results for {path_name}:")
            print(f"  Correlation: {metrics['correlation']:.4f}")
            print(f"  ARI: {metrics['ari_with_lineage']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            
            # Save per-path metrics
            per_path_dir = os.path.join(result_dir, path_name)
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
        df.to_csv(os.path.join(result_dir, 'vmm_scvi_results.csv'), index=False)
        print(f"\nSaved aggregated results to {result_dir}/vmm_scvi_results.csv")
        print(f"Processed {len(rows)} paths successfully")
    else:
        print("\nNo paths were processed")


if __name__ == "__main__":
    main()

