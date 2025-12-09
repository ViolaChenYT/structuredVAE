"""
Train Wasserstein Autoencoder (WAE) 1D on all C. elegans lineage paths.

This script loads pre-trained scVI data and trains WAE1D models on top of them.
It follows test_vmm_scvi.py but:
1. Loads pre-existing scVI data from scvi_path_{path}/trained.h5ad
2. Trains WAE1D model with PyTorch
3. Uses GMM prior with number of clusters based on unique lineages

Requirements:
- Pre-trained scVI models in data/scvi_path_*/trained.h5ad
- wae1d module must exist (wae.py, encoder.py, decoder.py)
- PyTorch with CUDA support (optional)
"""

import os
import sys
import json
import gzip
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add wae1d directory to path for imports
wae1d_dir = os.path.join(os.path.dirname(__file__), 'wae1d')
sys.path.insert(0, wae1d_dir)

import numpy as np
import pandas as pd
import anndata
import scipy.sparse as sp
import scanpy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical, Normal, MixtureSameFamily

# Import WAE1D from wae1d module
from wae import WAE1D

from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import entropy, spearmanr, kendalltau
from scipy.spatial.distance import jensenshannon
from sklearn.mixture import GaussianMixture as GaussianMixture_sklearn
import matplotlib.pyplot as plt
from itertools import product

# Check PyTorch device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: Using {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)


def train_wae(path_name, base_dir, result_dir, max_clusters=5,
              max_epochs=1500, warmup_epochs=500, batch_size=128,
              learning_rate=1e-3, h_dim=128, n_layers_enc=2, n_layers_dec=2,
              trial_seed=42, use_gpu=True):
    """
    Train WAE1D model on one lineage path.
    
    This loads pre-trained scVI data and trains WAE1D with GMM prior:
    1. Load pre-trained scVI data from scvi_path_{path_name}/trained.h5ad
    2. Set up GMM prior with n_clusters based on unique lineages
    3. Train WAE1D with Negative Binomial likelihood
    
    Args:
        path_name: Name of the path
        base_dir: Base directory for data
        result_dir: Directory to save results
        max_clusters: Maximum number of clusters for GMM prior
        max_epochs: Maximum training epochs
        warmup_epochs: Epochs with weight=0.0 before enabling Wasserstein loss
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        h_dim: Hidden dimension
        n_layers_enc: Number of encoder layers
        n_layers_dec: Number of decoder layers
        trial_seed: Random seed
        use_gpu: Whether to use GPU
        
    Returns:
        model: Trained WAE1D model
        embeddings: Latent embeddings
        data: AnnData object with results
        history: Training history dictionary
    """
    # Load pre-trained scVI data
    print(f"Loading pre-trained scVI data for {path_name}...")
    scvi_data_path = os.path.join(base_dir, f"scvi_path_{path_name}", "trained.h5ad")
    
    if not os.path.exists(scvi_data_path):
        raise FileNotFoundError(
            f"Pre-trained scVI file not found: {scvi_data_path}\n"
            f"Please run scVI training first (e.g., scvi_celegan_path.py)"
        )
    
    data = anndata.read_h5ad(scvi_data_path)
    
    # Check for scVI embeddings (optional, just for reference)
    if 'X_scVI' not in data.obsm:
        print(f"  Warning: No X_scVI embeddings found in {scvi_data_path}")
    else:
        print(f"  Loaded {len(data)} cells with pre-trained scVI embeddings")
    
    # Extract raw counts
    # Note: data.X should still be raw counts in the trained.h5ad
    if sp.issparse(data.X):
        counts = data.X.toarray()
    else:
        counts = data.X.copy()
    
    # Convert to PyTorch tensor
    X = torch.from_numpy(counts).float()
    
    # Get number of unique lineages
    n_components = len(set(data.obs['lineage']))
    print(f"  Found {n_components} unique lineages: {sorted(set(data.obs['lineage']))}")
    
    # Use min(n_components, max_clusters) for GMM prior
    n_clusters = min(n_components, max_clusters)
    if n_components < 2:
        print(f"  Warning: Only {n_components} unique lineage(s) found, results may be degenerate")
    
    # Create save directory
    save_path = os.path.join(result_dir, f"wae_{path_name}")
    os.makedirs(save_path, exist_ok=True)
    
    # Create train/validation split (90/10)
    train_ratio = 0.9
    n_cells = len(data)
    n_train = int(n_cells * train_ratio)
    
    # Set seed for reproducible split
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)
    
    indices = np.random.permutation(n_cells)
    i_train = indices[:n_train]
    i_valid = indices[n_train:]
    
    # Save indices
    np.save(os.path.join(save_path, 'train_indices.npy'), i_train)
    np.save(os.path.join(save_path, 'valid_indices.npy'), i_valid)
    
    print(f"  Train/valid split: {len(i_train)}/{len(i_valid)} cells")
    
    # ========== Set up GMM prior ==========
    print(f"Setting up GMM prior with {n_clusters} clusters...")
    
    # Initialize GMM prior with uniform mixture probabilities
    mix_probs = torch.tensor([1.0 / n_clusters for _ in range(n_clusters)])
    
    # Initialize component means (spread evenly)
    # Use a range that should cover the latent space after initialization
    initial_range = 30.0  # Similar to wae_prosstt.py
    means = torch.tensor([-initial_range/2 + (initial_range / (n_clusters - 1)) * i 
                          if n_clusters > 1 else 0.0 
                          for i in range(n_clusters)], dtype=torch.float32)
    
    # Initialize component stds
    stds = torch.ones(n_clusters, dtype=torch.float32)
    
    # Create GMM prior
    mix = Categorical(mix_probs)
    comp = Normal(means, stds)
    gmm_prior = MixtureSameFamily(mix, comp)
    
    # ========== Initialize WAE1D model ==========
    print(f"Initializing WAE1D model...")
    print(f"  Input dim: {X.shape[1]}, Hidden dim: {h_dim}")
    print(f"  Encoder layers: {n_layers_enc}, Decoder layers: {n_layers_dec}")
    print(f"  Likelihood: Negative Binomial")
    
    model = WAE1D(
        prior=gmm_prior,
        in_dim=X.shape[1],
        x_dim=X.shape[1],
        h_dim=h_dim,
        n_layers_enc=n_layers_enc,
        n_layers_dec=n_layers_dec,
        embedding_dim=1,
        likelihood="nb",
        auto_init=1,  # Allow prior initialization from data
        device=device
    )
    
    if use_gpu and torch.cuda.is_available():
        model = model.to(device)
        print(f"  Model moved to {device}")
    else:
        print(f"  Model on CPU")
    
    # ========== Training setup ==========
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create DataLoader
    train_dataset = TensorDataset(X[i_train])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataset = TensorDataset(X[i_valid])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    num_train_batches = len(train_loader)
    
    # ========== Training loop ==========
    print(f"Training WAE1D for {path_name}...")
    print(f"  Max epochs: {max_epochs}, Warmup epochs: {warmup_epochs}")
    
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_w2': [],
        'valid_loss': [],
        'valid_recon_loss': [],
        'valid_w2': []
    }
    
    best_valid_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, max_epochs + 1):
        # Determine Wasserstein weight
        if epoch <= warmup_epochs:
            w_weight = 0.0
        else:
            w_weight = 1.0
        
        # Training phase
        model.train()
        train_metrics = {'loss': 0.0, 'recon_loss': 0.0, 'wasserstein_distance': 0.0}
        
        for (xb,) in train_loader:
            xb = xb.to(device).float()
            
            optimizer.zero_grad()
            loss, metrics = model(xb, weight=w_weight)
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            for k in train_metrics.keys():
                train_metrics[k] += metrics[k].item()
        
        # Average metrics over batches
        for k in train_metrics.keys():
            train_metrics[k] /= num_train_batches
        
        # Validation phase
        model.eval()
        valid_metrics = {'loss': 0.0, 'recon_loss': 0.0, 'wasserstein_distance': 0.0}
        num_valid_batches = len(valid_loader)
        
        with torch.no_grad():
            for (xb,) in valid_loader:
                xb = xb.to(device).float()
                loss, metrics = model(xb, weight=w_weight)
                
                # Accumulate metrics
                for k in valid_metrics.keys():
                    valid_metrics[k] += metrics[k].item()
        
        # Average metrics over batches
        for k in valid_metrics.keys():
            valid_metrics[k] /= num_valid_batches
        
        # Track best model
        if valid_metrics['loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['loss']
            best_model_state = model.state_dict().copy()
        
        # Store history
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['train_w2'].append(train_metrics['wasserstein_distance'])
        history['valid_loss'].append(valid_metrics['loss'])
        history['valid_recon_loss'].append(valid_metrics['recon_loss'])
        history['valid_w2'].append(valid_metrics['wasserstein_distance'])
        
        # Print progress
        if epoch == 1 or epoch % 100 == 0 or epoch == max_epochs:
            print(f"  Epoch {epoch}/{max_epochs}: "
                  f"train_loss={train_metrics['loss']:.4f}, "
                  f"valid_loss={valid_metrics['loss']:.4f}, "
                  f"w_weight={w_weight}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Loaded best model (valid_loss={best_valid_loss:.4f})")
    
    # Save model weights and history
    torch.save(model.state_dict(), os.path.join(save_path, 'wae_best_checkpoint.pth'))
    with open(os.path.join(save_path, 'wae_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # Get final embeddings
    model.eval()
    with torch.no_grad():
        X_log1p = torch.log1p(X)
        if use_gpu and torch.cuda.is_available():
            latent_params = model.encoder(X_log1p.to(device))
        else:
            latent_params = model.encoder(X_log1p)
        wae_embeddings = latent_params.detach().cpu().numpy()
    
    np.save(os.path.join(save_path, 'wae_embeddings.npy'), wae_embeddings)
    
    # Add embeddings to AnnData
    data.obsm["wae"] = wae_embeddings
    
    return model, wae_embeddings, data, history


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
    plt.title(f"Histogram of WAE Z by lineage: {path}")
    plt.legend(title="lineage", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_dir = os.path.join(result_dir, f"wae_{path}")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'wae_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstruction_loss(history, path_name, result_dir):
    """Plot reconstruction loss from training history."""
    if history is None:
        return
    
    # Check if we have any reconstruction loss data
    has_train = 'train_recon_loss' in history and len(history['train_recon_loss']) > 0
    has_valid = 'valid_recon_loss' in history and len(history['valid_recon_loss']) > 0
    
    if not has_train and not has_valid:
        print(f"  Warning: No reconstruction loss data found in history for {path_name}")
        return
    
    plot_dir = os.path.join(result_dir, f"wae_{path_name}")
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    # Determine epoch range from available data
    if has_train:
        epochs = np.arange(1, len(history['train_recon_loss']) + 1)
        plt.plot(epochs, history['train_recon_loss'], label='Train Reconstruction Loss', alpha=0.7)
    
    if has_valid:
        # Use same epoch range or create new one if train doesn't exist
        if not has_train:
            epochs = np.arange(1, len(history['valid_recon_loss']) + 1)
        plt.plot(epochs, history['valid_recon_loss'], label='Validation Reconstruction Loss', alpha=0.7)
    
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title(f"Reconstruction Loss: {path_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'reconstruction_loss.png'), dpi=150, bbox_inches='tight')
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
        return abs(correlation), abs(tau), p_value, adata
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
    
    # NMI with lineage
    try:
        nmi_with_lineage = normalized_mutual_info_score(lineage_labels, cluster_assignments, average_method='arithmetic')
    except Exception as e:
        path_msg = f" (path: {path})" if path is not None else ""
        print(f"Warning: Could not calculate NMI{path_msg}: {e}")
        nmi_with_lineage = np.nan
    
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
        'nmi_with_lineage': nmi_with_lineage,
    }
    
    return gmm_metrics, mixture_proportion_metrics


def evaluate_wae(data, path_name, result_dir, latent_key="wae", history=None):
    """
    Evaluate WAE results and compute metrics.
    
    Args:
        data: AnnData object with embeddings
        path_name: Name of the path
        result_dir: Directory to save results
        latent_key: Key in obsm for latent embeddings
        history: Optional training history dictionary for plotting reconstruction loss
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Compute correlation with lineage
    correlation, tau, p_value, data = compute_correlation(data, latent_key)
    
    # Plot histogram
    plot_histogram(data, latent_key, path_name, result_dir)
    
    # Plot reconstruction loss if history is provided
    if history is not None:
        plot_reconstruction_loss(history, path_name, result_dir)
    
    # Fit GMM and analyze
    gmm_metrics, mixture_metrics = fit_gmm_and_analyze(data, latent_key, path=path_name)
    
    # Save annotated data
    save_path = os.path.join(result_dir, f"wae_{path_name}")
    data.write_h5ad(os.path.join(save_path, 'trained.h5ad'))
    
    # Combine metrics - focus on key metrics requested
    metrics = {
        'path_name': path_name,
        'abs_spearman': correlation,  # Already abs from compute_correlation
        'abs_kendall_tau': tau,  # Already abs from compute_correlation
        'ari': None,  # Will be filled from gmm_metrics
        'nmi': None,  # Will be filled from gmm_metrics
        'p_value': p_value,
    }
    
    # Handle None returns from GMM analysis (edge cases)
    if gmm_metrics is not None:
        # Update key metrics
        metrics['ari'] = gmm_metrics.get('ari_with_lineage', np.nan)
        metrics['nmi'] = gmm_metrics.get('nmi_with_lineage', np.nan)
        # Also include all other metrics
        metrics.update({k: (v.item() if hasattr(v, 'item') else v) for k, v in gmm_metrics.items()})
    else:
        metrics['ari'] = np.nan
        metrics['nmi'] = np.nan
        metrics.update({
            'entropy': np.nan,
            'bic': np.nan,
            'aic': np.nan,
            'log_likelihood': np.nan,
            'perplexity': np.nan,
            'silhouette': np.nan,
            'ari_with_lineage': np.nan,
            'nmi_with_lineage': np.nan,
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
    """Main function to run WAE on all paths."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train WAE1D on all lineage paths")
    parser.add_argument("--remake-plots-only", action="store_true",
                       help="If set, skip training and only remake plots/evaluations from existing results")
    parser.add_argument("--remake-reconstruction-plots-only", action="store_true",
                       help="If set, only remake reconstruction loss plots from existing wae_history.pkl files")
    args = parser.parse_args()
    
    # Configuration
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    result_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/wae"
    paths_dict_file = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree_new.json.gz"
    
    os.makedirs(result_dir, exist_ok=True)
    
    # Load paths
    paths_dict = load_paths_dict(paths_dict_file)
    print(f"Found {len(paths_dict)} paths to process")
    
    if args.remake_plots_only:
        print("Mode: Remake plots only (skipping training)")
    if args.remake_reconstruction_plots_only:
        print("Mode: Remake reconstruction loss plots only (skipping training and other plots)")
    
    # Training configuration
    max_clusters = 5
    max_epochs = 1500
    warmup_epochs = 500
    batch_size = 128
    learning_rate = 1e-3
    h_dim = 128
    n_layers_enc = 2
    n_layers_dec = 2
    trial_seed = 42
    use_gpu = True
    
    # Process each path
    rows = []
    for i, path_name in enumerate(sorted(paths_dict.keys())):
        # Check if already processed
        plot_path = os.path.join(result_dir, f"wae_{path_name}", 'wae_histogram.png')
        if not args.remake_plots_only and not args.remake_reconstruction_plots_only and os.path.exists(plot_path):
            print(f"[{i+1}/{len(paths_dict)}] Skipping {path_name} (already processed)")
            continue
        
        print(f"\n[{i+1}/{len(paths_dict)}] Processing {path_name}...")
        
        try:
            history = None
            if args.remake_reconstruction_plots_only:
                # Only remake reconstruction loss plots
                history_path = os.path.join(result_dir, f"wae_{path_name}", 'wae_history.pkl')
                if os.path.exists(history_path):
                    with open(history_path, 'rb') as f:
                        history = pickle.load(f)
                    plot_reconstruction_loss(history, path_name, result_dir)
                    print(f"✓ Successfully remade reconstruction loss plot for {path_name}")
                else:
                    print(f"  Warning: {history_path} not found, skipping")
                continue
            elif args.remake_plots_only:
                # Load existing data and remake plots/evaluations
                data_path = os.path.join(result_dir, f"wae_{path_name}", 'trained.h5ad')
                if not os.path.exists(data_path):
                    print(f"  Warning: {data_path} not found, skipping")
                    continue
                
                data = anndata.read_h5ad(data_path)
                if 'wae' not in data.obsm:
                    print(f"  Warning: No 'wae' embeddings found in {data_path}, skipping")
                    continue
                
                # Try to load history for plotting reconstruction loss
                history_path = os.path.join(result_dir, f"wae_{path_name}", 'wae_history.pkl')
                if os.path.exists(history_path):
                    with open(history_path, 'rb') as f:
                        history = pickle.load(f)
                
                print(f"  Remaking plots/evaluations for {path_name}...")
            else:
                # Train model
                model, embeddings, data, history = train_wae(
                    path_name=path_name,
                    base_dir=base_dir,
                    result_dir=result_dir,
                    max_clusters=max_clusters,
                    max_epochs=max_epochs,
                    warmup_epochs=warmup_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    h_dim=h_dim,
                    n_layers_enc=n_layers_enc,
                    n_layers_dec=n_layers_dec,
                    trial_seed=trial_seed,
                    use_gpu=use_gpu
                )
            
            # Evaluate
            metrics = evaluate_wae(data, path_name, result_dir, history=history)
            
            print(f"Results for {path_name}:")
            print(f"  Abs Spearman: {metrics['abs_spearman']:.4f}")
            print(f"  Abs Kendall's tau: {metrics['abs_kendall_tau']:.4f}")
            print(f"  ARI: {metrics['ari']:.4f}")
            print(f"  NMI: {metrics['nmi']:.4f}")
            
            # Save per-path metrics
            per_path_dir = os.path.join(result_dir, f"wae_{path_name}")
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
        df.to_csv(os.path.join(result_dir, 'wae_results.csv'), index=False)
        print(f"\nSaved aggregated results to {result_dir}/wae_results.csv")
        print(f"Processed {len(rows)} paths successfully")
    else:
        print("\nNo paths were processed")


if __name__ == "__main__":
    main()

