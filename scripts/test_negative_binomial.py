import os
import json
import gzip
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import anndata
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import argparse
from src.models import *
from src.priors import *
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, kendalltau
from sklearn.mixture import GaussianMixture as GaussianMixture_sklearn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from itertools import product
import scanpy as sc
def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)

def expand_choices(items, sep="/", strip=True, dedup=False):
    """
    items: list[str] possibly containing 'a/b' meaning choose one
    returns: list[list[str]] of all expanded lists
    """
    opts = []
    for s in items:
        parts = s.split(sep)
        if strip:
            parts = [p.strip() for p in parts]
        opts.append(parts if len(parts) > 1 else [parts[0]])
    combos = [list(choice) for choice in product(*opts)]
    if dedup:
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
    missing_node = int(len(label_idx) < len(path[-1])-len(path[0])+1)
    origin = {}
    for lab in label_list:
        val = next((label_idx[o] for o in lab.split(sep) if o in label_idx), None)
        if val is not None:
            origin[lab] = val
    return origin, missing_node
def scanpy_norm_log1p_from_torch(X: torch.Tensor) -> torch.Tensor:
    # move to CPU + numpy (Scanpy expects numpy/scipy)
    X_np = X.detach().cpu().numpy().astype(np.float32, copy=False)

    adata = sc.AnnData(X_np)                 # create AnnData
    sc.pp.normalize_total(adata, target_sum=None, inplace=True)  # Scanpy normalize_total
    sc.pp.log1p(adata)                       # Scanpy log1p (natural log)

    # back to torch, preserve original device & dtype
    X_out = torch.from_numpy(adata.X).to(X.device).type_as(X)
    return X_out 

def training_negative_binomial(path_name, base_dir, result_dir, device="cpu",batch_size=128,lr=1e-3,weight_decay=1e-5,early_stopping=True,patience=200,epochs=700,min_delta=1e-4):
    # data = anndata.read_loom(f"{base_dir}/{path_name}.loom")
    data = sc.read_h5ad(f"{base_dir}/scvi_path_{path_name}/trained.h5ad")
    n_components = len(set(data.obs['lineage']))
    model_prior = GaussianMixture(latent_dim=1, num_clusters=n_components)
    model_encoder = build_encoder(dim_x=2000, h_dim=64, n_layers=2)
    model_decoder = build_decoder_nb(dim_x=2000, latent_dim=1, h_dim=64, n_layers=2)
    model = EmpiricalBayesVariationalAutoencoder(encoder=model_encoder, enc_out_dim=64, decoder=model_decoder, prior=model_prior).to(device)
    
    # Convert sparse matrix to dense and then to tensor
    X_dense = torch.tensor(data.X.todense(), dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_dense), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    losses_history = []
    best_loss = float('inf')
    patience_counter = 0
    early_stop = False
    for epoch in range(epochs):
        kl_w = 1
        tot = 0.0
        n = 0
        epoch_losses = {}
        for (xb,) in dl:
            if xb.size(0) == 1:
                # print(f"Warning: xb.size(0) == 1, skipping batch")
                continue
            xb = xb.to(device).float()
            outputs = model.train_step(xb, opt)
            # print(outputs)
            loss = outputs["vae-loss"]
            losses = {"loss": loss}
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item() * xb.size(0)
            tot += losses["loss"].item() * xb.size(0)
            n += xb.size(0)
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n
        losses_history.append(epoch_losses)
        current_loss = tot/n
        # Early stopping check
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"[{epoch:03d}] loss={current_loss:.3f} (best: {best_loss:.3f}, patience: {patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience: {patience})")
            early_stop = True
            break
    if not early_stop:
        print("Training completed!")
    else:
        print(f"Training stopped early at epoch {epoch}")
    
    print(f"Final loss: {current_loss:.4f}, Best loss: {best_loss:.4f}")
    print(f"Total epochs: {epoch}")
    
    # Save losses_history to CSV
    per_path_dir = os.path.join(result_dir, "plots", path_name)
    os.makedirs(per_path_dir, exist_ok=True)
    losses_df = pd.DataFrame(losses_history)
    losses_df.to_csv(os.path.join(per_path_dir, 'losses_history.csv'), index=False)
    
    return model, losses_history

def plot_reconstruction_loss(losses_history, path, result_dir):
    """Plot reconstruction loss from training history."""
    if losses_history is None or len(losses_history) == 0:
        return
    
    per_path_dir = os.path.join(result_dir, "plots", path)
    os.makedirs(per_path_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(losses_history) + 1)
    
    # Extract loss values from losses_history
    loss_values = [epoch_loss.get('loss', np.nan) for epoch_loss in losses_history]
    
    plt.plot(epochs, loss_values, label='Training Loss', alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Reconstruction Loss: {path}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(per_path_dir, 'reconstruction_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_histogram(adata,latent_key, path, result_dir="/n/fs/ragr-data/users/viola/structuredVAE/results/plots/"):
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
    
    for i, lineage in enumerate(unique_lineages):
        lineage_coords = adata.obsm[latent_key][adata.obs["lineage"] == lineage]
        if len(lineage_coords) > 0:
            # Add cell count to legend label
            n_cells = len(lineage_coords)
            label = f"{lineage} (n={n_cells})"
            plt.hist(lineage_coords, bins=bin_edges, alpha=0.5, label=label, color=colors[i], density=True)
    plt.xlabel("Z")
    plt.ylabel("Density")
    plt.title("Histogram of GMMVAE Z by lineage(NB)")
    plt.legend(title="lineage")
    plt.grid(True, alpha=0.3)
    if not os.path.exists(f"{result_dir}/{path}"):
        os.makedirs(f"{result_dir}/{path}")
    plt.savefig(f"{result_dir}/{path}/NB_GMM_histogram.png")
    plt.close()
    return adata

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
        return np.nan, np.nan, adata
    
    lineage_numeric = pd.to_numeric(adata.obs[new_key]).values
    
    # Flatten latent embeddings to 1D for correlation
    latent_1d = adata.obsm[latent_key].flatten()
    
    # Check for degenerate cases
    if len(np.unique(lineage_numeric)) < 2 or len(np.unique(latent_1d)) < 2:
        print(f"Warning: Insufficient variance for correlation (unique lineages: {len(np.unique(lineage_numeric))}, unique latent values: {len(np.unique(latent_1d))})")
        return np.nan, np.nan, adata
    
    try:
        correlation, p_value = spearmanr(lineage_numeric, latent_1d)
        tau, _ = kendalltau(lineage_numeric, latent_1d)
        return abs(correlation), abs(tau), adata
    except Exception as e:
        print(f"Warning: Error computing correlation: {e}")
        return np.nan, np.nan, adata

def compute_clustering_metrics(adata, latent_key, path=None):
    """Fit GMM to latent embeddings and compute ARI and NMI with lineage."""
    if 'lineage_category' not in adata.obs.columns:
        return np.nan, np.nan
    
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
    gmm = GaussianMixture_sklearn(n_components=n_components, random_state=42)
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

def evaluate_negative_binomial(model, path, base_dir, result_dir, device="cpu", skip_training=False, losses_history=None):
    """
    Evaluate negative binomial model and compute metrics.
    
    Args:
        model: Trained model (can be None if skip_training=True)
        path: Path name
        base_dir: Base directory for data
        result_dir: Directory to save results
        device: Device to use
        skip_training: If True, skip model inference and use existing embeddings
        losses_history: Optional training history for plotting reconstruction loss
        
    Returns:
        metrics: Dictionary with abs_spearman, abs_kendall_tau, ari, nmi
    """
    data = anndata.read_h5ad(f"{base_dir}/scvi_path_{path}/trained.h5ad")
    latent_key = "Z_learned_nb"
    
    if skip_training:
        # Use existing embeddings
        if latent_key not in data.obsm:
            raise ValueError(f"No {latent_key} embeddings found in trained.h5ad")
        print(f"  Using existing {latent_key} embeddings")
    else:
        # Run model inference
        model.eval()
        X = torch.tensor(data.X.todense(), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            qz_x = model._define_variational_family(X.float().to(device))
            mu_q = qz_x.mean
        
        # Save latent embeddings to adata.obsm
        data.obsm[latent_key] = mu_q.detach().to("cpu").numpy()
        
        # Save updated data
        data.write_h5ad(f"{base_dir}/scvi_path_{path}/trained.h5ad")
    
    # Plot histogram
    plot_histogram(data, latent_key, path, result_dir)
    
    # Plot reconstruction loss if available
    if losses_history is not None:
        plot_reconstruction_loss(losses_history, path, result_dir)
    
    # Compute correlation with lineage
    correlation, tau, data = compute_correlation(data, latent_key)
    
    # Compute clustering metrics (ARI and NMI)
    ari, nmi = compute_clustering_metrics(data, latent_key, path=path)
    
    # Return only requested metrics
    metrics = {
        'path_name': path,
        'abs_spearman': correlation,
        'abs_kendall_tau': tau,
        'ari': ari,
        'nmi': nmi,
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train negative binomial VAE on all lineage paths")
    parser.add_argument("--remake-plots-only", action="store_true",
                       help="If set, skip training and only remake plots/evaluations from existing results")
    parser.add_argument("--remake-reconstruction-plots-only", action="store_true",
                       help="If set, only remake reconstruction loss plots from existing losses_history.csv files")
    args = parser.parse_args()
    
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    result_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/negative_binomial"
    paths_dict_file = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree_new.json.gz"
    
    os.makedirs(result_dir, exist_ok=True)
    
    paths_dict = load_paths_dict(paths_dict_file)
    print(f"Found {len(paths_dict)} paths to process")
    
    if args.remake_reconstruction_plots_only:
        print("Mode: Remake reconstruction loss plots only (skipping training and other plots)")
    
    device = "cpu"  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    
    for i, path in enumerate(sorted(paths_dict.keys())):
        # Check if already processed (check for metrics.csv file)
        metrics_path = os.path.join(result_dir, "plots", path, "metrics.csv")
        if not args.remake_plots_only and not args.remake_reconstruction_plots_only and os.path.exists(metrics_path):
            print(f"[{i+1}/{len(paths_dict)}] Skipping {path} (already processed)")
            continue
        
        print(f"\n[{i+1}/{len(paths_dict)}] Processing {path}...")
        
        try:
            losses_history = None
            if args.remake_reconstruction_plots_only:
                # Only remake reconstruction loss plots
                losses_path = os.path.join(result_dir, "plots", path, "losses_history.csv")
                if os.path.exists(losses_path):
                    losses_df = pd.read_csv(losses_path)
                    losses_history = losses_df.to_dict('records')
                    plot_reconstruction_loss(losses_history, path, result_dir)
                    print(f"✓ Successfully remade reconstruction loss plot for {path}")
                else:
                    print(f"  Warning: {losses_path} not found, skipping")
                continue
            elif args.remake_plots_only:
                # Load existing data and remake plots/evaluations
                data_path = os.path.join(base_dir, f"scvi_path_{path}", "trained.h5ad")
                if not os.path.exists(data_path):
                    print(f"  Warning: {data_path} not found, skipping")
                    continue
                
                data = anndata.read_h5ad(data_path)
                if 'Z_learned_nb' not in data.obsm:
                    print(f"  Warning: No 'Z_learned_nb' embeddings found in {data_path}, skipping")
                    continue
                
                # Try to load losses_history for plotting
                losses_path = os.path.join(result_dir, "plots", path, "losses_history.csv")
                if os.path.exists(losses_path):
                    losses_df = pd.read_csv(losses_path)
                    losses_history = losses_df.to_dict('records')
                
                print(f"  Remaking plots/evaluations for {path}...")
                # Evaluate without training
                metrics = evaluate_negative_binomial(None, path, base_dir, result_dir, device=device, skip_training=True, losses_history=losses_history)
            else:
                print(f"Training {path}...")
                model, losses_history = training_negative_binomial(path, base_dir, result_dir, device=device)
                
                # Evaluate
                metrics = evaluate_negative_binomial(model, path, base_dir, result_dir, device=device, losses_history=losses_history)
            
            print(f"Results for {path}:")
            print(f"  Abs Spearman: {metrics['abs_spearman']:.4f}")
            print(f"  Abs Kendall's tau: {metrics['abs_kendall_tau']:.4f}")
            print(f"  ARI: {metrics['ari']:.4f}")
            print(f"  NMI: {metrics['nmi']:.4f}")
            
            # Save per-path metrics
            per_path_dir = os.path.join(result_dir, "plots", path)
            os.makedirs(per_path_dir, exist_ok=True)
            pd.DataFrame([metrics]).to_csv(os.path.join(per_path_dir, 'metrics.csv'), index=False)
            
            rows.append(metrics)
            
        except Exception as e:
            print(f"ERROR processing {path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save aggregated results
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(result_dir, 'negative_binomial_results.csv'), index=False)
        print(f"\nSaved aggregated results to {result_dir}/negative_binomial_results.csv")
        print(f"Processed {len(rows)} paths successfully")
    else:
        print("\nNo paths were processed")
        

if __name__ == "__main__":
    main()