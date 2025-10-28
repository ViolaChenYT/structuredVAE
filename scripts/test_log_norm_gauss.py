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
from src.models import *
from src.priors import *
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy,spearmanr
from sklearn.mixture import GaussianMixture as GaussianMixture_sklearn
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import fcntl
import scanpy as sc
def safe_read_h5ad(filepath):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with open(filepath, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                return anndata.read_h5ad(filepath)
        except (IOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise IOError(f"Failed to read {filepath} after {max_retries} attempts")

def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)
def scanpy_norm_log1p_from_torch(X: torch.Tensor) -> torch.Tensor:
    # move to CPU + numpy (Scanpy expects numpy/scipy)
    X_np = X.detach().cpu().numpy().astype(np.float32, copy=False)

    adata = sc.AnnData(X_np)                 # create AnnData
    sc.pp.normalize_total(adata, target_sum=None, inplace=True)  # Scanpy normalize_total
    sc.pp.log1p(adata)                       # Scanpy log1p (natural log)

    # back to torch, preserve original device & dtype
    X_out = torch.from_numpy(adata.X).to(X.device).type_as(X)
    return X_out 
def training_log_norm_gauss(path_name, base_dir,device="cpu",batch_size=128,lr=1e-3,weight_decay=1e-5,early_stopping=True,patience=200,epochs=700,min_delta=1e-4):
    data = anndata.read_loom(f"{base_dir}/{path_name}.loom")
    n_components = len(set(data.obs['lineage']))
    model_prior = GaussianMixture(latent_dim=1, num_clusters=n_components)
    model_encoder = build_encoder(dim_x=2000, h_dim=64, n_layers=2)
    model_decoder = build_decoder_gaussian(dim_x=2000, latent_dim=1, h_dim=64, n_layers=2)
    model = EmpiricalBayesVariationalAutoencoder(encoder=model_encoder, enc_out_dim=64, decoder=model_decoder, prior=model_prior).to(device)
    # Convert sparse matrix to dense and then to tensor
    X_dense = torch.tensor(data.X.todense(), dtype=torch.float32)
    X_dense = scanpy_norm_log1p_from_torch(X_dense)
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
                print(f"Warning: xb.size(0) == 1, skipping batch")
                continue
            xb = xb.to(device).float()
            loss, _ = model.variational_inference_step(xb, opt)
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
    
    return model, losses_history

def plot_histogram(adata,latent_key, path, result_dir="/n/fs/ragr-data/users/viola/structuredVAE/results/plots/"):
    labels = adata.obs["lineage"]
    unique_items = set(labels)
    unique_lineages = sorted(unique_items, key=lambda s: len(s.split('/')[0]))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_lineages)))
    for i, lineage in enumerate(unique_lineages):
        lineage_coords = adata.obsm[latent_key][adata.obs["lineage"] == lineage]
        if len(lineage_coords) > 0:
            plt.hist(lineage_coords, bins=50, alpha=0.5, label=lineage, color=colors[i], density=True)
    plt.xlabel("Z")
    plt.ylabel("Frequency")
    plt.title("Histogram of GMMVAE Z by lineage(log-norm-gauss)")
    plt.legend(title="lineage")
    plt.grid(True, alpha=0.3)
    if not os.path.exists(f"{result_dir}/{path}"):
        os.makedirs(f"{result_dir}/{path}")
    plt.savefig(f"{result_dir}/{path}/log_norm_gauss_GMM_histogram.png")
    plt.close()
    return adata

def compute_correlation(adata, latent_key, labels):
    new_key = "lineage_category"
    if 'lineage' in adata.obs.columns:
        adata.obs[new_key] = adata.obs['lineage'].apply(
            lambda x: len(x.split("/")[0]) if "/" in x else len(x)
        )
    else:
        print(f"Warning: No lineage column found in")
        return None, None
    lineage_numeric = pd.to_numeric(adata.obs[new_key])
    correlation, p_value = spearmanr(lineage_numeric, adata.obsm[latent_key])
    return abs(correlation), p_value, adata

def fit_gmm_and_analyze(adata, latent_key):
    lineage_labels = pd.to_numeric(adata.obs['lineage_category']).values
    n_components = len(set(lineage_labels))
    gmm = GaussianMixture_sklearn(n_components=n_components, random_state=42)
    latent_data = adata.obsm[latent_key].reshape(-1, 1)
    gmm.fit(latent_data)
    probabilities = gmm.predict_proba(latent_data)
    point_entropies = [entropy(probs) for probs in probabilities]
    entropy_value = np.mean(point_entropies)
    ari_with_lineage=np.nan
    
    cluster_assignments = gmm.predict(latent_data)
    mixture_proportion_metrics = {}
    if adata is not None and 'lineage_category' in adata.obs.columns:
        lineage_labels = pd.to_numeric(adata.obs['lineage_category']).values
        ari_with_lineage = adjusted_rand_score(cluster_assignments, lineage_labels)
        gmm_proportions = gmm.weights_
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
    gmm_metrics = {
        'entropy': entropy_value,
        'bic': gmm.bic(latent_data),
        'aic': gmm.aic(latent_data),
        'log_likelihood': gmm.score(latent_data),
        'perplexity': np.exp(-gmm.score(latent_data) / len(latent_data)),
        'silhouette': silhouette_score(latent_data, gmm.predict(latent_data)),
        'ari_with_lineage': ari_with_lineage,
    }
    return gmm_metrics, mixture_proportion_metrics

def evaluate_log_norm_gauss(model, path, base_dir,device="cpu"):
    model.eval()
    data = safe_read_h5ad(f"{base_dir}/scvi_path_{path}/trained.h5ad")
    y = data.obs["lineage"]
    X = torch.tensor(data.X.todense(), dtype=torch.float32).to(device)
    with torch.no_grad():
        qz_x = model._define_variational_family(X.float().to(device))
        mu_q = qz_x.mean
        Z_learned = qz_x.sample()
        Z_learned = Z_learned.detach().flatten().to("cpu").numpy()
    latent_key = "Z_learned_log_norm_gauss"
    data.obsm[latent_key] = mu_q.detach().to("cpu").numpy()
    data.write_h5ad(f"{base_dir}/scvi_path_{path}/trained.h5ad")
    labels = y.detach().to("cpu").numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    plot_histogram(data, latent_key, path)
    correlation, p_value, data = compute_correlation(data, latent_key, labels)
    gmm_metrics, mixture_proportion_metrics = fit_gmm_and_analyze(data, latent_key)
    return mu_q, Z_learned, labels, correlation, p_value, gmm_metrics, mixture_proportion_metrics

def main():
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    result_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/"
    paths_dict_file = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree.json.gz"
    paths_dict = load_paths_dict(paths_dict_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    cnt = 0
    for i, path in enumerate(sorted(paths_dict.keys())):
        # if i % 2 == 1:
        if os.path.exists(f"{result_dir}/{path}/log_norm_gauss_GMM_histogram.png"):
            continue
        print(f"Training {path}...")
        model, losses_history = training_log_norm_gauss(path, base_dir,device=device)
        mu_q, z, labels, rho, pval, gmm_metrics, mixture_metrics = evaluate_log_norm_gauss(model, path, base_dir,device=device)
        row = {"path_name":path}
        print(rho, gmm_metrics)
        row.update({f"{k}":(v.item() if hasattr(v, 'item') else v) for k, v in gmm_metrics.items()})        
        row.update({f"{k}":(v.item() if hasattr(v, 'item') else v) for k, v in mixture_metrics.items()})
        row.update({"correlation":rho, "p_value":pval})
        rows.append(row)
        # cnt += 1
        # if cnt > 10:
        #     break
        
    df = pd.DataFrame(rows)
    df.to_csv(f"{result_dir}/log_norm_gauss_results.csv", index=False)
        

if __name__ == "__main__":
    main()