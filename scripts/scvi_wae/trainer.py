"""
Training and evaluation functions for SCVI-WAE model.
"""
import numpy as np
import scanpy as sc
import scvi
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import DataLoader, TensorDataset
import phate

from scvi_wae.clustering_utils import gmm_cluster_1d
from scvi_wae.losses import (
    mixture_uniform_reg,
    pairwise_distance_loss,
    wasserstein_distance_1d_learnable,
    wasserstein_distance_1d_mixture_sample,
)
from scvi_wae.path_utils import label_order_index


def train_and_eval(
    adata,
    device=None,
    # Model architecture parameters
    arches_params=None,
    # Prior training flags
    train_prior_means=True,
    train_prior_stds=True,
    train_prior_mix=False,
    # Training hyperparameters
    epochs=500,
    batch_size=128,
    kl_weight=0.0,
    wd_weight=10.0,
    dist_weight=1.0,
    tau_gumbel=1.0,
    mix_uniform_reg_weight=1.0,
    # Optimizer parameters
    lr=1e-3,
    weight_decay=1e-5,
    # PHATE parameters
    normalize_total_target_sum=1e4,
    scale_max_value=10,
    # Lineage column name
    lineage_key="lineage",
    batch_key="batch",
    # Verbose
    verbose=True,
    print_every=100,
):
    """
    Train and evaluate SCVI-WAE model on AnnData object.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with expression data and metadata
    device : torch.device, optional
        Device to run training on. If None, uses CUDA if available.
    arches_params : dict, optional
        Architecture parameters for SCVI model. Defaults to:
        {
            "use_layer_norm": "both",
            "use_batch_norm": "none",
            "encode_covariates": True,
            "dropout_rate": 0.2,
            "n_layers": 2,
            "n_hidden": 64,
            "n_latent": 1,
        }
    train_prior_means : bool
        Whether to train prior means
    train_prior_stds : bool
        Whether to train prior standard deviations
    train_prior_mix : bool
        Whether to train prior mixture weights
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    kl_weight : float
        Weight for KL divergence loss (if > 0)
    wd_weight : float
        Weight for Wasserstein distance loss
    dist_weight : float
        Weight for pairwise distance loss
    tau_gumbel : float
        Temperature for Gumbel-Softmax relaxation
    mix_uniform_reg_weight : float
        Weight for mixture uniform regularization
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer
    normalize_total_target_sum : float
        Target sum for normalization
    scale_max_value : float
        Maximum value for scaling
    lineage_key : str
        Key in adata.obs for lineage labels
    batch_key : str
        Key in adata.obs for batch labels
    verbose : bool
        Whether to print training progress
    print_every : int
        Print every N epochs
    
    Returns
    -------
    dict
        Dictionary containing:
        - "model": trained model
        - "vae": SCVI model wrapper
        - "centroids": learned centroids
        - "log_stds": learned log standard deviations
        - "mix_logits": learned mixture logits
        - "losses_history": list of loss dictionaries per epoch
        - "eval_results": dictionary with evaluation metrics
        - "lineage_label": lineage labels used for evaluation
        - "mu": latent representations
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if arches_params is None:
        arches_params = dict(
            use_layer_norm="both",
            use_batch_norm="none",
            encode_covariates=True,
            dropout_rate=0.2,
            n_layers=2,
            n_hidden=64,
            n_latent=1,
        )
    
    # Prepare data
    if hasattr(adata.X, "toarray"):
        X_np = adata.X.toarray()
    else:
        X_np = np.asarray(adata.X)
    
    count_mat_tensor = torch.from_numpy(X_np).float()
    N_tot = count_mat_tensor.size(0)
    unique_batches, batch_int = np.unique(np.asarray(adata.obs[batch_key].tolist()), return_inverse=True)
    batch_int_tensor = torch.from_numpy(batch_int).to(device).to(torch.int64).unsqueeze(-1)
    
    # Process lineage labels
    label_idx_dict, if_missing = label_order_index(adata.obs[lineage_key].unique().tolist())
    if verbose:
        print(label_idx_dict)
    lineage_label = adata.obs[lineage_key].apply(
        lambda x: label_idx_dict[x]
    ).tolist()
    n_clusters = len(np.unique(np.array(lineage_label)))
    
    # Compute PHATE to get pairwise distance matrix
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, target_sum=normalize_total_target_sum)
    sc.pp.log1p(adata_copy)
    sc.pp.scale(adata_copy, max_value=scale_max_value)
    sc.tl.pca(adata_copy, svd_solver="arpack")
    phate_operator = phate.PHATE()
    _ = phate_operator.fit_transform(adata_copy.obsm['X_pca'])
    pairwise_dist_np = squareform(pdist(phate_operator.diff_potential, "euclidean"))
    pairwise_dist_tensor = torch.from_numpy(pairwise_dist_np).to(device).float()
    
    # Initialize SCVI
    if hasattr(adata.X, "tocsr"):
        adata.layers["counts"] = adata.X.copy().tocsr()
    else:
        adata.layers["counts"] = adata.X.copy()
    
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    vae = scvi.model.SCVI(adata, **arches_params)
    model = vae.module.to(device)
    
    # Initialize data loader
    indices = torch.arange(N_tot)
    dataset = TensorDataset(count_mat_tensor, indices)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    # Initialize learnable prior parameters
    centroids = torch.nn.Parameter(torch.zeros(n_clusters, device=device))
    log_stds = torch.nn.Parameter(torch.zeros(n_clusters, device=device))
    mix_logits = torch.nn.Parameter(torch.zeros(n_clusters, device=device))
    
    centroids.requires_grad = train_prior_means
    log_stds.requires_grad = train_prior_stds
    mix_logits.requires_grad = train_prior_mix
    
    # Learnable scale for distance loss
    log_dist_scale = torch.nn.Parameter(torch.tensor(0.0, device=device))
    
    # Initialize optimizers
    opt_ae = torch.optim.Adam(
        list(model.parameters()) + [log_dist_scale],
        lr=lr,
        weight_decay=weight_decay,
    )
    
    prior_params = []
    if train_prior_means:
        prior_params.append(centroids)
    if train_prior_stds:
        prior_params.append(log_stds)
    if train_prior_mix:
        prior_params.append(mix_logits)
    
    opt_prior = (
        torch.optim.Adam(prior_params, lr=lr, weight_decay=weight_decay)
        if prior_params
        else None
    )
    
    def initialize_prior_from_encoder():
        """Initialize centroids / stds from aggregated posterior mean of current encoder."""
        model.eval()
        with torch.no_grad():
            X_tot = count_mat_tensor.to(device)
            N = X_tot.size(0)
            batch_all = torch.zeros((N, 1), device=device, dtype=torch.long)
            labels_all = torch.zeros((N, 1), device=device, dtype=torch.long)
            tensors_all = {"X": X_tot, "batch": batch_all, "labels": labels_all}
            inf_inputs_all = model._get_inference_input(tensors_all)
            inf_outputs_all = model._regular_inference(**inf_inputs_all)
            mu_q = inf_outputs_all["qz"].mean  # [N, 1]
            
            mu = mu_q.detach().flatten().cpu().numpy()
            mu_sort = np.sort(mu)
            
            means = []
            stds = []
            for k in range(n_clusters):
                start = int(k / n_clusters * len(mu_sort))
                end = int((k + 1) / n_clusters * len(mu_sort))
                data_batch = mu_sort[start:end]
                if data_batch.size == 0:
                    data_batch = mu_sort
                means.append(float(np.median(data_batch)))
                stds.append(float(np.std(data_batch)))
            
            means_t = torch.tensor(means, device=device, dtype=torch.float32)
            stds_t = torch.tensor(stds, device=device, dtype=torch.float32)
            stds_t = torch.clamp(stds_t, min=1e-3)
            
            # inverse softplus to initialize log_stds:
            # std = softplus(log_std)  =>  log_std = log(exp(std) - 1)
            log_stds_init = torch.log(torch.exp(stds_t) - 1.0)
            centroids.data.copy_(means_t)
            log_stds.data.copy_(log_stds_init)
        
        model.train()
    
    initialize_prior_from_encoder()
    
    # Training loop
    losses_history = []
    
    for ep in range(1, epochs + 1):
        model.train()
        epoch_ae = {"loss": 0.0, "recon_loss": 0.0, "wd": 0.0, "dist": 0.0}
        for xb, idx in dl:
            xb = xb.to(device).float()
            idx = idx.to(device)  # global indices for this batch
            
            bsz = xb.size(0)
            batch_tmp = batch_int_tensor[idx.long()]
            labels_tmp = torch.zeros((bsz, 1), device=device, dtype=torch.long)
            tensors = {"X": xb, "batch": batch_tmp, "labels": labels_tmp}
            
            opt_ae.zero_grad(set_to_none=True)
            inference_inputs = model._get_inference_input(tensors)
            inference_outputs = model._regular_inference(**inference_inputs)
            generative_inputs = model._get_generative_input(
                tensors=tensors, inference_outputs=inference_outputs
            )
            generative_outputs = model.generative(**generative_inputs)
            loss_info = model.loss(
                tensors=tensors,
                inference_outputs=inference_outputs,
                generative_outputs=generative_outputs,
                kl_weight=kl_weight,
            )
            recon_loss = loss_info.reconstruction_loss["reconstruction_loss"].mean()
            if kl_weight > 0.0:
                loss = loss_info.loss
            else:
                loss = recon_loss
            
            # Wasserstein distance to GMM prior
            if train_prior_mix:
                wd_ae = wasserstein_distance_1d_mixture_sample(
                    inference_outputs["z"],
                    centroids.detach(),
                    log_stds.detach(),
                    mix_logits.detach(),
                    tau=tau_gumbel,
                )
            else:
                wd_ae = wasserstein_distance_1d_learnable(
                    inference_outputs["z"],
                    centroids.detach(),
                    log_stds.detach(),
                )
            
            # Pairwise distance loss
            dist_scale = F.softplus(log_dist_scale) + 1e-6
            dist_loss = pairwise_distance_loss(
                inference_outputs["z"],
                idx,
                pairwise_dist_tensor,
                dist_scale=dist_scale,
                normalize=False,
            )
            
            # Total AE loss
            loss_ae = loss + wd_weight * wd_ae + dist_weight * dist_loss
            loss_ae.backward()
            opt_ae.step()
            
            epoch_ae["loss"] += loss_ae.item() * bsz
            epoch_ae["recon_loss"] += recon_loss.item() * bsz
            epoch_ae["wd"] += wd_ae.item() * bsz
            epoch_ae["dist"] += dist_loss.item() * bsz
        
        for k in epoch_ae:
            epoch_ae[k] /= float(N_tot)
        
        # Prior update step
        if opt_prior is not None and wd_weight > 0:
            model.eval()
            epoch_prior = {"loss": 0.0, "wd": 0.0}
            for xb, idx in dl:
                xb = xb.to(device).float()
                idx = idx.to(device)
                
                bsz = xb.size(0)
                batch_tmp = batch_int_tensor[idx.long()]
                labels_tmp = torch.zeros((bsz, 1), device=device, dtype=torch.long)
                tensors = {"X": xb, "batch": batch_tmp, "labels": labels_tmp}
                
                # compute z with encoder frozen
                with torch.no_grad():
                    inference_inputs = model._get_inference_input(tensors)
                    inference_outputs = model._regular_inference(**inference_inputs)
                    z_detached = inference_outputs["z"].detach()
                
                opt_prior.zero_grad(set_to_none=True)
                
                if train_prior_mix:
                    wd_prior = wasserstein_distance_1d_mixture_sample(
                        z_detached,
                        centroids,
                        log_stds,
                        mix_logits,
                        tau=tau_gumbel,
                    )
                else:
                    wd_prior = wasserstein_distance_1d_learnable(
                        z_detached,
                        centroids,
                        log_stds,
                    )
                # regularize mixture weights toward uniform (only matters if train_prior_mix=True)
                if mix_uniform_reg_weight > 0.0 and train_prior_mix:
                    reg_mix = mixture_uniform_reg(mix_logits)
                else:
                    reg_mix = z_detached.new_zeros(())
                # total prior loss
                loss_prior = wd_weight * wd_prior + mix_uniform_reg_weight * reg_mix
                loss_prior.backward()
                opt_prior.step()
                
                epoch_prior["loss"] += loss_prior.item() * bsz
                epoch_prior["wd"] += wd_prior.item() * bsz
            
            for k in epoch_prior:
                epoch_prior[k] /= float(N_tot)
        else:
            epoch_prior = None
        
        # Store losses
        if opt_prior is not None and wd_weight > 0:
            losses_history.append(
                {
                    "ae_loss": epoch_ae["loss"],
                    "ae_recon_loss": epoch_ae["recon_loss"],
                    "ae_wd": epoch_ae["wd"],
                    "ae_dist": epoch_ae["dist"],
                    "prior_loss": epoch_prior["loss"],
                    "prior_wd": epoch_prior["wd"],
                }
            )
        else:
            losses_history.append(
                {
                    "ae_loss": epoch_ae["loss"],
                    "ae_recon_loss": epoch_ae["recon_loss"],
                    "ae_wd": epoch_ae["wd"],
                    "ae_dist": epoch_ae["dist"],
                }
            )
        
        # Print progress
        if verbose and (ep == 1 or ep % print_every == 0 or ep == epochs):
            print(
                f"[{ep:03d}] "
                f"AE: loss={epoch_ae['loss']:.3f}, recon={epoch_ae['recon_loss']:.3f}, "
                f"wd={epoch_ae['wd']:.3f}, dist={epoch_ae['dist']:.3f}"
            )
            if opt_prior is not None and wd_weight > 0:
                print(
                    f"      Prior: loss={epoch_prior['loss']:.3f}, wd={epoch_prior['wd']:.3f}"
                )
    
    # After training evaluation
    model.eval()
    with torch.no_grad():
        X_tot = count_mat_tensor.to(device)
        labels_tmp = torch.zeros((len(X_tot), 1)).to(device)
        tensors = {"X": X_tot, "batch": batch_int_tensor.to(device), "labels": labels_tmp}
        inference_inputs = model._get_inference_input(tensors)
        inference_outputs = model._regular_inference(**inference_inputs)
        mu_q = inference_outputs["qz"].mean
        mu = mu_q.detach().flatten().to("cpu").numpy()
    
    # Compute evaluation metrics
    r, p = spearmanr(mu.reshape(-1), np.array(lineage_label))
    abs_r = abs(r)
    
    # Two different ways to compute NMI/ARI
    inbuilt_labels, inbuilt_resp, _ = gmm_cluster_1d(
        mu.reshape(-1),
        centroids.detach().to("cpu").numpy(),
        torch.exp(log_stds).detach().to("cpu").numpy(),
        torch.softmax(mix_logits, dim=0).detach().to("cpu").numpy()
    )
    nmi_inbuilt = normalized_mutual_info_score(
        np.array(lineage_label), inbuilt_labels, average_method="arithmetic"
    )
    ari_inbuilt = adjusted_rand_score(np.array(lineage_label), inbuilt_labels)
    
    model_gmm = GaussianMixture(
        n_components=n_clusters, covariance_type="full", random_state=0
    ).fit(mu.reshape(-1, 1))
    resp_gmm = model_gmm.predict_proba(mu.reshape(-1, 1))
    labels_gmm = model_gmm.predict(mu.reshape(-1, 1))
    nmi_gmm = normalized_mutual_info_score(
        np.array(lineage_label), labels_gmm, average_method="arithmetic"
    )
    ari_gmm = adjusted_rand_score(np.array(lineage_label), labels_gmm)
    
    eval_results = {
        "spearman_r": abs_r,
        "spearman_p": p,
        "nmi_inbuilt": nmi_inbuilt,
        "ari_inbuilt": ari_inbuilt,
        "nmi_gmm": nmi_gmm,
        "ari_gmm": ari_gmm,
    }
    
    if verbose:
        print(f"Spearman correlation: {abs_r:.4f}")
        print(f"Inbuilt GMM - NMI: {nmi_inbuilt:.4f}, ARI: {ari_inbuilt:.4f}")
        print(f"Sklearn GMM - NMI: {nmi_gmm:.4f}, ARI: {ari_gmm:.4f}")
    
    return {
        "model": model,
        "vae": vae,
        "centroids": centroids,
        "log_stds": log_stds,
        "mix_logits": mix_logits,
        "losses_history": losses_history,
        "eval_results": eval_results,
        "lineage_label": lineage_label,
        "mu": mu,
    }
