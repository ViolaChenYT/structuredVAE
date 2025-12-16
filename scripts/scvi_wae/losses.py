import math
import torch
import torch.nn.functional as F


def wasserstein_distance_1d_mixture_sample(
    encoded_samples: torch.Tensor,
    centroids: torch.Tensor,
    log_stds: torch.Tensor,
    mix_logits: torch.Tensor | None = None,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fast stochastic approximation of 1D W2^2 between:
      - empirical distribution of encoded_samples (uniform over batch),
      - Gaussian mixture prior with:
          means       = centroids [K]
          stds        = softplus(log_stds) [K]
          mix weights = softmax(mix_logits) if provided, else uniform.

    Strategy:
      - Draw B samples from the mixture prior using a Gumbel-Softmax relaxation
        (B = batch size).
      - Sort both sets and compute mean squared difference between order stats.

    Complexity: O(B K) + O(B log B), with B = batch_size, K = #components.
    """
    # flatten encoded_samples -> [B]
    z = encoded_samples.view(-1)
    B = z.size(0)
    device = z.device
    dtype = z.dtype

    K = centroids.shape[0]
    assert log_stds.shape[0] == K

    # stds > 0
    stds = F.softplus(log_stds) + eps  # [K]

    # mixture logits / probabilities
    if mix_logits is None:
        logits = torch.zeros(K, device=device, dtype=dtype)
    else:
        logits = mix_logits.to(device=device, dtype=dtype)

    # Gumbel-Softmax reparameterization for mixture assignment
    # sample Gumbel noise
    tiny = torch.finfo(dtype).tiny
    U = torch.rand(B, K, device=device, dtype=dtype)
    U = U.clamp(min=tiny, max=1.0 - tiny)
    g = -torch.log(-torch.log(U))  # [B, K]

    # relaxed one-hot weights for components, per sample
    y = F.softmax((logits.unsqueeze(0) + g) / tau, dim=-1)  # [B, K]

    # base Normal noise for each component/sample
    eps_base = torch.randn(B, K, device=device, dtype=dtype)  # [B, K]
    comp_samples = centroids.view(1, K) + stds.view(1, K) * eps_base  # [B, K]

    # mixture samples: convex combination of component samples
    prior_samples = (y * comp_samples).sum(dim=-1)  # [B]

    # 1D W2^2 approximation via sorted order statistics
    z_sorted, _ = torch.sort(z)
    prior_sorted, _ = torch.sort(prior_samples)
    w2 = torch.mean((z_sorted - prior_sorted) ** 2)
    return w2


def wasserstein_distance_1d_learnable(
    encoded_samples: torch.Tensor,
    centroids: torch.Tensor,
    log_stds: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    1D approximate W2 distance between:
      - empirical distribution of encoded_samples
      - Gaussian mixture with fixed centroids and learnable stds
    centroids: [K]
    log_stds: [K] (nn.Parameter); stds = softplus(log_stds)
    """
    # Flatten encoded_samples to [B, 1]
    z = encoded_samples.view(-1, 1)
    B = z.size(0)
    device = z.device
    dtype = z.dtype
    K = centroids.shape[0]
    assert log_stds.shape[0] == K

    # positive stds
    stds = F.softplus(log_stds) + eps  # [K]

    # --- Build deterministic "prior samples" ---
    # Use m quantiles from N(0,1) per component, then shift/scale
    m = math.ceil(B / K)
    u = (torch.arange(m, device=device, dtype=dtype) + 0.5) / m  # [m]
    base = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    ).icdf(u)  # [m]
    base = base.view(m, 1)  # [m, 1]

    prior_samples_list = []
    for k in range(K):
        mu_k = centroids[k]         # scalar
        std_k = stds[k]             # scalar
        prior_k = mu_k + std_k * base  # [m, 1]
        prior_samples_list.append(prior_k)

    prior_samples = torch.cat(prior_samples_list, dim=0)  # [K*m, 1]

    # Match number of samples to B
    if prior_samples.size(0) > B:
        prior_samples = prior_samples[:B]
    elif prior_samples.size(0) < B:
        repeat_times = math.ceil(B / prior_samples.size(0))
        prior_samples = prior_samples.repeat(repeat_times, 1)[:B]

    # --- Sort both and compute 1D W2 (MSE between quantiles) ---
    z_sorted, _ = torch.sort(z, dim=0)
    prior_sorted, _ = torch.sort(prior_samples, dim=0)
    wd = F.mse_loss(z_sorted, prior_sorted, reduction="sum") / B
    return wd

def sliced_wasserstein_2d_mixture(z, centroids, log_stds, mix_logits, n_projections=50):
    """
    Approximates 2D Wasserstein distance by projecting onto random 1D lines.
    """
    device = z.device
    batch_size = z.size(0)
    
    # 1. Sample from the GMM Prior
    # We need to generate a 'fake' batch from the prior to compare against
    z_prior = sample_gmm_2d(batch_size, centroids, log_stds, mix_logits)
    
    # 2. Generate Random Projections (theta)
    # Random vectors on the unit circle
    theta = torch.randn(n_projections, 2, device=device)
    theta = theta / torch.norm(theta, dim=1, keepdim=True) # Normalize
    
    # 3. Project both distributions
    # shapes: [Batch, 2] @ [2, Projections] -> [Batch, Projections]
    proj_z = z @ theta.t()
    proj_prior = z_prior @ theta.t()
    
    # 4. Sort and Compute L2 distance (1D Wasserstein)
    proj_z_sorted, _ = torch.sort(proj_z, dim=0)
    proj_prior_sorted, _ = torch.sort(proj_prior, dim=0)
    
    # Average over batch and projections
    wd = torch.mean((proj_z_sorted - proj_prior_sorted) ** 2)
    return wd

def sample_gmm_2d(batch_size, centroids, log_stds, mix_logits):
    """Helper to sample from differentiable GMM parameters"""
    # 1. Sample Cluster Assignments (Gumbel-Softmax or Categorical)
    probs = torch.softmax(mix_logits, dim=0)
    # For sampling, we can just use multinomial since we don't differentiate 
    # through the discrete choice, only the resulting Gaussian parameters
    indices = torch.multinomial(probs, batch_size, replacement=True)
    
    # 2. Gather Mean/Std for each sample
    mu = centroids[indices] # [Batch, 2]
    std = torch.exp(log_stds[indices]) # [Batch, 2] (assuming diagonal cov)
    
    # 3. Reparameterization Trick
    eps = torch.randn_like(mu)
    return mu + eps * std

def sample_gmm_2d(batch_size, centroids, log_stds, mix_logits):
    """Helper to sample from differentiable GMM parameters"""
    # 1. Sample Cluster Assignments (Gumbel-Softmax or Categorical)
    probs = torch.softmax(mix_logits, dim=0)
    # For sampling, we can just use multinomial since we don't differentiate 
    # through the discrete choice, only the resulting Gaussian parameters
    indices = torch.multinomial(probs, batch_size, replacement=True)
    
    # 2. Gather Mean/Std for each sample
    mu = centroids[indices] # [Batch, 2]
    std = torch.exp(log_stds[indices]) # [Batch, 2] (assuming diagonal cov)
    
    # 3. Reparameterization Trick
    eps = torch.randn_like(mu)
    return mu + eps * std
    
def pairwise_distance_loss(
    z: torch.Tensor,
    batch_indices: torch.Tensor,
    dist_mat: torch.Tensor,
    dist_scale: torch.Tensor = None,
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    z: [B, latent_dim]
    batch_indices: [B] global indices of the cells in this batch
    dist_mat: [N, N] precomputed pairwise distances between cells (no grad)
    dist_scale: scalar multiplier applied to dist_mat (learnable)
    """
    z_flat = z.view(z.size(0), -1)  # [B, latent_dim]

    # pairwise Euclidean distances in latent space
    d_latent = torch.cdist(z_flat, z_flat, p=2)  # [B, B]

    # corresponding target distances for this batch
    idx = batch_indices.long()
    d_target = dist_mat[idx][:, idx]  # [B, B]

    # apply learnable scaling
    if dist_scale is not None:
        d_target = d_target * dist_scale

    if normalize:
        d_latent = d_latent / (d_latent.mean() + eps)
        d_target = d_target / (d_target.mean() + eps)

    return F.mse_loss(d_latent, d_target)


def mixture_uniform_reg(mix_logits: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviation of mixture weights π from uniform.
    π = softmax(mix_logits); u_k = 1/K.

    Returns sum_k (π_k - 1/K)^2.
    """
    pi = torch.softmax(mix_logits, dim=0)
    K = pi.size(0)
    uniform = pi.new_full((K,), 1.0 / K)
    return torch.sum((pi - uniform) ** 2)

def dirichlet_prior_loss(probs, alpha=2.0):
    """
    Penalizes probabilities that become too small.
    alpha > 1.0 encourages uniform-ish distribution but allows variance.
    alpha = 1.0 is uniform (no penalty).
    alpha < 1.0 encourages sparsity.
    """
    # Probs is softmax(logits)
    # Add epsilon for numerical stability
    return -torch.sum((alpha - 1) * torch.log(probs + 1e-6))